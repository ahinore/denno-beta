#!/usr/bin/env python3
"""P4-1 (s103): scripted 10-turn conversation stability check (repetition / template collapse).

使い方 (最小):
  python3 scripts/long_conversation_check.py --dry-run

  何も起動せず、何も送らずに 10 ターン分のリクエスト本文と計画だけを表示する。

使い方 (実サーバに対して):
  python3 scripts/long_conversation_check.py --base-url http://127.0.0.1:8095 \\
      --out /tmp/lc_s103.json

  既に動いている Anthropic 形式の /v1/messages に 10 ターンを順に投げる。
  各ターンは「前の assistant 応答 + 次の user 発話」を積んで送るので会話は実際に伸びる。
  このスクリプト自身はサーバを起動しないし停止もしない。

使い方 (その他):
  # GPU もサーバも無しで、検出器と exit code 対応だけを検算する
  python3 scripts/long_conversation_check.py --self-test

  # stdlib unittest のテスト一式 (偽 HTTP サーバを ephemeral port で立てる)
  python3 scripts/test_long_conversation_check.py -v

なぜ要るのか:
  元の不具合報告 (「数ターンでモデルが崩れる」) は temperature = 1.0 が原因で、
  serve の --temperature-cap 0.7 で緩和された。しかしこの修正を守る再現可能な
  複数ターンの回帰テストが無かった。ここで崩れ方 (同一文の繰り返し・
  chat template の断片露出・偽の user ターンの混入) を機械的に検出して固定する。

決定性について:
  このチェックは「複数回の実行で同じ文章が返ること」を要求しない (サーバは標本化する)。
  決定的なのは「同じトランスクリプト JSON からは常に同じ判定が出ること」。
  検出器は文字列だけを見る純関数なので、この性質が成り立つ。--out で保存した
  トランスクリプトは後から判定し直せる (transcript_verdict())。

検出器 (各々独立したフラグを持ち、ターン番号と根拠の抜粋を報告する):
  repeat_line   同じ >= 24 文字の行が 3 回以上連続している
  repeat_sentence 同じ末尾文が 3 回以上連続している
  long_line     400 文字超の応答で、1 行が全体の 60 % 以上を占める
  template_marker  chat template の印 (user / assistant / system /
                <|im_start|> / <|im_end|> / <|endoftext|>) が行頭に単独で現れる
  fake_turn     回答の後に「\n\nuser\n」形式の偽の user 発話が続いている
  max_tokens    stop_reason == "max_tokens" (報告のみ・失敗にしない)
  empty         応答が空、または空白だけ

exit code:
  0  全ターンで (max_tokens を除く) フラグが立たなかった
  2  少なくとも 1 ターンでハードな失敗 (repeat_* / long_line / template_marker /
     fake_turn / empty)。(turn, flag, 根拠の抜粋) の表を出す
  3  インフラの問題 (接続拒否・HTTP 200 以外・本文が壊れている・fixture が読めない)

標準ライブラリのみ。ネットワークは urllib だけ。eval は使わない。
"""

import argparse
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

DEFAULT_BASE_URL = "http://127.0.0.1:8095"
DEFAULT_MODEL = "local/qwen35-9b"
DEFAULT_TURNS = 10
DEFAULT_MAX_TOKENS = 512
DEFAULT_FIXTURE = os.path.join(HERE, "fixtures", "long_conversation_turns.json")
DEFAULT_TIMEOUT_S = 300.0

# 8090 = 本番 (触らない) / 8080 = serve の既定 (触らない)。cold_ab.py と同じ扱い。
FORBIDDEN_PORTS = (8080, 8090)

# 検出器のしきい値 (cold_ab.py のように定数で名前を付けて自己テストから参照する)
MIN_REPEAT_LINE_CHARS = 24
REPEAT_RUN = 3
LONG_REPLY_CHARS = 400
DOMINANT_LINE_FRACTION = 0.60
EVIDENCE_CHARS = 60

TEMPLATE_MARKERS = ("user", "assistant", "system",
                    "<|im_start|>", "<|im_end|>", "<|endoftext|>")

# ハード失敗 (exit 2) と、報告だけのフラグ (exit 0) を分ける。
HARD_FLAGS = ("repeat_line", "repeat_sentence", "long_line", "template_marker",
              "fake_turn", "empty")
SOFT_FLAGS = ("max_tokens",)

# 偽の user ターン: 改行 2 つ + user + 改行 の後に本文がある
FAKE_TURN_RE = re.compile(r"\n[ \t]*\n[ \t]*user[ \t]*\n[ \t]*\S", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 純関数: 検出器 (すべてネットワーク非依存 = 自己テスト対象)
# ---------------------------------------------------------------------------

def split_lines(text):
    """行に割る。末尾の空行は捨てる (改行の数で判定がぶれないように)。"""
    return [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]


def content_lines(text):
    """空行を除いた行 (繰り返し判定は空行を無視する)。"""
    return [ln for ln in split_lines(text) if ln.strip()]


def normalize_line(line):
    """比較用に前後の空白と連続空白を潰す。表示は元の行のまま。"""
    return " ".join(line.split())


def longest_run(lines):
    """同じ行が連続する最大回数と、その行を返す。空なら (0, "")."""
    best_n, best_line = 0, ""
    prev, run = None, 0
    for ln in lines:
        key = normalize_line(ln)
        if key and key == prev:
            run += 1
        else:
            prev, run = key, 1
        if key and run > best_n:
            best_n, best_line = run, ln
    return best_n, best_line


def sentence_parts(text):
    """文に割る。日本語の句点と英語のピリオド/感嘆/疑問符、改行も区切りにする。

    句点は記号自体を文末に残す (「です。」と「です」を別文にしない)。
    """
    parts = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in "。．.!?！？\n":
            s = "".join(buf).strip()
            if s:
                parts.append(s)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def trailing_sentence_run(text):
    """末尾で同じ文が連続する回数と、その文を返す。空なら (0, "")."""
    parts = [normalize_line(p) for p in sentence_parts(text)]
    parts = [p for p in parts if p]
    if not parts:
        return 0, ""
    last = parts[-1]
    n = 0
    for p in reversed(parts):
        if p != last:
            break
        n += 1
    # 元の文 (正規化前) を根拠として返したいので、末尾から同じ長さだけ拾い直す
    raw = [p for p in sentence_parts(text) if p.strip()]
    return n, (raw[-1] if raw else "")


def dominant_line(text):
    """1 行が本文全体に占める割合が最大の行と割合を返す。"""
    total = len(text)
    if total == 0:
        return "", 0.0
    lines = content_lines(text)
    if not lines:
        return "", 0.0
    best = max(lines, key=lambda ln: len(ln))
    return best, len(best) / float(total)


def template_marker_hits(text):
    """行頭に単独で現れる chat template 印を列挙する (重複なし・出現順)。"""
    hits = []
    for ln in split_lines(text):
        s = ln.strip()
        low = s.lower()
        for m in TEMPLATE_MARKERS:
            if low == m and m not in hits:
                hits.append(m)
    return hits


def fake_turn_hit(text):
    """回答の後に偽の user 発話が続いていれば、その一致文字列を返す。"""
    # 「行頭に単独で user」は template_marker 側でも拾うが、こちらは本文が
    # 続いていることまで要求する (偽ターン)。両方立つのは正しい挙動。
    m = FAKE_TURN_RE.search(text)
    return m.group(0) if m else ""


def excerpt(text, limit=EVIDENCE_CHARS):
    """根拠の抜粋。1 行に畳んで表に載せられる形にする。"""
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "..."


def detect(reply):
    """1 応答を検査し、フラグのリスト [(flag, evidence)] を返す。

    同じフラグは 1 回だけ。順序は HARD_FLAGS の定義順 (報告を安定させる)。
    """
    flags = {}
    text = reply if isinstance(reply, str) else ""

    if not text.strip():
        # 空応答は他を見るまでもない
        return [("empty", "(whitespace only, %d bytes)" % len(text))]

    n_line, line = longest_run(content_lines(text))
    if n_line >= REPEAT_RUN and len(normalize_line(line)) >= MIN_REPEAT_LINE_CHARS:
        flags["repeat_line"] = "%dx %s" % (n_line, excerpt(line))

    n_sent, sent = trailing_sentence_run(text)
    if n_sent >= REPEAT_RUN:
        flags["repeat_sentence"] = "%dx %s" % (n_sent, excerpt(sent))

    dom, frac = dominant_line(text)
    if len(text) >= LONG_REPLY_CHARS and frac >= DOMINANT_LINE_FRACTION:
        flags["long_line"] = "%d/%d chars (%.0f%%) %s" % (
            len(dom), len(text), frac * 100.0, excerpt(dom, 40))

    for m in template_marker_hits(text):
        flags.setdefault("template_marker", "line == %r" % m)
        break

    hit = fake_turn_hit(text)
    if hit:
        flags["fake_turn"] = repr(excerpt(hit))

    out = []
    for name in HARD_FLAGS:
        if name in flags:
            out.append((name, flags[name]))
    return out


def detect_turn(turn):
    """transcript の 1 ターンから [(flag, evidence)] を返す (max_tokens を含む)。"""
    found = detect(turn.get("reply", ""))
    for name in SOFT_FLAGS:
        if name == "max_tokens" and turn.get("stop_reason") == "max_tokens":
            found.append((name, "stop_reason=max_tokens"))
    return found


def turn_is_hard(turn):
    """そのターンにハード失敗のフラグが立っているか。"""
    return any(flag in HARD_FLAGS for flag, _ in detect_turn(turn))


def transcript_verdict(transcript):
    """トランスクリプト全体の判定。同じ JSON からは常に同じ結果を返す (純関数)。

    返り値:
      {"clean": bool,            # ハード失敗が 0 件
       "failures": [{"turn": int, "flag": str, "evidence": str}, ...],
       "soft": [同じ形],          # max_tokens など報告だけのもの
       "turn_count": int,
       "exit_code": 0 | 2}
    """
    failures, soft = [], []
    turns = transcript.get("turns", []) if isinstance(transcript, dict) else []
    for turn in turns:
        idx = turn.get("index")
        for flag, evidence in detect_turn(turn):
            row = {"turn": idx, "flag": flag, "evidence": evidence}
            if flag in HARD_FLAGS:
                failures.append(row)
            else:
                soft.append(row)
    return {
        "clean": not failures,
        "failures": failures,
        "soft": soft,
        "turn_count": len(turns),
        "exit_code": 0 if not failures else 2,
    }


# ---------------------------------------------------------------------------
# fixture と計画 (純関数)
# ---------------------------------------------------------------------------

class FixtureError(Exception):
    """fixture が読めない / 形が違う (exit 3 で扱う)。"""


def load_fixture(path):
    """10 ターン分の user 文を読み、index 順の文字列リストにして返す。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise FixtureError("cannot read fixture %s: %s" % (path, e))
    except ValueError as e:
        raise FixtureError("fixture is not valid JSON (%s): %s" % (path, e))

    turns = data.get("turns") if isinstance(data, dict) else None
    if not isinstance(turns, list) or not turns:
        raise FixtureError("fixture has no non-empty 'turns' list: %s" % path)

    prompts = []
    for pos, t in enumerate(turns, 1):
        if not isinstance(t, dict):
            raise FixtureError("fixture turn #%d is not an object" % pos)
        text = t.get("text")
        if not isinstance(text, str) or not text.strip():
            raise FixtureError("fixture turn #%d has no non-empty 'text'" % pos)
        want = t.get("index", pos)
        if want != pos:
            raise FixtureError("fixture turn #%d has index %r (must be %d in order)"
                               % (pos, want, pos))
        prompts.append(text.strip())
    return prompts


def plan_turns(prompts, turns):
    """先頭 turns 個の user 文を返す。足りなければその事実を明示して中断する。"""
    if turns < 1:
        raise ValueError("turns must be >= 1")
    if turns > len(prompts):
        raise FixtureError("requested %d turns but the fixture only has %d prompts"
                           % (turns, len(prompts)))
    return list(prompts[:turns])


def build_messages(history, user_text):
    """これまでの (role, content) に次の user 発話を足した messages を作る。

    返り値は新しいリスト (呼び側の history は書き換えない)。
    """
    msgs = [{"role": r, "content": c} for r, c in history]
    msgs.append({"role": "user", "content": user_text})
    # Anthropic 形式は user / assistant の交互を要求する。assistant 応答が空だった
    # 場合もここでは空文字のまま積む (サーバ側のエラーは exit 3 で扱う)。
    return msgs


def build_request_body(model, messages, max_tokens):
    """Anthropic Messages API のリクエスト本文。

    temperature は送らない (cap はサーバ側の serve 設定が持つ。ここで上書きすると
    「cap が効いているか」を測れなくなる)。
    """
    return {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }


def response_text(body):
    """Anthropic 形式の応答から本文テキストを連結して返す。

    content は [{"type": "text", "text": "..."}] の配列。text 以外のブロック
    (thinking 等) は判定対象にしない (崩れの検出は可視テキストが対象)。
    """
    if not isinstance(body, dict):
        raise ValueError("response body is not a JSON object")
    content = body.get("content")
    if not isinstance(content, list):
        raise ValueError("response body has no 'content' list")
    texts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text")
            if isinstance(t, str):
                texts.append(t)
    return "".join(texts)


# ---------------------------------------------------------------------------
# 安全装置
# ---------------------------------------------------------------------------

def url_port(base_url):
    """base_url からポートを取る (省略時は scheme の既定)。"""
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("--base-url must start with http:// or https://")
    if not parsed.hostname:
        raise ValueError("--base-url has no host: %s" % base_url)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("--base-url has an invalid port: %s" % base_url)
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return port


def validate_args(args):
    """送信前に落とすべき条件を検査し、エラーメッセージのリストを返す。"""
    errors = []
    if args.turns < 1:
        errors.append("--turns must be >= 1 (got %d)" % args.turns)
    if args.max_tokens < 1:
        errors.append("--max-tokens must be >= 1 (got %d)" % args.max_tokens)
    if args.timeout_s <= 0:
        errors.append("--timeout-s must be > 0 (got %s)" % args.timeout_s)
    try:
        port = url_port(args.base_url)
    except ValueError as e:
        errors.append(str(e))
    else:
        if port in FORBIDDEN_PORTS:
            errors.append("--base-url port %d is reserved (8080 = serve default, "
                          "8090 = production); do not touch it" % port)
    return errors


def messages_url(base_url):
    """base_url + /v1/messages。末尾スラッシュと二重スラッシュを正す。"""
    return base_url.rstrip("/") + "/v1/messages"


# ---------------------------------------------------------------------------
# 実行 (ここから下はネットワークに出る。self-test からは呼ばない)
# ---------------------------------------------------------------------------

class InfraError(Exception):
    """接続拒否 / HTTP 200 以外 / 本文が壊れている (exit 3)。"""


def post_messages(url, body, timeout_s):
    """1 リクエストを送り、パース済み JSON を返す。失敗はすべて InfraError。"""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"content-type": "application/json",
                                          "accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = resp.getcode()
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:  # 本文が読めなくても status は伝える
            detail = ""
        raise InfraError("HTTP %d from %s%s" % (e.code, url,
                                                (": " + detail) if detail else ""))
    except urllib.error.URLError as e:
        raise InfraError("cannot reach %s: %s" % (url, e.reason))
    except socket.timeout:
        raise InfraError("timeout after %.1fs waiting for %s" % (timeout_s, url))
    except OSError as e:
        raise InfraError("cannot reach %s: %s" % (url, e))
    if status != 200:
        raise InfraError("HTTP %d from %s" % (status, url))
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise InfraError("malformed response body from %s: %s" % (url, e))


def run_conversation(base_url, prompts, model=DEFAULT_MODEL, max_tokens=DEFAULT_MAX_TOKENS,
                     timeout_s=DEFAULT_TIMEOUT_S, seed_note="", log=None):
    """turn 数だけ会話を回し、transcript dict を返す。turn の形:

      {"index": 1-based, "user": 送った文, "reply": 本文, "stop_reason": str,
       "usage": {...} | None, "duration_ms": int}

    history は assistant の応答を積んで伸びる。したがって turn N の
    messages 長は 2N-1 (user/assistant 交互 + 最後の user)。
    """
    url = messages_url(base_url)
    history = []
    turns = []
    for i, text in enumerate(prompts, 1):
        messages = build_messages(history, text)
        body = build_request_body(model, messages, max_tokens)
        if log:
            log("turn %d/%d: %d messages, %d chars of user text"
                % (i, len(prompts), len(messages), len(text)))
        started = _now_ms()
        resp = post_messages(url, body, timeout_s)
        elapsed = _now_ms() - started
        try:
            reply = response_text(resp)
        except ValueError as e:
            raise InfraError("turn %d: %s" % (i, e))
        stop_reason = resp.get("stop_reason")
        turns.append({
            "index": i,
            "user": text,
            "reply": reply,
            "stop_reason": stop_reason,
            "usage": resp.get("usage") if isinstance(resp.get("usage"), dict) else None,
            "duration_ms": elapsed,
        })
        history.append(("user", text))
        history.append(("assistant", reply))
    return {
        "schema": "denno.long_conversation.transcript/1",
        "base_url": base_url,
        "model": model,
        "max_tokens": max_tokens,
        "seed_note": seed_note,
        "turn_count": len(turns),
        "turns": turns,
    }


def _now_ms():
    import time
    return int(round(time.time() * 1000.0))


# ---------------------------------------------------------------------------
# 表示
# ---------------------------------------------------------------------------

def format_failure_table(result):
    """(turn, flag, evidence) の表。落ちたときだけ出す。"""
    rows = result.get("failures", []) + result.get("soft", [])
    if not rows:
        return []
    head = ("  %-6s %-16s %s" % ("turn", "flag", "evidence"),
            "  " + "-" * 6 + " " + "-" * 16 + " " + "-" * 40)
    out = list(head)
    for r in rows:
        out.append("  %-6s %-16s %s" % (str(r["turn"]), r["flag"], r["evidence"]))
    return out


def print_report(transcript, result, out=sys.stdout):
    print("conversation check: %d turns, base_url=%s, model=%s"
          % (result["turn_count"], transcript.get("base_url"),
             transcript.get("model")), file=out)
    if transcript.get("seed_note"):
        print("seed note: %s" % transcript["seed_note"], file=out)
    for turn in transcript.get("turns", []):
        flags = [f for f, _ in detect_turn(turn)]
        mark = "OK  " if not flags else "FLAG"
        print("  turn %2d  %s  %5d chars  stop=%s  flags=%s"
              % (turn.get("index", 0), mark, len(turn.get("reply", "")),
                 turn.get("stop_reason"), ",".join(flags) if flags else "-"), file=out)
    rows = format_failure_table(result)
    if rows:
        print("", file=out)
        print("flags (hard = exit 2, soft = reported only):", file=out)
        for line in rows:
            print(line, file=out)
    if result["clean"]:
        print("RESULT: CLEAN (%d turns, no hard failure)" % result["turn_count"], file=out)
    else:
        print("RESULT: FAILED (%d hard flag(s) in %d turns)"
              % (len(result["failures"]), result["turn_count"]), file=out)


def build_transcript(args, prompts):
    """仮のトランスクリプト (dry-run 用)。本文は空で、送る予定だけを持つ。"""
    history = []
    turns = []
    for i, text in enumerate(prompts, 1):
        messages = build_messages(history, text)
        turns.append({
            "index": i,
            "user": text,
            "request_messages": len(messages),
            "request_body": build_request_body(args.model, messages, args.max_tokens),
        })
        history.append(("user", text))
        history.append(("assistant", ""))
    return turns


def run_dry(args, prompts, out=sys.stdout):
    """何も送らず、ターンごとのリクエスト本文を出す (本文は 1 行 JSON)。"""
    print("DRY RUN: nothing is sent, no server is started or stopped", file=out)
    print("target: POST %s" % messages_url(args.base_url), file=out)
    print("model=%s turns=%d max_tokens=%d"
          % (args.model, args.turns, args.max_tokens), file=out)
    if args.seed_note:
        print("seed note: %s" % args.seed_note, file=out)
    for turn in build_transcript(args, prompts):
        print("", file=out)
        print("--- turn %d --- user: %s" % (turn["index"], turn["user"]), file=out)
        print("messages: %d (history grows by 2 per turn)" % turn["request_messages"],
              file=out)
        print("REQUEST: " + json.dumps(turn["request_body"], ensure_ascii=False,
                                       separators=(",", ":")), file=out)
    print("", file=out)
    print("DRY RUN OK: %d request bodies printed, no network access" % len(prompts),
          file=out)
    return 0


# ---------------------------------------------------------------------------
# self-test (完全オフライン: ソケットを開かない)
# ---------------------------------------------------------------------------

def self_test(out=sys.stdout):
    """検出器・exit code 対応・安全装置を固定文字列で検算する。"""
    failures = []

    def check(name, cond, detail=""):
        if cond:
            print("ok   %s%s" % (name, (" — " + detail) if detail else ""), file=out)
        else:
            print("FAIL %s%s" % (name, (" — " + detail) if detail else ""), file=out)
            failures.append(name)

    def flags_of(text, stop_reason="end_turn"):
        return [f for f, _ in detect_turn({"reply": text, "stop_reason": stop_reason})]

    # 1) きれいな応答
    clean = ("鎌倉の午前中は北鎌倉から入ると人が少ないです。"
             "建長寺を開門と同時に見てから、鶴岡八幡宮へ歩くと混雑を避けられます。"
             "昼は小町通りを外した路地の店にすると待ちません。")
    check("clean reply has no flags", flags_of(clean) == [], str(flags_of(clean)))

    # 2) 繰り返し行の崩壊
    # しきい値は >= 24 文字。ちょうど 24 文字の行で境界を確かめる。
    rep_line = "鎌倉の回り方をご案内しますので、ご確認ください。"  # 24 chars
    rep = "以下が候補です。\n" + (rep_line + "\n") * 4 + "以上です。"
    check("repeat line sample is exactly the threshold",
          len(rep_line) == MIN_REPEAT_LINE_CHARS, "%d chars" % len(rep_line))
    check("repeated >=24-char line detected", "repeat_line" in flags_of(rep),
          str(flags_of(rep)))
    short = "はい。\n" * 5
    check("repeated short line is not a collapse", "repeat_line" not in flags_of(short),
          str(flags_of(short)))

    # 3) 末尾文の繰り返し
    sent = ("初日は北鎌倉から歩きます。昼食は駅前の店にします。"
            "時間に余裕があります。時間に余裕があります。時間に余裕があります。")
    check("repeated trailing sentence detected", "repeat_sentence" in flags_of(sent),
          str(flags_of(sent)))
    ok_sent = "初日は北鎌倉から歩きます。昼食は駅前の店にします。二日目は早く出ます。"
    check("distinct sentences are not a collapse",
          "repeat_sentence" not in flags_of(ok_sent), str(flags_of(ok_sent)))

    # 4) 長い行が支配する
    # 単一文の繰り返しで repeat_sentence を同時に立てないよう、文ごとに変える。
    long_line = "前置きです。" + "".join(
        "この行の %02d 番目の説明は応答の大半を占めています。" % i for i in range(1, 21))
    check("long_line sample is long enough", len(long_line) >= LONG_REPLY_CHARS,
          "%d chars" % len(long_line))
    check("dominant long line detected", "long_line" in flags_of(long_line),
          str(flags_of(long_line)))
    many = "\n".join("行 %d: 短い説明です。" % i for i in range(40))
    check("many short lines are not a long_line", "long_line" not in flags_of(many),
          str(flags_of(many)))

    # 5) template 印 (6 種類すべて)
    for m in TEMPLATE_MARKERS:
        t = "回答です。\n%s\n次の回答です。" % m
        check("template marker %r detected" % m, "template_marker" in flags_of(t),
              str(flags_of(t)))
        inline = "回答です。%s という語を説明します。" % m
        check("inline %r is not a template leak" % m, "template_marker" not in flags_of(inline),
              str(flags_of(inline)))

    # 6) 偽の user ターン
    fake = "初日は北鎌倉から歩きます。\n\nuser\n二日目の予定も教えてください。"
    check("fake user turn detected", "fake_turn" in flags_of(fake), str(flags_of(fake)))
    no_fake = "初日は北鎌倉から歩きます。\n\nuser の意図は混雑回避です。"
    check("a sentence mentioning 'user' is not a fake turn",
          "fake_turn" not in flags_of(no_fake), str(flags_of(no_fake)))

    # 7) max_tokens は報告のみ
    trunc = {"index": 3, "reply": clean, "stop_reason": "max_tokens"}
    v = transcript_verdict({"turns": [trunc]})
    check("max_tokens is reported", v["soft"] and v["soft"][0]["flag"] == "max_tokens",
          str(v["soft"]))
    check("max_tokens does not fail the run", v["clean"] and v["exit_code"] == 0,
          "exit_code=%d" % v["exit_code"])

    # 8) 空の応答
    for empty in ("", "   \n\t\n  "):
        check("empty reply %r detected" % empty[:6], "empty" in flags_of(empty),
              str(flags_of(empty)))

    # 9) exit code の対応
    bad = {"turns": [{"index": 4, "reply": rep, "stop_reason": "end_turn"}]}
    vb = transcript_verdict(bad)
    check("hard flag -> exit 2", vb["exit_code"] == 2 and not vb["clean"],
          "exit_code=%d" % vb["exit_code"])
    check("hard flag carries the turn index", vb["failures"][0]["turn"] == 4,
          str(vb["failures"][0]))
    check("clean transcript -> exit 0",
          transcript_verdict({"turns": [{"index": 1, "reply": clean,
                                         "stop_reason": "end_turn"}]})["exit_code"] == 0)

    # 10) 判定の決定性 (同じトランスクリプト -> 同じ判定)
    t1 = {"turns": [{"index": 1, "reply": rep, "stop_reason": "end_turn"},
                    {"index": 2, "reply": clean, "stop_reason": "max_tokens"}]}
    a = transcript_verdict(t1)
    b = transcript_verdict(json.loads(json.dumps(t1)))
    check("verdict is deterministic for the same transcript", a == b,
          "failures=%d soft=%d" % (len(a["failures"]), len(a["soft"])))

    # 11) 安全装置
    from types import SimpleNamespace
    for port in FORBIDDEN_PORTS:
        args = SimpleNamespace(turns=10, max_tokens=512, timeout_s=300.0,
                               base_url="http://127.0.0.1:%d" % port)
        errs = validate_args(args)
        check("port %d refused" % port, any("reserved" in e for e in errs), str(errs))

    # 12) fixture
    try:
        prompts = load_fixture(DEFAULT_FIXTURE)
        check("committed fixture loads", len(prompts) == DEFAULT_TURNS,
              "%d prompts" % len(prompts))
        check("fixture prompts are non-empty",
              all(p.strip() for p in prompts))
    except FixtureError as e:
        check("committed fixture loads", False, str(e))

    # 13) 会話の組み立て (history が伸びる)
    msgs1 = build_messages([], "one")
    msgs2 = build_messages([("user", "one"), ("assistant", "ans")], "two")
    check("turn 1 sends 1 message", len(msgs1) == 1, str(len(msgs1)))
    check("turn 2 sends 3 messages", len(msgs2) == 3, str(len(msgs2)))
    check("turn 2 keeps the first turn's text", msgs2[0]["content"] == "one")
    check("build_messages does not mutate its input",
          msgs2[:1] == [{"role": "user", "content": "one"}])

    # 14) URL 組み立て
    check("messages_url adds /v1/messages",
          messages_url("http://127.0.0.1:8095") == "http://127.0.0.1:8095/v1/messages")
    check("messages_url normalises a trailing slash",
          messages_url("http://h:1/") == "http://h:1/v1/messages")

    if failures:
        print("SELF-TEST FAILED: %d check(s): %s" % (len(failures), ", ".join(failures)),
              file=out)
        return 1
    print("SELF-TEST PASSED (CPU only; no socket, no server, no GPU)", file=out)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def positive_int(s):
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


def build_parser():
    ap = argparse.ArgumentParser(
        prog="long_conversation_check.py",
        description="Scripted multi-turn conversation stability check (P4-1). "
                    "Sends N turns to an Anthropic-style /v1/messages endpoint and "
                    "flags repetition collapse, template leakage, fake user turns, "
                    "truncation and empty replies.",
        epilog="Exit: 0 clean, 2 at least one hard failure (repeat_line, "
               "repeat_sentence, long_line, template_marker, fake_turn, empty), "
               "3 infrastructure error (connection, HTTP != 200, malformed body, "
               "unreadable fixture). It reports; it does not restart the server.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help="server base URL (default %s; ports 8080 and 8090 are "
                         "refused)" % DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model id sent in the body (default %s)" % DEFAULT_MODEL)
    ap.add_argument("--turns", type=positive_int, default=DEFAULT_TURNS,
                    help="number of user turns to send (default %d)" % DEFAULT_TURNS)
    ap.add_argument("--max-tokens", type=positive_int, default=DEFAULT_MAX_TOKENS,
                    help="max_tokens per request (default %d)" % DEFAULT_MAX_TOKENS)
    ap.add_argument("--seed-note", default="",
                    help="free text recorded in the transcript/report "
                         "(seed, build, commit — not a measurement)")
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE,
                    help="turn payload fixture (default scripts/fixtures/"
                         "long_conversation_turns.json)")
    ap.add_argument("--out",
                    help="write the transcript JSON here (default: do not write)")
    ap.add_argument("--replay",
                    help="do not send anything: re-judge an existing transcript JSON "
                         "and print the same verdict")
    ap.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S,
                    help="per-request timeout in seconds (default %.0f)" % DEFAULT_TIMEOUT_S)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and request bodies; send nothing, start nothing")
    ap.add_argument("--self-test", action="store_true",
                    help="CPU-only self test (no socket, no server, no GPU)")
    ap.add_argument("--json", dest="json_out", action="store_true",
                    help="print the summary and verdict as one JSON object on stdout")
    return ap


def emit_json(transcript, result, out=sys.stdout):
    """標準出力にまとめた JSON を 1 つだけ出す (機械可読)。"""
    payload = {
        "verdict": result,
        "transcript": {
            "base_url": transcript.get("base_url"),
            "model": transcript.get("model"),
            "max_tokens": transcript.get("max_tokens"),
            "seed_note": transcript.get("seed_note"),
            "turn_count": transcript.get("turn_count"),
            "turns": [{"index": t.get("index"),
                       "chars": len(t.get("reply", "")),
                       "stop_reason": t.get("stop_reason")}
                      for t in transcript.get("turns", [])],
        },
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), file=out)


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    errors = validate_args(args)
    if errors:
        for e in errors:
            print("ERROR: %s" % e, file=sys.stderr)
        return 3

    # --replay: 既に取ったトランスクリプトを判定し直すだけ (ネットワークに出ない)
    if args.replay:
        try:
            with open(args.replay, "r", encoding="utf-8") as f:
                transcript = json.load(f)
        except (OSError, ValueError) as e:
            print("ERROR: cannot read transcript %s: %s" % (args.replay, e),
                  file=sys.stderr)
            return 3
        result = transcript_verdict(transcript)
        if args.json_out:
            emit_json(transcript, result)
        else:
            print_report(transcript, result)
        return result["exit_code"]

    try:
        prompts = plan_turns(load_fixture(args.fixture), args.turns)
    except FixtureError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 3

    if args.dry_run:
        return run_dry(args, prompts)

    try:
        transcript = run_conversation(args.base_url, prompts, model=args.model,
                                      max_tokens=args.max_tokens,
                                      timeout_s=args.timeout_s,
                                      seed_note=args.seed_note,
                                      log=lambda m: print(m, file=sys.stderr))
    except InfraError as e:
        print("INFRA ERROR: %s" % e, file=sys.stderr)
        return 3

    result = transcript_verdict(transcript)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

    if args.json_out:
        emit_json(transcript, result)
    else:
        print_report(transcript, result)
        if args.out:
            print("transcript written to %s" % args.out)

    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())