#!/usr/bin/env python3
"""P4-4 (s98): 長文 needle recall の回帰チェック。

s73 の調査では 4,671 token の長文 (needle-in-a-haystack) を手で入力して
「長い文脈の KV 経路が壊れていない」ことを確かめた (docs/HISTORY.en.md:1621)。
その 1 回きりの確認を、毎回同じ payload で回せる形に固定したものがこのスクリプト。

流れ:
  1. fixture (既定 scripts/needle_fixture.json) から filler を決定的に組み立て、
     needle (合言葉 1 文) を depth の位置に差し込み、最後に質問を置く。
  2. /v1/chat/completions (OpenAI 互換) を試し、404 なら /v1/messages
     (Anthropic 互換。denno serve の実経路はこちら) に落ちる。
  3. 応答に期待する合言葉が含まれているかで判定する。

判定 (exit code):
  0  期待する答えが見つかった (recall 成功)
  1  答えが無い / 間違っている (recall 失敗 = 回帰)
  2  通信失敗・応答の解釈失敗 (判定不能。1 と混同しない)

token 数は文字数から推測しない。payload の実 token 数はサーバ応答の
usage.input_tokens を採り、JSON 出力に記録する。目標帯 (既定 4,400-5,000) を
外れたら警告するが、判定は recall の成否だけで決める。

標準ライブラリのみ (urllib を使い requests は使わない)。

使い方:
  python3 scripts/needle_check.py --base-url http://localhost:8095 \\
      --model local/qwen35-9b --json-out /tmp/needle.json

  # 深さを変えて感度を見る (0.0 = 先頭, 1.0 = 末尾)
  python3 scripts/needle_check.py --depth 0.9

  # 経路を固定する (auto は chat/completions -> 404 -> messages)
  python3 scripts/needle_check.py --api anthropic

CPU だけで自己テストするには scripts/needle_selftest.py を使う。
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://localhost:8095"
DEFAULT_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "needle_fixture.json")

# fixture に値が無いときの控え。
DEFAULTS = {
    "target_input_tokens_min": 4400,
    "target_input_tokens_max": 5000,
    "orig_input_tokens": 4671,
    "depth_tolerance": 1.0,
    "filler_chars": 24000,
    "min_sentences": 40,
    "opener_ratio": 0.2,
    "seed": 20261019,
    "multiplier": 1103515245,
    "increment": 12345,
    "modulus": 2147483648,
    "subject_prefix": "",
}

EXIT_OK = 0
EXIT_RECALL_FAIL = 1
EXIT_TRANSPORT = 2


# ---------------------------------------------------------------- fixture

def load_fixture(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _lcg(seed, multiplier, increment, modulus):
    """決定的な擬似乱数 (線形合同法)。実行環境に依存しない数列を返す。"""
    state = int(seed)
    while True:
        state = (state * int(multiplier) + int(increment)) % int(modulus)
        yield state


def _filler_params(fixture):
    f = dict(DEFAULTS)
    f.update({k: v for k, v in (fixture.get("filler") or {}).items()
              if not k.startswith("_")})
    return f


def build_filler(fixture):
    """fixture の仕様から filler 本文を決定的に組み立てる。

    乱数は使わず、seed 付き LCG で語句表から選ぶ。同じ fixture からは
    何度組み立てても同じ文字列になる (selftest が 2 回作って照合する)。
    """
    f = _filler_params(fixture)
    for key in ("topics", "openers", "subjects", "verbs", "objects", "closers"):
        if not f.get(key):
            raise ValueError("fixture の filler.%s が空" % key)

    rng = _lcg(f["seed"], f["multiplier"], f["increment"], f["modulus"])

    def pick(lst):
        return lst[next(rng) % len(lst)]

    target = int(f["filler_chars"])
    min_sentences = int(f["min_sentences"])
    opener_cut = int(round(float(f["opener_ratio"]) * 1000.0))

    sentences = []
    total = 0
    while total < target or len(sentences) < min_sentences:
        # 文頭フラグを必ず 1 回引いてから組み立てる (引く順序を固定し、
        # fixture の一部を書き換えても後続の並びが崩れないようにする)。
        opener = pick(f["openers"]) if (next(rng) % 1000) < opener_cut else None
        topic = pick(f["topics"])
        subject = pick(f["subjects"])
        verb = pick(f["verbs"])
        obj = pick(f["objects"])
        closer = pick(f["closers"])

        if opener:
            head = opener + " "
            # 開幕句が文頭なので topic は小文字のまま
            head += topic
        else:
            head = topic[0].upper() + topic[1:]
        if f["subject_prefix"]:
            head += ", " + f["subject_prefix"] + " " + subject
        else:
            head += ", " + subject
        s = "%s %s %s %s." % (head, verb, obj, closer)
        sentences.append(s)
        total += len(s) + 1
    return " ".join(sentences)


def split_sentences(filler):
    """filler を文単位に分ける (needle を差し込む位置を決めるため)。"""
    return [s for s in re.split(r"(?<=\.)\s+", filler) if s]


def insert_needle(filler, needle, depth):
    """filler の depth (文字数比) の位置に needle を独立した段落として差し込む。

    戻り値: (payload_text, achieved_depth)。achieved_depth は実際に差し込んだ
    位置の文字数比で、文境界に丸めた分だけ要求値とずれる。
    """
    depth = min(max(float(depth), 0.0), 1.0)
    sentences = split_sentences(filler)
    if not sentences:
        raise ValueError("filler が空で needle を差し込めない")

    # 差し込み位置の文字数 (needle を除いた filler を基準にする)
    target_chars = depth * len(filler)
    pos = 0
    at = len(sentences)
    for i, s in enumerate(sentences):
        # 各文の終わり (空白込み) を越えたところで最初に target を跨ぐ位置を採る
        if pos >= target_chars:
            at = i
            break
        pos += len(s) + 1
    if depth <= 0.0:
        at = 0
    elif depth >= 1.0:
        at = len(sentences)

    head = sentences[:at]
    tail = sentences[at:]
    body = " ".join(head)
    rest = " ".join(tail)
    # needle は前後の本文と混ざらないよう空行で区切る (段落として独立させる)
    parts = [p for p in (body, needle, rest) if p]
    payload_text = "\n\n".join(parts)

    # 実位置 (needle の開始文字 / needle 本文を除いた全長)
    if body:
        achieved = len(body) / float(len(filler)) if filler else 0.0
    else:
        achieved = 0.0
    return payload_text, achieved


def build_prompt(fixture, depth):
    """fixture から (payload_text, 質問, 実 depth) を作る。

    戻り値: dict(prefix=..., question=..., achieved_depth=...)
    prefix は needle 入りの本文 (質問は別のユーザー発話として送る)。
    """
    filler = build_filler(fixture)
    needle = fixture["needle"]
    prefix, achieved = insert_needle(filler, needle, depth)
    return {
        "filler_chars": len(filler),
        "payload_chars": len(prefix),
        "achieved_depth": achieved,
        "text": prefix,
        "question": fixture["question"],
    }


# ---------------------------------------------------------------- 判定

_WS = re.compile(r"\s+")


def normalize(text):
    """比較用の正規化: 小文字化 + 空白の連続を 1 つに + 前後を落とす。"""
    if not isinstance(text, str):
        return ""
    return _WS.sub(" ", text).strip().lower()


def _contains_term(haystack, term):
    """正規化済み文字列に term が「語として」含まれるか。

    単純な部分文字列一致だと期待値 QX-4471 が QX-44715 に、別表記 QX4471 が
    QX44710 に当たって誤って recall 成功になる。英数字の境界を見て弾く。
    前後に英数字が続かない位置だけを一致とみなす。
    """
    if not term:
        return False
    start = 0
    while True:
        i = haystack.find(term, start)
        if i < 0:
            return False
        j = i + len(term)
        before_ok = (i == 0) or not haystack[i - 1].isalnum()
        after_ok = (j >= len(haystack)) or not haystack[j].isalnum()
        if before_ok and after_ok:
            return True
        start = i + 1


def judge(answer, expected, alternatives=(), forbidden=()):
    """応答が期待する答えを含むかを判定する (純関数)。

    戻り値: dict(matched=bool, reason=str, matched_term=str|None)。
      matched=True  -> exit 0
      matched=False -> exit 1 (理由: empty / forbidden_only / missing)

    照合は case-insensitive かつ空白正規化した上での語境界つき一致。
    "the code is QX-4471" のように前後に文が付いていても recall 成立とみなす
    (モデルに「コードだけ答えろ」と指示はするが、前置きを許容する)。
    """
    norm = normalize(answer)
    if not norm:
        return {"matched": False, "reason": "empty", "matched_term": None}

    terms = [expected] + [a for a in alternatives if a]
    for t in terms:
        if _contains_term(norm, normalize(t)):
            return {"matched": True, "reason": "found", "matched_term": t}

    forb = [normalize(x) for x in forbidden if x]
    if forb and any(_contains_term(norm, fb) for fb in forb):
        return {"matched": False, "reason": "forbidden_only",
                "matched_term": None}
    return {"matched": False, "reason": "missing", "matched_term": None}


# ---------------------------------------------------------------- HTTP

def _post_json(url, payload, timeout):
    """JSON を POST して (status, body_text) を返す。HTTP エラーも本文を返す。

    longconv_check.py と同じ扱い: 4xx/5xx は例外にせず status で返し、
    接続不能 (URLError/OSError) だけを例外として上げる。
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"content-type": "application/json", "accept": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:  # 404 / 500 も中身を見たい
        return e.code, e.read().decode("utf-8", "replace")


def extract_openai_text(obj):
    """OpenAI chat.completion 形式から本文を取り出す。"""
    try:
        choice = obj["choices"][0]
    except (KeyError, IndexError, TypeError):
        raise ValueError("OpenAI 形式の応答に choices[0] が無い")
    msg = choice.get("message") or {}
    content = msg.get("content")
    if content is None:
        content = choice.get("text")
    if not isinstance(content, str):
        raise ValueError("OpenAI 形式の応答に文字列の content が無い")
    return content


def extract_anthropic_text(obj):
    """Anthropic messages 形式 (content = [{type:text,...}]) から本文を取り出す。"""
    blocks = obj.get("content")
    if not isinstance(blocks, list):
        raise ValueError("Anthropic 形式の応答に content 配列が無い")
    parts = [b.get("text", "") for b in blocks
             if isinstance(b, dict) and b.get("type") == "text"]
    return "".join(parts)


def extract_usage(obj, api_format):
    """usage から入力 token 数を取り出す (無ければ None)。

    Anthropic は input_tokens、OpenAI は prompt_tokens。どちらも無ければ
    total_tokens にすら触らず None を返す (推測で埋めない)。
    """
    u = obj.get("usage")
    if not isinstance(u, dict):
        return {}
    out = {"raw": u, "input_tokens": None}
    key = "input_tokens" if api_format == "anthropic" else "prompt_tokens"
    v = u.get(key)
    if isinstance(v, int) and not isinstance(v, bool):
        out["input_tokens"] = v
    return out


def resolve_endpoint(base_url, api):
    """叩く URL と応答形式の候補を返す。

    api="auto" (既定) は両方の形を順に試す: まず OpenAI 互換
    /v1/chat/completions、404 なら /v1/messages (denno serve の実経路) に落ちる。
    denno serve に chat/completions は無いので auto は常に messages に着地する。
    """
    base = base_url.rstrip("/")
    if api == "openai":
        return [(base + "/v1/chat/completions", "openai")]
    if api == "anthropic":
        return [(base + "/v1/messages", "anthropic")]
    return [(base + "/v1/chat/completions", "openai"),
            (base + "/v1/messages", "anthropic")]


def build_payload(api_format, model, context_text, question, system_prompt,
                  temperature, max_tokens, top_p=None):
    if api_format == "openai":
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": context_text})
        msgs.append({"role": "user", "content": question})
        payload = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        return payload
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": context_text},
                     {"role": "user", "content": question}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if system_prompt:
        payload["system"] = system_prompt
    if top_p is not None:
        payload["top_p"] = top_p
    return payload


# ---------------------------------------------------------------- 本体

def _fmt(x):
    if x is None:
        return "None"
    return ("%.3f" % x).rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


def run(args, fixture):
    th = dict(DEFAULTS)
    th.update({k: v for k, v in (fixture.get("thresholds") or {}).items()
               if not k.startswith("_")})

    system_prompt = fixture.get("system_prompt") or ""
    gen = {k: v for k, v in (fixture.get("gen") or {}).items()
           if not k.startswith("_")}
    max_tokens = args.max_tokens if args.max_tokens is not None \
        else int(gen.get("max_tokens") or 32)
    top_p = gen.get("top_p")

    try:
        prompt = build_prompt(fixture, args.depth)
    except (ValueError, KeyError) as e:
        print("ERROR: fixture から payload を作れない: %s" % e, file=sys.stderr)
        return EXIT_TRANSPORT

    expected = fixture["expected_answer"]
    alternatives = fixture.get("expected_alternatives") or []
    forbidden = fixture.get("forbidden_answers") or []

    endpoints = resolve_endpoint(args.base_url, args.api)

    report = {
        "schema": "needle_check/1",
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url,
        "model": args.model,
        "fixture": os.path.basename(args.fixture),
        "fixture_id": fixture.get("id"),
        "requested_api": args.api,
        "temperature": args.temperature,
        "max_tokens": max_tokens,
        "depth_requested": args.depth,
        "depth_achieved": round(prompt["achieved_depth"], 4),
        "filler_chars": prompt["filler_chars"],
        "payload_chars": prompt["payload_chars"],
        "needle": fixture["needle"],
        "needle_in_payload": fixture["needle"] in prompt["text"],
        "question": prompt["question"],
        "expected_answer": expected,
        "api_format": None,
        "endpoint": None,
        "fallback_used": False,
        "status": None,
        "answer": None,
        "verdict": None,
        "usage": {},
        "usage_input_tokens": None,
        "timings_ms": {},
        "errors": [],
        "exit": None,
    }

    # ---- HTTP: auto のとき 404 で次の形式に落ちる
    t0 = time.monotonic()
    status, body, fmt, url = None, None, None, None
    tried = []
    for (u, f) in endpoints:
        payload = build_payload(f, args.model, prompt["text"], prompt["question"],
                                system_prompt, args.temperature, max_tokens, top_p)
        try:
            st, bd = _post_json(u, payload, args.timeout)
        except (urllib.error.URLError, OSError) as e:
            tried.append({"endpoint": u, "api": f, "error": str(e)})
            continue
        tried.append({"endpoint": u, "api": f, "status": st})
        if st == 404 and len(endpoints) > 1:
            continue  # 別形式を試す
        status, body, fmt, url = st, bd, f, u
        break
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    report["attempts"] = tried
    report["timings_ms"]["request"] = round(elapsed_ms, 1)

    if status is None:
        detail = "; ".join("%s -> %s" % (a["endpoint"],
                                         a.get("status") or a.get("error"))
                           for a in tried) or "エンドポイント無し"
        msg = "接続/エンドポイント失敗: %s" % detail
        report["errors"].append(msg)
        print("ERROR: %s" % msg, file=sys.stderr)
        report["exit"] = EXIT_TRANSPORT
        finish_report(report, args, 0.0)
        return EXIT_TRANSPORT

    report["api_format"] = fmt
    report["endpoint"] = url
    report["status"] = status
    report["fallback_used"] = bool(len(tried) > 1)

    if status != 200:
        msg = "HTTP %d: %s" % (status, body[:300].replace("\n", " "))
        report["errors"].append(msg)
        report["body_head"] = body[:400]
        print("ERROR: %s" % msg, file=sys.stderr)
        report["exit"] = EXIT_TRANSPORT
        finish_report(report, args, elapsed_ms)
        return EXIT_TRANSPORT

    try:
        obj = json.loads(body)
        answer = (extract_openai_text(obj) if fmt == "openai"
                  else extract_anthropic_text(obj))
    except (ValueError, json.JSONDecodeError) as e:
        msg = "応答の解釈に失敗: %s" % e
        report["errors"].append(msg)
        report["body_head"] = body[:400]
        print("ERROR: %s" % msg, file=sys.stderr)
        report["exit"] = EXIT_TRANSPORT
        finish_report(report, args, elapsed_ms)
        return EXIT_TRANSPORT

    usage = extract_usage(obj, fmt)
    report["usage"] = usage.get("raw", {})
    report["usage_input_tokens"] = usage.get("input_tokens")
    report["answer"] = answer

    verdict = judge(answer, expected, alternatives, forbidden)
    report["verdict"] = verdict

    # 目標 token 帯から外れたら警告する (判定は recall の成否だけで決める)
    warnings = []
    itok = report["usage_input_tokens"]
    lo, hi = th["target_input_tokens_min"], th["target_input_tokens_max"]
    if itok is None:
        warnings.append("応答に usage の入力 token 数が無い (payload 長は未確認)")
    elif itok < lo or itok > hi:
        warnings.append("usage.input_tokens = %d が目標帯 %d-%d の外 "
                        "(filler.filler_chars を調整する)" % (itok, lo, hi))
    if th.get("depth_tolerance") is not None and \
            abs(prompt["achieved_depth"] - args.depth) > float(th["depth_tolerance"]):
        warnings.append("実 depth %.3f が要求 %.3f から外れている"
                        % (prompt["achieved_depth"], args.depth))
    if not report["needle_in_payload"]:
        warnings.append("needle が payload に含まれていない (fixture 破損)")
    report["warnings"] = warnings

    code = EXIT_OK if verdict["matched"] else EXIT_RECALL_FAIL
    report["exit"] = code
    finish_report(report, args, elapsed_ms)
    return code


def finish_report(report, args, req_ms):
    """JSON と人要約を出す。"""
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["timings_ms"]["total"] = round(req_ms or 0.0, 1)

    if args.json_out:
        try:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print("json: %s" % args.json_out)
        except OSError as e:
            print("WARNING: --json-out に書けなかった: %s" % e, file=sys.stderr)

    print()
    print("== needle_check summary ==")
    print("endpoint:  %s (%s)%s"
          % (report.get("endpoint"), report.get("api_format"),
             "  [fallback]" if report.get("fallback_used") else ""))
    print("model:     %s   temperature: %s" % (report["model"], _fmt(report["temperature"])))
    print("depth:     requested %s / achieved %.3f"
          % (_fmt(report["depth_requested"]), report.get("depth_achieved") or 0.0))
    print("payload:   %d chars (filler %d chars, needle present: %s)"
          % (report.get("payload_chars") or 0, report.get("filler_chars") or 0,
             report.get("needle_in_payload")))
    itok = report.get("usage_input_tokens")
    print("tokens:    %s"
          % ("usage.input_tokens = %d" % itok if itok is not None
             else "usage.input_tokens = (未計測: 応答に usage が無い)"))
    print("request:   %.1f ms" % (req_ms or 0.0))
    if report.get("answer") is not None:
        head = report["answer"].replace("\n", " ")
        if len(head) > 200:
            head = head[:200] + "..."
        print("answer:    %r" % head)
    v = report.get("verdict") or {}
    if v:
        print("verdict:   %s (%s%s)"
              % ("MATCH" if v.get("matched") else "NO MATCH",
                 v.get("reason"),
                 ", term=%r" % v["matched_term"] if v.get("matched_term") else ""))
    for w in report.get("warnings") or []:
        print("warning:   %s" % w)
    for e in report.get("errors") or []:
        print("error:     %s" % e)
    print("RESULT: %s" % (
        "RECALL OK (exit 0)" if report.get("exit") == EXIT_OK else
        "RECALL FAIL (exit 1)" if report.get("exit") == EXIT_RECALL_FAIL else
        "ERROR (exit 2)"))


def build_parser():
    p = argparse.ArgumentParser(
        prog="needle_check.py",
        description=("長文 (既定で約 4,700 token) の needle recall を確認する回帰チェック。"
                     "固定の filler に合言葉を 1 つだけ埋め込み、その合言葉を答えられるかを"
                     "見る。答えが見つかれば exit 0、答えが無い/間違いは exit 1、"
                     "通信や応答の解釈に失敗したら exit 2。"),
        epilog=("例:\n"
                "  python3 scripts/needle_check.py \\\n"
                "      --base-url http://localhost:8095 --model local/qwen35-9b \\\n"
                "      --json-out /tmp/needle.json\n"
                "  python3 scripts/needle_check.py --depth 0.9   # 深さ感度\n"
                "  python3 scripts/needle_check.py --api anthropic\n"
                "\n"
                "exit code: 0 = recall 成功 / 1 = recall 失敗 (回帰) / "
                "2 = 通信・解釈の失敗\n"
                "CPU だけで自己テストするには scripts/needle_selftest.py を使う"
                " (偽サーバを立てて exit code を確認する)。"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help="接続先 (既定 %(default)s)")
    p.add_argument("--model", default="local/qwen35-9b",
                   help="モデル ID (serve の --name。既定 %(default)s)")
    p.add_argument("--depth", type=float, default=None,
                   help=("needle を置く位置 (filler 文字数に対する割合 0.0-1.0)。"
                         "0.0 = 先頭、1.0 = 末尾。既定は fixture の depth"))
    p.add_argument("--temperature", type=float, default=0.0,
                   help="送る温度 (既定 %(default)s)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="生成の上限 token (既定: fixture の gen.max_tokens)")
    p.add_argument("--fixture", default=DEFAULT_FIXTURE,
                   help="needle / filler 仕様 / 期待値の JSON (既定 scripts/needle_fixture.json)")
    p.add_argument("--json-out", default=None,
                   help="結果 JSON の書き出し先 (例 /tmp/needle.json)")
    p.add_argument("--api", choices=("auto", "openai", "anthropic"), default="auto",
                   help=("叩く API 形式。auto は /v1/chat/completions を試し、"
                         "404 なら /v1/messages (denno serve の実経路)。既定 %(default)s"))
    p.add_argument("--timeout", type=float, default=180.0,
                   help="リクエストのタイムアウト秒 (既定 %(default)s)")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        fixture = load_fixture(args.fixture)
    except (OSError, json.JSONDecodeError) as e:
        print("ERROR: fixture を読めない (%s): %s" % (args.fixture, e),
              file=sys.stderr)
        return EXIT_TRANSPORT
    if args.depth is None:
        args.depth = float(fixture.get("depth", 0.5))
    if not (0.0 <= args.depth <= 1.0):
        print("ERROR: --depth は 0.0-1.0 (got %s)" % args.depth, file=sys.stderr)
        return EXIT_TRANSPORT
    if args.temperature < 0:
        print("ERROR: --temperature は 0 以上 (got %s)" % args.temperature,
              file=sys.stderr)
        return EXIT_TRANSPORT
    return run(args, fixture)


if __name__ == "__main__":
    sys.exit(main())