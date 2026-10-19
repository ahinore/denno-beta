#!/usr/bin/env python3
"""P4-1 (s97): 9B 長会話の崩壊を検出する回帰チェック。

元の不具合報告「数ターンで崩壊する」は temperature = 1.0 が原因で、
serve 側の --temperature-cap 0.7 で解決した。しかし複数ターンの回帰テストが
無いので、cap が外れても誰も気づけない。このスクリプトがその検出器になる。

流れ:
  1. fixture (既定 scripts/longconv_fixture.json) から 10 ターン分の user 文を読む。
  2. OpenAI 互換 (/v1/chat/completions) に順に投げる。denno serve は
     Anthropic 互換 (/v1/messages) なので、そちらの経路も自動で使える。
  3. 各応答を 4 種類の崩壊について判定する (下記 detect_*)。
  4. 結果を JSON (--json-out) と人が読む要約の両方で出す。
     崩壊が 1 つでも見つかれば exit 1。通信自体が失敗したら exit 2。

標準ライブラリのみ (urllib を使い requests は使わない)。

使い方:
  python3 scripts/longconv_check.py --base-url http://localhost:8095 \\
      --model local/qwen35-9b --json-out /tmp/longconv.json

  # cap 回帰の確認 (送った温度とサーバの応答を突き合わせる)
  python3 scripts/longconv_check.py --expect-cap 0.7

検出する崩壊 (4 種):
  repetition   反復崩壊: 周期 n (既定 8 文字) 以上の並びが 3 回以上続き、
               応答長に占める割合がしきい値 (0.35) を超える。
  contamination テンプレ片の漏出: <|im_start|> 等の特殊トークンや
               "assistant:" のようなロール接頭辞が本文に現れる。
  degenerate   空応答 / 異常長: 空文字、空白のみ、min_chars 未満、max_chars 超過。
  duplicate    同一文の連続: 直前ターンの応答と完全一致、または
               difflib.SequenceMatcher の類似度がしきい値を超える。
"""

import argparse
import difflib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://localhost:8095"
DEFAULT_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "longconv_fixture.json")

# fixture に値が無いときの控え。判定の意味を変えない範囲の保守的な既定。
DEFAULTS = {
    "repeat_period_min": 8,
    "repeat_min_repeats": 3,
    "repeat_coverage_max": 0.35,
    "similarity_max": 0.90,
    "min_chars": 1,
    "max_chars": 4000,
    "contamination_markers": [
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
        "system:", "assistant:", "user:", "Human:", "Assistant:",
    ],
}

# 検出カテゴリ (JSON の categories と exit 判定で共用)
CATEGORIES = ("repetition", "contamination", "degenerate", "duplicate")


# ---------------------------------------------------------------- 検出器

def longest_repeat(text, period_min=8, min_repeats=3):
    """周期 period_min 以上で min_repeats 回以上続く並びのうち最長のものを返す。

    戻り値: (coverage, period, repeats, unit) / 見つからなければ (0.0, 0, 0, "")。
    coverage は「反復している部分の文字数 / 全長」。

    実装は素直に: 各周期 p について text[i:i+p] が何回続くかを数える。
    長会話の応答は高々数千文字なので O(n^2) でも実用上問題ない
    (period_min 未満と、残り文字数より長い周期は飛ばす)。
    """
    n = len(text)
    if n == 0 or period_min <= 0 or min_repeats < 2:
        return (0.0, 0, 0, "")
    best = (0, 0, 0, "")  # (covered, period, repeats, unit)
    p = period_min
    while p * min_repeats <= n:
        i = 0
        while i + p <= n:
            unit = text[i:i + p]
            reps = 1
            j = i + p
            while j + p <= n and text[j:j + p] == unit:
                reps += 1
                j += p
            if reps >= min_repeats:
                covered = reps * p
                # 同じ被覆長なら周期の短い方 (より「はっきりした」反復) を採る
                if covered > best[0]:
                    best = (covered, p, reps, unit)
                i = j  # この反復の先へ飛ばす
            else:
                i += 1
        p += 1
    covered, period, reps, unit = best
    if covered == 0:
        return (0.0, 0, 0, "")
    return (covered / float(n), period, reps, unit)


def find_contamination(text, markers):
    """本文に現れた特殊トークン / ロール接頭辞を返す (重複なし・出現順)。"""
    hits = []
    for m in markers:
        # "assistant:" は行頭か空白の直後だけを見る。本文中の普通の英単語
        # (例: "the assistant: a helper") を拾いすぎないため、行頭優先で判定し、
        # 特殊トークン (<|...|>) はどこに出ても漏出とみなす。
        if m.startswith("<|"):
            found = m in text
        else:
            low = text.lower()
            key = m.lower()
            found = False
            idx = low.find(key)
            while idx != -1:
                if idx == 0 or text[idx - 1] in "\n\r":
                    found = True
                    break
                idx = low.find(key, idx + 1)
        if found and m not in hits:
            hits.append(m)
    return hits


def similarity(a, b):
    """difflib の類似度 (0..1)。空同士は 1.0 (どちらも空 = 同一扱い)。"""
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def detect_turn(text, prev_text, th, prev_was_empty=False):
    """1 ターン分の崩壊を判定する。戻り値は (issues, detail)。

    issues は CATEGORIES の部分集合。prev_was_empty が真のときは、
    空応答が続いても「同一文の連続」としては数えない (degenerate で拾う)。
    """
    issues = []
    detail = {}

    # --- degenerate: 空 / 異常長
    stripped = text.strip() if isinstance(text, str) else ""
    n = len(stripped)
    degen = []
    if n == 0:
        degen.append("empty")
    elif n < th["min_chars"]:
        degen.append("too_short")
    if len(text) > th["max_chars"]:
        degen.append("too_long")
    if degen:
        issues.append("degenerate")
        detail["degenerate"] = {"reasons": degen, "chars": len(text)}

    # --- contamination: 特殊トークン / ロール接頭辞
    hits = find_contamination(text, th["contamination_markers"])
    if hits:
        issues.append("contamination")
        detail["contamination"] = {"markers": hits}

    # --- repetition: 最長反復の占有率
    cov, period, reps, unit = longest_repeat(
        text, th["repeat_period_min"], th["repeat_min_repeats"])
    detail["longest_repeat"] = {
        "coverage": round(cov, 4),
        "period": period,
        "repeats": reps,
        "unit_head": unit[:40],
    }
    if cov > th["repeat_coverage_max"]:
        issues.append("repetition")
        detail["repetition"] = {
            "coverage": round(cov, 4),
            "threshold": th["repeat_coverage_max"],
            "period": period,
            "repeats": reps,
        }

    # --- duplicate: 直前ターンと同一 / 高類似
    if prev_text is not None:
        same = (not prev_was_empty) and (text.strip() == prev_text.strip()) and text.strip()
        sim = similarity(text, prev_text)
        detail["similarity"] = round(sim, 4)
        if same or sim > th["similarity_max"]:
            issues.append("duplicate")
            detail["duplicate"] = {
                "identical": bool(same),
                "similarity": round(sim, 4),
                "threshold": th["similarity_max"],
            }

    return issues, detail


# ---------------------------------------------------------------- HTTP

def _post_json(url, payload, timeout):
    """JSON を POST して (status, body_text) を返す。HTTP エラーも本文を返す。"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"content-type": "application/json", "accept": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:  # 4xx/5xx も中身を見たい
        return e.code, e.read().decode("utf-8", "replace")


def _get_json(url, timeout):
    req = urllib.request.Request(url, headers={"accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
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


def resolve_endpoint(base_url, api):
    """叩く URL と応答形式の候補を返す。

    api="auto" (既定) は両方の形を順に試す: まず OpenAI 互換 /v1/chat/completions、
    404 なら /v1/messages (denno serve の実経路はこちら) に落ちる。
    どちらも 404 ならエラーで終わる。確定した形式は以降のターンで使い回す。
    """
    base = base_url.rstrip("/")
    if api == "openai":
        return [(base + "/v1/chat/completions", "openai")]
    if api == "anthropic":
        return [(base + "/v1/messages", "anthropic")]
    return [(base + "/v1/chat/completions", "openai"),
            (base + "/v1/messages", "anthropic")]


def build_payload(api_format, model, messages, system_prompt, temperature,
                  gen, max_tokens):
    if api_format == "openai":
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend(messages)
        payload = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if gen.get("top_p") is not None:
            payload["top_p"] = gen["top_p"]
        return payload
    # anthropic
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if system_prompt:
        payload["system"] = system_prompt
    if gen.get("top_p") is not None:
        payload["top_p"] = gen["top_p"]
    return payload


def usage_of(obj):
    u = obj.get("usage")
    return u if isinstance(u, dict) else {}


# ---------------------------------------------------------------- 本体

def run(args, fixture):
    th = dict(DEFAULTS)
    th.update({k: v for k, v in (fixture.get("thresholds") or {}).items()
               if not k.startswith("_")})
    turns = fixture.get("turns") or []
    if not turns:
        raise SystemExit("fixture に turns が無い: " + args.fixture)
    if args.turns and args.turns > 0:
        turns = turns[:args.turns]

    gen = {k: v for k, v in (fixture.get("gen") or {}).items()
           if not k.startswith("_")}
    system_prompt = fixture.get("system_prompt") or ""

    # 温度: --temperature が明示されればそれ、無ければ max_temp。
    # max_temp を超える要求は警告して丸める (cap を跨がないようにする)。
    requested = args.temperature if args.temperature is not None else args.max_temp
    clamped_from = None
    if requested > args.max_temp:
        clamped_from = requested
        requested = args.max_temp
        print("WARNING: --temperature %s は --max-temp %s を超えています。"
              "%s に丸めます。"
              % (_fmt(clamped_from), _fmt(args.max_temp), _fmt(requested)),
              file=sys.stderr)

    endpoints = resolve_endpoint(args.base_url, args.api)

    report = {
        "schema": "longconv_check/1",
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url,
        "model": args.model,
        "temperature_requested": requested,
        "temperature_clamped_from": clamped_from,
        "expect_cap": args.expect_cap,
        "fixture": os.path.basename(args.fixture),
        "turns_planned": len(turns),
        "api_format": None,
        "endpoint": None,
        "turns": [],
        "errors": [],
        "exit": None,
    }

    messages = []
    prev_text = None
    prev_was_empty = False
    chosen = None

    for idx, t in enumerate(turns, 1):
        user = t["user"] if isinstance(t, dict) else str(t)
        messages.append({"role": "user", "content": user})

        t0 = time.monotonic()
        status, body, fmt, url = None, None, None, None
        # 最初のターンで api_format を確定させる (auto のとき)
        candidates = [chosen] if chosen else endpoints
        last_err = None
        for (u, f) in candidates:
            p = build_payload(f, args.model, messages, system_prompt, requested,
                              gen, args.max_tokens)
            try:
                st, bd = _post_json(u, p, args.timeout)
            except (urllib.error.URLError, OSError) as e:
                last_err = "%s: %s" % (u, e)
                continue
            if st == 404 and chosen is None and len(candidates) > 1:
                last_err = "%s: 404" % u
                continue  # 別形式を試す
            status, body, fmt, url = st, bd, f, u
            break
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        if status is None:
            msg = "接続/エンドポイント失敗: %s" % (last_err or "不明")
            report["errors"].append({"turn": idx, "error": msg})
            print("ERROR: turn %d: %s" % (idx, msg), file=sys.stderr)
            report["exit"] = 2
            finish_report(report, args)
            return 2

        if chosen is None:
            chosen = (url, fmt)
            report["api_format"] = fmt
            report["endpoint"] = url

        entry = {"turn": idx, "user": user, "status": status,
                 "elapsed_ms": round(elapsed_ms, 1)}
        if status != 200:
            entry["error"] = "HTTP %d" % status
            entry["body_head"] = body[:400]
            report["errors"].append({"turn": idx, "error": "HTTP %d" % status,
                                     "body_head": body[:400]})
            report["turns"].append(entry)
            print("ERROR: turn %d: HTTP %d %s"
                  % (idx, status, body[:200].replace("\n", " ")), file=sys.stderr)
            report["exit"] = 2
            finish_report(report, args)
            return 2

        try:
            obj = json.loads(body)
            text = (extract_openai_text(obj) if fmt == "openai"
                    else extract_anthropic_text(obj))
        except (ValueError, json.JSONDecodeError) as e:
            entry["error"] = "応答の解釈に失敗: %s" % e
            entry["body_head"] = body[:400]
            report["errors"].append({"turn": idx, "error": entry["error"]})
            report["turns"].append(entry)
            print("ERROR: turn %d: %s" % (idx, entry["error"]), file=sys.stderr)
            report["exit"] = 2
            finish_report(report, args)
            return 2

        issues, detail = detect_turn(text, prev_text, th, prev_was_empty)
        entry["reply"] = text
        entry["chars"] = len(text)
        entry["issues"] = issues
        entry["detail"] = detail
        entry["usage"] = usage_of(obj)
        # --expect-cap: 応答に温度が返る実装なら記録して後で突き合わせる。
        # denno の /v1/messages は温度を返さないので、その場合は送信値だけを見る。
        entry["accepted_temperature"] = obj.get("temperature")
        report["turns"].append(entry)

        messages.append({"role": "assistant", "content": text})
        prev_text = text
        prev_was_empty = (len(text.strip()) == 0)

        print("turn %2d: %4d chars  %7.1f ms  %s"
              % (idx, len(text), elapsed_ms,
                 "ok" if not issues else "COLLAPSE:" + ",".join(issues)))

    # --expect-cap の判定: 送った温度が max_temp を超えていないこと、
    # およびサーバが受理温度を返す場合はそれが expect_cap 以下であること。
    if args.expect_cap is not None:
        cap_issues = []
        if requested > args.expect_cap + 1e-6:
            cap_issues.append(
                "送信温度 %s が --expect-cap %s を超えている (max-temp を下げるか "
                "--temperature を見直す)" % (_fmt(requested), _fmt(args.expect_cap)))
        accepted_vals = [e.get("accepted_temperature") for e in report["turns"]
                         if e.get("accepted_temperature") is not None]
        for v in accepted_vals:
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v > args.expect_cap + 1e-6:
                cap_issues.append(
                    "サーバが受理した温度 %s が cap %s を超えている (cap 回帰)"
                    % (_fmt(v), _fmt(args.expect_cap)))
        report["cap_check"] = {
            "expect_cap": args.expect_cap,
            "sent_temperature": requested,
            "accepted_temperatures": accepted_vals,
            "issues": cap_issues,
        }
        print("cap check: sent=%s expect_cap=%s accepted=%s"
              % (_fmt(requested), _fmt(args.expect_cap),
                 accepted_vals if accepted_vals else "(応答に無し)"))

    collapsed = [e for e in report["turns"] if e.get("issues")]
    code = 1 if (collapsed or report["errors"] or
                 report.get("cap_check", {}).get("issues")) else 0
    report["summary"] = {
        "turns": len(report["turns"]),
        "collapsed_turns": [e["turn"] for e in collapsed],
        "categories": sorted({i for e in collapsed for i in e["issues"]}),
        "errors": len(report["errors"]),
    }
    report["exit"] = code
    finish_report(report, args)
    return code


def _fmt(x):
    if x is None:
        return "None"
    return ("%.3f" % x).rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


def finish_report(report, args):
    """JSON と人要約を出す。"""
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.json_out:
        try:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print("json: %s" % args.json_out)
        except OSError as e:
            print("WARNING: --json-out に書けなかった: %s" % e, file=sys.stderr)

    print()
    print("== longconv_check summary ==")
    print("endpoint: %s (%s)" % (report.get("endpoint"), report.get("api_format")))
    print("model:    %s   temperature sent: %s"
          % (report["model"], _fmt(report["temperature_requested"])))
    if report.get("temperature_clamped_from") is not None:
        print("note:     %s -> %s に丸めた (--max-temp)"
              % (_fmt(report["temperature_clamped_from"]),
                 _fmt(report["temperature_requested"])))
    s = report.get("summary") or {}
    if s:
        print("turns:    %d   collapsed: %s"
              % (s["turns"], s["collapsed_turns"] or "none"))
        if s["categories"]:
            print("categories: %s" % ", ".join(s["categories"]))
    for e in report["turns"]:
        if e.get("issues"):
            d = e.get("detail", {})
            bits = []
            if "repetition" in d:
                r = d["repetition"]
                bits.append("repetition cov=%.2f (period=%d x%d)"
                            % (r["coverage"], r["period"], r["repeats"]))
            if "contamination" in d:
                bits.append("contamination %s"
                            % ",".join(d["contamination"]["markers"]))
            if "degenerate" in d:
                bits.append("degenerate %s (%d chars)"
                            % (",".join(d["degenerate"]["reasons"]),
                               d["degenerate"]["chars"]))
            if "duplicate" in d:
                dd = d["duplicate"]
                bits.append("duplicate sim=%.3f identical=%s"
                            % (dd["similarity"], dd["identical"]))
            print("  turn %2d: %s" % (e["turn"], "; ".join(bits)))
    if report.get("errors"):
        for e in report["errors"]:
            print("  error turn %s: %s" % (e.get("turn"), e.get("error")))
    cc = report.get("cap_check")
    if cc and cc.get("issues"):
        print("cap check: FAIL")
        for i in cc["issues"]:
            print("  " + i)
    print("RESULT: %s" % ("COLLAPSE (exit 1)" if report.get("exit") == 1 else
                          "ERROR (exit 2)" if report.get("exit") == 2 else
                          "OK (exit 0)"))


def load_fixture(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_parser():
    p = argparse.ArgumentParser(
        prog="longconv_check.py",
        description=("9B の長会話 (既定 10 ターン) を OpenAI 互換 API に流し、"
                     "反復崩壊・テンプレ片の漏出・空応答/異常長・同一文の連続を"
                     "検出する回帰チェック。崩壊があれば exit 1、通信失敗は exit 2。"),
        epilog=("例:\n"
                "  python3 scripts/longconv_check.py \\\n"
                "      --base-url http://localhost:8095 --model local/qwen35-9b \\\n"
                "      --json-out /tmp/longconv.json\n"
                "  python3 scripts/longconv_check.py --expect-cap 0.7\n"
                "\n"
                "CPU だけで自己テストするには scripts/longconv_selftest.py を使う"
                " (偽サーバを立てて検出器の exit code を確認する)。"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help="接続先 (既定 %(default)s)")
    p.add_argument("--model", default="local/qwen35-9b",
                   help="モデル ID (serve の --name。既定 %(default)s)")
    p.add_argument("--turns", type=int, default=10,
                   help="流すターン数 (fixture の先頭から。既定 %(default)s)")
    p.add_argument("--max-temp", type=float, default=0.7,
                   help="送る温度の上限。超える要求は警告して丸める (既定 %(default)s)")
    p.add_argument("--temperature", type=float, default=None,
                   help="送る温度を明示 (既定: --max-temp と同じ)")
    p.add_argument("--expect-cap", type=float, default=None,
                   help=("serve の --temperature-cap の期待値。指定すると送信温度と"
                         "サーバが受理した温度を突き合わせ、cap 回帰を検出する"))
    p.add_argument("--max-tokens", type=int, default=None,
                   help="1 応答の生成上限 (既定: fixture の gen.max_tokens)")
    p.add_argument("--fixture", default=DEFAULT_FIXTURE,
                   help="台本としきい値の JSON (既定 scripts/longconv_fixture.json)")
    p.add_argument("--json-out", default=None,
                   help="結果 JSON の書き出し先 (例 /tmp/longconv.json)")
    p.add_argument("--api", choices=("auto", "openai", "anthropic"), default="auto",
                   help=("叩く API 形式。auto は /v1/chat/completions を試し、"
                         "404 なら /v1/messages (denno serve の実経路)。既定 %(default)s"))
    p.add_argument("--timeout", type=float, default=180.0,
                   help="1 リクエストのタイムアウト秒 (既定 %(default)s)")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        fixture = load_fixture(args.fixture)
    except (OSError, json.JSONDecodeError) as e:
        print("ERROR: fixture を読めない (%s): %s" % (args.fixture, e),
              file=sys.stderr)
        return 2
    if args.max_tokens is None:
        args.max_tokens = int((fixture.get("gen") or {}).get("max_tokens") or 256)
    if args.max_temp < 0 or math.isnan(args.max_temp):
        print("ERROR: --max-temp は 0 以上 (got %s)" % args.max_temp, file=sys.stderr)
        return 2
    if args.turns <= 0:
        print("ERROR: --turns は 1 以上 (got %d)" % args.turns, file=sys.stderr)
        return 2
    return run(args, fixture)


if __name__ == "__main__":
    sys.exit(main())