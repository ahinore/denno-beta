#!/usr/bin/env python3
"""P5-1 (s109): prefill and decode dashboard for cold A/B measurements.

使い方:
  # cold A/B の --json を 1 つ以上集めたディレクトリから HTML を作る
  python3 scripts/dashboard.py --cold-ab-dir /tmp/cold_ab_runs --out /tmp/dashboard.html

  # PF_BENCH のログを prefill / decode の両パネルに重ねる
  # (PF_BENCH.sh の payload は max_tokens=32 なので --max-tokens 32 を渡す)
  python3 scripts/dashboard.py --cold-ab-dir /tmp/cold_ab_runs \
      --pf-bench-log /tmp/pf_bench.log --max-tokens 32 --out /tmp/dashboard.html

  # プロトコル変更の era 名を渡すと、そこに縦線を引く (繰り返し可)
  python3 scripts/dashboard.py --cold-ab-dir /tmp/cold_ab_runs \
      --protocol-change s104 --protocol-change s107 --out /tmp/dashboard.html

  # GPU 無しの自己テスト
  python3 scripts/dashboard.py --self-test

なぜ進捗ログを読まないのか:
  docs の進捗ログは自由文なのでパースしない。機械可読な正は scripts/cold_ab.py の
  --json 出力で、その形は run_ab() が返す dict そのもの:

    {"payload_sha256": str, "payload_bytes": int,
     "measurements": [{"round": int, "binary": "old"|"new", "binary_path": str,
                       "wall": float|null, "input_tokens": int|null,
                       "output_tokens": int|null, "boot_s": float|null,
                       "ok": bool, "error": str, "log": str}, ...],
     "stats": {"old": {"n": int, "median": float|null, "runs": [float, ...]},
               "new": {"n": int, "median": float|null, "runs": [float, ...]},
               "delta_pct": float|null},
     "guard_tripped": bool, "leftover_pids": [int, ...],
     "binaries": {"old": str, "new": str}, "rounds": int,
     "protocol": {"model": str, "quant": str, "port": int, "seed": int,
                  "max_tokens": int, "expect_input_tokens": int|null,
                  "source_doc": str}}

パネルと軸:
  prefill = cold A/B の wall 秒 (成功した measurement だけを、ファイル名順 -> ラウンド順
  に並べた「測定順」) と、PF_BENCH ログの PF wall 秒 (ファイル順)。
  decode  = ms/token。cold A/B の JSON には decode の数値が無いので、decode の点は
  すべて PF_BENCH ログ行から作る: wall 秒 * 1000 / tokens。tokens はその行に
  output_tokens があればその実測値、無ければ --max-tokens で渡した値。
  どちらも無い行は点を作らず、理由を HTML に出す (推測しない)。

exit code:
  0  出力した
  1  入力が使えない (JSON が壊れている / dir が無い / 点が 1 つも無い)
  2  引数の誤り

標準ライブラリのみ。HTML はインライン SVG だけで、外部 CSS/JS/フォント/画像を参照しない。
出力は決定的 (同じ入力 -> byte 一致)。現在時刻・乱数・dict の反復順・ロケール依存の
小数表示に依存しない (数値はすべて明示的な書式で出す)。
"""

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# 数値の書式は明示する (locale に依存させない)
FMT_WALL = "%.3f"
FMT_DECODE = "%.2f"

# SVG のレイアウト (px)
WIDTH = 760
HEIGHT = 260
PAD_LEFT = 78
PAD_RIGHT = 26
PAD_TOP = 34
PAD_BOTTOM = 46
PLOT_W = WIDTH - PAD_LEFT - PAD_RIGHT
PLOT_H = HEIGHT - PAD_TOP - PAD_BOTTOM


class InputError(Exception):
    """入力が使えない (exit 1)。"""


# ---------------------------------------------------------------------------
# 入力の読み取り (すべて純関数)
# ---------------------------------------------------------------------------

def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(v):
    """float にして有限かを確かめる。bool / 文字列 / NaN / inf は None。"""
    if not _is_number(v):
        return None
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _finite_text(v):
    """ログ文字列用: 数値として読めて有限なら float、それ以外は None。

    JSON の値 (数値でなければならない) には _finite を使い、こちらは
    PF_BENCH のログ行 ("12.345s" のような単位付きの文字列) にだけ使う。
    """
    if v is None:
        return None
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def parse_cold_ab_payload(payload, name="<payload>"):
    """cold_ab.py --json の dict を検査し、正規化した測定に直す。

    中央値 / delta の構造 (stats.old.median, stats.new.median, stats.delta_pct) が
    無いものは「使えない入力」として InputError を投げる。フィールドを推測しない。
    """
    if not isinstance(payload, dict):
        raise InputError("%s: the JSON top level is not an object" % name)

    stats = payload.get("stats")
    if not isinstance(stats, dict):
        raise InputError("%s: no 'stats' object (is this cold_ab.py --json output?)" % name)
    for side in ("old", "new"):
        s = stats.get(side)
        if not isinstance(s, dict):
            raise InputError("%s: stats.%s is missing" % (name, side))
        if "median" not in s:
            raise InputError("%s: stats.%s.median is missing" % (name, side))
        if s["median"] is not None and _finite(s["median"]) is None:
            raise InputError("%s: stats.%s.median is not a finite number" % (name, side))
    if "delta_pct" not in stats:
        raise InputError("%s: stats.delta_pct is missing" % name)
    if stats["delta_pct"] is not None and _finite(stats["delta_pct"]) is None:
        raise InputError("%s: stats.delta_pct is not a finite number" % name)

    measurements = payload.get("measurements")
    if not isinstance(measurements, list):
        raise InputError("%s: 'measurements' is not a list" % name)

    points = []
    for m in measurements:
        if not isinstance(m, dict):
            raise InputError("%s: a measurement is not an object" % name)
        wall = _finite(m.get("wall"))
        # 失敗ラウンド / wall が無いラウンドは点にしない (cold_ab.summarise と同じ扱い)
        if not m.get("ok") or wall is None or wall <= 0:
            continue
        points.append({"round": m.get("round"), "binary": m.get("binary"), "wall": wall})

    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        protocol = {}
    # era の出どころ: cold_ab.py --json --era LABEL が "era" を直接書く (s110 以降)。
    # s110 以前の JSON や手で足した JSON のために "label" も読み、どちらも無ければ
    # ファイル名を使う (HTML の表にも出すが、それは fallback であることを表す)。
    label = payload.get("era")
    era_source = "era"
    if label is None:
        label = payload.get("label")
        era_source = "label"
    if label is None:
        label = name
        era_source = "filename"
    return {
        "name": name,
        "points": points,
        "old_median": _finite(stats["old"].get("median")),
        "new_median": _finite(stats["new"].get("median")),
        "delta_pct": _finite(stats["delta_pct"]),
        "label": label,
        "era_source": era_source,
        "model": protocol.get("model"),
        "quant": protocol.get("quant"),
        "payload_sha256": payload.get("payload_sha256"),
    }


def load_cold_ab_dir(path):
    """ディレクトリ内の *.json をファイル名順に読む (決定的)。"""
    if not os.path.isdir(path):
        raise InputError("%s: not a directory" % path)
    names = sorted(n for n in os.listdir(path) if n.endswith(".json"))
    runs = []
    for n in names:
        full = os.path.join(path, n)
        if not os.path.isfile(full):
            raise InputError("%s: not a regular file" % full)
        try:
            with open(full, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except ValueError as exc:
            raise InputError("%s: malformed JSON (%s)" % (full, exc))
        except OSError as exc:
            raise InputError("%s: cannot read (%s)" % (full, exc))
        runs.append(parse_cold_ab_payload(payload, name=n))
    return runs


def parse_pf_bench_log(text, name="<log>"):
    """PF_BENCH のログ行を点にする。wall / input_tokens / tok/s を読む。

    返り値は [{"wall": float, "input_tokens": int|None, "tok_per_s": float|None,
              "output_tokens": int|None, "line": int}, ...] (ファイル順)。
    """
    points = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line.startswith("PF wall="):
            continue
        fields = {}
        for token in line.split():
            if "=" in token:
                k, _, v = token.partition("=")
                fields[k] = v
            elif token.endswith("s") and _finite_text(token[:-1]) is not None:
                # PF_BENCH.sh は echo "PF wall=${OUT}s ..." なので、値は "12.345s" と
                # 単位付きで出る。単位を落として読む。
                fields.setdefault("wall", token[:-1])
        # wall の値に単位が残っている ("wall=12.345s") 場合も同じく落とす。
        wall = _finite_text(str(fields.get("wall", "")).rstrip("s"))
        if wall is None:
            continue
        it = fields.get("input_tokens")
        input_tokens = int(it) if it is not None and it.lstrip("-").isdigit() else None
        tps_raw = fields.get("prefill_tok_per_s")
        if tps_raw is None:
            tps_raw = fields.get("tok/s")
            if tps_raw is None:
                tps_raw = fields.get("tok_per_s")
        tok_per_s = _finite_text(tps_raw)
        ot = fields.get("output_tokens")
        output_tokens = int(ot) if ot is not None and ot.lstrip("-").isdigit() else None
        points.append({"wall": wall, "input_tokens": input_tokens,
                       "tok_per_s": tok_per_s, "output_tokens": output_tokens,
                       "line": lineno})
    return points


def load_pf_bench_log(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as exc:
        raise InputError("%s: cannot read (%s)" % (path, exc))
    return parse_pf_bench_log(text, name=os.path.basename(path))


def prefill_points(runs, pf_points):
    """prefill パネルの点: cold A/B の成功 wall (ファイル名順 -> ラウンド順) + PF wall。"""
    pts = []
    for run in runs:
        for p in run["points"]:
            pts.append({"value": p["wall"], "source": "cold_ab", "run": run["name"],
                        "binary": p["binary"], "era": run["label"]})
    for i, p in enumerate(pf_points, start=1):
        pts.append({"value": p["wall"], "source": "pf_bench", "run": "PF line %d" % p["line"],
                    "binary": None, "era": None})
    return pts


def decode_points(pf_points, max_tokens=None):
    """decode パネルの点 (ms/token)。PF 行の wall 秒から作る。

    output_tokens が行にあればそれを使い、無ければ max_tokens を使う (PF_BENCH.sh の
    payload は max_tokens=32 で固定)。どちらも無ければ点を作らず、理由を返す。
    """
    pts, skipped = [], 0
    for p in pf_points:
        n = p["output_tokens"] if p["output_tokens"] else max_tokens
        if not n or n <= 0:
            skipped += 1
            continue
        pts.append({"value": p["wall"] * 1000.0 / float(n), "source": "pf_bench",
                    "run": "PF line %d" % p["line"], "binary": None, "era": None,
                    "tokens": n})
    return pts, skipped


# ---------------------------------------------------------------------------
# レイアウト (純関数)
# ---------------------------------------------------------------------------

def axis_range(values):
    """0 起点で上端に余白を付けた (lo, hi)。全値が同じでも lo==hi にしない。"""
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    # prefill / decode とも 0 を下辺にする (棒ではなく折れ線だが、0 基準のほうが読みやすい)
    lo = 0.0 if lo >= 0 else lo
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    return lo, hi + span * 0.10


def nice_ticks(lo, hi, count=5):
    """lo..hi を等分した目盛り値。最大値の 10% 余白の上で決めるので単調。"""
    if count < 2:
        count = 2
    step = (hi - lo) / float(count - 1)
    return [lo + step * i for i in range(count)]


def project(points, lo, hi, x_lo, x_hi):
    """(index, value) を SVG 座標に直す。点が 1 つでも中央に置く (NaN にしない)。"""
    out = []
    n = len(points)
    span = hi - lo
    if span <= 0:
        span = 1.0
    for i, p in enumerate(points):
        if n <= 1:
            fx = 0.5
        else:
            fx = i / float(n - 1)
        x = x_lo + fx * (x_hi - x_lo)
        fy = (p["value"] - lo) / span
        y = (PLOT_H) - fy * PLOT_H   # パネル内座標 (上が小さい)
        out.append((x, y))
    return out


# ---------------------------------------------------------------------------
# SVG / HTML (純関数・すべて明示書式)
# ---------------------------------------------------------------------------

def esc(text):
    s = "" if text is None else str(text)
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def fmt_num(v, spec):
    try:
        return spec % float(v)
    except (TypeError, ValueError):
        return "-"


def render_panel(title, unit, points, markers, top, x_label, empty_note=None,
                 mark_axis_unit=None):
    """1 パネル分の SVG 断片を返す。top はパネル群の y オフセット。

    points: [{"value": float, "source": str, "run": str, "binary": str|None,
              "era": str|None}, ...]  (測定順)
    markers: [index, ...] — この x 位置に縦線を引く
    """
    out = []
    y0 = top + PAD_TOP
    x0 = PAD_LEFT
    lo, hi = axis_range([p["value"] for p in points])
    out.append('<g transform="translate(0,%d)">' % top)
    out.append('<text x="%d" y="%d" class="ptitle">%s (%s)</text>'
               % (PAD_LEFT, 18, esc(title), esc(unit)))

    # y 軸の目盛りと数値
    ticks = nice_ticks(lo, hi)
    span = hi - lo if hi > lo else 1.0
    for t in ticks:
        fy = (t - lo) / span
        y = y0 + PLOT_H - fy * PLOT_H
        out.append('<line x1="%d" y1="%s" x2="%d" y2="%s" class="grid"/>'
                   % (x0, fmt_num(y, "%.2f"), x0 + PLOT_W, fmt_num(y, "%.2f")))
        label = fmt_num(t, FMT_WALL) if unit.startswith("s") else fmt_num(t, FMT_DECODE)
        out.append('<text x="%d" y="%s" class="tick" text-anchor="end">%s</text>'
                   % (x0 - 8, fmt_num(y + 4, "%.2f"), esc(label)))
    out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="axis"/>'
               % (x0, y0 + PLOT_H, x0 + PLOT_W, y0 + PLOT_H))
    out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="axis"/>'
               % (x0, y0, x0, y0 + PLOT_H))
    out.append('<text x="%d" y="%d" class="axtitle">%s</text>'
               % (x0, y0 + PLOT_H + 18, esc(x_label)))
    out.append('<text x="%d" y="%d" class="axtitle" transform="rotate(-90 %d %d)">%s</text>'
               % (16, y0 + PLOT_H // 2, 16, y0 + PLOT_H // 2, esc(unit)))

    if len(points) < 2:
        note = empty_note or "not enough points"
        out.append('<text x="%d" y="%d" class="warn">%s</text>'
                   % (x0 + PLOT_W // 2, y0 + PLOT_H // 2, esc(note)))
        if len(points) == 1:
            xy = project(points, lo, hi, x0, x0 + PLOT_W)[0]
            out.append('<circle cx="%s" cy="%s" r="4" class="dot"/>'
                       % (fmt_num(xy[0], "%.2f"), fmt_num(y0 + xy[1], "%.2f")))
    else:
        xy = project(points, lo, hi, x0, x0 + PLOT_W)
        path = " ".join("%s,%s" % (fmt_num(x, "%.2f"), fmt_num(y0 + y, "%.2f"))
                        for x, y in xy)
        out.append('<polyline points="%s" class="series"/>' % path)
        for (x, y), p in zip(xy, points):
            out.append('<circle cx="%s" cy="%s" r="3.2" class="dot"/>'
                       % (fmt_num(x, "%.2f"), fmt_num(y0 + y, "%.2f")))
        # 点の数 (x 軸の両端に index)
        out.append('<text x="%d" y="%d" class="tick" text-anchor="middle">0</text>'
                   % (x0, y0 + PLOT_H + 14))
        out.append('<text x="%d" y="%d" class="tick" text-anchor="middle">%d</text>'
                   % (x0 + PLOT_W, y0 + PLOT_H + 14, len(points) - 1))

    # プロトコル変更の縦線 (era が一致した点の直前に入れる)
    for idx, label in markers:
        if len(points) <= 1:
            fx = 0.5
        else:
            fx = idx / float(len(points) - 1)
        x = x0 + fx * PLOT_W
        out.append('<line x1="%s" y1="%d" x2="%s" y2="%d" class="marker"/>'
                   % (fmt_num(x, "%.2f"), y0, fmt_num(x, "%.2f"), y0 + PLOT_H))
        out.append('<text x="%s" y="%d" class="marklabel">%s</text>'
                   % (fmt_num(x + 4, "%.2f"), y0 - 6, esc(label)))
    out.append("</g>")
    return out


def marker_indices(points, changes):
    """--protocol-change に一致する era が始まる (index, label) を返す (昇順)。

    点の era は cold A/B の JSON の "era" / "label" から来る。無ければファイル名。
    同じ era が連続する点の途中では線を引かない (era の境目だけに 1 本)。
    """
    if not changes:
        return []
    wanted = set(changes)
    out = []
    prev = None
    for i, p in enumerate(points):
        era = p.get("era")
        if era in wanted and era != prev:
            out.append((i, str(era)))
        prev = era
    return out


def render_html(runs, pf_points, pf_name, changes, max_tokens=None):
    """自己完結 HTML を返す (byte 決定的)。"""
    pre = prefill_points(runs, pf_points)
    dec, skipped = decode_points(pf_points, max_tokens)
    pre_markers = marker_indices(pre, changes)
    dec_markers = marker_indices(dec, changes)

    # ヘッダ表 (cold A/B の run ごとの中央値と delta)。era は JSON の "era" / "label"
    # が無いときだけファイル名になる (その場合は era_source == "filename")。
    rows = []
    for r in runs:
        rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
            "</tr>" % (esc(r["name"]), esc(r["label"]),
                       fmt_num(r["old_median"], FMT_WALL) if r["old_median"] is not None else "n/a",
                       fmt_num(r["new_median"], FMT_WALL) if r["new_median"] is not None else "n/a",
                       fmt_num(r["delta_pct"], "%+.1f") if r["delta_pct"] is not None else "n/a",
                       esc(r["quant"] or "-")))
    if not rows:
        rows.append('<tr><td colspan="6">no cold A/B measurements</td></tr>')

    svg = []
    svg.append('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
               'viewBox="0 0 %d %d" role="img">' % (WIDTH, HEIGHT,
                                                    WIDTH, HEIGHT))
    svg.extend(render_panel("Prefill", "s", pre, pre_markers, 0,
                            "measurement order (index)"))
    svg.append('</svg>')
    svg.append('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
               'viewBox="0 0 %d %d" role="img">' % (WIDTH, HEIGHT,
                                                    WIDTH, HEIGHT))
    dec_note = None
    if not dec and skipped:
        dec_note = ("not enough points: %d PF_BENCH line(s) carry no output_tokens "
                    "and no token count was given (pass --max-tokens)" % skipped)
    svg.extend(render_panel("Decode", "ms/token", dec, dec_markers, 0,
                            "measurement order (index)", empty_note=dec_note))
    svg.append('</svg>')

    css = ("svg{background:#fff} .series{fill:none;stroke:#1f5fbf;stroke-width:2}"
           " .dot{fill:#1f5fbf} .grid{stroke:#e3e6ea;stroke-width:1}"
           " .axis{stroke:#444;stroke-width:1} .tick{fill:#444;font:11px monospace}"
           " .axtitle{fill:#444;font:11px monospace}"
           " .ptitle{fill:#111;font:bold 13px monospace}"
           " .marker{stroke:#c03030;stroke-width:2;stroke-dasharray:5 4}"
           " .marklabel{fill:#c03030;font:11px monospace}"
           " .warn{fill:#8a5a00;font:12px monospace;text-anchor:middle}"
           " body{font:13px monospace;color:#111;margin:16px}"
           " table{border-collapse:collapse;margin-top:8px}"
           " td,th{border:1px solid #ccc;padding:3px 8px;text-align:left}")

    parts = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append("<title>denno prefill / decode dashboard</title>")
    parts.append("<style>%s</style></head><body>" % css)
    parts.append("<h1>denno prefill and decode</h1>")
    parts.append("<p>prefill: cold A/B wall seconds (successful measurements only) "
                 "and PF_BENCH wall seconds. decode: ms/token from PF_BENCH lines, "
                 "wall_seconds * 1000 / tokens (tokens = the line's output_tokens, "
                 "else --max-tokens=%s). Each panel has its own y axis.</p>"
                 % esc("not given" if max_tokens is None else "%d" % max_tokens))
    if changes:
        parts.append("<p>protocol change marker at era: %s</p>"
                     % esc(", ".join(changes)))
    else:
        parts.append("<p>no protocol change marker (pass --protocol-change)</p>")
    parts.append("".join(svg))
    parts.append("<h2>cold A/B runs</h2>")
    parts.append("<table><tr><th>file</th><th>era</th><th>old median (s)</th>"
                 "<th>new median (s)</th><th>delta %</th><th>quant</th></tr>")
    parts.extend(rows)
    parts.append("</table>")
    parts.append("<h2>points</h2>")
    parts.append("<p>prefill: %d point(s); decode: %d point(s); PF_BENCH log: %s</p>"
                 % (len(pre), len(dec), esc(pf_name or "not given")))
    parts.append("<ul>")
    for i, p in enumerate(pre):
        parts.append("<li>prefill[%d] = %s s (%s: %s%s)</li>"
                     % (i, fmt_num(p["value"], FMT_WALL), esc(p["source"]),
                        esc(p["run"]),
                        (", " + esc(p["binary"])) if p["binary"] else ""))
    for i, p in enumerate(dec):
        parts.append("<li>decode[%d] = %s ms/token (%s: %s, %d tokens)</li>"
                     % (i, fmt_num(p["value"], FMT_DECODE), esc(p["source"]),
                        esc(p["run"]), p["tokens"]))
    parts.append("</ul>")
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# 自己テスト (CPU のみ)
# ---------------------------------------------------------------------------

def self_test(out=sys.stdout):
    failures = []

    def check(name, cond, detail=""):
        if cond:
            print("ok   %s%s" % (name, (" — " + detail) if detail else ""), file=out)
        else:
            print("FAIL %s%s" % (name, (" — " + detail) if detail else ""), file=out)
            failures.append(name)

    log = ("PF wall=12.345s input_tokens=5686 prefill_tok_per_s=460.6\n"
           "noise line\n"
           "PF wall=11.000s input_tokens=5686 prefill_tok_per_s=516.9\n")
    pts = parse_pf_bench_log(log)
    check("pf log: two points", len(pts) == 2, str(len(pts)))
    check("pf log: wall parsed", pts and pts[0]["wall"] == 12.345, str(pts[:1]))
    check("pf log: input_tokens parsed", pts and pts[0]["input_tokens"] == 5686)
    check("pf log: tok/s parsed", pts and abs(pts[0]["tok_per_s"] - 460.6) < 1e-9)
    check("pf log: old spelling accepted",
          len(parse_pf_bench_log("PF wall=3.5s input_tokens=100 tok/s=28.6\n")) == 1)

    dec, skipped = decode_points(pts, max_tokens=32)
    check("decode: ms/token from wall", dec and abs(dec[0]["value"] - 385.78125) < 1e-6,
          str(dec[:1]))
    dec2, _ = decode_points(parse_pf_bench_log(
        "PF wall=1.000s input_tokens=10 output_tokens=200\n"))
    check("decode: output_tokens wins", dec2 and abs(dec2[0]["value"] - 5.0) < 1e-9)
    _, sk = decode_points(pts, max_tokens=None)
    check("decode: no token count -> skipped", sk == 2, str(sk))

    payload = {"payload_sha256": "abc", "payload_bytes": 1,
               "measurements": [{"round": 1, "binary": "old", "wall": 10.0, "ok": True},
                                {"round": 1, "binary": "new", "wall": 9.5, "ok": True},
                                {"round": 2, "binary": "new", "wall": None, "ok": False}],
               "stats": {"old": {"n": 1, "median": 10.0, "runs": [10.0]},
                         "new": {"n": 1, "median": 9.5, "runs": [9.5]},
                         "delta_pct": -5.0},
               "guard_tripped": False, "leftover_pids": [], "rounds": 1,
               "binaries": {"old": "/a", "new": "/b"},
               "protocol": {"model": "m", "quant": "int8", "port": 8095, "seed": 42,
                            "max_tokens": 32, "expect_input_tokens": 5686,
                            "source_doc": "d"}, "era": "s109"}
    run = parse_cold_ab_payload(payload, name="r1.json")
    check("cold ab: failed round dropped", len(run["points"]) == 2, str(len(run["points"])))
    check("cold ab: era read", run["label"] == "s109")
    bad = dict(payload)
    bad.pop("stats")
    try:
        parse_cold_ab_payload(bad, name="broken.json")
        check("cold ab: missing stats rejected", False)
    except InputError as exc:
        check("cold ab: missing stats rejected", "broken.json" in str(exc))

    check("axis: equal values do not divide by zero", axis_range([5.0, 5.0])[1] > 5.0)
    check("axis: empty is safe", axis_range([]) == (0.0, 1.0))
    xy = project([{"value": 1.0}], 0.0, 2.0, 0, 100)
    check("project: single point is finite", all(x == x for x in xy[0]), str(xy))

    html = render_html([run], pts, "pf.log", ["s109"])
    check("html: both panels titled", "Prefill (s)" in html and "Decode (ms/token)" in html)
    check("html: no script tag", "<script" not in html.lower())
    check("html: no external refs",
          "http://" not in html.replace('xmlns="http://www.w3.org/2000/svg"', "")
          and "https://" not in html and "cdn" not in html.lower())
    check("html: deterministic", html == render_html([run], pts, "pf.log", ["s109"]))
    check("html: marker present", 'class="marker"' in html)
    check("html: no marker without a match",
          'class="marker"' not in render_html([run], pts, "pf.log", []))
    check("html: no nan/inf", "nan" not in html.lower() and "inf" not in html.lower())

    print("", file=out)
    if failures:
        print("SELF-TEST FAILED: %d check(s)" % len(failures), file=out)
        return 1
    print("SELF-TEST PASSED (CPU only; no GPU, no server, no cargo)", file=out)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        prog="dashboard.py",
        description="Prefill and decode on separate axes, from cold_ab.py --json "
                    "runs and an optional PF_BENCH log (P5-1).",
        epilog="Exit: 0 ok, 1 unusable input, 2 usage error.")
    ap.add_argument("--cold-ab-dir", metavar="DIR",
                    help="directory of cold_ab.py --json outputs (*.json, name order)")
    ap.add_argument("--pf-bench-log", metavar="FILE",
                    help="PF_BENCH log with 'PF wall=<sec> input_tokens=<n> ...' lines")
    ap.add_argument("--max-tokens", type=int, metavar="N",
                    help="token count per PF_BENCH line when the line has no "
                         "output_tokens (PF_BENCH.sh sends max_tokens=32)")
    ap.add_argument("--protocol-change", metavar="LABEL", action="append", default=[],
                    help="era/label value; a vertical marker is drawn where it starts "
                         "(repeatable)")
    ap.add_argument("--out", metavar="FILE", help="self-contained HTML to write")
    ap.add_argument("--self-test", action="store_true",
                    help="CPU-only self test (no GPU, no server, no cargo)")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.max_tokens is not None and args.max_tokens < 1:
        ap.error("--max-tokens must be >= 1")

    if not args.cold_ab_dir:
        ap.error("--cold-ab-dir is required (or use --self-test)")
    if not args.out:
        ap.error("--out is required")

    try:
        runs = load_cold_ab_dir(args.cold_ab_dir)
        pf_points = load_pf_bench_log(args.pf_bench_log) if args.pf_bench_log else []
        pre = prefill_points(runs, pf_points)
        if not pre and not pf_points:
            raise InputError("%s: no measurements and no PF_BENCH points"
                             % args.cold_ab_dir)
        html = render_html(runs, pf_points, args.pf_bench_log, args.protocol_change,
                           args.max_tokens)
    except InputError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        # 失敗時に中途半端な --out を残さない
        try:
            if os.path.exists(args.out):
                os.remove(args.out)
        except OSError:
            pass
        return 1

    # 書き込みも一時ファイル経由 (失敗時に部分ファイルを残さない)
    tmp = args.out + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(html)
        os.replace(tmp, args.out)
    except OSError as exc:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        print("ERROR: cannot write %s (%s)" % (args.out, exc), file=sys.stderr)
        return 1
    print("wrote %s (%d bytes)" % (args.out, len(html.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())