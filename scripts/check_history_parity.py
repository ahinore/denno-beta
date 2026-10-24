#!/usr/bin/env python3
"""docs/history の日英同期チェッカ (CPU のみ・標準ライブラリのみ)。

英語版は日本語版の翻訳なので、日本語側を直したときに英語側が取り残される。
このスクリプトは「対応が崩れていないか」だけを機械的に見る。内容は直さない。

検査項目 (ja ファイルと en ファイルの対ごと):
  1. ファイル対応   ja にしか無い / en にしか無いファイルは FAIL
  2. 見出し数       ^# で始まる行の数が一致するか
  3. 表の行数       ^| で始まる行の数が一致するか
  4. コードブロック フェンスで囲まれた区間の数が一致するか (閉じ忘れは FAIL)
  5. commit SHA     ja 側で ` ` 囲みの 7〜40 桁 16 進トークンの集合。
                    en 側に無いものは FAIL (訳し漏れ)。en にしか無いものは WARN。
  6. 数値トークン   [0-9]+(\\.[0-9]+)? の個数。en が ja より 10% 超少なければ WARN、
                    30% 超少なければ FAIL (訳し漏れの疑い)。
  7. en の冒頭      先頭 10 行以内に "Japanese original:" 行があり、
                    そのリンク先が実在するか (無ければ FAIL)。
  8. ja のナビ行    "> [索引]" で始まる行に [English](en/...md) があり、
                    リンク先が実在するか (無ければ FAIL)。索引 README.md は
                    ナビ行の代わりに en/README.md へのリンクを見る。

判定と終了コード:
  0  FAIL が 1 件も無い (PASS / WARN のみ)
  1  FAIL が 1 件以上ある
  2  前提が壊れている (history ディレクトリや en/ が無い、読めない)

数値は「件数」しか見ない。訳文の質も、数値の中身が正しいかも見ない。
ネットワーク・外部依存は使わない。

使い方:
  python3 scripts/check_history_parity.py
  python3 scripts/check_history_parity.py --history-dir /tmp/copy/docs/history
"""
import argparse
import os
import re
import sys

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_SETUP = 2

HEADING_RE = re.compile(r"^#")
TABLE_ROW_RE = re.compile(r"^\|")
FENCE_RE = re.compile(r"^```")
SHA_RE = re.compile(r"`([0-9A-Fa-f]{7,40})`")
NUM_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")
NAV_PREFIX = "> [索引]"
EN_LINK_RE = re.compile(r"\[English\]\(([^)]+)\)")
BACKLINK_MARK = "Japanese original:"
MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

INDEX_NAME = "README.md"
WARN_SHORTFALL = 0.10   # en がこれより少ないと WARN
FAIL_SHORTFALL = 0.30   # en がこれより少ないと FAIL

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class SetupError(Exception):
    """前提が壊れている (exit 2)。"""


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def count_headings(lines):
    return sum(1 for line in lines if HEADING_RE.match(line))


def count_table_rows(lines):
    return sum(1 for line in lines if TABLE_ROW_RE.match(line))


def count_code_blocks(lines):
    """閉じたフェンス区間の数と、閉じ忘れの有無を返す。"""
    blocks = 0
    inside = False
    for line in lines:
        if FENCE_RE.match(line):
            if inside:
                inside = False
            else:
                inside = True
                blocks += 1
    return blocks, inside


def sha_tokens(text):
    return set(m.group(1).lower() for m in SHA_RE.finditer(text))


def count_numbers(text):
    return len(NUM_RE.findall(text))


def _shortfall(ja_count, en_count):
    if ja_count == 0:
        return None
    return (ja_count - en_count) / float(ja_count)


def _pct(value):
    return "%.1f%%" % (value * 100.0)


def compare_pair(name, ja_text, en_text):
    """1 対の比較。[(verdict, 項目, メッセージ)] を返す。"""
    out = []
    ja_lines = ja_text.splitlines()
    en_lines = en_text.splitlines()

    ja_h, en_h = count_headings(ja_lines), count_headings(en_lines)
    out.append((
        PASS if ja_h == en_h else FAIL, "headings",
        "見出し %d/%d (ja/en)" % (ja_h, en_h),
    ))

    ja_t, en_t = count_table_rows(ja_lines), count_table_rows(en_lines)
    out.append((
        PASS if ja_t == en_t else FAIL, "table",
        "表の行 %d/%d (ja/en)" % (ja_t, en_t),
    ))

    ja_b, ja_unclosed = count_code_blocks(ja_lines)
    en_b, en_unclosed = count_code_blocks(en_lines)
    if ja_unclosed or en_unclosed:
        out.append((
            FAIL, "fences",
            "閉じていないフェンスがある (ja=%s en=%s)"
            % ("あり" if ja_unclosed else "なし", "あり" if en_unclosed else "なし"),
        ))
    else:
        out.append((
            PASS if ja_b == en_b else FAIL, "fences",
            "コードブロック %d/%d (ja/en)" % (ja_b, en_b),
        ))

    ja_sha, en_sha = sha_tokens(ja_text), sha_tokens(en_text)
    missing = sorted(ja_sha - en_sha)
    extra = sorted(en_sha - ja_sha)
    if missing:
        out.append((
            FAIL, "sha",
            "en に無い SHA %d 件: %s" % (len(missing), ", ".join(missing)),
        ))
    elif extra:
        out.append((
            WARN, "sha",
            "ja に無い SHA %d 件: %s" % (len(extra), ", ".join(extra)),
        ))
    else:
        out.append((PASS, "sha", "SHA %d/%d (ja/en)" % (len(ja_sha), len(en_sha))))

    ja_n, en_n = count_numbers(ja_text), count_numbers(en_text)
    gap = _shortfall(ja_n, en_n)
    if gap is None:
        out.append((PASS, "numbers", "数値トークン ja=0 en=%d (比較対象外)" % en_n))
    elif gap > FAIL_SHORTFALL:
        out.append((
            FAIL, "numbers",
            "数値トークン %d/%d (ja/en): en が %s 少ない (>%s)"
            % (ja_n, en_n, _pct(gap), _pct(FAIL_SHORTFALL)),
        ))
    elif gap > WARN_SHORTFALL:
        out.append((
            WARN, "numbers",
            "数値トークン %d/%d (ja/en): en が %s 少ない (>%s)"
            % (ja_n, en_n, _pct(gap), _pct(WARN_SHORTFALL)),
        ))
    elif gap < 0:
        out.append((
            PASS, "numbers",
            "数値トークン %d/%d (ja/en): en が %s 多い" % (ja_n, en_n, _pct(-gap)),
        ))
    else:
        out.append((
            PASS, "numbers",
            "数値トークン %d/%d (ja/en): en が %s 少ない" % (ja_n, en_n, _pct(gap)),
        ))

    return out


def check_backlink(en_dir, name, en_text):
    """en 側の先頭 10 行に Japanese original: と、実在するリンクがあるか。"""
    head = en_text.splitlines()[:10]
    hits = [line for line in head if BACKLINK_MARK in line]
    if not hits:
        return (FAIL, "backlink", '先頭 10 行に "%s" の行が無い' % BACKLINK_MARK)
    links = MD_LINK_RE.findall(hits[0])
    if not links:
        return (FAIL, "backlink", '"%s" の行にリンクが無い: %s' % (BACKLINK_MARK, hits[0]))
    target = links[0]
    path = os.path.normpath(os.path.join(en_dir, target))
    if not os.path.isfile(path):
        return (FAIL, "backlink", "リンク先が無い: %s" % target)
    if os.path.basename(path) != name:
        return (
            WARN, "backlink",
            "リンク先 %s が対応する ja ファイル %s と違う" % (target, name),
        )
    return (PASS, "backlink", "リンク先 %s が実在" % target)


def check_nav(history_dir, name, ja_text):
    """ja 側のナビ行に [English](en/...) があり、リンク先が実在するか。"""
    if name == INDEX_NAME:
        # 索引はナビ行を持たないので、本文の en/README.md リンクで代替する。
        if "en/" + INDEX_NAME in ja_text:
            return (PASS, "nav", "索引: en/%s へのリンクあり" % INDEX_NAME)
        return (FAIL, "nav", "索引: en/%s へのリンクが無い" % INDEX_NAME)

    navs = [line for line in ja_text.splitlines() if line.startswith(NAV_PREFIX)]
    if not navs:
        return (FAIL, "nav", '"%s" で始まるナビ行が無い' % NAV_PREFIX)

    problems = []
    for nav in navs:
        links = EN_LINK_RE.findall(nav)
        if not links:
            problems.append("ナビ行に [English](...) が無い")
            continue
        target = links[0]
        path = os.path.normpath(os.path.join(history_dir, target))
        if not os.path.isfile(path):
            problems.append("リンク先が無い: %s" % target)
        elif os.path.basename(path) != name:
            problems.append("リンク先 %s が %s と違う" % (target, name))
    if problems:
        return (FAIL, "nav", " / ".join(problems))
    return (PASS, "nav", "ナビ行 -> %s" % EN_LINK_RE.findall(navs[0])[0])


def run(history_dir):
    """全検査を回して (行リスト, PASS/WARN/FAIL 件数, 欠落ファイル数) を返す。"""
    en_dir = os.path.join(history_dir, "en")
    if not os.path.isdir(history_dir):
        raise SetupError("history ディレクトリが無い: %s" % history_dir)
    if not os.path.isdir(en_dir):
        raise SetupError("en ディレクトリが無い: %s" % en_dir)

    ja_names = sorted(n for n in os.listdir(history_dir)
                      if n.endswith(".md") and os.path.isfile(os.path.join(history_dir, n)))
    en_names = sorted(n for n in os.listdir(en_dir)
                      if n.endswith(".md") and os.path.isfile(os.path.join(en_dir, n)))

    rows = []
    counts = {PASS: 0, WARN: 0, FAIL: 0}
    missing = 0

    def emit(verdict, label, message):
        counts[verdict] += 1
        rows.append((verdict, label, message))

    only_ja = [n for n in ja_names if n not in en_names]
    only_en = [n for n in en_names if n not in ja_names]
    if only_ja or only_en:
        parts = []
        if only_ja:
            parts.append("ja にしか無い: %s" % ", ".join(only_ja))
        if only_en:
            parts.append("en にしか無い: %s" % ", ".join(only_en))
        emit(FAIL, "files", " / ".join(parts))
        missing = len(only_ja) + len(only_en)
    else:
        emit(PASS, "files", "ja %d 本 / en %d 本 / %d 対"
             % (len(ja_names), len(en_names), len(ja_names)))

    for name in ja_names:
        if name not in en_names:
            continue
        ja_text = read_text(os.path.join(history_dir, name))
        en_text = read_text(os.path.join(en_dir, name))
        results = compare_pair(name, ja_text, en_text)
        results.append(check_backlink(en_dir, name, en_text))
        results.append(check_nav(history_dir, name, ja_text))
        for verdict, item, message in results:
            emit(verdict, "%s %s" % (name, item), message)

    return rows, counts, missing


def main(argv=None):
    default_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "history")
    parser = argparse.ArgumentParser(
        description="docs/history の日英同期を検査する (内容は変更しない)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history-dir", default=default_dir,
                        help="検査する history ディレクトリ (既定: <repo>/docs/history)")
    args = parser.parse_args(argv)

    history_dir = os.path.abspath(args.history_dir)
    print("history parity: %s" % history_dir)
    try:
        rows, counts, missing = run(history_dir)
    except SetupError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return EXIT_SETUP

    width = max(len(label) for _, label, _ in rows) if rows else 0
    for verdict, label, message in rows:
        print("%s %-*s  %s" % (verdict, width, label, message))

    total = counts[PASS] + counts[WARN] + counts[FAIL]
    ok = counts[FAIL] == 0
    print("-" * 60)
    print("summary: %d PASS / %d WARN / %d FAIL (%d checks, %d file missing) -> %s"
          % (counts[PASS], counts[WARN], counts[FAIL], total, missing,
             "OK" if ok else "FAIL"))
    return EXIT_OK if ok else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
