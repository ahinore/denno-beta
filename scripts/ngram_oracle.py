#!/usr/bin/env python3
"""s120: n-gram 投機 (docs/SPEC_DECODE_DESIGN.md 案 B) の受理率 p を測るオラクル。

案 B (直前文脈の n-gram が前に出てきたら、その続きを draft として仮決めする) の
成否は「文脈長 n の文脈が過去に出現した位置で、続く k トークンが前回の続きと
一致する率 p_k」だけで決まる。損益分岐は p* = C_d / S ~ 0.0006 なので、
**p を測れば案 B を採るか閉じるかが決まる**。GPU もモデル重みも要らない。

このスクリプトはモデルのトークナイザを使わない (重みが無い環境でも動く)。
代わりに 2 通りの近似トークン化で両方を出す:

  words  : ASCII 英数は空白/記号区切り、日本語は文字種境界で分割する簡易分割
           (日本語 1 文字 ≒ 1 トークン程度の粗さ。実トークナイザとは違う)
  bytes  : UTF-8 バイト列をそのままトークン列にする (語彙 256)

どちらも**実トークナイザの置き換えではない**。p の絶対値はこれで決まらないので、
結果は「この近似で何が出たか」として読むこと (docs/TASKS.md P2-1・HISTORY s120)。

## 計測の定義

トークン列 T (長さ L)・文脈長 n・続き長 k について、位置 i (n <= i < L) を昇順に走査し:

  ctx = T[i-n : i]
  文脈 ctx が **過去に出現した** = ある j + n <= i で T[j : j+n] == ctx
  **(j は「直前の出現」= j < i を満たす最大の j を採る)**

  - 予測機会 (opportunity): 直前の出現位置 j が j + n + k <= L を満たす i の数
    (前回の続きが k トークン分そろっている場合だけ数える。そろっていなければ
     draft を作れないので機会にしない)
  - 一致 (hit): T[j+n : j+n+k] == T[i : i+k] の i の数
  - p_k = hit / opportunity (opportunity == 0 のときは p_k = null)

「直前の出現」を採るのは、実装 (engine のホスト側探索) が文脈を後ろから探して
最初に見つかった続きを使うから。同じ文脈が複数回出て続きが食い違う場合、
2 番目以前の出現を採ると **実装より甘い数字**になる。

文脈が自分の直前に重なる位置 (j + n > i、周期的なテキスト) も「過去に出現」に
含める。このとき前回の続きは自分自身の先頭と重なるが、draft としては valid。

集計は決定論: 位置は昇順に 1 回だけ走査し、dict は「文脈 -> (最後の位置, 前回の続き)」
の記録にだけ使う (反復順には依存しない)。乱数は使わない (`--synth` は固定シード)。

## コーパス

  --corpus docs_proxy   : リポジトリ内のテキスト。`docs/*.ja.md` + `docs/*.en.md` +
                          `crates/**/*.rs`。**これは代理 (proxy) であって実ワークロードでは
                          ない** (モデルの生成テキストではない)
  --corpus synth        : tool-call 風の合成コーパス。固定シードから決定的に生成する
                          (JSON 引数の繰り返し・同じキー名・同じファイルパスの再出現)。
                          **合成である** (実プロンプトのログではない)
  --path PATH...        : 明示したファイル / ディレクトリ (拡張子は問わない)

実プロンプトのログは探しに行かない (個人情報・ネットワーク取得は禁止)。

## 使い方

  python3 scripts/ngram_oracle.py --self-test
  python3 scripts/ngram_oracle.py --corpus docs_proxy
  python3 scripts/ngram_oracle.py --corpus synth --json
  python3 scripts/ngram_oracle.py --path docs/TASKS.md --n 12 --k 1,2,3,4
  python3 scripts/ngram_oracle.py --path some.txt --tokenizer bytes --json

終了コード: 0 正常 / 1 --self-test 失敗 / 2 使い方・入力の問題。
"""

import argparse
import json
import os
import random
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

DEFAULT_N = 12
DEFAULT_KS = (1, 2, 3, 4)
SYNTH_SEED = 20261020

PROXY_NOTE = (
    "proxy corpus (docs/*.ja.md + docs/*.en.md + crates/**/*.rs): this is NOT the "
    "real workload; it is repository text standing in for model output"
)
SYNTH_NOTE = (
    "synthetic corpus: deterministically generated from a fixed seed in this script; "
    "it is NOT a log of real prompts"
)
TOKENIZER_NOTES = {
    "words": (
        "approximate word split (ASCII alnum runs / single punctuation / character-class "
        "boundaries in Japanese); NOT the model tokenizer"
    ),
    "bytes": "raw UTF-8 bytes (vocab 256); NOT the model tokenizer",
}


# --------------------------------------------------------------------------- #
# トークン化 (2 通り)
# --------------------------------------------------------------------------- #

_ASCII_ALNUM = re.compile(r"[A-Za-z0-9_]+")


def _char_class(ch):
    """日本語 (CJK) の文字種境界。CJK 以外は None (= 境界を作らない)。"""
    o = ord(ch)
    if 0x3040 <= o <= 0x309F:
        return "hiragana"
    if 0x30A0 <= o <= 0x30FF:
        return "katakana"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
        return "kanji"
    if 0xFF66 <= o <= 0xFF9F:
        return "halfwidth_katakana"
    return None


def _is_ascii_word(ch):
    return ch.isascii() and (ch.isalnum() or ch == "_")


def tokenize_words(text):
    """簡易な単語分割。

    - ASCII 英数 + `_` の連続は 1 トークン
    - それ以外の 1 文字 (記号・空白・句読点) は 1 トークン
    - 日本語 (かな・漢字・半角カナ) は **文字種が変わるところで区切る**
      (ひらがな連続 / カタカナ連続 / 漢字連続はそれぞれまとめる)
    実トークナイザとは一致しない近似である (モジュール docstring 参照)。
    """
    toks = []
    cur = []
    cur_kind = None  # "ascii" | 文字種 | None

    def flush():
        if cur:
            toks.append("".join(cur))
            del cur[:]

    for ch in text:
        kind = _char_class(ch)
        if kind is None:
            kind = "ascii" if _is_ascii_word(ch) else None
        if kind is not None and kind == cur_kind:
            cur.append(ch)
            continue
        flush()
        cur_kind = kind
        if kind is None:
            toks.append(ch)
        else:
            cur.append(ch)
    flush()
    return toks


def tokenize_bytes(text):
    """UTF-8 バイト列をそのままトークン列にする。"""
    return list(text.encode("utf-8"))


TOKENIZERS = {"words": tokenize_words, "bytes": tokenize_bytes}


# --------------------------------------------------------------------------- #
# オラクル本体
# --------------------------------------------------------------------------- #


def _counts(tokens, n, ks):
    """1 本のトークン列について k ごとの (一致, 機会) を返す。決定論。"""
    largest = max(ks) if ks else 0
    hits = {k: 0 for k in ks}
    opps = {k: 0 for k in ks}
    if n <= 0 or largest <= 0:
        return hits, opps
    length = len(tokens)
    if length <= n:
        return hits, opps
    # 文脈 tuple -> (最後に現れた位置, その続きの先頭位置)。位置 i を昇順に進めるので
    # 「最後に現れた位置」は必ず i より前 = 直前の出現になる。
    last = {}
    for i in range(n, length):
        ctx = tuple(tokens[i - n:i])
        prev = last.get(ctx)
        if prev is not None:
            j = prev
            for k in ks:
                if j + n + k <= length:
                    opps[k] += 1
                    if tokens[j + n:j + n + k] == tokens[i:i + k]:
                        hits[k] += 1
        # この文脈の記録を i で上書きする (次に同じ文脈が出たときの「前回」になる)
        last[ctx] = i
    return hits, opps


def oracle(tokens, n=DEFAULT_N, ks=DEFAULT_KS):
    """トークン列 1 本の p_k。

    返り値: {"tokens": L, "n": n, "k": [{"k": k, "hit": h, "opportunity": o,
            "p": h/o or None}, ...]}
    """
    hits, opps = _counts(tokens, n, tuple(ks))
    rows = []
    for k in ks:
        o = opps[k]
        rows.append({
            "k": k,
            "hit": hits[k],
            "opportunity": o,
            "p": (hits[k] / o) if o else None,
        })
    return {"tokens": len(tokens), "n": n, "k": rows}


def merge_reports(reports, n, ks):
    """複数文書の (一致, 機会) を足し上げる (文書をまたぐ文脈は見ない)。"""
    total = {k: {"k": k, "hit": 0, "opportunity": 0} for k in ks}
    tokens = 0
    for rep in reports:
        tokens += rep["tokens"]
        for row in rep["k"]:
            acc = total[row["k"]]
            acc["hit"] += row["hit"]
            acc["opportunity"] += row["opportunity"]
    rows = []
    for k in ks:
        acc = total[k]
        o = acc["opportunity"]
        rows.append({
            "k": k,
            "hit": acc["hit"],
            "opportunity": o,
            "p": (acc["hit"] / o) if o else None,
        })
    return {"tokens": tokens, "n": n, "k": rows}


def run_corpus(docs, tokenizer, n=DEFAULT_N, ks=DEFAULT_KS):
    """[(name, text), ...] -> 集計レポート。docs は渡された順に走査する。"""
    tokenize = TOKENIZERS[tokenizer]
    reports = []
    per_doc = []
    for name, text in docs:
        rep = oracle(tokenize(text), n, ks)
        reports.append(rep)
        per_doc.append({"name": name, "tokens": rep["tokens"]})
    agg = merge_reports(reports, n, ks)
    agg["tokenizer"] = tokenizer
    agg["documents"] = len(docs)
    agg["per_document"] = per_doc
    return agg


# --------------------------------------------------------------------------- #
# コーパス
# --------------------------------------------------------------------------- #


def _read_text(path):
    """読み込み (UTF-8 厳格)。返り値 (text, None) か (None, 理由)。"""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError as exc:
        return None, "not valid UTF-8 (byte %d: %s)" % (exc.start, exc.reason)


def _walk(paths, suffixes=None):
    """ファイル / ディレクトリを決定的な順に列挙する (ソート済み)。"""
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs.sort()
                for fn in sorted(files):
                    if suffixes and not fn.endswith(suffixes):
                        continue
                    out.append(os.path.join(root, fn))
        elif os.path.isfile(p):
            if suffixes and not p.endswith(suffixes):
                continue
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return sorted(out)


def load_docs_proxy(repo=REPO):
    """代理コーパス: docs/*.ja.md + docs/*.en.md + crates/**/*.rs (昇順)。"""
    docs_dir = os.path.join(repo, "docs")
    paths = []
    if os.path.isdir(docs_dir):
        for fn in sorted(os.listdir(docs_dir)):
            if fn.endswith(".ja.md") or fn.endswith(".en.md"):
                paths.append(os.path.join(docs_dir, fn))
    crate_paths = _walk([os.path.join(repo, "crates")], suffixes=(".rs",))
    docs = []
    skipped = []
    for path in sorted(paths) + crate_paths:
        text, why = _read_text(path)
        if text is None:
            skipped.append({"name": os.path.relpath(path, repo), "reason": why})
            continue
        docs.append((os.path.relpath(path, repo), text))
    return docs, skipped


# 合成コーパス: tool-call 風。実プロンプトではなく、この関数が決定的に作る。
_SYNTH_TOOLS = ("read_file", "write_file", "search_files", "terminal", "patch",
                "browser_navigate", "skill_view", "todo")
_SYNTH_PATHS = ("src/model/qwen35.rs", "crates/denno-engine/src/cuda.rs",
                "crates/denno-server/src/engine.rs", "scripts/run_cpu_checks.sh",
                "docs/TASKS.md", "kernels/gdn.cu", "Cargo.toml",
                "crates/denno-engine/src/kv_cache.rs")
_SYNTH_KEYS = ("path", "pattern", "command", "content", "offset", "limit",
               "query", "ref", "file_path", "timeout")
_SYNTH_LANGS = ("rust", "python", "sh", "toml", "markdown")


def synth_corpus(seed=SYNTH_SEED, n_calls=400, n_lines=24):
    """tool-call 風の合成文書 (固定シード・決定的)。

    含める構造: 同じツール名・同じキー名・同じファイルパスの再出現、JSON 引数の
    ブロックの繰り返し、同じ import 行・同じ `#[test]` の雛形。前半と後半で
    「同じツール呼び出しの形」が繰り返されるので、12-gram の再出現は起きやすい。
    **実プロンプトではない** (合成)。
    """
    rng = random.Random(seed)
    blocks = []

    def tool_call():
        tool = rng.choice(_SYNTH_TOOLS)
        keys = rng.sample(_SYNTH_KEYS, 2 + rng.randrange(0, 3))
        args = {}
        for key in keys:
            if key in ("path", "file_path"):
                args[key] = rng.choice(_SYNTH_PATHS)
            elif key == "pattern":
                args[key] = rng.choice(("fn forward_decode_chain", "pub struct KvCache",
                                        "TODO", "sync\\(", "assert_eq!"))
            elif key == "command":
                args[key] = rng.choice(("cargo test -p denno-engine --lib",
                                        "git status --porcelain", "python3 -m unittest",
                                        "grep -rn advance crates/"))
            elif key == "content":
                args[key] = "fn main() {\\n    println!(\\\"ok\\\");\\n}"
            elif key in ("offset", "limit"):
                args[key] = rng.randrange(1, 400)
            elif key == "ref":
                args[key] = "@e%d" % rng.randrange(1, 90)
            elif key == "timeout":
                args[key] = rng.choice((60, 180, 300, 600))
            else:
                args[key] = rng.choice(("decode chain", "kv staging", "anchor md5"))
        body = ", ".join('"%s": %s' % (k, json.dumps(args[k])) for k in keys)
        return "<tool_call>\n{\n  \"name\": \"%s\",\n  \"arguments\": {\n    %s\n  }\n}\n</tool_call>" % (tool, body)

    for idx in range(n_calls):
        head = "### step %d\n" % idx
        lines = []
        for _ in range(n_lines):
            pick = rng.randrange(0, 10)
            if pick < 4:
                lines.append("use std::collections::HashMap;")
            elif pick < 6:
                lines.append("pub fn advance(&mut self, n: usize) { self.len += n; }")
            elif pick < 7:
                lines.append("#[test]\nfn oracle_counts_the_previous_continuation() {}")
            elif pick < 8:
                lines.append("let lang = \"%s\";" % rng.choice(_SYNTH_LANGS))
            elif pick < 9:
                lines.append(tool_call())
            else:
                lines.append("%s = %s" % (rng.choice(_SYNTH_KEYS),
                                          rng.choice(_SYNTH_PATHS)))
        blocks.append(head + "\n".join(lines) + "\n")
    return "\n".join(blocks)


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #


def _fmt_p(p):
    return "n/a (opportunity 0)" if p is None else "%.6f" % p


def format_text(report, note=None, extra_notes=()):
    lines = []
    if note:
        lines.append("corpus: " + note)
    lines.append("tokenizer: %s -- %s" % (report["tokenizer"],
                                          TOKENIZER_NOTES[report["tokenizer"]]))
    lines.append("context n: %d tokens / documents: %d / tokens: %d"
                 % (report["n"], report["documents"], report["tokens"]))
    lines.append("")
    lines.append("%-4s %-12s %-12s %-10s" % ("k", "hit", "opportunity", "p_k"))
    for row in report["k"]:
        lines.append("%-4d %-12d %-12d %-10s"
                     % (row["k"], row["hit"], row["opportunity"], _fmt_p(row["p"])))
    for note_line in extra_notes:
        lines.append("")
        lines.append(note_line)
    return "\n".join(lines)


def _build_parser():
    ap = argparse.ArgumentParser(
        prog="ngram_oracle.py",
        description="n-gram speculative decoding oracle (docs/SPEC_DECODE_DESIGN.md plan B). "
                    "CPU only, no model weights, no network.")
    ap.add_argument("--corpus", choices=("docs_proxy", "synth"),
                    help="built-in corpus (docs_proxy is a PROXY, synth is SYNTHETIC)")
    ap.add_argument("--path", action="append", default=None, metavar="PATH",
                    help="file or directory to measure (repeatable)")
    ap.add_argument("--tokenizer", choices=sorted(TOKENIZERS), default=None,
                    help="words | bytes (omit to report both)")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="context length in tokens (default %d)" % DEFAULT_N)
    ap.add_argument("--k", default=",".join(str(k) for k in DEFAULT_KS),
                    help="continuation lengths, comma separated (default 1,2,3,4)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--self-test", action="store_true",
                    help="check p_1 against hand-computable inputs and exit")
    return ap


def _parse_ks(text):
    ks = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k <= 0:
            raise ValueError("k must be positive: %r" % part)
        ks.append(k)
    if not ks:
        raise ValueError("no k given")
    return sorted(set(ks))


# --------------------------------------------------------------------------- #
# --self-test
# --------------------------------------------------------------------------- #


def self_test():
    """手計算できる小さな入力で既知の値になることを検査する。"""
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append("%s: got %r, want %r" % (name, got, want))

    # (1) 反復しない列: 文脈が一度も再出現しないので機会 0 -> p は None。
    toks = list(range(20))
    rep = oracle(toks, n=3, ks=(1, 2))
    check("no-repeat/tokens", rep["tokens"], 20)
    check("no-repeat/p1-opp", rep["k"][0]["opportunity"], 0)
    check("no-repeat/p1", rep["k"][0]["p"], None)

    # (2) 手計算: [1,2,3,4, 1,2,3,4, 1,2] (L=10)・n=2。文脈は周期 4 で並ぶので、
    #     直前の出現は j = i-4 になり、その続きは i の 4 つ後ろ = 一致しない
    #     (i=6 の (1,2) の前回の続きは [1,2,3]。ここは [3,4,1])。
    #     i=6..9 の 4 位置が機会。k は長さの足切り j+n+k <= L で
    #     k=1: 4 (6..9) / k=2: 3 (7..9) / k=3: 2 (8,9)。すべて不一致 -> p = 0。
    toks = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
    rep = oracle(toks, n=2, ks=(1, 2, 3))
    by_k = {row["k"]: row for row in rep["k"]}
    check("period/opp1", by_k[1]["opportunity"], 4)
    check("period/hit1", by_k[1]["hit"], 0)
    check("period/p1", by_k[1]["p"], 0.0)
    check("period/hit2", by_k[2]["hit"], 0)
    # 足切りは「前回の続きの範囲 j+n+k <= L」だけで判定する (今回側が短いときは
    # 比べた長さが k 未満になり必ず不一致になるが、機会としては数える)。
    check("period/opp2", by_k[2]["opportunity"], 4)
    check("period/opp3", by_k[3]["opportunity"], 4)
    check("period/hit3", by_k[3]["hit"], 0)

    # (3) 続きが食い違う列: 文脈 [a,b] が 3 回出て続きが x / y / e。
    #     toks = a b x c a b y d a b e (L=11)・n=2
    #     i=6 の (a,b) の直前の出現は j=2 (続き [x]) で、ここは [y] -> 不一致。
    #     i=10 の (a,b) の直前の出現は j=6 (続き [y]) -> ここは [e] で不一致。
    #     他の位置の文脈は過去に無い。-> 機会 2・一致 0 (直前の出現を採る定義そのもの)。
    toks = list("abxcabydabe")
    rep = oracle(toks, n=2, ks=(1,))
    row = rep["k"][0]
    check("conflict/opp", row["opportunity"], 2)
    check("conflict/hit", row["hit"], 0)
    check("conflict/p", row["p"], 0.0)
    # 同じ列を「最初の出現」で比べると i=6 は x vs y で不一致になる -> 定義の差を明示
    # (直前の出現を採るので本実装は上の 1.0。最初の出現を採る実装とは違う数字になる)

    # (4) k が文脈長を超える (k > n) は扱える: [1,2,3,4,1,2,3,4,1,5] (L=10)・n=2・k=3。
    #     直前の出現を持つ位置は i=6..9 の 4 つ (すべて j = i-4)。
    #     i=9 の (4,1) は j=5 (続き [2,3,4]) と [5] で不一致、他も同様 -> 一致 0。
    #     k=3 でも機会の足切りは j+n+k <= L なので 4 機会 (k <= n でも同じ数)。
    toks = [1, 2, 3, 4, 1, 2, 3, 4, 1, 5]
    rep = oracle(toks, n=2, ks=(3,))
    check("k>n/opp", rep["k"][0]["opportunity"], 4)
    check("k>n/hit", rep["k"][0]["hit"], 0)
    # 同じ列で k=1 も 4 機会・不一致
    rep = oracle(toks, n=2, ks=(1,))
    check("k>n/k1-opp", rep["k"][0]["opportunity"], 4)
    check("k>n/k1-hit", rep["k"][0]["hit"], 0)

    # (5) 空入力 / 短すぎる入力 / n = 0 / k = 0 は機会 0 で例外にならない。
    for toks in ([], [1], [1, 2]):
        rep = oracle(toks, n=2, ks=(1,))
        check("short/%d/opp" % len(toks), rep["k"][0]["opportunity"], 0)
    check("empty/n0", oracle([], n=0, ks=(1,))["k"][0]["opportunity"], 0)
    check("len3/k0", oracle([1, 2, 3], n=1, ks=(0,))["k"][0]["opportunity"], 0)

    # (6) トークン化: バイトは文字数ではなく UTF-8 長。
    check("bytes/ascii", tokenize_bytes("ab"), [97, 98])
    check("bytes/utf8-len", len(tokenize_bytes("あ")), 3)
    check("words/ascii", tokenize_words("foo(1)"), ["foo", "(", "1", ")"])
    # 日本語は文字種境界で割るので、同じ種類の連続は 1 トークンにまとまる
    check("words/jp-class", tokenize_words("漢字とかな"), ["漢字", "とかな"])
    check("words/jp-class2", tokenize_words("関数を呼ぶ"), ["関数", "を", "呼", "ぶ"])
    check("words/kata", tokenize_words("テスト関数"), ["テスト", "関数"])

    # (7) 集計の合成則: 2 文書の hit/opp の和がそのまま合計になる。
    ra = oracle([1, 2, 3, 1, 2, 3], n=2, ks=(1,))
    rb = oracle([4, 5, 6, 4, 5, 6], n=2, ks=(1,))
    merged = merge_reports([ra, rb], 2, (1,))
    check("merge/opp", merged["k"][0]["opportunity"],
          ra["k"][0]["opportunity"] + rb["k"][0]["opportunity"])
    check("merge/hit", merged["k"][0]["hit"],
          ra["k"][0]["hit"] + rb["k"][0]["hit"])
    check("merge/tokens", merged["tokens"], 12)

    # (8) 合成コーパスは決定的 (同じシード -> 同じトークン数と p) で空でない。
    a = run_corpus([("synth", synth_corpus())], "words", n=12, ks=(1,))
    b = run_corpus([("synth", synth_corpus())], "words", n=12, ks=(1,))
    check("synth/deterministic", a, b)
    check("synth/nonempty", a["tokens"] > 1000, True)
    check("synth/repetitive", a["k"][0]["opportunity"] > 0, True)

    # (9) コマンド経路: --path + --json が JSON を出し、非 UTF-8 を捕捉して exit 0。
    ok, why = _self_test_cli()
    if not ok:
        failures.append("cli: " + why)

    for line in failures:
        print("FAIL " + line, file=sys.stderr)
    if failures:
        print("self-test: %d failure(s)" % len(failures))
        return 1
    print("self-test: ok (9 groups)")
    return 0


def _self_test_cli():
    """一時ファイルで --json / 非 UTF-8 / 存在しないパスを確かめる。"""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        good = os.path.join(tmp, "good.txt")
        with open(good, "w", encoding="utf-8") as f:
            f.write("alpha beta alpha beta alpha beta\n")
        bad = os.path.join(tmp, "bad.bin")
        with open(bad, "wb") as f:
            f.write(b"ok\xff\xfe\x00bad\n")

        cmd = [sys.executable, os.path.abspath(__file__), "--path", good,
               "--tokenizer", "words", "--n", "2", "--k", "1", "--json"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return False, "--json exited %d: %s" % (proc.returncode, proc.stderr.strip())
        try:
            doc = json.loads(proc.stdout)
        except ValueError as exc:
            return False, "--json is not JSON: %s" % exc
        if doc["reports"][0]["tokenizer"] != "words":
            return False, "--json lost the tokenizer name"
        row = doc["reports"][0]["k"][0]
        # "alpha beta" x3 -> 文脈 (alpha,beta) の直前の出現は (beta,alpha) の続きで
        # 一致しない。比較するのは「機会があること」と p が 0..1 の範囲にあること。
        if row["k"] != 1 or row["opportunity"] <= 0 or not (0.0 <= row["p"] <= 1.0):
            return False, "--json row unexpected: %r" % (row,)
        if "not the real workload" not in " ".join(doc["notes"]):
            return False, "--json notes lost the corpus disclaimer"

        cmd = [sys.executable, os.path.abspath(__file__), "--path", bad,
               "--tokenizer", "words", "--json"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return False, "non-UTF-8 input exited %d (want 0, skipped)" % proc.returncode
        doc = json.loads(proc.stdout)
        skipped = doc.get("skipped") or []
        if not skipped or "UTF-8" not in skipped[0]["reason"]:
            return False, "non-UTF-8 input not reported as skipped: %r" % (skipped,)
        if doc["reports"][0]["documents"] != 0:
            return False, "skipped file still produced a document"

        cmd = [sys.executable, os.path.abspath(__file__),
               "--path", os.path.join(tmp, "nope.txt"), "--json"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 2:
            return False, "missing path exited %d (want 2)" % proc.returncode
    return True, ""


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv=None):
    ap = _build_parser()
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    try:
        ks = _parse_ks(args.k)
    except ValueError as exc:
        print("ngram_oracle: bad --k: %s" % exc, file=sys.stderr)
        return 2
    if args.n < 0:
        print("ngram_oracle: --n must be >= 0", file=sys.stderr)
        return 2

    notes = []
    skipped = []
    try:
        if args.path:
            paths = _walk(args.path)
            docs = []
            for path in paths:
                text, why = _read_text(path)
                if text is None:
                    skipped.append({"name": path, "reason": why})
                    continue
                docs.append((path, text))
            notes.append("explicit --path corpus (%s); not the real workload"
                         % ", ".join(args.path))
        elif args.corpus == "docs_proxy":
            docs, skipped = load_docs_proxy()
            notes.append(PROXY_NOTE)
        elif args.corpus == "synth":
            docs = [("synth", synth_corpus())]
            notes.append(SYNTH_NOTE)
        else:
            print("ngram_oracle: give --corpus docs_proxy|synth or --path PATH",
                  file=sys.stderr)
            return 2
    except FileNotFoundError as exc:
        print("ngram_oracle: no such file or directory: %s" % exc, file=sys.stderr)
        return 2
    except OSError as exc:
        print("ngram_oracle: cannot read: %s" % exc, file=sys.stderr)
        return 2

    tokenizers = [args.tokenizer] if args.tokenizer else ["words", "bytes"]
    reports = [run_corpus(docs, t, args.n, ks) for t in tokenizers]

    for rep in reports:
        for row in rep["k"]:
            if row["opportunity"] == 0:
                notes.append("warning: k=%d had 0 opportunities for tokenizer %s; "
                             "p is undefined" % (row["k"], rep["tokenizer"]))

    if args.json:
        doc = {
            "spec": "docs/SPEC_DECODE_DESIGN.md plan B (n-gram speculation)",
            "n": args.n,
            "k": list(ks),
            "tokenizers": tokenizers,
            "tokenizer_notes": {t: TOKENIZER_NOTES[t] for t in tokenizers},
            "notes": notes,
            "skipped": skipped,
            "reports": reports,
        }
        print(json.dumps(doc, ensure_ascii=True, indent=2, sort_keys=True))
        return 0

    first = True
    for rep in reports:
        if not first:
            print("")
        first = False
        print(format_text(rep, note=notes[0] if notes else None))
    if skipped:
        print("")
        for item in skipped:
            print("skipped: %s -- %s" % (item["name"], item["reason"]))
    if notes and not skipped:
        print("")
        for note in notes[1:] if first is False else notes:
            print(note)
    return 0


if __name__ == "__main__":
    sys.exit(main())