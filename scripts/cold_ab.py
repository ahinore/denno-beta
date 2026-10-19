#!/usr/bin/env python3
"""P3-4 (s102): cold single-shot A/B benchmark driver for prefill acceptance.

使い方 (最小):
  python3 scripts/cold_ab.py --old target/bin_canonical --new target/bin_candidate

  1 コマンドで「古いバイナリ / 新しいバイナリ」を交互にコールド起動し、1 リクエスト
  だけ投げて wall 時間を測り、中央値と delta % を出す。

使い方 (その他):
  # 3 ラウンド (既定)、ポート 8095、9B int8、5,686 token の payload 固定
  python3 scripts/cold_ab.py --old /tmp/denno_old --new /tmp/denno_new --rounds 3

  # 何も起動せず、payload の sha256・ラウンド計画・起動コマンド行だけ表示する
  python3 scripts/cold_ab.py --old ... --new ... --dry-run

  # GPU 無しでも走る自己テスト (ラウンド計画 / 中央値 / delta / payload 決定性 / 安全装置)
  python3 scripts/cold_ab.py --self-test

  # stdlib unittest のテスト一式
  python3 scripts/test_cold_ab.py -v

なぜ交互 (alternating) なのか:
  周波数スロットリングや熱の影響で同じバイナリでも時間が流れると遅くなる。毎ラウンドを
  old -> new、次のラウンドを new -> old、と順序を反転させると、ドリフトが両方に均等に
  乗って差からは消える。片側だけ先に測るとドリフトがそのまま delta に混入する
  (過去に「2 回目の暖まったラン」を冷えた値と取り違えて 8.71 s という誤った数字が出た)。

プロトコル (cold single shot):
  1. ポート上のサーバを全部止め、ポートが応答しなくなるのを待つ
  2. DENNO_CUDA=1 DENNO_PROFILE=1 HOME=<repo>/data <bin> serve <model> --port <port>
     --host 127.0.0.1 --seed 42 --quant <quant> で起動
  3. /health が ok を返すまで待つ (既定タイムアウト 80 s)
  4. リクエストを「ちょうど 1 回」投げ、wall 時間を測る
  5. wall / input_tokens / tok/s を記録し、サーバを止める -> 1 に戻る

    prefix cache はサーバ起動直後なので常に空 = 毎回コールド prefill。

token 数の扱い:
  文字数から token 数を推測しない。payload の実 token 数はサーバが返す
  usage.input_tokens を採る。payload はリポジトリ内の docs/ARCHITECTURE.en.md から
  決定的に組み立てる (PF_BENCH.sh の 5,686 token payload と同源) ので、期待値
  --expect-input-tokens と違えば「payload が静かに変わった」として中断する (exit 2)。
  payload の sha256 も一緒に出すので、過去の数字と同じ条件かを後から照合できる。

出力と判定:
  測定ごとに 1 行、続いて表、最後にバイナリごとの中央値と
  delta = (new_med - old_med) / old_med * 100 (小数 1 桁)。
  このスクリプトは「報告だけ」する。合否 (">= 0.3 s faster") は supervisor が決める。

exit code:
  0  最後まで走った (測定の成否は問わない)
  1  1 ラウンド以上が失敗した (health 待ちタイムアウト / リクエスト失敗 / input_tokens 取得不可)
  2  payload ガードが落ちた (input_tokens が期待値と違う) = 測る前に中断
  3  起動前の安全装置 (--old と --new が同一 / 予約ポート / バイナリが同じ)
     ※ 3 は argparse の引数エラーに合わせた値で、2 や 1 と区別するため。

timeout:
  --timeout-s (既定 600 s) は 1 リクエストの待ち時間で、health 待ちにも
  max(--timeout-s, --health-timeout) として効く。既定の health 待ちは従来どおり
  80 s のまま (遅い機で 300 s を超える応答を切らないために既定を 600 s にしてある)。

標準ライブラリのみ (requests は使わない)。GPU は不要だが、実測には 9B が載る GPU 機が要る。
"""

import argparse
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# PF_BENCH.sh と同じ素材 (リポジトリ内文書 / 個人情報なし)
SOURCE_DOC = os.path.join(REPO, "docs", "ARCHITECTURE.en.md")
FILLER_BYTES = 20000
# 置換は PF_BENCH.sh と同じ 1 語だけ。ここを増やすと payload が変わり比較不能になる。
FILLER_SUBST = (("ahino", "user"),)

SYSTEM_LINE = "あなたは親切なアシスタントです。\n\n"
USER_TEXT = "東京の観光地を3つ教えて"

# 8090 = 本番 (触らない) / 8080 = serve の既定 (触らない)
FORBIDDEN_PORTS = (8080, 8090)
DEFAULT_MODEL = "local/qwen35-9b"
DEFAULT_QUANT = "int8"
DEFAULT_PORT = 8095
DEFAULT_ROUNDS = 3
DEFAULT_EXPECT_INPUT_TOKENS = 5686
HEALTH_TIMEOUT_S = 80.0
REQUEST_TIMEOUT_S = 300.0
DEFAULT_TIMEOUT_S = 600.0
PORT_FREE_TIMEOUT_S = 30.0

# GPU_VERIFY.sh の kill_servers と同じ判定材料。実体で判定するのは、cmdline を
# 書き換える実装では /proc/PID/cmdline が空になり pkill が取り逃すため。
SERVE_PATH_MARKERS = ("bin_canonical", "bin_candidate", "release/denno")


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------

def build_filler(doc_path=SOURCE_DOC):
    """docs 文書の先頭 20,000 バイトを PF_BENCH.sh と同じ規則で加工して返す。

    `head -c 20000` はバイト単位で切る (UTF-8 の文字境界は保証されない) ので、
    ここもバイトで切り、デコードできない端数は捨てる。返り値は決定的。
    """
    with open(doc_path, "rb") as f:
        raw = f.read(FILLER_BYTES)
    # head -c は「最大」なので短い文書ならその全部。切り詰めで壊れた末尾の
    # 不完全な UTF-8 バイト列は errors="ignore" で落とす (毎回同じ結果になる)。
    text = raw.decode("utf-8", errors="ignore")
    for old, new in FILLER_SUBST:
        text = text.replace(old, new)
    return text


def build_payload(doc_path=SOURCE_DOC, model=DEFAULT_MODEL, max_tokens=32):
    """PF_BENCH.sh と同内容の payload。JSON は実エンコーダで組む (手エスケープしない)。"""
    filler = build_filler(doc_path)
    # PF_BENCH.sh は awk で 1 行ずつ printf "%s\n" するので、末尾に改行が 1 つ付く
    # (元が改行で終わっていても 2 つにはならない: awk は最終行を 1 回だけ出す)。
    doc = filler if filler.endswith("\n") else filler + "\n"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "thinking": "off",
        "system": SYSTEM_LINE + doc,
        "messages": [{"role": "user", "content": USER_TEXT}],
    }
    return payload


def encode_payload(payload):
    """送信バイト列。区切り最小 + キー順は構築順のまま (決定的)。

    本文を含めて UTF-8 のまま出す。PF_BENCH.sh は messages の 1 文だけ `\\u6771...`
    とエスケープして書いていたが、JSON ではどちらも同じ文字列に解決されるので
    サーバが受ける token 列は同じ。sha256 はこのスクリプト自身の基準値として
    記録する (PF_BENCH.sh の出力バイト列とはエスケープ表記の分だけ一致しない)。
    """
    return json.dumps(payload, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def payload_sha256(body):
    return hashlib.sha256(body).hexdigest()


# ---------------------------------------------------------------------------
# ラウンド計画 / 統計 (すべて純関数 = 自己テスト対象)
# ---------------------------------------------------------------------------

def round_plan(rounds):
    """各ラウンドの実行順を返す。偶数ラウンドは old->new、奇数は new->old。

    rounds=3 -> [("old","new"), ("new","old"), ("old","new")]
    こうすると各ラウンド内で 2 つのバイナリが「先/後」を交代し、時間方向の
    ドリフトが両者に均等に乗る。
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    out = []
    for i in range(rounds):
        out.append(("old", "new") if i % 2 == 0 else ("new", "old"))
    return out


def median(values):
    """中央値 (偶数個は中央 2 つの平均)。空なら None。"""
    vals = [float(v) for v in values]
    if not vals:
        return None
    return statistics.median(vals)


def compute_delta_pct(old_med, new_med):
    """delta = (new - old) / old * 100。負 = 新が速い。算出不能なら None。"""
    if old_med is None or new_med is None:
        return None
    if old_med == 0:
        return None
    return (new_med - old_med) / old_med * 100.0


def summarise(measurements):
    """[{"binary":"old","wall":...,"ok":bool}, ...] -> {"old": {..}, "new": {..}} と delta。

    失敗ラウンドと wall<=0 は中央値から除く (0 を混ぜると中央値が壊れる)。
    """
    stats = {}
    for name in ("old", "new"):
        vals = [m["wall"] for m in measurements
                if m["binary"] == name and m.get("ok") and m.get("wall", 0) > 0]
        stats[name] = {"n": len(vals), "median": median(vals), "runs": vals}
    stats["delta_pct"] = compute_delta_pct(stats["old"]["median"], stats["new"]["median"])
    return stats


def fmt_seconds(v):
    return "n/a" if v is None else "%.3f" % v


def fmt_delta(v):
    return "n/a" if v is None else "%+.1f" % v


# ---------------------------------------------------------------------------
# プロセス / ポート (副作用あり。自己テストからは呼ばない)
# ---------------------------------------------------------------------------

def port_answers(port, host="127.0.0.1", timeout=2.0):
    """HTTP が返るか (/health) を見る。返らなければ False。"""
    url = "http://%s:%d/health" % (host, port)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            resp.read()
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def tcp_in_use(port, host="127.0.0.1", timeout=0.5):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) == 0
    except OSError:
        return False
    finally:
        s.close()


def proc_state(pid):
    """/proc/<pid>/status の State 1 文字 (Z = ゾンビ)。取れなければ None。"""
    try:
        with open("/proc/%d/status" % pid, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("State:"):
                    return line.split()[1]
    except OSError:
        return None
    return None


def list_serve_pids(targets=(), markers=SERVE_PATH_MARKERS):
    """kill 対象の PID を列挙する。

    /proc/<pid>/exe の実体が、こちらが起動したバイナリ (targets) か、GPU_VERIFY.sh が
    使うパス (markers) に一致するもの。ゾンビは除外する — 未回収のゾンビは exe リンクを
    保持して ETXTBSY の原因になるため。
    """
    targets = set(t for t in targets if t)
    pids = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return pids
    for ent in entries:
        if not ent.isdigit():
            continue
        pid = int(ent)
        if pid == os.getpid():
            continue
        exe = os.path.join("/proc", ent, "exe")
        try:
            real = os.path.realpath(exe)
        except OSError:
            continue
        if not os.path.exists(exe):
            continue
        match = real in targets or any(m in real for m in markers)
        if not match:
            continue
        if proc_state(pid) == "Z":
            continue
        pids.append(pid)
    return pids


def stop_servers(port, targets=(), host="127.0.0.1", wait_s=PORT_FREE_TIMEOUT_S,
                 verbose=True, pids_extra=()):
    """ポート上のサーバを止め、ポートが応答しなくなるまで待つ。

    1) /proc/<pid>/exe で対象を絞って SIGTERM -> 少し残れば SIGKILL
       (自分が起動した Popen の pid は pids_extra で必ず含める。スクリプトや
       interpreter 経由で起動した場合は exe がバイナリのパスにならないため)
    2) /health が答える間は待つ (応答しなくなったら解放とみなす)
    3) 最後に TCP で connect できないことを確かめる
    """
    extra = [p for p in pids_extra if p]
    pids = list(dict.fromkeys(list_serve_pids(targets) + extra))
    # exe が対象と一致しない (script/interpreter) ものは /proc では拾えないので、
    # 自分が起動した pid を alive 判定つきで個別に扱う
    def alive(pid):
        return proc_state(pid) not in (None, "Z")

    pids = [p for p in pids if alive(p)]
    if pids and verbose:
        print("  stop: SIGTERM %s" % pids)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.time() + 10.0
    while time.time() < deadline and any(alive(p) for p in pids):
        time.sleep(0.2)
    left = [p for p in pids if alive(p)]
    if left:
        if verbose:
            print("  stop: SIGKILL %s" % left)
        for pid in left:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        deadline = time.time() + 5.0
        while time.time() < deadline and any(alive(p) for p in left):
            time.sleep(0.1)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if not port_answers(port, host, timeout=1.0) and not tcp_in_use(port, host):
            return True
        # こちらの pid が全部死んだら、待たずに「解放」とみなしてよい
        # (ping が来ないサーバは health で判別できない)
        if pids and not any(alive(p) for p in pids):
            return True
        time.sleep(0.5)
    if verbose:
        print("  stop: WARNING port %d still busy after %.0fs" % (port, wait_s))
    return False


def serve_cmd(binary, model, port, quant, host="127.0.0.1", seed=42):
    return [binary, "serve", model,
            "--port", str(port), "--host", host,
            "--seed", str(seed), "--quant", quant]


def serve_env(repo=REPO):
    env = dict(os.environ)
    env["DENNO_CUDA"] = "1"
    env["DENNO_PROFILE"] = "1"
    env["HOME"] = os.path.join(repo, "data")
    return env


def start_server(binary, model, port, quant, log_path, repo=REPO):
    """起動して (Popen, logfile) を返す。stdout/stderr は個別ログへ (プロファイル行を後で見る)。"""
    cmd = serve_cmd(binary, model, port, quant)
    logf = open(log_path, "wb")
    proc = subprocess.Popen(cmd, env=serve_env(repo), stdout=logf,
                            stderr=subprocess.STDOUT, start_new_session=True)
    return proc, logf


def wait_health(port, timeout=HEALTH_TIMEOUT_S, proc=None, host="127.0.0.1"):
    """/health が ok を返すまで待つ。バイナリが先に死んだら即 False。"""
    deadline = time.time() + timeout
    url = "http://%s:%d/health" % (host, port)
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            if "ok" in body:
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    return False


def one_request(body, port, host="127.0.0.1", timeout=REQUEST_TIMEOUT_S):
    """リクエストを 1 回投げ、(wall 秒, 応答 dict or None, エラー文字列) を返す。

    wall は time.monotonic の差分 = 接続〜本文読み切りの実時間。
    """
    url = "http://%s:%d/v1/messages" % (host, port)
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"content-type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        wall = time.monotonic() - t0
    except (urllib.error.URLError, OSError) as exc:
        return None, None, "request failed: %s" % exc
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError as exc:
        return wall, None, "response is not JSON: %s" % exc
    return wall, data, ""


def extract_usage(data):
    """usage.input_tokens / output_tokens を int で取り出す (無ければ None)。"""
    usage = {"input_tokens": None, "output_tokens": None}
    if isinstance(data, dict):
        u = data.get("usage")
        if isinstance(u, dict):
            usage["input_tokens"] = u.get("input_tokens")
            usage["output_tokens"] = u.get("output_tokens")
    return usage


# ---------------------------------------------------------------------------
# 安全装置
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_args(args):
    """起動前に落とすべき条件を検査し、エラーメッセージのリストを返す。"""
    errors = []
    if args.rounds < 1:
        errors.append("--rounds must be >= 1 (got %d)" % args.rounds)
    if args.port in FORBIDDEN_PORTS:
        errors.append("--port %d is reserved (8080 = serve default, 8090 = production); "
                      "do not touch it" % args.port)
    if not (1 <= args.port <= 65535):
        errors.append("--port %d is out of range" % args.port)
    if args.max_tokens < 1:
        errors.append("--max-tokens must be >= 1")
    if args.expect_input_tokens < 1:
        errors.append("--expect-input-tokens must be >= 1")
    if getattr(args, "timeout_s", DEFAULT_TIMEOUT_S) <= 0:
        errors.append("--timeout-s must be > 0 (got %s)" % args.timeout_s)

    # 自己テスト / dry-run ではバイナリの存在を要求しない (計画の表示だけ)
    need_files = not (args.self_test or args.dry_run)
    for name in ("old", "new"):
        p = getattr(args, name)
        if not p:
            errors.append("--%s is required" % name)
            continue
        if os.path.abspath(p) == os.path.abspath(getattr(args, "new" if name == "old" else "old") or ""):
            errors.append("--old and --new are the same path (%s)" % p)
            continue
        if need_files:
            if not os.path.exists(p):
                errors.append("--%s does not exist: %s" % (name, p))
            elif not os.access(p, os.X_OK):
                errors.append("--%s is not executable: %s" % (name, p))
            elif not os.path.isfile(p):
                errors.append("--%s is not a regular file: %s" % (name, p))
    if need_files and not errors:
        a, b = os.path.realpath(args.old), os.path.realpath(args.new)
        if a == b:
            errors.append("--old and --new resolve to the same file: %s" % a)
        else:
            sa, sb = sha256_file(args.old), sha256_file(args.new)
            if sa == sb:
                errors.append("--old and --new have the same sha256 (%s); "
                              "the A/B would measure nothing" % sa[:16])
    return errors


def warn_foreign_listener(port, verbose=True):
    """ポートに既に denno serve らしいものが居たら警告する (止めてから測るのは呼び側)。"""
    if tcp_in_use(port):
        pids = list_serve_pids()
        if verbose:
            msg = "WARNING: something is already listening on port %d" % port
            if pids:
                msg += " (denno serve pids: %s)" % pids
            print(msg + " — it will be stopped before the first round")
        return True
    return False


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------

def run_ab(args, out=sys.stdout):
    """交互ラウンドを実行し、結果 dict を返す。"""
    bins = {"old": os.path.abspath(args.old), "new": os.path.abspath(args.new)}
    body = encode_payload(build_payload(args.source_doc, args.model, args.max_tokens))
    sha = payload_sha256(body)
    expect = args.expect_input_tokens
    guard_on = expect > 0

    plan = round_plan(args.rounds)
    def say(msg=""):
        print(msg, file=out)

    say("cold A/B protocol: start server -> one request (cold prefix cache) -> stop")
    say("  payload sha256 = %s (%d bytes)" % (sha, len(body)))
    say("  model=%s quant=%s port=%d seed=%d max_tokens=%d"
        % (args.model, args.quant, args.port, args.seed, args.max_tokens))
    say("  expected input_tokens = %s" % (expect if guard_on else "not checked"))
    for name in ("old", "new"):
        say("  %-3s %s  sha256=%s"
            % (name, bins[name], sha256_file(bins[name])[:16]))
    say("  round plan: " + " | ".join("%d:%s>%s" % (i + 1, p[0], p[1])
                                     for i, p in enumerate(plan)))

    measurements = []
    guard_tripped = False
    targets = (bins["old"], bins["new"])
    # --timeout-s は health 待ちとリクエストの両方に効く。既定では health の
    # 80 s のまま (max を取るので --health-timeout を明示すればそちらが優先)。
    health_timeout = max(float(getattr(args, "timeout_s", DEFAULT_TIMEOUT_S)),
                         float(args.health_timeout))

    for rnd, order in enumerate(plan, start=1):
        for name in order:
            binary = bins[name]
            log_path = os.path.join(args.log_dir, "cold_ab_r%d_%s.log" % (rnd, name))
            say(" [r%d/%d] %s start (%s)" % (rnd, args.rounds, name, binary))
            stop_servers(args.port, targets=targets, verbose=False)
            rec = {"round": rnd, "binary": name, "binary_path": binary,
                   "wall": None, "input_tokens": None, "output_tokens": None,
                   "boot_s": None, "ok": False, "error": "", "log": log_path}
            proc = logf = None
            try:
                t_start = time.monotonic()
                proc, logf = start_server(binary, args.model, args.port,
                                          args.quant, log_path)
                ready = wait_health(args.port, health_timeout, proc=proc)
                rec["boot_s"] = time.monotonic() - t_start
                if not ready:
                    rec["error"] = ("health timeout (%.0fs) or server exited early"
                                    % health_timeout)
                    say(" [r%d/%d] %s FAIL: %s (log: %s)"
                        % (rnd, args.rounds, name, rec["error"], log_path))
                    measurements.append(rec)
                    continue

                wall, data, err = one_request(body, args.port, timeout=args.timeout_s)
                rec["wall"] = wall
                usage = extract_usage(data)
                rec["input_tokens"] = usage.get("input_tokens")
                rec["output_tokens"] = usage.get("output_tokens")
                if err or data is None:
                    rec["error"] = err or "no response"
                elif rec["input_tokens"] is None:
                    rec["error"] = "response has no usage.input_tokens"
                elif guard_on and rec["input_tokens"] != expect:
                    guard_tripped = True
                    rec["error"] = ("payload guard: input_tokens %d != expected %d"
                                    % (rec["input_tokens"], expect))
                else:
                    rec["ok"] = True

                if rec["ok"]:
                    say(" [r%d/%d] %s wall=%.3fs input_tokens=%d output_tokens=%s"
                        % (rnd, args.rounds, name, wall, rec["input_tokens"],
                           rec["output_tokens"]))
                elif guard_tripped:
                    say(" [r%d/%d] %s GUARD TRIP: %s — payload changed; aborting"
                        % (rnd, args.rounds, name, rec["error"]))
                else:
                    say(" [r%d/%d] %s FAIL: %s" % (rnd, args.rounds, name, rec["error"]))
                measurements.append(rec)
            finally:
                stop_servers(args.port, targets=targets, verbose=False,
                             pids_extra=(proc.pid if proc is not None else (),))
                if logf is not None:
                    logf.close()
                if proc is not None and proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except OSError:
                        pass
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            if guard_tripped:
                break
        if guard_tripped:
            break

    left = list_serve_pids(targets)
    if left:
        say("WARNING: serve processes still alive after the run: %s" % left)

    stats = summarise(measurements)
    return {"payload_sha256": sha, "payload_bytes": len(body),
            "measurements": measurements, "stats": stats,
            "guard_tripped": guard_tripped, "leftover_pids": left,
            "binaries": bins, "rounds": args.rounds,
            "protocol": {"model": args.model, "quant": args.quant,
                         "port": args.port, "seed": args.seed,
                         "max_tokens": args.max_tokens,
                         "expect_input_tokens": expect if guard_on else None,
                         "source_doc": args.source_doc}}


def print_report(result, out=sys.stdout):
    stats = result["stats"]
    print("", file=out)
    print("round  binary  wall(s)   input_tokens  ok", file=out)
    for m in result["measurements"]:
        wall = "   -    " if m["wall"] is None else "%8.3f" % m["wall"]
        it = "     -" if m["input_tokens"] is None else "%6d" % m["input_tokens"]
        print("%5d  %-6s  %s  %s  %s" % (m["round"], m["binary"], wall, it,
                                         "yes" if m["ok"] else "NO %s" % m["error"]),
              file=out)
    print("", file=out)
    for name in ("old", "new"):
        vals = stats[name]["runs"]
        shown = ", ".join("%.3f" % v for v in vals) if vals else "none"
        print("%-4s n=%d  median=%ss   runs=[%s]"
              % (name, stats[name]["n"], fmt_seconds(stats[name]["median"]), shown),
              file=out)
    d = stats["delta_pct"]
    if d is None:
        print("delta = n/a (need at least one successful measurement per binary)",
              file=out)
    else:
        print("delta = %s%%  (new %ss vs old %ss)"
              % (fmt_delta(d), fmt_seconds(stats["new"]["median"]),
                 fmt_seconds(stats["old"]["median"])), file=out)
        print("note: negative delta = new is faster. Acceptance (>= 0.3 s faster) is "
              "decided by the supervisor, not this script.", file=out)
    print("payload sha256 = %s" % result["payload_sha256"], file=out)


def run_dry(args, out=sys.stdout):
    payload = build_payload(args.source_doc, args.model, args.max_tokens)
    body = encode_payload(payload)
    sha = payload_sha256(body)
    def say(msg=""):
        print(msg, file=out)
    say("DRY RUN — no server is started and no request is sent.")
    say("  source doc   = %s" % args.source_doc)
    say("  payload sha256 = %s" % sha)
    say("  payload bytes  = %d" % len(body))
    say("  system bytes   = %d (fixed line %d + filler)"
        % (len(payload["system"].encode("utf-8")), len(SYSTEM_LINE.encode("utf-8"))))
    say("  messages       = %s" % json.dumps(payload["messages"], ensure_ascii=False))
    say("  max_tokens=%d temperature=%s thinking=%s"
        % (payload["max_tokens"], payload["temperature"], payload["thinking"]))
    say("  expect input_tokens = %d" % args.expect_input_tokens)
    say("  port=%d model=%s quant=%s seed=%d health_timeout=%.0fs"
        % (args.port, args.model, args.quant, args.seed, args.health_timeout))
    # 従来の行は変えない。--timeout-s を既定から変えたときだけ 1 行足す。
    if getattr(args, "timeout_s", DEFAULT_TIMEOUT_S) != DEFAULT_TIMEOUT_S:
        say("  timeout = %.0fs (per request; health wait uses max(timeout, "
            "health_timeout))" % args.timeout_s)
    say("")
    for name in ("old", "new"):
        p = getattr(args, name)
        say("  %-3s binary: %s" % (name, os.path.abspath(p) if p else "(missing)"))
    say("")
    say("round plan (alternating so thermal drift cancels):")
    for i, order in enumerate(round_plan(args.rounds), start=1):
        say("  round %d: %s" % (i, " -> ".join(order)))
    say("")
    say("commands per round:")
    for name in ("old", "new"):
        p = getattr(args, name)
        p = os.path.abspath(p) if p else "<%s>" % name
        say("  DENNO_CUDA=1 DENNO_PROFILE=1 HOME=%s/data \\" % REPO)
        say("    %s" % " ".join(serve_cmd(p, args.model, args.port, args.quant)))
    say("  POST http://127.0.0.1:%d/v1/messages  (payload sha256 above)" % args.port)
    say("  stop the server, wait until port %d no longer answers, next round" % args.port)
    if tcp_in_use(args.port):
        say("")
        say("WARNING: port %d is already in use on this machine" % args.port)
    return {"payload_sha256": sha, "payload_bytes": len(body),
            "plan": round_plan(args.rounds)}


# ---------------------------------------------------------------------------
# 自己テスト (CPU のみ・サーバも cargo も起動しない)
# ---------------------------------------------------------------------------

def self_test(out=sys.stdout):
    """計画・統計・payload 決定性・安全装置をその場で検算する。"""
    failures = []

    def check(name, cond, detail=""):
        if cond:
            print("ok   %s%s" % (name, (" — " + detail) if detail else ""), file=out)
        else:
            print("FAIL %s%s" % (name, (" — " + detail) if detail else ""), file=out)
            failures.append(name)

    # 1) 交互計画
    check("plan 1 round", round_plan(1) == [("old", "new")])
    check("plan 2 rounds", round_plan(2) == [("old", "new"), ("new", "old")],
          str(round_plan(2)))
    p3 = round_plan(3)
    check("plan 3 rounds order", p3 == [("old", "new"), ("new", "old"), ("old", "new")])
    for r in (1, 2, 5, 8):
        plan = round_plan(r)
        check("plan %d: old ran %d, new ran %d" % (r, r, r),
              sum(o.count("old") for o in plan) == r == sum(o.count("new") for o in plan))
        firsts = [o[0] for o in plan]
        check("plan %d: first position alternates" % r,
              all(firsts[i] != firsts[i + 1] for i in range(len(firsts) - 1)))
    try:
        round_plan(0)
        check("plan(0) rejected", False)
    except ValueError:
        check("plan(0) rejected", True)

    # 2) 中央値
    check("median odd", median([3, 1, 2]) == 2.0)
    check("median even", median([4, 1, 3, 2]) == 2.5)
    check("median single", median([7.5]) == 7.5)
    check("median empty is None", median([]) is None)
    check("median ignores input order", median([10, 2, 6]) == 6.0)

    # 3) delta
    check("delta faster", abs(compute_delta_pct(10.0, 9.0) - (-10.0)) < 1e-9,
          "%+.1f%%" % compute_delta_pct(10.0, 9.0))
    check("delta slower", abs(compute_delta_pct(10.0, 11.0) - 10.0) < 1e-9)
    check("delta equal", compute_delta_pct(10.0, 10.0) == 0.0)
    check("delta zero guard", compute_delta_pct(0.0, 1.0) is None)
    check("delta missing guard", compute_delta_pct(None, 1.0) is None)
    # 0.3 s の改善幅を % でも出せること (supervisor の判定は秒で行う)
    d = compute_delta_pct(13.18, 12.88)
    check("delta 13.18 -> 12.88 is about -2.3%", abs(d - (-2.2762)) < 0.01,
          "%+.1f%%" % d)

    # 4) summarise (失敗ラウンド除外)
    ms = [
        {"binary": "old", "wall": 10.0, "ok": True},
        {"binary": "new", "wall": 9.0, "ok": True},
        {"binary": "old", "wall": 11.0, "ok": True},
        {"binary": "new", "wall": None, "ok": False},
        {"binary": "old", "wall": None, "ok": False},
        {"binary": "new", "wall": 20.0, "ok": False},  # 失敗は無視
    ]
    st = summarise(ms)
    check("summarise old median", st["old"]["median"] == 10.5, str(st["old"]["median"]))
    check("summarise new median", st["new"]["median"] == 9.0, str(st["new"]["median"]))
    check("summarise counts", (st["old"]["n"], st["new"]["n"]) == (2, 1))
    check("summarise delta", abs(st["delta_pct"] - (-14.2857)) < 0.01,
          "%+.1f%%" % st["delta_pct"])
    check("summarise all-failed gives None",
          summarise([{"binary": "old", "wall": None, "ok": False},
                     {"binary": "new", "wall": None, "ok": False}])["delta_pct"] is None)

    # 5) payload 決定性と形
    p1 = build_payload()
    body1 = encode_payload(p1)
    body2 = encode_payload(build_payload())
    check("payload is byte-identical across builds", body1 == body2,
          "sha256=" + payload_sha256(body1))
    check("payload sha256 is stable", payload_sha256(body1) == payload_sha256(body2))
    check("payload has thinking off", p1["thinking"] == "off")
    check("payload has temperature 0", p1["temperature"] == 0)
    check("payload has max_tokens 32", p1["max_tokens"] == 32)
    check("payload has one user message",
          p1["messages"] == [{"role": "user", "content": USER_TEXT}])
    check("payload system starts with the fixed line",
          p1["system"].startswith(SYSTEM_LINE))
    check("filler has no 'ahino'", "ahino" not in p1["system"])
    filler = build_filler()
    check("filler is non-empty and <= 20000 bytes of source",
          0 < len(filler) and len(filler.encode("utf-8")) <= FILLER_BYTES + 10,
          "%d chars / %d bytes" % (len(filler), len(filler.encode("utf-8"))))
    check("system ends with exactly one newline",
          p1["system"].endswith("\n") and not p1["system"].endswith("\n\n"))
    check("payload round-trips through json", json.loads(body1.decode("utf-8")) == p1)
    check("payload is not hand-escaped (quotes survive)",
          '"' in p1["system"])

    # 6) 安全装置 (純関数部分)。dry_run=True でファイル検査を飛ばし、計画だけ見る。
    def mk(**kw):
        base = {"rounds": 3, "port": DEFAULT_PORT, "max_tokens": 32,
                "expect_input_tokens": DEFAULT_EXPECT_INPUT_TOKENS,
                "old": "/tmp/x_old", "new": "/tmp/x_new",
                "self_test": False, "dry_run": True}
        base.update(kw)
        return argparse.Namespace(**base)
    check("guard: 8080 refused", bool(validate_args(mk(port=8080))))
    check("guard: 8090 refused", bool(validate_args(mk(port=8090))))
    check("guard: 8095 allowed", validate_args(mk(port=8095)) == [])
    check("guard: identical paths refused",
          bool(validate_args(mk(old="/tmp/same", new="/tmp/same"))))
    check("guard: rounds 0 refused", bool(validate_args(mk(rounds=0))))
    check("guard: bad port refused", bool(validate_args(mk(port=70000))))
    e = validate_args(mk(old="/tmp/nope_old_a", new="/tmp/nope_new_b", dry_run=False))
    check("guard: missing binaries refused (not dry-run)", bool(e), str(e))

    # 同じファイルを指す symlink / 同一内容の別ファイル (実ファイルで確認)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "denno")
        with open(p, "wb") as f:
            f.write(b"#!/bin/sh\nexit 0\n")
        os.chmod(p, 0o755)
        link = os.path.join(td, "denno_link")
        os.symlink(p, link)
        copy = os.path.join(td, "denno_copy")
        with open(p, "rb") as src, open(copy, "wb") as dst:
            dst.write(src.read())
        os.chmod(copy, 0o755)
        other = os.path.join(td, "denno_other")
        with open(other, "wb") as f:
            f.write(b"#!/bin/sh\nexit 0\n# other\n")
        os.chmod(other, 0o755)
        check("guard: symlink to the same file refused",
              bool(validate_args(mk(old=p, new=link, dry_run=False))))
        check("guard: same sha256 refused",
              bool(validate_args(mk(old=p, new=copy, dry_run=False))))
        check("guard: distinct binaries accepted",
              validate_args(mk(old=p, new=other, dry_run=False)) == [])

    # 7) usage 抽出
    u = extract_usage({"usage": {"input_tokens": 5686, "output_tokens": 32}})
    check("usage extraction", u == {"input_tokens": 5686, "output_tokens": 32})
    check("usage missing", extract_usage({"id": "x"}) ==
          {"input_tokens": None, "output_tokens": None})

    print("", file=out)
    if failures:
        print("SELF-TEST FAILED: %d check(s)" % len(failures), file=out)
        return 1
    print("SELF-TEST PASSED (CPU only; no GPU, no server, no cargo)", file=out)
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
        prog="cold_ab.py",
        description="Cold single-shot A/B for prefill acceptance (P3-4). "
                    "Alternating rounds, one request per server start, "
                    "cold prefix cache.",
        epilog="Exit: 0 completed, 1 a round failed, 2 payload guard tripped, "
               "3 argument/safety error. Reports only — it does not decide "
               "acceptance.")
    ap.add_argument("--old", help="old (baseline) denno binary")
    ap.add_argument("--new", help="new (candidate) denno binary")
    ap.add_argument("--rounds", type=positive_int, default=DEFAULT_ROUNDS,
                    help="number of A/B rounds (default %d)" % DEFAULT_ROUNDS)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help="serving port (default %d; 8080 and 8090 are refused)"
                         % DEFAULT_PORT)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model repo id passed to serve (default %s)" % DEFAULT_MODEL)
    ap.add_argument("--quant", default=DEFAULT_QUANT, choices=("int8", "fp32"),
                    help="quant mode (default %s)" % DEFAULT_QUANT)
    ap.add_argument("--seed", type=int, default=42, help="serve --seed (default 42)")
    ap.add_argument("--max-tokens", type=positive_int, default=32,
                    help="max_tokens in the payload (default 32, same as PF_BENCH)")
    ap.add_argument("--expect-input-tokens", type=int,
                    default=DEFAULT_EXPECT_INPUT_TOKENS,
                    help="abort (exit 2) if usage.input_tokens differs "
                         "(default %d; 0 disables the check)"
                         % DEFAULT_EXPECT_INPUT_TOKENS)
    ap.add_argument("--source-doc", default=SOURCE_DOC,
                    help="document the filler is cut from (default docs/ARCHITECTURE.en.md)")
    ap.add_argument("--json", dest="json_out",
                    help="write the full result (payload sha256, measurements, "
                         "medians, delta) to this path")
    ap.add_argument("--log-dir", default="/tmp",
                    help="where per-round server logs go (default /tmp)")
    ap.add_argument("--health-timeout", type=float, default=HEALTH_TIMEOUT_S,
                    help="seconds to wait for /health (default %.0f)" % HEALTH_TIMEOUT_S)
    ap.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S,
                    help="per-request timeout in seconds, also used for the health "
                         "wait when it exceeds --health-timeout (default %.0f)"
                         % DEFAULT_TIMEOUT_S)
    ap.add_argument("--dry-run", action="store_true",
                    help="print payload sha256, round plan and command lines only")
    ap.add_argument("--self-test", action="store_true",
                    help="CPU-only self test (no GPU, no server, no cargo)")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.old or not args.new:
        ap.error("--old and --new are required (or use --self-test / --dry-run with both)")

    errors = validate_args(args)
    if errors:
        for e in errors:
            print("ERROR: %s" % e, file=sys.stderr)
        return 3

    if args.dry_run:
        run_dry(args)
        return 0

    warn_foreign_listener(args.port)
    print("stopping any server on port %d before the first round" % args.port)
    stop_servers(args.port, targets=(os.path.abspath(args.old), os.path.abspath(args.new)))

    result = run_ab(args)
    print_report(result)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        print("json written to %s" % args.json_out)

    if result["guard_tripped"]:
        return 2
    if any(not m["ok"] for m in result["measurements"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())