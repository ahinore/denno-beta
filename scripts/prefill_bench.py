#!/usr/bin/env python3
"""prefill の cold 計測 (8095 の 27B q4 用)。

同じプロンプトの再送は prefix cache に当たるため、cold 計測には使えない。
毎回ユニークな前置き (nonce) を付けて必ず cache miss にする。

使い方:
    python3 scripts/prefill_bench.py [chars] [runs]
    python3 scripts/prefill_bench.py 16000 3    # 3,161 token 相当
    python3 scripts/prefill_bench.py 65000 2    # 12,850 token 相当

判定は中央値。起動直後の 1 回は重みの遅延アップロードを含むので捨てる。
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8095/v1/messages"
MODEL = "local/qwen35-27b"

chars = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def one(nonce: str) -> float:
    filler = ("The quick brown fox jumps over the lazy dog near the river bank. "
              * (chars // 60 + 1))[:chars]
    text = "Nonce %s.\n" % nonce + filler + "\nReply with OK."
    body = json.dumps({
        "model": MODEL,
        "max_tokens": 4,
        "messages": [{"role": "user", "content": text}],
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    })
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        json.loads(r.read())
    return time.time() - t0


walls = []
for i in range(runs):
    w = one("%d-%d" % (int(time.time()), i))
    walls.append(w)
    print("run%d wall=%.2fs" % (i + 1, w), flush=True)

s = sorted(walls)
med = s[len(s) // 2]
print("median=%.2fs  min=%.2fs  max=%.2fs" % (med, s[0], s[-1]))
