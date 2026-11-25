#!/usr/bin/env python3
"""task 878: lab の SCENARIO=gen 用に、長文脈 (約 6k / 13k / 26k token) の
2 ターン分の /v1/messages body を作る決定論的な生成器。

`gen` シナリオは commit 中の `scripts/lab_gen/*.json` を**全部**送るので、
このスクリプトが吐く JSON だけを置くこと (他の用途の json を混ぜない)。

各サイズについて 2 本:
  <name>_t1.json : 長い文書 + 末尾の質問 (cold prefill)
  <name>_t2.json : 同じ messages + assistant の回答 + 追加質問 (KV 再利用)

token 数は文字数から推定でしか作れない (このコンテナに Q4_K_XL の
tokenizer が無い)。実 token 数は lab の REQ 行 `prompt=` を読むこと。
英語の擬似乱数語列で約 4.9 文字/token を狙う。

model フィールドは入れない (t792 の lab_gen と同じ形。lab 側が補う)。
"""
import json
import os
import random

WORDS = """
alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike
november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee
zulu anchor bridge candle domain engine fabric garden harbor island jungle kernel
ladder matrix needle object parcel quartz ribbon saddle timber umbrella valley
wagon xenon yellow zephyr archive buffer cluster digest entropy fragment glacier
horizon invoice junction kettle lattice meadow nucleus orchard paddle quiver
rocket spindle tunnel upward vessel walnut yard zebra abacus balance circuit
density element formula gravity habitat impulse juncture kinetic leverage
magnetic neutron orbital particle quantum radiator signal tension uniform
vector wavelength yield zone beacon canyon dolphin ember falcon granite hollow
""".split()

SECRETS = {
    "ctx6k": "SILVER-TORTOISE-41",
    "ctx13k": "AMBER-KESTREL-77",
    "ctx26k": "COPPER-LYNX-93",
}


def filler(seed: int, n_chars: int) -> str:
    rng = random.Random(seed)
    out = []
    n = 0
    while n < n_chars:
        w = rng.choice(WORDS)
        out.append(w)
        n += len(w) + 1
    return " ".join(out)


def build(name: str, n_chars: int, seed: int) -> dict:
    body = filler(seed, n_chars)
    secret = SECRETS[name]
    # 針は文書の 40% 地点に 1 回だけ置く
    at = int(len(body) * 0.4)
    at = body.find(" ", at)
    if at < 0:
        at = len(body) // 2
    needle = f" Remember this passphrase: {secret}. "
    doc = body[:at] + needle + body[at:]
    head = (
        "Read the document below carefully. It is a long synthetic record. "
        "Answer the question at the very end with only the requested value.\n\n"
        "--- BEGIN DOCUMENT ---\n"
    )
    tail = (
        "\n--- END DOCUMENT ---\n\n"
        "Question: what passphrase was stated in the document? "
        "Answer with the passphrase only."
    )
    user = head + doc + tail
    return {
        "max_tokens": 32,
        "temperature": 0,
        "messages": [{"role": "user", "content": user}],
    }


def turn2(t1: dict, answer: str) -> dict:
    msgs = list(t1["messages"])
    msgs.append({"role": "assistant", "content": answer})
    msgs.append(
        {
            "role": "user",
            "content": "Now repeat that passphrase once more, and then say DONE.",
        }
    )
    return {"max_tokens": 32, "temperature": 0, "messages": msgs}


PLAN = [
    ("ctx6k", 28_000, 601),
    ("ctx13k", 62_000, 1301),
    ("ctx26k", 132_000, 2601),
]

HERE = os.path.dirname(os.path.abspath(__file__))

for i, (name, chars, seed) in enumerate(PLAN):
    t1 = build(name, chars, seed)
    with open(os.path.join(HERE, f"t878_{name}_t1.json"), "w") as f:
        json.dump(t1, f, ensure_ascii=True)
    ans = SECRETS[name]
    t2 = turn2(t1, ans)
    with open(os.path.join(HERE, f"t878_{name}_t2.json"), "w") as f:
        json.dump(t2, f, ensure_ascii=True)
    print(name, "chars", len(t1["messages"][0]["content"]), "approx_tokens",
          round(len(t1["messages"][0]["content"]) / 4.9))
