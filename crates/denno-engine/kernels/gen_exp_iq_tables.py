#!/usr/bin/env python3
"""kquant_tables.rs の表を CUDA 用ヘッダへ写す (値は 1 bit も変えない)。

    python3 gen_exp_iq_tables.py > kernels_exp_iq_tables.h

`crates/denno-engine/src/kquant_tables.rs` の static 配列を読み、同じ並びの
C 配列を出す。表の値は llama.cpp ggml (MIT) 由来 (kquant_tables.rs の冒頭を参照)。
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src", "kquant_tables.rs")


def parse(src, name):
    m = re.search(r"pub static " + name + r": \[[^\]]+\] = \[(.*?)\];", src, re.S)
    if not m:
        raise SystemExit("table not found: " + name)
    # 接尾辞 (u8 / u64 / i8 / usize ...) は値の一部ではない。先に食う
    # (食わないと "216u8" の "8" を別の値として拾う)。
    return re.findall(
        r"(0x[0-9a-fA-F]+|-?\d+)(?:u8|u16|u32|u64|i8|i16|i32|i64|usize|isize)?", m.group(1)
    )


def emit_u64(out, cname, name, vals):
    out.append("__device__ const unsigned long long %s[%d] = {" % (cname, len(vals)))
    for i in range(0, len(vals), 8):
        out.append("    " + ", ".join(v + "ull" for v in vals[i : i + 8]) + ",")
    out.append("};")
    out.append("")


def emit_u32(out, cname, name, vals):
    out.append("__device__ const unsigned int %s[%d] = {" % (cname, len(vals)))
    for i in range(0, len(vals), 8):
        out.append("    " + ", ".join(v + "u" for v in vals[i : i + 8]) + ",")
    out.append("};")
    out.append("")


def emit_u8(out, cname, vals):
    out.append("__device__ const unsigned char %s[%d] = {" % (cname, len(vals)))
    for i in range(0, len(vals), 16):
        out.append("    " + ", ".join(v + "u" for v in vals[i : i + 16]) + ",")
    out.append("};")
    out.append("")


def emit_i8(out, cname, vals):
    s = []
    for v in vals:
        x = int(v, 0)
        if x > 127:
            x -= 256
        s.append(str(x))
    out.append("__device__ const signed char %s[%d] = {" % (cname, len(s)))
    for i in range(0, len(s), 8):
        out.append("    " + ", ".join(s[i : i + 8]) + ",")
    out.append("};")
    out.append("")


def main():
    src = open(SRC, encoding="utf-8").read()
    out = [
        "// 自動生成: kquant_tables.rs から写した GPU 用の表 (値は 1 bit も変えない)。",
        "// 手で編集しないこと。生成: python3 gen_exp_iq_tables.py",
        "#pragma once",
        "",
    ]
    emit_u64(out, "dno_iq2s_grid", "IQ2S_GRID", parse(src, "IQ2S_GRID"))
    emit_u64(out, "dno_iq3s_grid", "IQ3S_GRID", parse(src, "IQ3S_GRID"))
    emit_u32(out, "dno_iq3xxs_grid", "IQ3XXS_GRID", parse(src, "IQ3XXS_GRID"))
    emit_u8(out, "dno_ksigns_iq2xs", parse(src, "KSIGNS_IQ2XS"))
    emit_i8(out, "dno_kvalues_iq4nl", parse(src, "KVALUES_IQ4NL"))
    sys.stdout.write("\n".join(out) + "\n")


main()
