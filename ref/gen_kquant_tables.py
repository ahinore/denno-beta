# GGML テーブル (ggml-common.h) → Rust ソース変換スクリプト。
# 用法: python gen_kquant_tables.py <ggml-common.h> <出力.rs>
#
# 参照元: ggml-common.h は llama.cpp (https://github.com/ggml-org/llama.cpp)
# MIT License — Copyright (c) 2023-2026 The ggml authors から抽出したもの。
# 生成物には MIT 表示 (上記著作権表示) を埋め込む。
import re
import sys

src_path, out_path = sys.argv[1], sys.argv[2]
with open(src_path, "r", encoding="utf-8") as f:
    src = f.read()

# NGRID_IQ1S マクロ値
m = re.search(r"#define NGRID_IQ1S\s+(\d+)", src)
ngrid = int(m.group(1)) if m else None

# テーブル本体抽出: GGML_TABLE_BEGIN(type, name, size) ... GGML_TABLE_END()
pat = re.compile(
    r"GGML_TABLE_BEGIN\((\w+),\s*(\w+),\s*([A-Z0-9_]+|\d+)\)\s*(.*?)GGML_TABLE_END\(\)",
    re.S,
)

WANT = {
    "kmask_iq2xs": "u8",
    "ksigns_iq2xs": "u8",
    "iq2xxs_grid": "u64",
    "iq2xs_grid": "u64",
    "iq2s_grid": "u64",
    "iq3xxs_grid": "u32",
    "iq3s_grid": "u64",
    "kvalues_iq4nl": "i8",
    "iq1s_grid": "u64",
}

out = [
    "// 自動生成: ref/ggml/ggml-common.h から gen_kquant_tables.py で変換。",
    "// 手で編集しないこと (llama.cpp の ggml テーブルとビット一致が契約)。",
    "//",
    "// テーブル値の参照元: llama.cpp ggml (https://github.com/ggml-org/llama.cpp)",
    "// MIT License — Copyright (c) 2023-2026 The ggml authors。",
    "// ggml-common.h からの抽出データであり、MIT ライセンスの表示義務に従い",
    "// 上記著作権表示を保持する。",
    "// #![allow(dead_code)] 相当: 未使用テーブルは将来の型拡張用。",
    "#![allow(dead_code)]",
    "",
]

for ctype, name, size_decl, body in pat.findall(src):
    if name not in WANT:
        continue
    size = ngrid if size_decl == "NGRID_IQ1S" else int(size_decl)
    nums = re.findall(r"-?\b0x[0-9a-fA-F]+\b|-?\b\d+\b", body)
    assert len(nums) == size, f"{name}: {len(nums)} != {size}"
    rust_ty = WANT[name]
    out.append(f"pub static {name.upper()}: [{rust_ty}; {size}] = [")
    line = []
    for i, n in enumerate(nums):
        line.append(n + ("u64" if rust_ty == "u64" else "u32" if rust_ty == "u32" else "i8" if rust_ty == "i8" else "u8"))
        if len(line) == 8:
            out.append("    " + ", ".join(line) + ",")
            line = []
    if line:
        out.append("    " + ", ".join(line) + ",")
    out.append("];")
    out.append("")

with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out))
print(f"OK: {out_path} ({len(out)} lines)")
