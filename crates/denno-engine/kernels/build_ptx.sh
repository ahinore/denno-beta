#!/usr/bin/env bash
# denno の CUDA カーネル (.cu) を PTX に落とす。
#
# このコンテナには nvcc / ptxas が無いので、**実行は denno-gpu 側**で行う
# (PTX の再生成はメインセッション / オペレーターの担当)。
#
# 使い方 (このリポジトリは .sh を 100644 で持つので `bash` で呼ぶ):
#   bash build_ptx.sh              # kernels_moe_v3 だけ (既定)
#   bash build_ptx.sh moe_v3       # 同上
#   bash build_ptx.sh moe_v3 moe   # 指定したものだけ
#   bash build_ptx.sh all          # 全部 (bit-exact 経路も作り直す。差分を見てから commit)
#   bash build_ptx.sh --list       # 対象の一覧
#
# -fmad はカーネルごとに違う。**bit-exact 契約のカーネルは -fmad=false が必須**
# (FMA 融合が入ると CPU の逐次加算とビット一致しなくなる)。kernels_gemm*.cu は
# 契約外 (prefill のバッチ GEMM) なので既定の FMA ありでコンパイルする。
# kernels_moe_v3.cu は v2 と意図的にビット不一致 (4 レーン分割) だが、dequant の
# 演算列を v2 と揃えるため -fmad=false で作る。
#
# アーキは既存の PTX と同じ compute_75 (JIT で実機の arch に落ちる)。
# 変えたいときは DENNO_PTX_ARCH=compute_120 など。
#
# 終了コード: 0 = 生成した / 2 = 引数が不正 / 127 = nvcc が無い。
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${DENNO_PTX_ARCH:-compute_75}"
targets_all=(int8 elem gdn moe moe_batch moe_v2 gemm gemm_grouped moe_v3)

# 名前 -> "ソース|fmad" (fmad: nofmad = -fmad=false / fmad = 既定)。未知は 1。
spec() {
    case "$1" in
        int8)         echo "kernels_int8.cu|nofmad" ;;
        elem)         echo "kernels_elem.cu|nofmad" ;;
        gdn)          echo "gdn.cu|nofmad" ;;
        moe)          echo "kernels_moe.cu|nofmad" ;;
        moe_batch)    echo "kernels_moe_batch.cu|nofmad" ;;
        moe_v2)       echo "kernels_moe_v2.cu|nofmad" ;;
        moe_v3)       echo "kernels_moe_v3.cu|nofmad" ;;
        gemm)         echo "kernels_gemm.cu|fmad" ;;
        gemm_grouped) echo "kernels_gemm_grouped.cu|fmad" ;;
        *)            return 1 ;;
    esac
}

require() {
    spec "$1" >/dev/null || { echo "unknown target: $1 (--list を見る)" >&2; exit 2; }
}

compile() {
    local name="$1" src mode
    require "$name"
    src=$(spec "$name")
    mode="${src##*|}"
    src="${src%%|*}"
    [ -f "$src" ] || { echo "missing source: $src" >&2; exit 2; }
    case "$mode" in
        nofmad) set -- -ptx -arch="$ARCH" -fmad=false -O3 "$src" -o "kernels_$name.ptx" ;;
        fmad)   set -- -ptx -arch="$ARCH" -O3 "$src" -o "kernels_$name.ptx" ;;
    esac
    echo "+ nvcc $*"
    nvcc "$@"
}

case "${1:-}" in
    --list) printf '%s\n' "${targets_all[@]}"; exit 0 ;;
esac
[ "$#" -gt 0 ] || set -- moe_v3

# 引数は nvcc の有無より先に検証する (タイポを「nvcc が無い」と誤報しない)
for arg in "$@"; do
    [ "$arg" = all ] || require "$arg"
done

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc が無い (このコンテナでは生成できない。denno-gpu 側で実行する)" >&2
    exit 127
fi

if [ "$1" = all ]; then
    set -- "${targets_all[@]}"
fi
for arg in "$@"; do
    compile "$arg"
done
