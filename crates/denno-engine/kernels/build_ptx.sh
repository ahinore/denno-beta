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
targets_all=(int8 elem gdn moe moe_batch moe_v2 gemm gemm_grouped moe_v3 moe_fp16 exp_iq exp_iq_pf pf attn_dec gemm_tc)

# 名前 -> "ソース|fmad[|arch]" (fmad: nofmad = -fmad=false / fmad = 既定。
# arch を書いたときはその target だけ $ARCH を上書きする)。未知は 1。
spec() {
    case "$1" in
        int8)         echo "kernels_int8.cu|nofmad" ;;
        elem)         echo "kernels_elem.cu|nofmad" ;;
        gdn)          echo "gdn.cu|nofmad" ;;
        moe)          echo "kernels_moe.cu|nofmad" ;;
        moe_batch)    echo "kernels_moe_batch.cu|nofmad" ;;
        moe_v2)       echo "kernels_moe_v2.cu|nofmad" ;;
        moe_v3)       echo "kernels_moe_v3.cu|nofmad" ;;
        moe_fp16)     echo "kernels_moe_fp16.cu|nofmad" ;;
        # task 144: IQ 系 (IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS / IQ4_NL / Q2_0) の
        # expert GEMV。数値契約は CPU とビット一致ではない (1 行を 32 レーンで
        # 分割して足す) ので FMA を許可する。
        exp_iq)       echo "kernels_exp_iq.cu|fmad" ;;
        # task 148: prefill の m 行 expert GEMV。decode と同じくビット一致契約では
        # ない (1 行を 32 レーンで分割) ので FMA を許可する。
        exp_iq_pf)    echo "kernels_exp_iq_pf.cu|fmad" ;;
        # task 154: prefill の hc mix / combine の要素カーネル。CPU 参照と 1 演算
        # 1 丸めを合わせるため -fmad=false で作る (明示 __*_rn のみ)。
        pf)           echo "kernels_pf.cu|nofmad" ;;
        # task 160: decode の flash decoding (split-K・GQA まとめ読み)。
        # attention の緩和枠 (P17g) なので FMA は許可しない: 既存の attention
        # カーネル (elem) と同じ -fmad=false で丸めを揃え、A/B の差を
        # 「どこを読むか」だけにする。
        attn_dec)     echo "kernels_attn_dec.cu|nofmad" ;;
        gemm)         echo "kernels_gemm.cu|fmad" ;;
        gemm_grouped) echo "kernels_gemm_grouped.cu|fmad" ;;
        # task 169: prefill の射影を BF16 の tensor core (wmma) で回す。
        # bf16 の mma は compute_80 以降にしか無いので、この target だけ arch を
        # 上げる (PTX は前方互換なので sm_120 で JIT される)。数値契約は無し
        # (prefill の射影はビット一致契約の外)。
        gemm_tc)      echo "kernels_gemm_tc.cu|fmad|compute_90" ;;
        *)            return 1 ;;
    esac
}

require() {
    spec "$1" >/dev/null || { echo "unknown target: $1 (--list を見る)" >&2; exit 2; }
}

compile() {
    local name="$1" spec_str src mode arch
    require "$name"
    spec_str=$(spec "$name")
    # "src|mode[|arch]" を分解する (arch 省略時は $ARCH)。
    IFS='|' read -r src mode arch <<<"$spec_str"
    [ -n "${arch:-}" ] || arch="$ARCH"
    [ -f "$src" ] || { echo "missing source: $src" >&2; exit 2; }
    case "$mode" in
        nofmad) set -- -ptx -arch="$arch" -fmad=false -O3 "$src" -o "kernels_$name.ptx" ;;
        fmad)   set -- -ptx -arch="$arch" -O3 "$src" -o "kernels_$name.ptx" ;;
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
