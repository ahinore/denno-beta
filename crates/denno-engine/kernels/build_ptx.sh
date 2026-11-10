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
targets_all=(int8 elem gdn gdn_pf moe moe_batch moe_v2 gemm gemm_grouped moe_v3 moe_v4 moe_fp16 exp_iq exp_iq_v2 exp_iq_pf pf attn_dec gemm_tc gemm_tc2 dec_dev pf2 pf3 gemm_sn gemm_sn2 gemm_sn3 attn_pf2 moe_v5 mix_dev gdn_scan2 dec_epi mix_pf gdn_pf2 mix_inj mix_split mix_inj2 moe_v5_q5k mix_dev2 mix_up mix_down mix_upmix steer gemm_ba225 router225 gdn_pf3)

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
        # task 177: v3 のレーン数を可変にした版 (decode GDN の GEMV 用)。
        # データの読み方と dequant の演算列は v3 と同一なので -fmad=false。
        moe_v4)       echo "kernels_moe_v4.cu|nofmad" ;;
        moe_fp16)     echo "kernels_moe_fp16.cu|nofmad" ;;
        # task 144: IQ 系 (IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS / IQ4_NL / Q2_0) の
        # expert GEMV。数値契約は CPU とビット一致ではない (1 行を 32 レーンで
        # 分割して足す) ので FMA を許可する。
        exp_iq)       echo "kernels_exp_iq.cu|fmad" ;;
        # task 190: IQ expert GEMV の v2 (v5 の手法: ベクタロード / 次ブロックの
        # 先読み / アキュムレータ 4 本 / 表を shared へ)。数値契約は v1 と同じ
        # (CPU とビット一致しない。v1 とも合計順が違う) なので FMA を許可する。
        exp_iq_v2)    echo "kernels_exp_iq_v2.cu|fmad" ;;
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
        # task 179: decode の full attention 1 層をデバイスで閉じる (q/k 正規化 +
        # rope + gate 取り出し + 出力ゲート)。CPU と 1 演算 1 丸めを合わせるため
        # -fmad=false (明示 __*_rn のみ)。
        dec_dev)      echo "kernels_dec_dev.cu|nofmad" ;;
        # task 173: prefill の GDN スキャン (状態レジスタ常駐 + 4 レーン分割)。
        # 丸めは要素ごとに既存カーネルと同一にしたいので -fmad=false。
        # bf16 を使う予定は無いが、gemm_tc と同じ arch に揃えておく
        # (compute_75 でも動くが、lab の JIT 差を作らない)。
        gdn_pf)       echo "kernels_gdn_pf.cu|nofmad|compute_90" ;;
        # task 172: prefill の full attention 後処理 (per-head RMSNorm + partial
        # NeoX rope / 出力ゲート)。CPU 参照と 1 演算 1 丸めを合わせるため
        # -fmad=false で作る (明示 __*_rn のみ。expf だけは CUDA の libm)。
        pf2)          echo "kernels_pf2.cu|nofmad" ;;
        # task 187: decode の expert 寄与 + shared の足し込み / silu*up。
        # ホストの足し込みと 1 演算 1 丸めを合わせる (-fmad=false。明示演算のみ)。
        dec_epi)      echo "kernels_dec_epi.cu|nofmad" ;;
        # task 176: 同じ wmma 経路の第 2 版 (出力行 128・2 段プリフェッチ・
        # smem の行ストライド 40・Q4_K / Q5_K も扱う)。arch は gemm_tc と同じ理由で
        # compute_90。数値契約は無し (prefill の射影はビット一致契約の外)。
        gemm_tc2)     echo "kernels_gemm_tc2.cu|fmad|compute_90" ;;
        # task 181: prefill の N が小さい射影 (shared expert 640 / out 2560) 用の
        # BN=64 タイル。tc2 と同じ bf16 wmma なので arch も同じ compute_90。
        # 数値契約は無し (prefill の射影はビット一致契約の外)。
        gemm_sn)      echo "kernels_gemm_sn.cu|fmad|compute_90" ;;
        # task 183: 同じ BN=64 タイルで 1 ブロック 64 トークンにした版
        # (shared expert の 40 ブロック → 80 ブロック)。数値は sn と同じ
        # (k 昇順のアキュムレート) で、タイルの切り方だけが違う。
        gemm_sn2)     echo "kernels_gemm_sn2.cu|fmad|compute_90" ;;
        # task 222: sn2 のタイルのまま重みの型を Q5_K / BF16 まで広げた版。
        # shared expert の Q5_K (12 本) と hc mix の BF16 (down / up) を
        # テンソルコアに載せる。bf16 の mma が要るので arch は compute_90。
        # 数値契約は無し (prefill の射影 / mix はビット一致契約の外)。
        gemm_sn3)     echo "kernels_gemm_sn3.cu|fmad|compute_90" ;;
        # task 183: 融合 qkv 1 本の GEMM の後処理 (出力ゲートと行の gather)。
        # CPU 参照と 1 演算 1 丸めを合わせるため -fmad=false (明示 __*_rn のみ)。
        pf3)          echo "kernels_pf3.cu|nofmad" ;;
        # task 222: prefill attention v2 (スコアを bf16 の wmma、PV を 4 行 x 16 次元
        # のレジスタタイル)。bf16 の mma が要るので arch は compute_90。
        # 丸めは要素ごとに読めるように -fmad=false (PV は明示 fmaf)。
        attn_pf2)     echo "kernels_attn_pf2.cu|nofmad|compute_90" ;;
        # task 184: decode の GEMV をベクタロード + 複数アキュムレータ +
        # 次ブロック先読みで回す第 5 版 (Q8_0 / Q4_K)。dequant の演算列は
        # v3 / v4 と同じにするため -fmad=false。
        moe_v5)       echo "kernels_moe_v5.cu|nofmad" ;;
        # task 184: decode の hc mix の grouped RMS 正規化と inject の内積
        # (ツリー縮約)。1 演算 1 丸めを保つため -fmad=false。
        mix_dev)      echo "kernels_mix_dev.cu|nofmad" ;;
        # task 184: decode の GDN スキャン第 2 版 (4 レーン + 列レジスタ)。
        # 既存 dno_gdn_scan_sig と 1 演算 1 丸めを揃えるため -fmad=false。
        gdn_scan2)    echo "kernels_gdn_scan2.cu|nofmad" ;;
        # task 186: prefill の hc mix (down / up) を BF16 重みのまま回す GEMM。
        # 出力 1 要素 1 アキュムレータ・k 昇順の __fmaf_rn で、既存の
        # dno_gemm_bf16_mt とビット一致 (値は丸め順まで同一)。FMA は明示
        # intrinsic なので -fmad は既定 (gemm 系と同じ) でよい。
        mix_pf)       echo "kernels_mix_pf.cu|fmad" ;;
        # task 186: prefill GDN スキャン v2 (ヘッドあたり 2 ブロック・barrier 1 回)。
        # 値は gdn_pf とビット一致させたいので -fmad=false (式を変えないこと)。
        # arch は gdn_pf と同じ compute_90 に揃える (JIT 差を作らない)。
        # task 186 (3/n): -fmad を許可する版も lab t186b で測ったが、scan_pf は
        # 1295 -> 1277 ms (-1.4%) だけで wall は動かなかった。このカーネルは
        # 命令数ではなく SM あたりの仕事量で決まるので、丸めを変える価値が無い。
        # **nofmad に戻す** (scan128 とビット一致を保つ)。
        gdn_pf2)      echo "kernels_gdn_pf2.cu|nofmad|compute_90" ;;
        # task 189: prefill の mix inject (n = hc <= 8) を 1 スレッド 1 出力で
        # 回す GEMV。出力 1 要素 1 アキュムレータ・k 昇順の明示 __fmaf_rn なので
        # -fmad=false で作る (既存 dno_gemm_bf16_mt と丸め順まで同一)。
        mix_inj)      echo "kernels_mix_inj.cu|nofmad" ;;
        # task 195: prefill の mix down / up を n 方向に細かく割る GEMM
        # (1 ブロック 32 行 x 32 トークン・8 アキュムレータ)。出力 1 要素
        # 1 アキュムレータ・k 昇順の明示 __fmaf_rn なので既存カーネルと
        # ビット一致する (タイルの切り方だけが違う) → 既定の FMA でよい。
        mix_split)    echo "kernels_mix_split.cu|fmad" ;;
        # task 195: prefill の mix inject を k 方向にレーン分割する GEMV。
        # 丸め順が変わる (ビット一致ではない) ので、明示 __fmaf_rn のまま
        # -fmad=false で作る (演算列を読めるようにする)。
        mix_inj2)     echo "kernels_mix_inj2.cu|nofmad" ;;
        # task 192: v5 の Q5_K (ベクタロード / 4 アキュムレータ / 先読み)。
        # dequant の演算列は v4 と同じにするため -fmad=false。
        moe_v5_q5k)   echo "kernels_moe_v5_q5k.cu|nofmad" ;;
        # task 192: decode の hc mix の norm と inject を 1 本に畳む (hc 方向を
        # 1 ブロックへ)。現行 2 本と 1 演算 1 丸め・同じ縮約木を保つので -fmad=false。
        mix_dev2)     echo "kernels_mix_dev2.cu|nofmad" ;;
        # task 192: decode の mix up (BF16) を 1 warp = 1 行で回す版。
        # 1 演算 1 丸め (明示 __*_rn) なので -fmad=false。
        mix_up)       echo "kernels_mix_up.cu|nofmad" ;;
        # task 216: decode の mix down (BF16 + silu) を uint4 読みで回す版。
        # 1 演算 1 丸め (明示 __*_rn) なので -fmad=false
        # (silu の epilogue も dno_expf のまま = dno_matvec_bf16_r_silu と同一)。
        mix_down)     echo "kernels_mix_down.cu|nofmad" ;;
        # task 216: decode の mix の up と hc_mixed を 1 本に畳む版。
        # 2 本版と同じ鎖・同じ木・同じ式なので -fmad=false。
        mix_upmix)    echo "kernels_mix_upmix.cu|nofmad" ;;
        # task 225: prefill の ssm beta/alpha 射影 (F32 重み 2,560 x 48) を
        # 48 行 x 64 トークンのタイルで回す GEMM。出力 1 要素 1 アキュムレータ・
        # k 昇順の明示 __fmaf_rn なので -fmad=false (演算列を読めるようにする)。
        gemm_ba225)   echo "kernels_gemm_ba225.cu|nofmad" ;;
        # task 225: prefill の router softmax + top-k を 1 ワープ = 1 トークンで
        # 回す。選択規則と再正規化の加算順はホストと同一 (和の木だけ違う) なので
        # -fmad=false (明示演算のみ)。
        router225)    echo "kernels_router225.cu|nofmad" ;;
        # task 225: prefill GDN スキャン v3 (1 ヘッド 4 ブロック x 8 レーン)。
        # 丸めは要素ごとに v2 と同一にしたいので -fmad=false。arch も v2 と
        # 同じ compute_90 に揃える (JIT 差を作らない)。
        gdn_pf3)      echo "kernels_gdn_pf3.cu|nofmad|compute_90" ;;
        # task 218: 残差の平均の記録 (加算) とステアリング (引き算 / 射影)。
        # 加算は要素ごとに t 昇順、適用は h - alpha*v / h - alpha*(h.vhat)*vhat で、
        # ホスト側の実装と同じ演算列にするため -fmad=false (明示 __*_rn のみ)。
        steer)        echo "kernels_steer.cu|nofmad" ;;
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
