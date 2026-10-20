#!/bin/bash
# 本番起動: 27B を Q4_0 ネイティブ経路で 8095 に立てる (2026-10-20 以降の本番構成)。
#
# 9B int8 とは同居できない (27B 18.4 GB + 9B 18.0 GB > 32.6 GB) ので、
# 先に 8095 の serve を落としてから起動する。GPU_VERIFY はアンカー測定のため
# 9B int8 を起動するので、検証後はこのスクリプトで 27B に戻す。
#
# 使い方: bash scripts/prod_serve_27b.sh
# 注意: target/bin_canonical が無い、または古い場合は先にビルドして配備する。
set -u
cd "$(dirname "$0")/.." || exit 1

for p in /proc/[0-9]*; do
  pid=${p#/proc/}
  [ "$pid" = "$$" ] && continue
  c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)
  case "$c" in *serve*local/qwen35*) kill "$pid" 2>/dev/null ;; esac
done
sleep 2

export HOME=/work/denno-beta/data DENNO_CUDA=1
unset DENNO_PROFILE
# メモリ配分 (32,607 MiB のカード): 重み 16.7 GB + 基礎 1.7 GB + KV 10.0 GB +
# prefix cache 1.05 GB = 29.5 GB で、prefill の一時バッファ用に約 3 GB を残す。
# 12,288 MB の既定予算 + 32k token の prefix cache だと 32,073 MiB まで膨らみ、
# 続くリクエストが cuMemAlloc error 2 で CPU フォールバックに落ちた (2026-10-20 実測)。
export DENNO_KV_BUDGET_MB=10240
export DENNO_PREFIX_CACHE_MAX_TOKENS=8192
# クライアント都合の中断 (curl -m / アプリ側キャンセル) でスロットが busy のまま
# 残ることがあるため、単独利用でも 4 本持っておく。2 本だと中断 2 回で
# "no free KV cache slot" になる。
export DENNO_BATCH_SLOTS=4

exec ./target/bin_canonical serve local/qwen35-27b --port 8095 --host 0.0.0.0 \
  --seed 42 --quant q4 --repetition-penalty 1.05 --temperature-cap 0.7