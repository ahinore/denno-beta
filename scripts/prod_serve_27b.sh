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
# 単独利用のデスクトップ用途なので同時セッションは 4 本に絞り、
# 余った KV 予算を prefix cache (32k token = 4.3 GB) に回す。
export DENNO_BATCH_SLOTS=4
export DENNO_PREFIX_CACHE_MAX_TOKENS=32768

exec ./target/bin_canonical serve local/qwen35-27b --port 8095 --host 0.0.0.0 \
  --seed 42 --quant q4 --repetition-penalty 1.05 --temperature-cap 0.7