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
# メモリ配分 (32,607 MiB のカード): 重み 16.7 GB + 基礎 1.7 GB + KV 8.0 GB +
# prefix cache 3.2 GB = 29.6 GB で、prefill の一時バッファ用に約 3 GB を残す。
# 12,288 MB の既定予算 + 32k token の prefix cache だと 32,073 MiB まで膨らみ、
# 続くリクエストが cuMemAlloc error 2 で CPU フォールバックに落ちた (2026-10-20 実測)。
# 2026-10-21: prefix cache を 8,192 → 24,576 token に広げた。デスクトップの 1 ターンは
# 約 13,000 token で、8,192 の予算ではエントリが 1 つも作れず、毎ターン全 prefill に
# なっていた (同一プロンプトの再送でも 151.6 s、予算 24,576 では 2.62 s)。KV を
# 10,240 → 8,192 MB に下げて合計をほぼ据え置き、prefill バッファの余白を維持する。
export DENNO_KV_BUDGET_MB=8192
export DENNO_PREFIX_CACHE_MAX_TOKENS=24576
# クライアント都合の中断 (curl -m / アプリ側キャンセル) でスロットが busy のまま
# 残ることがあるため、単独利用でも 4 本持っておく。2 本だと中断 2 回で
# "no free KV cache slot" になる。
export DENNO_BATCH_SLOTS=4

exec ./target/bin_canonical serve local/qwen35-27b --port 8095 --host 0.0.0.0 \
  --seed 42 --quant q4 --repetition-penalty 1.05 --temperature-cap 0.7