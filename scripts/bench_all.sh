#!/bin/bash
# scripts/bench_all.sh - 27B の主要ボトルネックを 1 本ずつ測る (モデル起動なし)
#
# 目的: カーネルを書き換えるたびに 6 分の serve 計測を回さない。ここは 20 秒で終わる。
# 本番 8095 には触らない (小さな確保だけなので同居できる)。
#
# 使い方:
#   docker exec denno-gpu bash /work/denno-beta/scripts/bench_all.sh
#   docker exec denno-gpu bash /work/denno-beta/scripts/bench_all.sh --quick
#
# 判定: 出力の ms/call を docs/TASKS.md の基準値と比較する。悪化したら戻す。
set -u
cd /work/denno-beta || exit 1
export DENNO_CUDA=1
T=${CARGO_TARGET_DIR:-/tmp/denno-target}
BIN=$T/release/examples
QUICK=""
[ "${1:-}" = "--quick" ] && QUICK="--iters 10"

echo "== build (examples) =="
CARGO_TARGET_DIR=$T /opt/cargo/bin/cargo build --release -p denno-engine --features cuda \
  --example bench_q4_line --example bench_gdn 2>&1 | grep -E "^error" -A4 && exit 1

echo
echo "== gate_up [n=34816, k=5120] x 64 layers (decode m=1 / prefill m=300) =="
"$BIN/bench_q4_line" --layers 64 $QUICK

echo
echo "== down_proj [n=5120, k=17408] x 64 layers =="
"$BIN/bench_q4_line" --k 17408 --n 5120 --layers 64 $QUICK

echo
echo "== lm_head [n=248320, k=5120] x 1 =="
"$BIN/bench_q4_line" --n 248320 --layers 1 --raw $QUICK

echo
echo "== GDN scan (1 layer, state 3.3 MB) x 48 layers =="
"$BIN/bench_gdn" --layers 48 $QUICK

echo
echo "== done (8095 は稼働したまま) =="