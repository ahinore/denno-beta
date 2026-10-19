#!/bin/sh
# One-command CPU-only gate: kernels, engine lib, server, then every python self-test.
#
# Stops at the first failure, prints one line per stage plus a final PASS/FAIL,
# and exits 0 only if everything passed. No network, no server start/stop, no
# process inspection: it is stateless and safe to run anywhere with a rust
# toolchain and python3.
#
# CARGO_TARGET_DIR is honoured; it defaults to /tmp/denno-test-target because
# target/debug in a bind-mounted checkout can be owned by another user (EACCES).
#
# Usage:
#   scripts/run_cpu_checks.sh            # the whole gate
#   scripts/run_cpu_checks.sh --list     # print the stages and exit
#
# Exit: 0 all stages passed / 1 a stage failed / 2 usage or setup problem.

set -u

REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
cd "$REPO" || exit 2

: "${CARGO_TARGET_DIR:=/tmp/denno-test-target}"
export CARGO_TARGET_DIR

PYTHON=${PYTHON:-python3}
LOGDIR=$CARGO_TARGET_DIR/run_cpu_checks_logs

# name:command. Parsed with `read` on a here-document so the loop stays in the
# current shell (no pipeline subshell -> the exit status below is real).
STAGES='
kernel_invariants:cargo test -p denno-engine --test kernel_invariants
engine_lib:cargo test -p denno-engine --lib
server:cargo test -p denno-server
cold_ab_selftest:python3 scripts/cold_ab.py --self-test
cold_ab_tests:python3 scripts/test_cold_ab.py
longconv_selftest:python3 scripts/long_conversation_check.py --self-test
longconv_tests:python3 scripts/test_long_conversation_check.py
needle_selftest:python3 scripts/needle_selftest.py
needle_help:python3 scripts/needle_check.py --help
'

case "${1:-}" in
    "")
        ;;
    --list)
        printf '%s\n' "$STAGES" | while IFS= read -r line; do
            [ -n "$line" ] || continue
            printf '%s\t%s\n' "${line%%:*}" "${line#*:}"
        done
        exit 0
        ;;
    *)
        echo "usage: $0 [--list]" >&2
        exit 2
        ;;
esac

if ! command -v cargo >/dev/null 2>&1; then
    echo "run_cpu_checks: cargo not found in PATH" >&2
    exit 2
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "run_cpu_checks: $PYTHON not found in PATH" >&2
    exit 2
fi

mkdir -p "$LOGDIR" || { echo "run_cpu_checks: cannot create $LOGDIR" >&2; exit 2; }

echo "== CPU gate =="
echo "repo:   $REPO"
echo "target: $CARGO_TARGET_DIR"
echo "python: $($PYTHON -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo unknown)"
echo ""

n_stages=0
fail_stage=""
fail_cmd=""
fail_log=""

old_ifs=$IFS
IFS='
'
for line in $STAGES; do
    IFS=$old_ifs
    [ -n "$line" ] || continue
    name=${line%%:*}
    cmd=${line#*:}
    # PYTHON=... lets a caller point at a specific interpreter (e.g. a venv);
    # the stage table stays literal python3 so it is readable and greppable.
    cmd=$(printf '%s\n' "$cmd" | sed "s|^python3 |$PYTHON |")
    n_stages=$((n_stages + 1))
    log=$LOGDIR/$name.log

    printf '%-20s ' "$name"
    if sh -c "$cmd" >"$log" 2>&1; then
        printf 'ok\n'
    else
        rc=$?
        printf 'FAIL (exit %d)\n' "$rc"
        fail_stage=$name
        fail_cmd=$cmd
        fail_log=$log
        break
    fi
done

echo ""
if [ -n "$fail_stage" ]; then
    echo "--- $fail_stage failed: $fail_cmd"
    tail -n 40 "$fail_log" | sed 's/^/    /'
    echo ""
    echo "RESULT: FAIL ($fail_stage failed, $n_stages stage(s) run; logs in $LOGDIR)"
    exit 1
fi

echo "RESULT: PASS (all $n_stages stages; logs in $LOGDIR)"
exit 0