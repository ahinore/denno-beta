#!/bin/sh
# denno serve 起動ラッパ。モデルが未ダウンロードなら pull してから serve する。
set -e

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
NAME="${DENNO_NAME:-claude-local}"
PORT="${DENNO_PORT:-8080}"
SEED="${DENNO_SEED:-42}"
HOST="${DENNO_HOST:-0.0.0.0}"
MAX_TOKENS="${DENNO_MAX_TOKENS:-}"
QUANT="${DENNO_QUANT:-int8}"

mkdir -p "$HOME/.cache/denno/hub"

DIR="$HOME/.cache/denno/hub/models--$(echo "$MODEL" | tr '/' '--')"
if [ ! -f "$DIR/model.safetensors" ]; then
    echo "[entrypoint] downloading $MODEL ..."
    denno pull "$MODEL"
fi

set -- serve "$MODEL" --name "$NAME" --host "$HOST" --port "$PORT" --seed "$SEED"
if [ -n "$QUANT" ] && [ "$QUANT" != "fp32" ]; then
    set -- "$@" --quant "$QUANT"
fi
if [ -n "$MAX_TOKENS" ]; then
    set -- "$@" --max-tokens "$MAX_TOKENS"
fi
if [ -n "$DENNO_API_KEY" ]; then
    set -- "$@" --api-key "$DENNO_API_KEY"
fi
echo "[entrypoint] starting: denno $*"
exec denno "$@"
