#!/usr/bin/env bash
# task 220: /2d ビューワが使う実 API のスモークテスト。
#
# 実サーバー (既定 http://127.0.0.1:8098) に対して、ビューワと同じ順序で
#   shape → 介入の set / 全置換 / clear → 記録 (2 つの snapshot を各 2 チャット)
#   → apply (A / A-B) → 解除 → snapshot 削除 → 後始末
# を叩き、**返ってきた status と本文**を 1 つずつ検査する。
#
# 終了時にサーバーを「介入なし・ステアリングなし・記録なし」に戻す (trap)。
# 使い方:
#   crates/denno-view/examples/smoke_2d.sh                    # 8098
#   BASE=http://127.0.0.1:8098 crates/denno-view/examples/smoke_2d.sh
#   NAME_A=X NAME_B=Y ...                                      # snapshot 名を変える
#
# 前提: curl と python3。モデルが読み込み済みのサーバー (起動直後でもよい)。
set -uo pipefail

BASE=${BASE:-http://127.0.0.1:8098}
BASE=${BASE%/}
NAME_A=${NAME_A:-A}
NAME_B=${NAME_B:-B}
MAX_TOKENS=${MAX_TOKENS:-8}
PROMPT_A=${PROMPT_A:-"Name one colour."}
PROMPT_B=${PROMPT_B:-"Name one animal."}
CURL=${CURL:-curl}

TMP=$(mktemp -d)
CODE=""
PASS=0
FAIL=0
DID_INTERVENE=0
DID_RECORD=0
DID_APPLY=0
HAVE_A=0
HAVE_B=0

ok() { PASS=$((PASS + 1)); printf 'ok   %s\n' "$*"; }
no() { FAIL=$((FAIL + 1)); printf 'FAIL %s\n' "$*"; }
say() { printf '# %s\n' "$*"; }

cleanup() {
  # 途中で失敗しても「介入なし・ステアリングなし・記録なし」で終わる。
  # 自分が作ったものだけを消す (他人の snapshot / 適用を壊さない)。
  if [ "$DID_APPLY" = 1 ]; then
    $CURL -sS -o /dev/null -X DELETE "$BASE/v1/steer/apply" || true
  fi
  if [ "$DID_RECORD" = 1 ]; then
    $CURL -sS -o /dev/null -X POST "$BASE/v1/steer/record/stop" || true
  fi
  if [ "$HAVE_A" = 1 ]; then
    $CURL -sS -o /dev/null -X DELETE "$BASE/v1/steer/snapshots/$NAME_A" || true
  fi
  if [ "$HAVE_B" = 1 ]; then
    $CURL -sS -o /dev/null -X DELETE "$BASE/v1/steer/snapshots/$NAME_B" || true
  fi
  if [ "$DID_INTERVENE" = 1 ]; then
    $CURL -sS -o /dev/null -X DELETE "$BASE/v1/intervene" || true
  fi
  rm -rf "$TMP"
}
trap cleanup EXIT

# req METHOD PATH [BODY] -> $CODE, 本文は $TMP/body
req() {
  local m=$1 p=$2 b=${3-}
  if [ -n "$b" ]; then
    CODE=$($CURL -sS -m 300 -o "$TMP/body" -w '%{http_code}' -X "$m" \
      -H 'content-type: application/json' --data "$b" "$BASE$p") || CODE=000
  else
    CODE=$($CURL -sS -m 300 -o "$TMP/body" -w '%{http_code}' -X "$m" "$BASE$p") || CODE=000
  fi
}

# pycheck EXPR — $TMP/body を JSON として読み、EXPR が真なら 0。
# EXPR からは `body` (dict) と `json` が見える。
pycheck() {
  python3 - "$TMP/body" "$1" <<'PY'
import json, sys
try:
    body = json.load(open(sys.argv[1]))
except Exception as e:  # noqa: BLE001
    print("  not JSON: %s" % e)
    sys.exit(1)
try:
    # 複数行の式を受けるので括弧で包む (行頭の空白は括弧内なら無視される)。
    good = eval("(\n" + sys.argv[2] + "\n)", {"body": body, "json": json})  # noqa: S307 (テスト用)
except Exception as e:  # noqa: BLE001
    print("  check raised: %r (body=%.300r)" % (e, body))
    sys.exit(1)
if not good:
    print("  check false: %s (body=%.400r)" % (sys.argv[2], body))
    sys.exit(1)
PY
}

# check DESC WANT_CODE EXPR — 失敗しても続ける (集計する)。
check() {
  local desc=$1 want=$2 expr=$3
  if [ "$CODE" != "$want" ]; then
    no "$desc (HTTP $CODE != $want; body $(head -c 300 "$TMP/body" 2>/dev/null))"
    return 1
  fi
  if ! pycheck "$expr"; then
    no "$desc (body)"
    return 1
  fi
  ok "$desc"
  return 0
}

# fatal DESC — 依存する前提が崩れたら以降を止める (cleanup は trap がやる)。
fatal() { no "$* — 以降を中止"; exit 1; }

say "BASE=$BASE NAME_A=$NAME_A NAME_B=$NAME_B"

# ---- 1. 形 ---------------------------------------------------------------
req GET /v1/model/shape
check "GET /v1/model/shape 200 と形" 200 '
  isinstance(body.get("layers"), int) and body["layers"] > 0
  and isinstance(body.get("layer_kinds"), list) and len(body["layer_kinds"]) == body["layers"]
  and all(isinstance(k, str) and k for k in body["layer_kinds"])
  and isinstance(body.get("n_expert"), int) and body["n_expert"] > 0
  and isinstance(body.get("n_expert_used"), int) and 0 < body["n_expert_used"] <= body["n_expert"]
  and isinstance(body.get("n_hc"), int) and body["n_hc"] > 0
  and isinstance(body.get("n_embd"), int) and body["n_embd"] > 0
  and isinstance(body.get("ple_layers"), list)
  and all(isinstance(l, int) and 0 <= l < body["layers"] for l in body["ple_layers"])
' || fatal "shape が足りない (この版のサーバーには /2d 用 API が無い)"
LAYERS=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["layers"])' "$TMP/body")
N_HC=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["n_hc"])' "$TMP/body")
say "layers=$LAYERS n_hc=$N_HC"

req GET /v1/limits
check "GET /v1/limits 200 (画面が上限をここから読む)" 200 '
  isinstance(body.get("max_tokens_default"), int) and body["max_tokens_default"] > 0
' || true

# ---- 2. 初期状態は空でなければならない (他人の状態を壊さない) ------------
req GET /v1/intervene
check "GET /v1/intervene 200 と空" 200 '
  isinstance(body.get("epoch"), int)
  and body.get("experts_off") == [] and body.get("blocks_off") == [] and body.get("streams_off") == []
' || fatal "介入が既にある (このスクリプトは空の状態から始める)"
req GET /v1/steer
check "GET /v1/steer 200 (recording null / apply null / epoch)" 200 '
  isinstance(body.get("epoch"), int)
  and body.get("recording") is None and body.get("apply") is None
  and isinstance(body.get("snapshots"), list)
' || fatal "ステアリングが既に効いている (空の状態から始める)"
if python3 -c 'import json,sys;b=json.load(open(sys.argv[1]));sys.exit(0 if any(s.get("name") in (sys.argv[2],sys.argv[3]) for s in b["snapshots"]) else 1)' "$TMP/body" "$NAME_A" "$NAME_B"; then
  fatal "snapshot $NAME_A / $NAME_B が既にある (NAME_A / NAME_B を変えるか消してから)"
fi
E0=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["epoch"])' "$TMP/body")
say "epoch0=$E0"

# ---- 3. 介入の set / 全置換 / clear -------------------------------------
# ビューワの「switch off selected」と同じ本文 (expert / block / stream を 1 つずつ)。
L=$((LAYERS / 2))
S=$((N_HC > 1 ? 1 : 0))
req POST /v1/intervene "{\"experts_off\":[[$L,0]],\"blocks_off\":[[$L,\"moe\"]],\"streams_off\":[[$L,$S]]}"
check "POST /v1/intervene 200 (set と epoch +1)" 200 "
  body.get('epoch') == $E0 + 1
  and body.get('experts_off') == [[$L, 0]]
  and body.get('blocks_off') == [[$L, 'moe']]
  and body.get('streams_off') == [[$L, $S]]
" || fatal "介入を設定できない"
DID_INTERVENE=1
E1=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["epoch"])' "$TMP/body")

req GET /v1/intervene
check "GET /v1/intervene が set した内容を返す" 200 "
  body.get('epoch') == $E1 and body.get('experts_off') == [[$L, 0]]
  and body.get('blocks_off') == [[$L, 'moe']] and body.get('streams_off') == [[$L, $S]]
" || true

# 未知のフィールドは 400 で、状態も epoch も変えない (実 API の契約)
req POST /v1/intervene '{"nope":1}'
check "POST /v1/intervene は未知フィールドを 400 にする" 400 '
  isinstance(body.get("error"), dict) and isinstance(body["error"].get("message"), str) and body["error"]["message"]
' || true
req GET /v1/intervene
check "400 のあと状態と epoch は据え置き" 200 "body.get('epoch') == $E1 and body.get('experts_off') == [[$L, 0]]" || true

# 全置換: 省略した配列は消える (ビューワの「switch on selected」が使う)
req POST /v1/intervene "{\"experts_off\":[[$L,0]]}"
check "POST /v1/intervene は全置換 (blocks / streams が消える)" 200 "
  body.get('experts_off') == [[$L, 0]] and body.get('blocks_off') == [] and body.get('streams_off') == []
" || true

req DELETE /v1/intervene
check "DELETE /v1/intervene 200 (全消しと epoch +1)" 200 "
  body.get('epoch') >= $E1 + 1 and body.get('experts_off') == []
  and body.get('blocks_off') == [] and body.get('streams_off') == []
" || true
DID_INTERVENE=0
E2=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["epoch"])' "$TMP/body")
say "epoch after intervene = $E2"

# ---- 4. 記録 (2 つの snapshot を各 2 チャット) ---------------------------
chat() { # chat PROMPT
  req POST /v1/messages "{\"model\":\"denno\",\"max_tokens\":$MAX_TOKENS,\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}]}"
  check "POST /v1/messages 200 (chat: $1)" 200 '
    isinstance(body.get("content"), list) and body["content"]
    and isinstance(body["content"][0].get("text"), str)
    and isinstance(body.get("usage"), dict)
  '
}

record_start() { # record_start NAME
  req POST /v1/steer/record "{\"name\":\"$1\"}"
  check "POST /v1/steer/record 200 ($1)" 200 "
    body.get('recording') is not None and body['recording'].get('name') == '$1'
    and body['recording'].get('tokens', 0) == 0 and isinstance(body.get('snapshots'), list)
  " || fatal "記録を開始できない ($1)"
  DID_RECORD=1
}

record_stop() { # record_stop NAME
  req POST /v1/steer/record/stop
  check "POST /v1/steer/record/stop 200 ($1)" 200 "
    body.get('recording') is None
    and isinstance(body.get('saved'), dict) and body['saved'].get('name') == '$1'
    and body['saved'].get('tokens', 0) >= 2
    and body['saved'].get('layers') == $LAYERS
  " || fatal "記録を止められない ($1)"
  DID_RECORD=0
}

say "-- snapshot $NAME_A (2 chats) --"
record_start "$NAME_A"
chat "$PROMPT_A" || true
chat "$PROMPT_A again" || true
req GET /v1/steer
check "GET /v1/steer の recording.tokens >= 2 (チャットで進む)" 200 "
  body.get('recording') is not None and body['recording'].get('tokens', 0) >= 2
" || true
record_stop "$NAME_A"
HAVE_A=1

say "-- snapshot $NAME_B (2 chats) --"
record_start "$NAME_B"
chat "$PROMPT_B" || true
chat "$PROMPT_B again" || true
record_stop "$NAME_B"
HAVE_B=1

req GET /v1/steer
check "GET /v1/steer に $NAME_A と $NAME_B が並ぶ" 200 "
  isinstance(body.get('snapshots'), list)
  and sorted(s.get('name') for s in body['snapshots']) == sorted(['$NAME_A', '$NAME_B'])
  and all(isinstance(s.get('tokens'), int) and s['tokens'] >= 2 for s in body['snapshots'])
" || true

# ---- 5. apply (A / A-B) と解除 -----------------------------------------
ALPHA=${ALPHA:-0.01}
req POST /v1/steer/apply "{\"terms\":[{\"snapshot\":\"$NAME_A\",\"coef\":1}],\"layers\":[0,$LAYERS],\"alpha\":$ALPHA,\"mode\":\"subtract\"}"
check "POST /v1/steer/apply 200 (A, layers [0,$LAYERS))" 200 "
  body.get('epoch') >= $E2 + 1 and isinstance(body.get('apply'), dict)
  and body['apply'].get('terms') == [{'snapshot': '$NAME_A', 'coef': 1.0}]
  and body['apply'].get('layers') == [0, $LAYERS]
  and body['apply'].get('mode') == 'subtract'
  and abs(body['apply'].get('alpha', 0) - $ALPHA) < 1e-9
" || fatal "A を適用できない"
DID_APPLY=1
E3=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["epoch"])' "$TMP/body")

req GET /v1/steer
check "GET /v1/steer が適用中の A を返す" 200 "
  isinstance(body.get('apply'), dict) and body['apply'].get('terms') == [{'snapshot': '$NAME_A', 'coef': 1.0}]
" || true

req POST /v1/steer/apply "{\"terms\":[{\"snapshot\":\"$NAME_A\",\"coef\":1},{\"snapshot\":\"$NAME_B\",\"coef\":-1}],\"layers\":[0,$LAYERS],\"alpha\":$ALPHA,\"mode\":\"subtract\"}"
check "POST /v1/steer/apply 200 (A - B)" 200 "
  body.get('epoch') >= $E3 + 1 and isinstance(body.get('apply'), dict)
  and body['apply'].get('terms') == [{'snapshot': '$NAME_A', 'coef': 1.0}, {'snapshot': '$NAME_B', 'coef': -1.0}]
  and body['apply'].get('layers') == [0, $LAYERS]
" || true
E4=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["epoch"])' "$TMP/body")

# 不正な apply は 400 で、適用中の設定を変えない
req POST /v1/steer/apply "{\"terms\":[{\"snapshot\":\"nope\",\"coef\":1}],\"layers\":[0,$LAYERS]}"
check "POST /v1/steer/apply は未知の snapshot を 400 にする" 400 '
  isinstance(body.get("error"), dict) and body["error"].get("message")
' || true
req GET /v1/steer
check "400 のあと適用は A-B のまま / epoch 据え置き" 200 "
  body.get('epoch') == $E4 and len(body['apply']['terms']) == 2
" || true

req POST /v1/steer/apply "{\"terms\":[{\"snapshot\":\"$NAME_A\",\"coef\":1}],\"layers\":[0,$((LAYERS + 1))]}"
check "POST /v1/steer/apply は to > 層数を 400 にする" 400 '
  isinstance(body.get("error"), dict) and body["error"].get("message")
' || true

req DELETE /v1/steer/apply
check "DELETE /v1/steer/apply 200 (解除と epoch +1)" 200 "
  body.get('apply') is None and body.get('epoch') >= $E4 + 1 and body.get('recording') is None
" || fatal "適用を解除できない"
DID_APPLY=0
E5=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["epoch"])' "$TMP/body")

# 記録していないのに stop は 400 (実 API の契約)
req POST /v1/steer/record/stop
check "POST /v1/steer/record/stop は記録していなければ 400" 400 '
  isinstance(body.get("error"), dict) and body["error"].get("message")
' || true

# ---- 6. snapshot の削除と後始末 ---------------------------------------
for N in "$NAME_A" "$NAME_B"; do
  req DELETE "/v1/steer/snapshots/$N"
  check "DELETE /v1/steer/snapshots/$N 200" 200 '
    body.get("recording") is None and body.get("apply") is None
  ' || true
  [ "$N" = "$NAME_A" ] && HAVE_A=0
  [ "$N" = "$NAME_B" ] && HAVE_B=0
done

if [ "$DID_INTERVENE" = 1 ]; then
  req DELETE /v1/intervene
  check "DELETE /v1/intervene 200 (後始末)" 200 '
    body.get("experts_off") == [] and body.get("blocks_off") == [] and body.get("streams_off") == []
  ' || true
  DID_INTERVENE=0
fi

# ---- 7. 最終状態: 介入なし / ステアリングなし / 記録なし ---------------
req GET /v1/intervene
check "final: GET /v1/intervene は空" 200 '
  body.get("experts_off") == [] and body.get("blocks_off") == [] and body.get("streams_off") == []
  and isinstance(body.get("epoch"), int)
' || true
req GET /v1/steer
check "final: GET /v1/steer は clean (recording/apply null, snapshot 無し)" 200 '
  body.get("recording") is None and body.get("apply") is None and body.get("snapshots") == []
  and isinstance(body.get("epoch"), int)
' || true

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
if [ "$FAIL" != 0 ]; then
  exit 1
fi
say "smoke_2d: all checks passed"
exit 0
