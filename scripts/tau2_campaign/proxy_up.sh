#!/usr/bin/env bash
# Start the tau2 campaign's litellm proxy -- or REFUSE, loudly, before binding anything.
#
# WHAT THIS GUARDS AGAINST, MEASURED. The shared proxy (PID 75168) was launched at 13:52 on
# 2026-09-20 without `PINQ_GATEWAY_BASE_URL`. Every gateway route in `conf/serving/litellm.yaml`
# reads `api_base: os.environ/PINQ_GATEWAY_BASE_URL`; litellm treats an unresolved `api_base` as
# unspecified and falls back to the default OpenAI endpoint. So the proxy sent a gateway key
# to `api.openai.com` and relayed back, verbatim:
#
#   401  Incorrect API key provided: sk-...h3tw.   (body names https://platform.openai.com)
#
# For a day and a half that was read as a revoked credential. It was not: probed directly against
# the gateway, both keys returned 200 on both frozen roles, all four combinations. One unset
# variable, and no check anywhere could tell it apart from an auth failure -- because from the
# caller's side it IS an auth failure, just against the wrong vendor.
#
# The lesson is not "remember the variable". It is that a misconfiguration which impersonates an
# authentication error must be made to announce itself. Hence: refuse, name the variable, exit.
#
#   PINQ_GATEWAY_BASE_URL   required   the REMOTE gateway, e.g. https://gateway.example.com
#   PINQ_TAU2_PORT          default 4010    loopback port to bind
#   PINQ_PROXY_CHECK_ONLY   set to 1 to validate and exit 0 WITHOUT binding (tests use this)
#   PINQ_PROXY_VENV         optional   a venv holding a litellm binary; falls back to PATH.
#                                      Passed at RUNTIME -- no absolute home path lives in this file.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CONFIG="$REPO/conf/serving/litellm.tau2.yaml"
PORT="${PINQ_TAU2_PORT:-4010}"

die() { printf 'proxy_up: REFUSING: %s\n' "$*" >&2; exit 3; }

# Keys live in .env and nothing in this project auto-loads it. Sourcing here cannot mask the
# variable under test: .env does not define PINQ_GATEWAY_BASE_URL (checked -- 0 occurrences).
if [ -f "$REPO/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO/.env"
  set +a
fi

# ---------------------------------------------------------------- the guard
GW="${PINQ_GATEWAY_BASE_URL:-}"
[ -n "$GW" ] || die "PINQ_GATEWAY_BASE_URL is unset or empty.
  Every gateway route in $(basename "$CONFIG") reads api_base: os.environ/PINQ_GATEWAY_BASE_URL.
  litellm reads an unresolved api_base as UNSPECIFIED and falls back to the OpenAI default
  endpoint, which answers 401 'Incorrect API key provided' -- a misconfiguration wearing the
  costume of a dead key. Set it to the REMOTE gateway base URL and try again."

case "$GW" in
  http://127.0.0.1*|http://localhost*|https://127.0.0.1*|https://localhost*|http://0.0.0.0*)
    die "PINQ_GATEWAY_BASE_URL='$GW' is a LOOPBACK address.
  The gateway is remote; a loopback value points the wildcard at a local proxy -- possibly at
  this very process, which forwards to itself. The recipe
  PINQ_GATEWAY_BASE_URL=\"\$LITELLM_BASE_URL\" is a trap whenever the sweep has already pointed
  LITELLM_BASE_URL at the proxy (artifacts/night_readouts_20260919/RESULT.md:106)." ;;
  http://*|https://*) : ;;
  *) die "PINQ_GATEWAY_BASE_URL='$GW' is not an http(s) URL." ;;
esac

[ -f "$CONFIG" ] || die "config $CONFIG is missing."

# Both distinct gateway keys are required: the config shards the wildcard across them, and a
# missing one makes half the deployments in that pattern fail under simple-shuffle -- an
# intermittent 401 on a random half of calls, which is worse to debug than a total one.
for k in LITELLM_API_KEY PI_AGENT_LITELLM_API_KEY; do
  [ -n "${!k:-}" ] || die "$k is unset. The wildcard is sharded across both gateway keys; with
  one missing, simple-shuffle fails roughly half of gateway calls at random."
done

if [ "${PINQ_PROXY_CHECK_ONLY:-}" = "1" ]; then
  printf 'proxy_up: checks pass (gateway=%s, port=%s, config=%s). CHECK_ONLY -- not binding.\n' \
    "$GW" "$PORT" "$(basename "$CONFIG")"
  exit 0
fi

# ------------------------------------------------------------- bind and verify
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  die "port $PORT already has a listener. Pick another PINQ_TAU2_PORT rather than displacing
  whatever is there -- a proxy port is inside run identity via base_url_sha."
fi

PROXY_BIN="${PINQ_PROXY_VENV:-}"
if [ -n "$PROXY_BIN" ]; then
  PROXY_BIN="$PROXY_BIN/bin/litellm"
else
  PROXY_BIN="$(command -v litellm || true)"
fi
[ -x "$PROXY_BIN" ] || die "no litellm binary. Set PINQ_PROXY_VENV to a venv that has one."

LOG="${PINQ_TAU2_PROXY_LOG:-$REPO/artifacts/tau2_airline_relaunch_20260920/proxy.$PORT.log}"
mkdir -p "$(dirname "$LOG")"

printf 'proxy_up: starting %s on 127.0.0.1:%s (config %s)\n' \
  "$(basename "$PROXY_BIN")" "$PORT" "$(basename "$CONFIG")"
nohup "$PROXY_BIN" --config "$CONFIG" --port "$PORT" --host 127.0.0.1 >>"$LOG" 2>&1 &
PROXY_PID=$!
disown "$PROXY_PID" 2>/dev/null || true
printf 'proxy_up: pid=%s log=%s\n' "$PROXY_PID" "$LOG"

for i in $(seq 1 60); do
  if curl -s -m 5 "http://127.0.0.1:$PORT/health/liveliness" >/dev/null 2>&1; then
    printf 'proxy_up: live after %ss\n' "$i"; exit 0
  fi
  if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    printf 'proxy_up: the proxy EXITED during startup; last log lines:\n' >&2
    tail -20 "$LOG" >&2
    exit 4
  fi
  sleep 1
done
printf 'proxy_up: no /health/liveliness after 60s; last log lines:\n' >&2
tail -20 "$LOG" >&2
exit 5
