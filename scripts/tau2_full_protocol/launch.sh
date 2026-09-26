#!/usr/bin/env bash
# Launch one sharded tau2-bench campaign, detached, retail before airline.
#
# A foreground sweep dies with the shell that started it, so every shard is nohup'd and
# disowned and this script returns immediately. Nothing here is idempotent by accident:
# resume is `run_tau2_unit`'s own by-existence check, so re-running this script re-attaches to
# the same run directories and costs nothing for the units that finished.
#
# NO PATH IN THIS FILE IS A HOME PATH. Every root arrives as an environment variable, because
# `scripts/check_no_home_paths.sh` scans this tree and because a campaign that hardcodes one
# machine's layout cannot be re-run on the cluster.
#
# Required in the environment (set by the caller, which also sources the repo .env):
#   PI_L41_REPO        the repository whose runs/ and cache/ the campaign writes to
#   PI_L41_SRC         the src/ tree the campaign's code_version refers to (PYTHONPATH)
#   PI_L41_LOGS        a directory for one log per shard
#   PI_L41_CODE        the commit every unit is stamped with
#   PI_L41_SHARDS      how many shards to start
#   LITELLM_BASE_URL   the proxy
#   TAU2_DATA_DIR      the upstream tau2 checkout
set -u

die() { echo "FATAL: $*" >&2; exit 1; }

: "${PI_L41_REPO:?}" ; : "${PI_L41_SRC:?}" ; : "${PI_L41_LOGS:?}"
: "${PI_L41_CODE:?}" ; : "${PI_L41_SHARDS:?}"
: "${LITELLM_BASE_URL:?}" ; : "${TAU2_DATA_DIR:?}"

mkdir -p "$PI_L41_LOGS" || die "cannot create $PI_L41_LOGS"

# ONE UNCONDITIONAL TRAP, REGISTERED IMMEDIATELY AFTER THE OUTPUT DIRECTORY. Traps do not
# stack, so a second one later would silently replace this; and a trap registered after the
# first `|| die` cannot fire for it.
trap 'echo "[launch] exiting with $?"' EXIT

# LAYER 3 OF THE FIREWALL. A rollout worker that can read gold is the leak; the raise it gets
# instead is the firewall working.
unset PI_GOLD_ROOT

PY="${PI_L41_PY:-$PI_L41_REPO/.venv/bin/python}"
[ -x "$PY" ] || die "no interpreter at $PY"
CAMPAIGN="$PI_L41_SRC/../scripts/tau2_full_protocol/campaign.py"
[ -f "$CAMPAIGN" ] || die "no campaign script at $CAMPAIGN"

for i in $(seq 0 $((PI_L41_SHARDS - 1))); do
  PYTHONPATH="$PI_L41_SRC" nohup "$PY" -u "$CAMPAIGN" \
      --shard "$i" --n-shards "$PI_L41_SHARDS" \
      --domain retail --domain airline \
      --trials "${PI_L41_TRIALS:-4}" \
      --runs-root "$PI_L41_REPO/runs" \
      --cache-root "$PI_L41_REPO/cache" \
      --code-version "$PI_L41_CODE" \
      --grid-name "${PI_L41_GRID:-tau2_full_protocol_20260918}" \
      --timeout-s "${PI_L41_TIMEOUT:-2400}" \
      --bridge \
      --out "$PI_L41_LOGS/shard$i.jsonl" \
      > "$PI_L41_LOGS/shard$i.log" 2>&1 &
  disown
  echo "[launch] shard $i pid $!"
done

echo "[launch] $PI_L41_SHARDS shards at $PI_L41_CODE, logs under $PI_L41_LOGS"
