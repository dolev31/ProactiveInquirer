#!/usr/bin/env bash
# The tau2 airline relaunch, and every precondition it must not be launched without.
#
# WHY THIS IS IN THE REPOSITORY AND NOT IN A SCRATCHPAD. It carries the spending hold. A control
# that lives off the committed line of history is not a control -- that is the whole finding of
# artifacts/tau2_template_restore_20260919/RESULT.md, where a prompt fix on an unmerged branch
# let a campaign measure a dead channel for weeks. This script asserts that the campaign's
# worktree is pinned to a commit that is an ancestor of main, and it would be absurd for the
# assertion itself to be untracked. It was, for about an hour, and the integrator caught it.
#
# NO ABSOLUTE PATHS. scripts/check_no_home_paths.sh fails on a committed `/Users/<name>/`, and
# it is right to: a launcher with one author's home directory baked in runs for one person.
# Every location comes from the environment, and a RELATIVE root is refused rather than
# resolved against whatever directory the caller happened to be in -- `PI_CACHE_ROOT=./cache`
# once had two instruments reading different caches and agreeing with each other about it.
#
#   PINQ_CAMPAIGN_WT   required   the pinned, detached worktree the shards run from
#   PI_RUNS_ROOT       required   absolute
#   PI_CACHE_ROOT      required   absolute
#   PINQ_PY            optional   python to run the launcher with (default .venv/bin/python)
#   PINQ_PROXY         optional   default http://127.0.0.1:4000
#   MAXPAR SEEDS ARMS  optional
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WT=${PINQ_CAMPAIGN_WT:?set PINQ_CAMPAIGN_WT to the pinned campaign worktree}
RUNS=${PI_RUNS_ROOT:?set PI_RUNS_ROOT (absolute)}
CACHE=${PI_CACHE_ROOT:?set PI_CACHE_ROOT (absolute)}
PROXY=${PINQ_PROXY:-http://127.0.0.1:4000}
MAXPAR=${MAXPAR:-16}
ARMS=${ARMS:-"1_base_qa 2_trained_qa 5_trained_frag 6_base_frag"}
SEEDS=${SEEDS:-"0 1 2"}
LOGS=${PINQ_CAMPAIGN_LOGS:-$REPO/artifacts/tau2_phase5_logs}

for v in WT RUNS CACHE; do
  case "${!v}" in
    /*) ;;
    *) echo "REFUSING: $v must be ABSOLUTE, got '${!v}'. A relative root resolves against the" >&2
       echo "caller's cwd, which is how two instruments read different caches and agreed." >&2
       exit 2 ;;
  esac
done

# ---------------------------------------------------------------- the pinned tree's identity
[ -d "$WT" ] || { echo "REFUSING: pinned worktree $WT does not exist" >&2; exit 2; }
HEADSHA=$(git -C "$WT" rev-parse HEAD) || exit 2
if ! git -C "$WT" merge-base --is-ancestor "$HEADSHA" main; then
  echo "REFUSING TO LAUNCH: $WT is at $HEADSHA, which is NOT an ancestor of main." >&2
  echo "The tau2 templates would come from a side branch -- the failure this lane repaired." >&2
  exit 3
fi
if [ -n "$(git -C "$WT" status --porcelain)" ]; then
  echo "REFUSING TO LAUNCH: $WT is dirty; every run would be stamped dev- and training-only." >&2
  exit 4
fi
for f in inquirer_prompted_tau2_base inquirer_prompted_tau2 retriever_select; do
  [ -f "$WT/src/pinq/prompts/$f.txt" ] || { echo "REFUSING: $f.txt missing in $WT" >&2; exit 5; }
done
[ -e "$WT/data/traces/tau2" ] || { echo "REFUSING: $WT/data/traces/tau2 not linked" >&2; exit 6; }

# THE INTERPRETER, CHECKED HERE RATHER THAN DISCOVERED TWELVE TIMES. `.venv/` is gitignored, so
# a pinned worktree has no interpreter of its own -- measured. Without this the shards each die
# instantly on "no such file", which reads as a campaign-wide failure rather than a missing
# path. Checked by RUNNING it and importing the package, because an executable that cannot
# import `pinq` fails later and less legibly than one that is absent.
PY=${PINQ_PY:-$REPO/.venv/bin/python}
[ -x "$PY" ] || { echo "REFUSING: no interpreter at $PY; set PINQ_PY" >&2; exit 10; }
if ! PYTHONPATH="$WT/src" "$PY" -c 'import pinq, pi_run' 2>/dev/null; then
  echo "REFUSING: $PY cannot import pinq/pi_run with PYTHONPATH=$WT/src." >&2
  echo "A worktree venv is silently incomplete more often than it is absent." >&2
  exit 10
fi
export PINQ_PY="$PY" PINQ_REPO="$REPO"
echo "interpreter ok: $PY (imports pinq, pi_run against $WT/src)"

# ---------------------------------------------------------------- the spending hold
# ReconcileError killed 25% of this lane's airline probe units AFTER their dialogues were
# billed, and at that rate only ~32% of (point, seed) cells return all four arms. The diagnosis
# must be COMMITTED AND MERGED, because $WT is pinned: a file in a working tree cannot satisfy
# this, which is exactly the mistake this script itself was guilty of.
DIAG="$WT/artifacts/tau2_template_restore_20260919/RECONCILE_DIAGNOSIS.md"
if [ ! -s "$DIAG" ]; then
  echo "REFUSING TO LAUNCH: no reconcile-error diagnosis at" >&2
  echo "  artifacts/tau2_template_restore_20260919/RECONCILE_DIAGNOSIS.md (in $WT)." >&2
  exit 7
fi
grep -q '^VERDICT:' "$DIAG" || {
  echo "REFUSING TO LAUNCH: $DIAG states no 'VERDICT:' line; a placeholder must not pass." >&2
  exit 8; }
echo "diagnosis present: $(grep -m1 '^VERDICT:' "$DIAG")"

# ---------------------------------------------------------------- routes, BY GENERATION
# THE MODEL LIST IS NOT AUTHORITATIVE ON THIS PROXY, AND IT IS WRONG IN BOTH DIRECTIONS.
# Measured: aws/claude-sonnet-5 and aws/gpt-oss-120b are ABSENT from /v1/models and answer 200
# (they resolve through a wildcard the listing does not enumerate), while a checkpoint the
# listing DID contain answered 404. A membership test condemns working roles and vouches for
# dead ones. Only a generation answers the question.
#
# AND ONE GENERATION IS NOT ENOUGH. A live adapter was measured returning a connection error on
# first probe and 200 on retry, while a genuinely absent name returns a not-found NAMING THE
# MODEL every time. A lone probe that catches a transient fault condemns a working checkpoint --
# the same false reading as the listing, one layer down.
#
# THE RULE DISCRIMINATES ON HTTP STATUS, NEVER ON EXCEPTION CLASS. That distinction is the
# FIFTH false reading of this shape tonight and the first produced by a FIX rather than by an
# original check. The listing read a live route as absent; a single completion read a live
# route as absent; and then the correction to the single completion read a PERMANENT WALL as
# transient, because a permission refusal arrives as the same exception class as a connection
# error and a class-based rule files it under "retry". A class is a fact about the client
# library. A status is a fact about the server.
#
#   404, or a not-found naming the model   -> ABSENT.     Never retry: the adapter is unserved.
#   403, or "team not allowed to access"   -> PERMISSION. Never retry: a configuration fault
#                                             that cannot clear, so retrying is pure delay.
#   no status, 5xx, or a connection error  -> TRANSIENT.  Retry.
ENVF=${PINQ_ENV_FILE:-$REPO/.env}
[ -f "$ENVF" ] || { echo "REFUSING: no env file at $ENVF; set PINQ_ENV_FILE" >&2; exit 11; }
set -a; . "$ENVF"; set +a
for v in TAU2_DATA_DIR LITELLM_API_KEY; do
  [ -n "${!v:-}" ] || {
    echo "REFUSING: $v is unset after sourcing $ENVF." >&2
    echo "A pinned worktree has no .env of its own -- gitignored -- and the first phase-5" >&2
    echo "launch raised RuntimeError on all 408 units for exactly this reason." >&2
    exit 11; }
done
export PINQ_ENV_FILE="$ENVF"

probe() {
  curl -s -o /tmp/pinq_route_probe.json -w '%{http_code}' -m 60 \
    -H "Authorization: Bearer ${LITELLM_API_KEY:-}" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
    "$PROXY/chat/completions"
}
for m in ${PINQ_REQUIRED_MODELS:-qwen3-8b-base qwen3-8b-dpo-stacked-notdone-both aws/claude-sonnet-5 aws/gpt-oss-120b}; do
  ok=0
  for attempt in 1 2 3 4; do
    code=$(probe "$m")
    [ "$code" = "200" ] && { ok=1; break; }
    body=$(head -c 300 /tmp/pinq_route_probe.json 2>/dev/null)
    if [ "$code" = "404" ] || printf '%s' "$body" | grep -qi "does not exist\|NotFoundError"; then
      echo "REFUSING TO LAUNCH: $m is ABSENT from the backend (HTTP $code)." >&2
      echo "  $body" >&2
      echo "A 404 / not-found naming the model is stable across retries: it is not loaded." >&2
      echo "Reload the adapter and re-verify BY GENERATION, never by the model list." >&2
      exit 9
    fi
    if [ "$code" = "403" ] || printf '%s' "$body" | grep -qi "not allowed to access\|team not allowed"; then
      echo "REFUSING TO LAUNCH: $m is behind a PERMISSION WALL (HTTP $code)." >&2
      echo "  $body" >&2
      echo "This is a configuration fault, not absence and not a transient fault. It will" >&2
      echo "never clear by waiting, so it is refused immediately rather than retried." >&2
      exit 12
    fi
    if [ "$code" = "401" ] || printf '%s' "$body" | grep -qi "authenticationerror\|incorrect api key\|invalid api key"; then
      echo "REFUSING TO LAUNCH: $m failed UPSTREAM AUTHENTICATION (HTTP $code)." >&2
      echo "  $body" >&2
      echo "An invalid or revoked gateway key is exactly as permanent as a permission wall: it" >&2
      echo "will never clear by waiting, so it is refused immediately rather than retried. This" >&2
      echo "is NOT absence -- the model name is fine and the proxy answered -- and it is NOT a" >&2
      echo "budget or rate fault, which return different statuses and bodies." >&2
      echo "MEASURED 2026-09-20: both frozen roles this campaign requires returned 401 from the" >&2
      echo "upstream gateway while a LOCAL vLLM pin on the same proxy returned 200, which is what" >&2
      echo "localises the fault to the gateway credential rather than to the route or the model." >&2
      echo "The fix is a rotated key, and it takes effect only when the proxy is restarted, which" >&2
      echo "is an OPERATOR action -- not something a lane should do to shared infrastructure." >&2
      exit 14
    fi
    echo "  $m: HTTP ${code:-none} on attempt $attempt (transient: no status/5xx/connection), retrying" >&2
    sleep 5
  done
  [ "$ok" = 1 ] || {
    echo "REFUSING TO LAUNCH: $m did not answer after 4 attempts (last HTTP ${code:-none})." >&2
    echo "Neither a 404/not-found nor a 403/permission wall, so this is a transient fault and" >&2
    echo "NOT evidence that the checkpoint is gone. Do not conclude absence from this." >&2
    exit 9; }
  echo "  route ok (by generation): $m"
done

echo "pinned worktree $WT at $HEADSHA, ancestor of main, clean, templates and traces present"

# --dry-run PERFORMS EVERY CHECK AND INVOKES NO SHARD.
# A launcher that can only be exercised by running it will eventually launch during a test:
# someone verifying this script against current main sailed past every precondition -- correctly,
# because they are now all satisfied -- and would have started twelve paid shards had a path
# quirk not stopped it. The verification of the thing that spends money must not be "let it try".
# Everything above this line has already run, so a dry run is a real exercise of the guards and
# not a simulation of them.
if [ "${PINQ_DRY_RUN:-0}" = "1" ] || [ "${1:-}" = "--dry-run" ]; then
  echo "--dry-run: every precondition above was executed for real; NO shard invoked."
  echo "would launch: arms [$ARMS] x seeds [$SEEDS] over the 34 recorded ${PINQ_SUITE:-tau2_airline} cuts"
  exit 0
fi

mkdir -p "$LOGS"

# ---------------------------------------------------------------- preflight: pins, BY GENERATION
# A route probe proves a model ANSWERS; it says nothing about which model a UNIT RECORDED pinning
# to -- and that gap is exactly how 408/408 units of the first phase-5 launch shipped with
# drafter/answerer pinned to gpt-oss-120b while every route probe passed. This runs ONE real,
# paid unit per arm config into a scratch runs root and reads ITS OWN manifest back with
# verify_pins.py, refusing the full launch if what got recorded does not match what
# phase5_shard.sh requests for that arm. A `--dry-run` cannot stand in for this: run_tau2_forks.py
# prints the unit and writes no manifest, so there would be nothing to compare -- this check can
# only exist as real spend, deliberately, and it is four units, not 408.
#
# ONE UNIT COMES FROM A SCRATCH ONE-POINT SELECTION, NEVER `--n 1`. The real forkpoints file is
# a dict -- an EXPLICIT SELECTION -- and run_tau2_forks.py refuses to drop any of an explicit
# selection's points; `--n 1` against 34 named points is not "sample 1", it is an unconditional
# refusal (rc=2, before any unit is built), on every arm, every time. Measured 2026-09-19: this
# is what killed the first L9 preflight attempt outright. The fix is a real subset file of the
# same shape, holding exactly one recorded point, handed to phase5_shard.sh via
# PINQ_FORKPOINTS_OVERRIDE with no `--n`/`--min-k`/`--max-k` at all -- see
# tests/test_preflight_forkpoints_selection.py, which pins the guard's refusal (unchanged) and
# this override file's shape (the actual fix) against unmodified run_tau2_forks.py.
DPO=qwen3-8b-dpo-stacked-notdone-both
PREFLIGHT_RUNS=${PINQ_PREFLIGHT_RUNS:-$RUNS/.preflight_$$}
mkdir -p "$PREFLIGHT_RUNS"
PREFLIGHT_FORKPOINTS="$PREFLIGHT_RUNS/one_point.json"
"$PY" - "$WT/conf/forks/${PINQ_FORKS_FILE:-tau2_airline_test.recovered34.json}" "$PREFLIGHT_FORKPOINTS" <<'PYEOF'
import json
import sys

raw = json.load(open(sys.argv[1]))
if not isinstance(raw, dict) or not raw.get("fork_points"):
    sys.exit("REFUSING: real forkpoints file is not a non-empty explicit selection")
raw["fork_points"] = raw["fork_points"][:1]
raw["n_fork_points"] = 1
json.dump(raw, open(sys.argv[2], "w"))
PYEOF
[ -s "$PREFLIGHT_FORKPOINTS" ] || {
  echo "REFUSING TO LAUNCH: could not build the one-point preflight file $PREFLIGHT_FORKPOINTS" >&2
  exit 13; }
echo "preflight forkpoints override: $PREFLIGHT_FORKPOINTS (1 of the 34 recorded points)"
for arm in $ARMS; do
  case "$arm" in
    1_base_qa)      req_inq=qwen3-8b-base ;;
    2_trained_qa)   req_inq=$DPO ;;
    5_trained_frag) req_inq=$DPO ;;
    6_base_frag)    req_inq=qwen3-8b-base ;;
    *) echo "REFUSING: unknown arm config '$arm' -- preflight has no requested pins for it" >&2
       exit 13 ;;
  esac
  echo "preflight: launching ONE real unit for arm $arm (seed 0) into $PREFLIGHT_RUNS"
  before=$(find "$PREFLIGHT_RUNS" -name manifest.json 2>/dev/null | sort)
  PI_RUNS_ROOT="$PREFLIGHT_RUNS" PINQ_FORKPOINTS_OVERRIDE="$PREFLIGHT_FORKPOINTS" \
    "$REPO/scripts/tau2_campaign/phase5_shard.sh" "$arm" 0 \
    > "$LOGS/preflight.$arm.log" 2>&1
  rc=$?
  after=$(find "$PREFLIGHT_RUNS" -name manifest.json 2>/dev/null | sort)
  new=$(comm -13 <(echo "$before") <(echo "$after"))
  if [ "$rc" != 0 ] || [ -z "$new" ]; then
    echo "REFUSING TO LAUNCH: preflight unit for $arm produced no manifest (rc=$rc)." >&2
    echo "  see $LOGS/preflight.$arm.log" >&2
    exit 13
  fi
  manifest=$(printf '%s\n' "$new" | head -1)
  echo "preflight: $arm recorded $manifest -- checking pins"
  if ! "$PY" "$REPO/scripts/tau2_campaign/verify_pins.py" "$manifest" \
       --inquirer "$req_inq" --drafter openai/aws/claude-sonnet-5 \
       --answerer openai/aws/claude-sonnet-5 --user-sim openai/aws/gpt-oss-120b; then
    echo "REFUSING TO LAUNCH: preflight unit for $arm recorded the WRONG pins (see above)." >&2
    echo "This is the exact mechanism that mis-pinned 408 units on 2026-09-19 -- caught before" >&2
    echo "spend this time on a single unit, not after 408 of them." >&2
    exit 14
  fi
done
echo "preflight ok: one real unit per arm config in [$ARMS] recorded the requested pins"

for arm in $ARMS; do
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do sleep 20; done
    nohup "$REPO/scripts/tau2_campaign/phase5_shard.sh" "$arm" "$seed" \
      > "$LOGS/$arm.airline.seed$seed.log" 2>&1 &
    echo "launched $arm airline seed$seed pid=$!"
    sleep 3
  done
done
wait
echo "=== PHASE 5 ALL SHARDS RETURNED $(date -u +%H:%M:%S)"
