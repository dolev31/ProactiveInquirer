#!/usr/bin/env bash
# ONE Phase 5 shard: (arm config, seed) over the 34 recorded AIRLINE fork points.
#
#   $1 = arm config   $2 = seed
#
# AIRLINE ONLY. Retail's retrieval channel is shut at every 8B pin WITH the restored template in
# place -- measured, 1/96 answered asks at the base pin -- so a retail cell describes the channel
# and not a policy.
#
# THE TRAINED ARMS PIN THE DPO CHECKPOINT, not sft-headline. Measured on six airline cuts under
# the restored template, against the published Inquirer's rate on the same cuts:
#     qwen3-8b-base                      0.28x   shut
#     qwen3-8b-sft-headline              0.58x   below the two-thirds bar
#     qwen3-8b-dpo-stacked-notdone-both  1.02x   indistinguishable from the frontier model
#
# ARMS 3 AND 4 ARE CUT. Their distinguishing field is `target == "user"`, which the parser
# rewrites for every other arm and which is zero in 5,184 recorded asks at both campaign pins.
#
# PI_GOLD_ROOT IS UNSET: this is a rollout worker, and `gold_root()` raising is the firewall.
set -u
WT=${PINQ_CAMPAIGN_WT:?set PINQ_CAMPAIGN_WT}
RUNS=${PI_RUNS_ROOT:?set PI_RUNS_ROOT}
# THE INTERPRETER IS NOT IN THE PINNED WORKTREE. `.venv/` is gitignored, so a worktree created
# by `git worktree add` has no interpreter at all -- measured, `$WT/.venv/bin/python` does not
# exist -- and defaulting to it would have every one of the twelve shards die instantly on
# "no such file", which reads as a campaign-wide failure of unclear cause rather than as a
# missing path. The launcher checks this and passes PINQ_PY down; the fallback here is the
# REPO's interpreter, never the worktree's.
PY=${PINQ_PY:-${PINQ_REPO:-$WT}/.venv/bin/python}
[ -x "$PY" ] || { echo "no interpreter at $PY; set PINQ_PY" >&2; exit 9; }
CONF=$WT/conf/forks

# THE ENV FILE IS NOT IN THE PINNED WORKTREE EITHER. `.env` is gitignored, so `$WT/.env` does
# not exist -- and sourcing it conditionally meant TAU2_DATA_DIR and the gateway keys were
# simply unset. Measured: all 408 units of the first phase-5 launch raised
# `RuntimeError: TAU2_DATA_DIR is unset` in under a minute, which reads as the campaign being
# broken rather than as one file being absent. Third instance tonight of the same class
# (traces, interpreter, now env). Source the REPO's, and REFUSE if the result is unusable --
# a conditional source cannot tell "no file" from "file with nothing in it".
ENVF=${PINQ_ENV_FILE:-${PINQ_REPO:-$WT}/.env}
[ -f "$ENVF" ] || { echo "no env file at $ENVF; set PINQ_ENV_FILE" >&2; exit 9; }

# THE SUITE AND ITS FORK RECORD ARE PARAMETERS, DEFAULTING TO AIRLINE. They were hardcoded, which
# meant the confirmatory suite (tau2_retail) could not be run through this launcher at all without
# editing it -- and editing it is what you cannot do while an airline campaign is reading it, because
# this script is re-read per shard invocation. Defaults reproduce the previous behaviour exactly, so an
# existing caller is unaffected.
PINQ_SUITE="${PINQ_SUITE:-tau2_airline}"
PINQ_FORKS_FILE="${PINQ_FORKS_FILE:-tau2_airline_test.recovered34.json}"

unset PI_GOLD_ROOT
export PI_RUNS_ROOT="$RUNS"
export PYTHONPATH="$WT/src"
export LITELLM_BASE_URL=${PINQ_PROXY:-http://127.0.0.1:4000}

# THE ROLE PINS ARE SET BY PLAIN EXPORT, BEFORE `load_env.sh` SOURCES `.env` -- NEVER BY
# `${VAR:-default}` AFTER IT. `.env` pins every PI_MODEL_* role to gpt-oss-120b; `:-` only
# applies to an EMPTY variable, and `.env` never leaves these empty, so a default written after
# sourcing never fires. This ordering is the whole fix measured 2026-09-19 (see
# `load_env.sh` and `test_model_role_pin_precedence.py`): a plain export set BEFORE the source,
# survived by the helper's own capture-restore, is what decides. Do not move these below the
# `load_env.sh` call, and do not reintroduce `:-` here.
export PI_MODEL_DRAFTER=openai/aws/claude-sonnet-5
export PI_MODEL_ANSWERER=openai/aws/claude-sonnet-5
export PI_MODEL_USERSIM=openai/aws/gpt-oss-120b

DPO=qwen3-8b-dpo-stacked-notdone-both
case "$1" in
  1_base_qa)      export PI_MODEL_INQUIRER=qwen3-8b-base; ARM=inquirer_prompted; VARIANT=tau2_base ;;
  2_trained_qa)   export PI_MODEL_INQUIRER=$DPO;          ARM=inquirer_trained;  VARIANT=tau2_base ;;
  5_trained_frag) export PI_MODEL_INQUIRER=$DPO;          ARM=inquirer_trained;  VARIANT=tau2_stop ;;
  6_base_frag)    export PI_MODEL_INQUIRER=qwen3-8b-base; ARM=inquirer_prompted; VARIANT=tau2_stop ;;
  *) echo "unknown arm config $1 (3_trained_mayask and 4_base_mayask are CUT)" >&2; exit 9 ;;
esac

# NOW source `.env`, through the helper that restores every role pin set above AFTER the file
# has had its say -- so TAU2_DATA_DIR/LITELLM_API_KEY still load, and the five roles above still
# win. A raw `set -a; . "$ENVF"; set +a` here would silently re-open the exact bug this shard
# exists to not have.
PINQ_ENV_FILE="$ENVF" . "$WT/scripts/tau2_campaign/load_env.sh" || exit $?
: "${TAU2_DATA_DIR:?TAU2_DATA_DIR unset after sourcing $ENVF}"
: "${LITELLM_API_KEY:?LITELLM_API_KEY unset after sourcing $ENVF}"

cd "$WT" || exit 9
echo "=== shard $1 $PINQ_SUITE seed$2 start $(date -u +%H:%M:%S) inq=$PI_MODEL_INQUIRER drafter=$PI_MODEL_DRAFTER answerer=$PI_MODEL_ANSWERER arm=$ARM var=$VARIANT"
# PINQ_FORKPOINTS_OVERRIDE: how preflight gets ONE real unit, not `--n`. The fixed file below is
# a dict -- an EXPLICIT SELECTION -- and run_tau2_forks.py refuses to drop any of an explicit
# selection's points; `--n 1` against it does not sample 1 of 34, it refuses with rc=2, on EVERY
# arm, every time (measured 2026-09-19: this is what killed the first L9 preflight attempt,
# before a single unit was constructed -- no manifest, no ledger, no spend). The override must
# be a REAL subset file of the same shape (a dict with exactly the wanted points under
# `fork_points`), passed with NO `--n`/`--min-k`/`--max-k`, so the guard's own "no k flags ->
# use every point" branch fires and "every point" is however many the override file names. See
# tests/test_preflight_forkpoints_selection.py, which pins both halves of this: the guard's
# refusal (unchanged, correct, not weakened) and the override file's shape (the actual fix).
# Unset in a live launch, where the real, full 34-point file governs, unchanged.
"$PY" scripts/run_tau2_forks.py \
  --suite "$PINQ_SUITE" \
  --forkpoints "${PINQ_FORKPOINTS_OVERRIDE:-$CONF/${PINQ_FORKS_FILE}}" \
  --arm "$ARM" --seeds "$2" \
  --prompt-variant "$VARIANT" \
  --runs-root "$RUNS" ${DRY:+--dry-run}
rc=$?
echo "=== shard $1 $PINQ_SUITE seed$2 done rc=$rc $(date -u +%H:%M:%S)"
