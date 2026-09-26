#!/usr/bin/env bash
# Lane L6.1 step 4: the dev gate sweep and the held-out test sweep for the answer-node arm.
#
# RUN FROM A PINNED, CLEAN WORKTREE. `pi run --sweep` refuses a dirty tree, and a dirty tree
# stamps every run `dev-` (training-only, never an eval arm) -- including from a PEER's
# uncommitted file, because the checkout is shared. The pin recipe is `pi_run.cli`'s own
# refusal message:
#
#   git worktree add --detach <PIN> <sha>
#   ln -s <REPO>/data/corpora <PIN>/data/corpora     # EXACT subpaths only, never data/
#   ln -s <REPO>/data/raw     <PIN>/data/raw
#   ln -s <REPO>/data/canaries <PIN>/data/canaries
#
# Linking the parent `data/` would put gold inside the sweep's own root, which is the one thing
# the process split exists to prevent.
#
# WHAT THIS SCRIPT DOES NOT DO. It does not restart the proxy. `conf/serving/litellm.yaml` is
# shared and a restart lands on whichever peer sweep is in flight; the route is registered and
# the restart is scheduled by the operator, once.
#
# USAGE
#   PIN=<pinned worktree> REPO=<main checkout> MODEL=qwen3-8b-sft-headline-answernode \
#   PHASE=dev|test bash scripts/answer_node_stop/sweep_answernode.sh
set -uo pipefail

: "${PIN:?PIN=<pinned clean worktree>}"
: "${REPO:?REPO=<main checkout>}"
: "${MODEL:?MODEL=<served route name>}"
: "${PHASE:?PHASE=dev|test}"
LOGDIR="${LOGDIR:-$REPO/artifacts/answer_node_stop_20260919/logs}"
mkdir -p "$LOGDIR" || exit 9

# ABSOLUTE ROOTS, PASSED AS FLAGS. `.env` ships `PI_CACHE_ROOT=./cache`, which resolves against
# the READER's cwd -- from a pinned worktree that addresses an empty cache inside the worktree
# and reads as a 16% hit rate against the runs' own 97.1%, as absence rather than as an error.
# `PI_RUNS_ROOT` is not read by `pi run` at all, so it must be `--runs-root`.
CACHE="$REPO/cache"
RUNS="$REPO/runs"
# The frozen roles (Answerer, Drafter) must resolve to the SAME base_url as the Inquirer, or
# `base_url_sha` re-pins them too and the contrast is silently broken.
export LITELLM_BASE_URL="http://127.0.0.1:4000"
unset PI_GOLD_ROOT   # firewall layer 3: a rollout worker that can read gold is the leak

if [ "$PHASE" = "test" ]; then
  GRIDS=(conf/grids/tier1_trained_qa_base.yaml)
else
  GRIDS=(conf/grids/dev_select_musique.yaml conf/grids/dev_select_strategyqa.yaml conf/grids/dev_select_wiki2.yaml)
fi

# ASSERT EVERY ROUTE WITH A ONE-TOKEN COMPLETION, AND NEVER BY LOOKING FOR THE NAME IN
# /v1/models. Two independent reasons, and the second is the one that bites.
#
#  1. A ROUTER ENTRY IS NOT A LOADED ADAPTER. Three arms have been present in /v1/models AND in
#     conf/checkpoints.json while the vLLM backend answered `NotFoundError: The model ... does
#     not exist`. Only a generation says the weights are there; `$ART/serve/SERVED.<job>.json`
#     is what says WHICH weights answered.
#
#  2. THE LIST IS NOT A MEMBERSHIP TEST, AND ABSENCE FROM IT MEANS NOTHING. The shared proxy
#     enumerates 500+ ids and DOES NOT list the gpt-oss-120b id that serves as Drafter,
#     Answerer and Judge for every arm: that id resolves through a WILDCARD route the listing
#     does not enumerate, and real generations against it succeed (verified live on three ids,
#     2026-09-19). So grepping the listing returns absence for a route that works perfectly,
#     and reads as "the frozen roles are unrouted and this sweep will fail" -- plausible,
#     alarming, and false. It is the check anybody would assume is correct, which is why the
#     reason is written here rather than left to be rediscovered at 3am.
_probe() {  # $1 = model id, $2 = what it is, for the message
  local body
  body=$(curl -sS -m 60 http://127.0.0.1:4000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" 2>&1) \
    || { echo "REFUSE: the completion request to $1 ($2) failed outright" >&2; return 9; }
  case "$body" in
    *'"choices"'*) echo "route OK ($2): $1" ;;
    *) echo "REFUSE: $1 ($2) returned no choices -- a route can exist while the weights do not
answer. Response: ${body:0:400}" >&2; return 9 ;;
  esac
}

_probe "$MODEL" "the arm under test" || exit 9
# THE FROZEN ROLES RIDE ON THE SAME PROXY and must answer too, or the contrast breaks in a way
# no per-arm check would catch. `PI_FROZEN_ROLE_MODELS` is a bash array the caller sets; these
# are exactly the ids the listing does not enumerate, so they are probed and never grepped.
for ROLE_MODEL in "${PI_FROZEN_ROLE_MODELS[@]:-}"; do
  [ -n "$ROLE_MODEL" ] || continue
  _probe "$ROLE_MODEL" "frozen role" || exit 9
done

for G in "${GRIDS[@]}"; do
  NAME="$(basename "$G" .yaml).$PHASE"
  LOG="$LOGDIR/$NAME.log"
  # nohup AND disown, and then VERIFIED alive across a turn: `setsid` does not exist on macOS,
  # so survival is something to observe rather than something to assume from having typed
  # nohup. A foreground sweep dies with the shell that launched it, half-written.
  PYTHONPATH="$PIN/src" PI_MODEL_INQUIRER="$MODEL" \
    nohup "$REPO/.venv/bin/pi" run --root "$PIN" --sweep "$G" \
      --arm inquirer_trained \
      --runs-root "$RUNS" --cache-root "$CACHE" \
      --spend-cap 20 --concurrency 8 \
      > "$LOG" 2>&1 &
  echo "launched $NAME pid=$! log=$LOG"
  disown
done

echo "VERIFY: ps -p <pid> across a turn, and grep usd_billed in each log before believing any of them."
