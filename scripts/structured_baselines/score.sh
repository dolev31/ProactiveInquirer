#!/usr/bin/env bash
# Compact and score the clean structured-baselines grid. Run the gates FIRST; this does not.
#
# WHY PI_GOLD_ROOT IS EXPORTED HERE AND NOWHERE ELSE. The rollouts ran with it UNSET, and that is
# the firewall working rather than an oversight: `gold_root()` raises in a rollout worker, so a
# policy cannot reach the need graphs. Scoring is the stage that is supposed to see gold, so it is
# the only stage that exports it. If a rollout ever succeeds with this set, that is the leak.
#
# ORDER, and the gates are not optional: scripts/structured_baselines/gate_and_read.py refuses on a
# mixture of code_versions, two base_url_shas, a single arm, a selected intersection, re-runs
# counted as progress, and an absent root. Scoring a population that fails those produces a store
# that looks exactly like a good one.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:?usage: score.sh <artifact dir holding runs/>}"
# One artifact dir holds several roots (runs/, runs_8b/, runs_frontier/, runs_s1/, runs_s2/), one per
# invocation, each scored into ITS OWN store so an arm_id never spans two inquirer models in one store.
# Unset, these are exactly the original paths.
RUNS="${PINQ_RUNS_DIR:-$OUT/runs}"
PARQUET="${PINQ_PARQUET_DIR:-$OUT/scores_parquet}"
PY="$REPO/.venv/bin/python"

[ -d "$RUNS" ] || { echo "score.sh: REFUSING: no runs directory at $RUNS" >&2; exit 3; }
[ -x "$PY" ] || { echo "score.sh: REFUSING: no interpreter at $PY" >&2; exit 3; }

# The worktree the campaign ran from, so the scorer matches the code that produced the runs. Passed
# in rather than guessed, because scoring under a different code_version than the rollouts is the
# same class of error as pooling two of them.
WT="${PINQ_SCORE_WORKTREE:-$REPO}"
export PYTHONPATH="$WT/src"

mkdir -p "$PARQUET"

echo "=== compact $(date -u +%FT%TZ) ==="
"$PY" -m pi_run.cli compact --root "$WT" --runs-root "$RUNS" --out "$PARQUET"

echo "=== score, with PI_GOLD_ROOT exported $(date -u +%FT%TZ) ==="
PI_GOLD_ROOT="$REPO/data/gold" "$PY" -m pi_run.cli score \
  --root "$WT" \
  --parquet "$PARQUET" \
  --runs-root "$RUNS" \
  --gold-root "$REPO/data/gold" \
  --corpora-root "$REPO/data/corpora" \
  --graph-version v1 \
  --allow-no-judge

echo "=== scored store $(date -u +%FT%TZ) ==="
ls -la "$PARQUET"

# matches.parquet is what a matched-cost ladder is reconstructed from. Without it the contrast
# cannot be read at all, and its absence is quieter than an error: the next stage would fall back
# to whatever ladder it can build and verify_instrument would then have nothing to lock against.
[ -f "$PARQUET/matches.parquet" ] || {
  echo "score.sh: REFUSING: no matches.parquet; the matched-cost ladder cannot be rebuilt" >&2
  exit 3
}
echo "score.sh: matches.parquet present, the ladder can be rebuilt and locked."
