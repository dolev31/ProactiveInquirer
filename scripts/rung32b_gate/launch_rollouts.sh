#!/usr/bin/env bash
# THE 32B ROLLOUT CAMPAIGN. Launched detached 2026-09-19 (night), rung32b-gate lane.
#
# WHERE IT RUNS. From a PINNED WORKTREE at 993a38a, not the main checkout: main's HEAD moved
# twice in the twenty minutes before launch (c9f8378 -> 1b8176b -> 993a38a) as peer lanes
# committed. `code_version` is inside `semantic_hash` and so inside `run_id`, and a worker
# imports from disk as it starts, so a merge into src/ mid-sweep puts NEW code under the OLD
# stamp -- a run set that is no longer one thing. The worktree's HEAD cannot move under it.
#
# WHAT THE WORKTREE LACKS, and how it is closed. `data/` is git-ignored in its entirety, so a
# worktree has none. ONE read-only symlink is added, `data/corpora` -> main's, and nothing
# else: never the parent `data/` (gold, live data/rl, rm-through-link hazard).
# VERIFIED, not assumed: all four packages import from the worktree's src/ under PYTHONPATH,
# and _resolve_corpus finds tasks.jsonl for both suites.
#
# ROOTS ARE ABSOLUTE CLI FLAGS. PI_RUNS_ROOT is NOT read by `pi run` (only --runs-root is),
# and .env's PI_CACHE_ROOT=./cache resolves against the READER's cwd -- from this worktree
# that would be a second, empty cache. Both are passed absolute, at the MAIN checkout.
#
# ONE BASE URL FOR EVERY ARM AND ROLE. base_url_sha is inside run identity; a per-arm port
# would re-pin the frozen Drafter/Answerer too and silently break the contrast.
set -uo pipefail
# DERIVED, NOT HARDCODED. The executed copy of this file carried the two paths as literals;
# scripts/check_no_home_paths.sh refuses a committed absolute home path, on the ground that it
# is the commonest way a research repo stops being reproducible for anyone but its author.
# Only these two lines differ from what ran -- `diff` against
# artifacts/rung32b_gate_20260919/launch_rollouts.executed.sha256 in RESULT.md.
MAIN="${PINQ_MAIN:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
WT="${PINQ_WT:-$(dirname "$MAIN")/pinq-wt-rung32b-20260919}"
A=$MAIN/artifacts/rung32b_gate_20260919
cd "$WT"
set -a; . "$MAIN/.env"; set +a
unset PI_GOLD_ROOT                      # the firewall: rollout workers must not see gold
export LITELLM_BASE_URL=http://127.0.0.1:4000
export PYTHONPATH="$WT/src"
PI="$MAIN/.venv/bin/pi"
COMMON=(--runs-root "$MAIN/runs" --cache-root "$MAIN/cache" --concurrency 16)

run_one() {  # $1=model  $2=grid  $3=cap
  echo "===== $(date -u +%FT%TZ)  MODEL=$1  GRID=$2  CAP=\$$3 ====="
  PI_MODEL_INQUIRER="$1" "$PI" run --sweep "$WT/$2" "${COMMON[@]}" --spend-cap "$3"
  echo "===== $(date -u +%FT%TZ)  exit=$?  $2 ====="
}

# THE COMPARATOR FIRST: it is the bigger half (930 of 1,395 units) and the gate is undefined
# without it. Counts MEASURED with pi_run.sweep.plan() over the real grid loader, not retyped
# from the grids' declared n_tasks.
run_one qwen3-32b-base          conf/grids/reroll/dev_musique.yaml      12   # 264 units
run_one qwen3-32b-base          conf/grids/reroll/dev_strategyqa.yaml   18   # 666 units
run_one qwen3-32b-sft-headline  conf/grids/dev_select_musique.yaml       8   # 132 units
run_one qwen3-32b-sft-headline  conf/grids/dev_select_strategyqa.yaml   12   # 333 units
echo "ALL DONE $(date -u +%FT%TZ)"
