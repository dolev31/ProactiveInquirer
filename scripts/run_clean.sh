#!/usr/bin/env bash
# Run a sweep with a CLEAN code_version while the working tree is dirty.
#
# WHY THIS EXISTS. `git_info` marks a run `dirty` from `git status --porcelain
# --untracked-files=no`, so ANY uncommitted tracked file anywhere in the repo -- including one
# belonging to a concurrent workstream that this sweep does not import -- stamps every run with
# a `dev-` prefix. `pi_eval.report`'s ELIGIBLE predicate excludes `is_dev_run = TRUE`, so those
# runs can never appear in a reported table. Measured while another session held five tracked
# files open: 58 of 62 tau2 runs came out dev- and unreportable.
#
# The fix is not to weaken the dirty check -- it is correct, an unknown code version is exactly
# what the prefix is for -- but to run from a worktree pinned at HEAD, which genuinely IS
# clean. Artifacts still land in the main tree, so nothing is split across two locations.
#
# Usage:  scripts/run_clean.sh <pi-run args...>
#   e.g.  scripts/run_clean.sh --suite musique --arm drafter_only --n 5 --seeds 0
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WT="${PI_CLEAN_WORKTREE:-${TMPDIR:-/tmp}/pi-clean-wt}"

# SYNC THE WORKTREE TO THE MAIN REPO'S HEAD, BY SHA.
#
# This read `git -C "$WT" checkout --detach HEAD`, which in an already-detached worktree
# resolves HEAD to the WORKTREE's own commit and is therefore a NO-OP. Every invocation after
# the first silently ran STALE CODE while stamping runs with whatever `code_version` that old
# commit had -- the precise failure `git_info` exists to make visible, reintroduced by the
# script meant to protect it.
#
# Naming the sha explicitly is what makes it move. Refusing when a run is already using the
# worktree is what stops a checkout changing files under a live sweep.
HEAD_SHA="$(git -C "$MAIN" rev-parse HEAD)"

# A WORKTREE UNDER /tmp OUTLIVES ITS REGISTRATION. The default location is $TMPDIR, which the
# OS clears; git then still lists the worktree and refuses to re-add it:
#   "is a missing but already registered worktree; use 'add -f' to override, or 'prune'"
# Observed after the machine was away for an hour: the directory was gone, the registration
# was not, and the sweep would not start. `prune` drops registrations whose directory has
# vanished and touches nothing that still exists.
git -C "$MAIN" worktree prune >/dev/null 2>&1 || true

if [ ! -d "$WT/.git" ] && [ ! -f "$WT/.git" ]; then
  git -C "$MAIN" worktree add --detach "$WT" "$HEAD_SHA" >&2
elif [ "$(git -C "$WT" rev-parse HEAD)" != "$HEAD_SHA" ]; then
  if pgrep -f "bin/pi run --root $WT" >/dev/null 2>&1; then
    echo "refusing: a sweep is running from $WT and moving it would change files underneath it." >&2
    echo "          wait for it, or set PI_CLEAN_WORKTREE to a second worktree." >&2
    exit 3
  fi
  git -C "$WT" checkout --detach "$HEAD_SHA" >/dev/null 2>&1 || {
    echo "could not move $WT to $HEAD_SHA" >&2; exit 3; }
fi

# REFUSE rather than silently produce dev- runs anyway: a script whose whole purpose is a
# clean code_version must not quietly fail open.
if [ -n "$(git -C "$WT" status --porcelain --untracked-files=no)" ]; then
  echo "worktree $WT is itself dirty; refusing (it would stamp dev- exactly as the main tree does)" >&2
  exit 2
fi

export PI_GOLD_ROOT="${PI_GOLD_ROOT:-$MAIN/data/gold}"
exec "$MAIN/.venv/bin/pi" run \
  --root "$WT" \
  --runs-root "$MAIN/runs" \
  --cache-root "$MAIN/cache" \
  "$@"
