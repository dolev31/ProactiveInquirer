#!/usr/bin/env bash
# Assemble the environment in an order that cannot undo a check. SOURCE this, do not execute it.
#
# THE DEFECT THIS EXISTS FOR, MEASURED 2026-09-19. `.env` in this repository carries
# `PI_RUNS_ROOT=./runs` and `PI_CACHE_ROOT=./cache` -- RELATIVE. Every lane sources `.env`,
# because a missing gateway key does not error, it hangs in retry backoff, so sourcing it is the
# standing instruction. The tau2 campaign launcher validated an ABSOLUTE runs root at startup
# and refused relative ones; then, further down, it sourced `.env` to get the key, and the
# file's `./runs` silently replaced the value the guard had just approved. Twelve shards
# inherited it and wrote 117 run directories into the pinned worktree, where no reporting path
# looks.
#
# A GUARD THAT VALIDATES AND IS THEN OVERWRITTEN BEHIND ITS BACK IS WORSE THAN NO GUARD,
# BECAUSE IT REPORTS SUCCESS. That is the second time in one night that A CORRECTION INTRODUCED
# THE NEXT FALSE STATE: the route rule's fix filed a permanent permission wall under "retry",
# and the fix that sourced `.env` for the key undid the root check that ran before it. Put a
# guard AFTER the environment is fully assembled, never before.
#
# WHY THE ROOTS ARE NOT SIMPLY REMOVED FROM `.env`: most lane worktrees have `runs` as a
# SYMLINK to the shared root, so `./runs` resolves through the link and lands correctly. The
# defect is latent for them and live only where the root is a real directory. Changing the file
# would move every one of those lanes at once; protecting an explicit caller value does not.
#
# THE RULE: a value the CALLER set explicitly wins over the file's, and whatever survives must
# be absolute -- a relative root resolves against the caller's working directory, which is how
# a probe once read a 16% cache hit rate against a cache that held 97.1% and reported it as a
# finding about the cache.
#
# THE SAME DEFECT, ONE LEVEL UP, MEASURED 2026-09-19. `.env` pins every `PI_MODEL_*` role to
# `openai/aws/gpt-oss-120b`. `phase5_shard.sh` wanted the drafter and answerer frozen at
# `claude-sonnet-5` and wrote `export PI_MODEL_DRAFTER=${PI_MODEL_DRAFTER:-claude-sonnet-5}`
# AFTER sourcing `.env` -- but `:-` only applies to an EMPTY variable, and `.env` had already
# set it, so the default never fired. All 408 units of the first phase-5 launch recorded
# `pins.drafter.model_id = openai/aws/gpt-oss-120b`, voiding the comparison against the
# published claude-sonnet-5 column that was the entire point of the campaign. The Inquirer
# role escaped only because its branch used a plain `export`, no `:-`, ever. THE `:-` IDIOM
# IS THE DEFECT, not a implementation detail of it: it means "the file wins if it set
# anything", and `.env` always sets these five. Every `PI_MODEL_*` role gets the identical
# capture-source-restore treatment as the roots, so a caller's plain `export` (no `:-`
# anywhere) is what decides, never the sourced file.
#
#   PINQ_ENV_FILE   the file to source (required)

__pinq_caller_runs=${PI_RUNS_ROOT:-}
__pinq_caller_cache=${PI_CACHE_ROOT:-}
__pinq_caller_inquirer=${PI_MODEL_INQUIRER:-}
__pinq_caller_drafter=${PI_MODEL_DRAFTER:-}
__pinq_caller_answerer=${PI_MODEL_ANSWERER:-}
__pinq_caller_usersim=${PI_MODEL_USERSIM:-}
__pinq_caller_judge=${PI_MODEL_JUDGE:-}

if [ -z "${PINQ_ENV_FILE:-}" ] || [ ! -f "${PINQ_ENV_FILE}" ]; then
  echo "load_env.sh: no env file at '${PINQ_ENV_FILE:-<unset>}'" >&2
  return 1 2>/dev/null || exit 1
fi

set -a
# shellcheck disable=SC1090
. "$PINQ_ENV_FILE"
set +a

# THE RESTORE. This is the whole point of the file: it happens AFTER the source, so nothing the
# file contains can win over a root -- or a role pin -- the caller chose deliberately.
[ -n "$__pinq_caller_runs" ] && PI_RUNS_ROOT=$__pinq_caller_runs
[ -n "$__pinq_caller_cache" ] && PI_CACHE_ROOT=$__pinq_caller_cache
[ -n "$__pinq_caller_inquirer" ] && PI_MODEL_INQUIRER=$__pinq_caller_inquirer
[ -n "$__pinq_caller_drafter" ] && PI_MODEL_DRAFTER=$__pinq_caller_drafter
[ -n "$__pinq_caller_answerer" ] && PI_MODEL_ANSWERER=$__pinq_caller_answerer
[ -n "$__pinq_caller_usersim" ] && PI_MODEL_USERSIM=$__pinq_caller_usersim
[ -n "$__pinq_caller_judge" ] && PI_MODEL_JUDGE=$__pinq_caller_judge
export PI_RUNS_ROOT PI_CACHE_ROOT
export PI_MODEL_INQUIRER PI_MODEL_DRAFTER PI_MODEL_ANSWERER PI_MODEL_USERSIM PI_MODEL_JUDGE

for __pinq_v in PI_RUNS_ROOT PI_CACHE_ROOT; do
  eval "__pinq_val=\${$__pinq_v:-}"
  case "$__pinq_val" in
    /*) ;;
    "") echo "load_env.sh: $__pinq_v is unset after assembling the environment" >&2
        unset __pinq_v __pinq_val __pinq_caller_runs __pinq_caller_cache __pinq_caller_inquirer __pinq_caller_drafter __pinq_caller_answerer __pinq_caller_usersim __pinq_caller_judge
        return 2 2>/dev/null || exit 2 ;;
    *)  echo "load_env.sh: $__pinq_v='$__pinq_val' is not ABSOLUTE." >&2
        echo "  A relative root resolves against the caller's working directory, so two" >&2
        echo "  instruments can read different trees and agree with each other about it." >&2
        echo "  Pass an absolute path; it will survive sourcing $PINQ_ENV_FILE." >&2
        unset __pinq_v __pinq_val __pinq_caller_runs __pinq_caller_cache __pinq_caller_inquirer __pinq_caller_drafter __pinq_caller_answerer __pinq_caller_usersim __pinq_caller_judge
        return 2 2>/dev/null || exit 2 ;;
  esac
done
unset __pinq_v __pinq_val __pinq_caller_runs __pinq_caller_cache __pinq_caller_inquirer __pinq_caller_drafter __pinq_caller_answerer __pinq_caller_usersim __pinq_caller_judge
