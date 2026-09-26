#!/usr/bin/env bash
# Wait until no tau2 fork WORKER is alive, bounded. Committed, because a campaign watchdog that
# lives in a scratchpad is a watchdog nobody inherits.
#
# WHY A SCRIPT FILE AND WHY THIS PATTERN -- both halves are measured failures from 2026-09-19.
#
#   1. An inline `bash -c 'until [ "$(pgrep -f run_tau2_forks ...)" -eq 0 ]; ...'` MATCHES
#      ITSELF: the pattern is in the waiting shell's own argv, so the count can never reach zero
#      and the wait runs silently to its deadline. Four such loops were left running before it
#      was noticed, and each one also poisoned every other process count taken while it lived.
#      In a script FILE the waiting shell's argv is the script path, so the same pattern is
#      safe -- the bug is the inline form, not the pattern.
#
#   2. `pgrep -c` DOES NOT EXIST on this platform (BSD pgrep). It prints a usage message to
#      stderr and nothing to stdout, so `n=$(pgrep -c -f pat)` is empty and `[ "$n" -eq 0 ]`
#      reads that as "nothing is running". A wait built on it returned "no workers left" with
#      two live shards. Hence `pgrep -f ... | wc -l`.
#
# Both failures are silent and both look like a result, which is the whole reason they are
# written down here rather than fixed quietly.
#
#   $1 = bound in seconds (default 14 hours)
set -u
PAT=${PINQ_WORKER_PATTERN:-'python scripts/run_tau2_forks.py'}
DEADLINE=$(( $(date +%s) + ${1:-50400} ))
while :; do
  n=$(pgrep -f "$PAT" | wc -l | tr -d ' ')
  [ "$n" -eq 0 ] && { echo "NO WORKERS LEFT $(date -u +%H:%M:%S)"; break; }
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "BOUND REACHED with $n worker(s) alive $(date -u +%H:%M:%S)"; break
  fi
  sleep 60
done
