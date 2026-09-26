"""Score an isolated store for `evidence_coverage` only. $0, offline, no gateway calls.

`PI_GOLD_ROOT` must be set before this process imports `pi_eval` (the firewall raises
`GoldAccessError` otherwise -- that raise is the firewall working, not a bug to route around).
No judge is passed and none is configured in this repo's `.env` (`PI_JUDGE_CLIENT` unset), so
`judge_client_from_env()` resolves to `judge=None` and no LLM/network call happens.

Usage: PI_GOLD_ROOT=<repo>/data/gold python3 scripts/baseline_repro/run_score.py <store_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    store_dir = Path(sys.argv[1])

    from pi_eval.gold import gold_root
    from pi_eval.score import score

    # Fail loudly here rather than let the firewall's own raise be the first sign of trouble.
    print(f"PI_GOLD_ROOT resolves to: {gold_root()}")

    result = score(
        parquet_dir=store_dir,
        runs_root=None,  # evidence_coverage does not read answers; no judge, so no answers needed.
        graph_version="v1",
        judge=None,
        seed=0,
    )
    print(f"scorer_hash: {result.scorer_hash}")
    print(f"n_runs_scored: {result.n_runs_scored}")
    print(f"n_runs_skipped_no_graph: {result.n_runs_skipped_no_graph}")
    print(f"judging: {result.judging}")


if __name__ == "__main__":
    main()
