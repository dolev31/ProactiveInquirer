"""Exact unit counts for the 32B rung, from the SAME code `pi run --sweep` dispatches.

Not retyped from the grid's declared n_tasks: the grids' own comments say musique's
declared 132 yields 125 after the dev-split filter, so the declared number overstates.
"""

from pathlib import Path

from pi_run import grids
from pi_run.cli import _resolve_corpus, _select_task_ids
from pi_run.sweep import plan
from pi_run.worker import load_suite

# The repo root, DERIVED: this file lives at <root>/scripts/rung32b_gate/, and a committed
ROOT = Path(__file__).resolve().parents[2]
GRIDS = [
    ("baseline", ROOT / "conf/grids/reroll/dev_musique.yaml"),
    ("baseline", ROOT / "conf/grids/reroll/dev_strategyqa.yaml"),
    ("checkpoint", ROOT / "conf/grids/dev_select_musique.yaml"),
    ("checkpoint", ROOT / "conf/grids/dev_select_strategyqa.yaml"),
]

tot = {"baseline": 0, "checkpoint": 0}
for role, p in GRIDS:
    g = grids.load(str(p)) if hasattr(grids, "load") else grids.Grid.from_yaml(str(p))
    for suite_id in g.suites:
        corpus = _resolve_corpus(ROOT, suite_id, None)
        suite = load_suite(suite_id, str(corpus))
        tids = list(g.task_ids) or _select_task_ids(
            suite, suite_id, n=g.n_tasks, split=g.split, offset=g.task_offset or 0
        )
        units = plan(
            suite_id=suite_id,
            corpus_dir=str(corpus),
            task_ids=tids,
            arm_ids=list(g.arms),
            seeds=list(g.seeds),
            runs_root="/dev/null",
            cache_root="/dev/null",
            max_turns=g.max_turns,
            k=g.k_for(suite_id),
            budget_cap=g.budget_cap,
            grid_name=g.name,
            exploratory=g.exploratory,
        )
        tot[role] += len(units)
        print(
            f"{role:11s} {g.name:26s} suite={suite_id:12s} declared_n={g.n_tasks} "
            f"selected_tasks={len(tids)} arms={list(g.arms)} seeds={list(g.seeds)} "
            f"k={g.k_for(suite_id)} cap={g.budget_cap} split={g.split} "
            f"exploratory={g.exploratory} UNITS={len(units)}"
        )
print()
for r, n in tot.items():
    print(f"TOTAL {r}: {n}")
print(f"TOTAL all: {sum(tot.values())}")
