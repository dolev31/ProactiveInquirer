"""Pure sizing check for the three new cap32 frontier grids -- no network call, no spend.

Mirrors exactly what `pi run --sweep` does before it launches anything: resolve the corpus,
load the suite, select task_ids at split=test (head-n after the split filter, per
`pi_run.cli._select_task_ids`), then call `pi_run.sweep.plan()` once per pin-invocation (the
two/three-pin protocol is three or four separate `pi run --arm ...` calls, never one call
covering every arm under one ambient PI_MODEL_INQUIRER). Reports unit counts only; computes no
dollar figure (no historical $/unit basis exists yet for strategyqa or frames at this cap).

Run with PYTHONPATH=<ROOT>/src, and PI_GOLD_ROOT unset (this script never touches gold;
split_of works off the corpus/manifest split assignment, not gold). ROOT is overridable via
the SIZE_ROOT env var so the same script can size against the shared checkout or against a
pinned detached worktree -- the two are expected to agree, and disagreement is itself a
finding worth reporting, not something to paper over.
"""

import os
import sys
from pathlib import Path

from pi_run.cli import _resolve_corpus, _select_task_ids
from pi_run.sweep import plan
from pi_run.worker import load_suite

# Repo-relative fallback (this file lives at <repo>/scripts/) with an environment override --
# never a hardcoded home path, per scripts/check_no_home_paths.sh.
_DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent)
ROOT = os.environ.get("SIZE_ROOT", _DEFAULT_ROOT)
print(f"sizing against ROOT={ROOT}")

GRIDS = {
    "frontier_trained_musique_cap32": dict(
        suite="musique",
        n_tasks=200,
        seeds=[0, 1],
        pins=["inquirer_prompted:teacher", "inquirer_prompted:base8b", "inquirer_trained:trained"],
    ),
    "frontier_trained_strategyqa_cap32": dict(
        suite="strategyqa",
        n_tasks=200,
        seeds=[0, 1],
        pins=["inquirer_prompted:teacher", "inquirer_prompted:base8b", "inquirer_trained:trained"],
    ),
    "frames_trained_cap32": dict(
        suite="frames",
        n_tasks=824,
        seeds=[0],
        pins=[
            "drafter_only:drafter_only",
            "inquirer_prompted:teacher",
            "inquirer_prompted:base8b",
            "inquirer_trained:trained",
        ],
    ),
}

grand_total = 0
for grid_name, cfg in GRIDS.items():
    suite_id = cfg["suite"]
    from pathlib import Path

    corpus = _resolve_corpus(Path(ROOT), suite_id, None)
    suite = load_suite(suite_id, str(corpus))
    task_ids = _select_task_ids(suite, suite_id, n=cfg["n_tasks"], split="test", offset=0)
    print(
        f"{grid_name}: suite={suite_id} selected {len(task_ids)} test task_ids (grid asked for {cfg['n_tasks']})"
    )
    if len(task_ids) != cfg["n_tasks"]:
        print(
            f"  ZERO-TO-INTERROGATE: selected {len(task_ids)} != grid n_tasks {cfg['n_tasks']}",
            file=sys.stderr,
        )

    grid_total = 0
    for pin_label in cfg["pins"]:
        arm_id, pin_name = pin_label.split(":")
        specs = plan(
            suite_id=suite_id,
            corpus_dir=str(corpus),
            task_ids=task_ids,
            arm_ids=[arm_id],
            seeds=cfg["seeds"],
            runs_root=str(Path(ROOT) / "runs"),
            cache_root=str(Path(ROOT) / "cache"),
            max_turns=32,
            k=5,
            budget_cap=32,
            code_version="sizing-only",
            dirty=True,
            grid_name=grid_name,
        )
        print(f"  pin={pin_name:12s} arm={arm_id:18s} units={len(specs)}")
        grid_total += len(specs)
    print(f"  {grid_name} TOTAL units (all pins): {grid_total}")
    grand_total += grid_total

print(f"\nGRAND TOTAL units across all three cap32 grids, all pins: {grand_total}")
