"""Probe: can `pi_eval.build.synth_build.build` be scaled and parameterised for a horizontal
suite, and what does that cost?

Runs the real generator (no LLM calls, no gateway, $0) at several (n_tasks, n_facets, depth)
points into a throwaway root, timing wall-clock and measuring output size. Writes nothing
into the repo's own `data/gold/synth` -- every run below takes an explicit `root` under a
caller-supplied temp directory so the real synth v1 gold is never touched.

This is a read/measure tool, not a build step: it does not register anything under the
repo's `data/gold` or `data/corpora` trees, and the caller is expected to point `root` at a
scratch directory outside the repository.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pi_eval.build.synth_build import build  # noqa: E402

POINTS = [
    {"n_tasks": 12, "n_facets": 3, "depth": 3, "n_distractors": 6},  # shipped default
    {"n_tasks": 500, "n_facets": 4, "depth": 4, "n_distractors": 10},
    {"n_tasks": 2000, "n_facets": 8, "depth": 6, "n_distractors": 10},
]


def main(scratch_root: str) -> None:
    root = Path(scratch_root)
    rows = []
    for point in POINTS:
        sub = root / f"{point['n_tasks']}_{point['n_facets']}_{point['depth']}"
        t0 = time.time()
        corpus_path, gold_path, corpus_hash = build(
            n_tasks=point["n_tasks"],
            n_facets=point["n_facets"],
            depth=point["depth"],
            n_distractors=point["n_distractors"],
            seed=7,
            root=sub,
        )
        dt = time.time() - t0
        rows.append(
            {
                **point,
                "wall_s": round(dt, 3),
                "corpus_bytes": corpus_path.stat().st_size,
                "gold_bytes": gold_path.stat().st_size,
                "corpus_hash": corpus_hash,
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: synth_scale_probe.py <scratch_root_outside_repo>")
    main(sys.argv[1])
