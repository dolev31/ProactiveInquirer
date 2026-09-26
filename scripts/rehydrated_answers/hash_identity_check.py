"""Coordinator gate, reproducible: does adding a column to pi_eval.schema.RUNS move any
identity hash? Loads `src/pi_eval/schema.py` from two git refs (default: HEAD^ and HEAD) as two
separate modules via `git show`, computes `schema_hash()` on each, then replicates
`pi_eval.score`'s own `metric_defs_hash()`/`scorer_hash()` FORMULAS (same METRICS, DWR_WEIGHTS,
h, canon -- all from the one real, unmodified `pi_eval.score`, which this lane never edited)
with each schema module's `schema_hash()` substituted in turn. `graph_hash`/`matcher_hash`/
`METRIC_DEFS_VERSION` never read `pi_eval.schema`, so they are computed once and printed on both
sides for completeness rather than recomputed twice.

Run from the repo root with the project's venv (kept under scripts/, not artifacts/: results
belong under artifacts/, the tooling that produces them belongs somewhere ruff reads --
tests/test_artifacts_hold_no_python.py):
    .venv/bin/python scripts/rehydrated_answers/hash_identity_check.py [before] [after]
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from types import SimpleNamespace


def _git_show(ref: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=True
    ).stdout


def _load_module(source: str, name: str):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(source)
        tmp_path = fh.name
    spec = importlib.util.spec_from_file_location(name, tmp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    before_ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD^"
    after_ref = sys.argv[2] if len(sys.argv) > 2 else "HEAD"

    schema_before = _load_module(
        _git_show(before_ref, "src/pi_eval/schema.py"), "schema_before_scratch"
    )
    schema_after = _load_module(
        _git_show(after_ref, "src/pi_eval/schema.py"), "schema_after_scratch"
    )

    from pi_eval import score as sc
    from pinq.ids import canon, h

    def metric_defs_hash_with(schema_mod) -> str:
        return h(
            "metric_defs",
            sc.METRIC_DEFS_VERSION,
            canon(
                [
                    (m.name, m.family, m.binary, m.judge_derived, m.higher_is_better, m.indexed)
                    for m in sc.METRICS
                ]
            ),
            canon(sorted(sc.DWR_WEIGHTS.items())),
            schema_mod.schema_hash(),
        )

    schema_hash_before = schema_before.schema_hash()
    schema_hash_after = schema_after.schema_hash()
    metric_defs_before = metric_defs_hash_with(schema_before)
    metric_defs_after = metric_defs_hash_with(schema_after)

    graph_val = sc.graph_hash({})
    matcher_val = sc.matcher_hash(SimpleNamespace(matcher_id="dummy", matcher_family="dummy"))

    scorer_before = sc.scorer_hash(
        metric_defs=metric_defs_before, graph=graph_val, matcher=matcher_val, judge_pins=[]
    )
    scorer_after = sc.scorer_hash(
        metric_defs=metric_defs_after, graph=graph_val, matcher=matcher_val, judge_pins=[]
    )

    rows = [
        ("schema_hash", schema_hash_before, schema_hash_after),
        ("metric_defs_hash", metric_defs_before, metric_defs_after),
        ("graph_hash (fixed input)", graph_val, graph_val),
        ("matcher_hash (fixed input)", matcher_val, matcher_val),
        ("scorer_hash (fixed graph/matcher/judge_pins)", scorer_before, scorer_after),
        ("METRIC_DEFS_VERSION", sc.METRIC_DEFS_VERSION, sc.METRIC_DEFS_VERSION),
    ]
    width = max(len(r[0]) for r in rows)
    print(f"before={before_ref} after={after_ref}")
    print(f"{'quantity':{width}}  {'before':66}  {'after':66}  same?")
    for name, before, after in rows:
        print(f"{name:{width}}  {before:66}  {after:66}  {before == after}")


if __name__ == "__main__":
    main()
