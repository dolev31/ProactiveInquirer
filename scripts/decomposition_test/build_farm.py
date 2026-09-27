"""Build an isolated run-directory symlink farm for `pi compact` / `pi score`.

WHY A FARM RATHER THAN THE SHARED STORE. Measured 2026-09-18: the shared `scores/parquet`
mixes runs from several `code_version`s under one `grid_name`, and the SAME nominal
(suite, task, seed) baseline cell carries different `evidence_coverage` values across those
commits (221 of 1200 sampled keys disagreed, some by the full range 0.0..1.0) -- traced to a
compaction defect that dropped `turns`/`evidence`/`calls` rows for 21,001 runs while the run
directories on disk still hold them (Lane L0.6 / coordinator correction, 2026-09-18). Reading
gold-scored values back out of a store that lost the rows it scored them from is exactly the
"instrument error that looks like a finding" this repo's CONTRIBUTING.md warns about.

The fix used here: build a private directory of symlinks to EXACTLY the run ids this lane
needs (named by an explicit run-id-list file, never a live glob), point a fresh `pi compact`
and `pi score --allow-no-judge` at that directory alone, and read `_matched_cost` off the
private parquet it produces. One isolated store per lane, both arms of every contrast in it,
named per `artifacts/frames/FRAMES.md`'s convention:

    pi compact --runs-root <farm>/runs --out <farm>/scores_parquet --exclude-dev
    pi score --parquet <farm>/scores_parquet --runs-root <farm>/runs \\
        --gold-root data/gold --corpora-root data/corpora --allow-no-judge

This module only builds the farm (a pure filesystem operation); the two commands above are
run separately so their real stdout/exit code is captured verbatim in RESULT.md rather than
wrapped in a subprocess call this file would then have to re-test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def read_run_id_list(path: Path) -> list[str]:
    """One run id per line; blank lines and lines starting with '#' are skipped."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def build_farm(
    farm_runs_dir: Path,
    run_ids: Iterable[str],
    *,
    source_runs_root: Path,
) -> dict[str, list[str]]:
    """Symlink `source_runs_root/<run_id>` into `farm_runs_dir/<run_id>` for every id.

    Returns ``{"linked": [...], "missing": [...], "already_present": [...]}`` -- MISSING ids
    (no such directory under `source_runs_root`) are reported rather than silently skipped,
    because a run id list is the selector of record (see the module this feeds,
    `scripts/decomposition_test/contrast.py`'s ARM_MAP) and a silently-shrunk population reads
    as a smaller but otherwise normal-looking n.
    """
    farm_runs_dir = Path(farm_runs_dir)
    farm_runs_dir.mkdir(parents=True, exist_ok=True)
    # RESOLVED, NOT JUST `Path(...)`-WRAPPED. A relative `source_runs_root` (e.g. `Path("runs")`
    # from a caller whose cwd is the repo root) produces a symlink whose TARGET TEXT is that
    # same relative string, and a relative symlink resolves against the SYMLINK'S OWN
    # directory, not the cwd that was active when it was created. Measured 2026-09-18: built
    # exactly this way against `Path("runs")`, every one of 7,200 farm entries pointed at
    # `<farm>/runs/runs/<id>` -- one `runs/` too many -- and `pi compact` silently discovered
    # zero run directories (`"run_dirs": 0`), not an error. `.resolve()` here is what makes the
    # farm's symlinks absolute and therefore correct regardless of the caller's cwd.
    source_runs_root = Path(source_runs_root).resolve()

    linked: list[str] = []
    missing: list[str] = []
    already_present: list[str] = []
    for rid in run_ids:
        src = source_runs_root / rid
        dst = farm_runs_dir / rid
        if dst.exists() or dst.is_symlink():
            already_present.append(rid)
            continue
        if not src.exists():
            missing.append(rid)
            continue
        dst.symlink_to(src, target_is_directory=True)
        linked.append(rid)
    return {"linked": linked, "missing": missing, "already_present": already_present}
