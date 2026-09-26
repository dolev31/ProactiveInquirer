"""The four populations this lane compares, and how each one is pinned.

Every population is identified the way `artifacts/banking/RESULT.md` insists a tau2 population
must be: `arm_id` alone is poisoned (the same string names different models across campaigns),
so every selector here also pins `code_version` (or an explicit run-id list) and the result is
checked against the published `run_ids_sha` digest where one has already been printed
(`paper/appendix_instruments.tex`, `artifacts/forks_test/forks.tau2_*.test.json`). A population
whose digest does not match what was published is a bug in this file, not a new reading.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from scripts.tau2_failure_taxonomy.runs_io import (
    RunRecord,
    load_run,
    load_runs_by_id,
    read_run_id_list,
    scan_manifests,
)

_TAU2_SUITE_IDS = frozenset({"tau2_retail", "tau2_airline", "tau2"})


@lru_cache(maxsize=4)
def _scan_all_tau2(runs_root: Path) -> tuple[tuple[Path, dict], ...]:
    """One pass over `runs_root` for every tau2* manifest, cached per root for this process.

    `published_retail`/`published_airline`/`degraded` each used to run their own independent
    `scan_manifests` pass (four full walks of a ~170k-entry directory for one `build_taxonomy`
    run). Manifests are immutable once written, so caching the walk for the lifetime of this
    process is safe: nothing here writes to `runs_root`, and a run directory that appears
    mid-process (a peer campaign still writing into the shared tree) is exactly the kind of
    population drift `runs-dir-is-shared-filter-on-split-arm-code` warns is invisible unless
    this lane names its own selectors precisely -- which every caller below still does on top
    of this cache. `lru_cache` only hashes the ARGUMENT (`runs_root`, a `Path`); the returned
    manifests can and do stay plain, mutable dicts.
    """
    return tuple(scan_manifests(runs_root, suite_ids=_TAU2_SUITE_IDS))


def _iter_cached_manifests(runs_root: Path):
    yield from _scan_all_tau2(runs_root)


# NO ABSOLUTE PATH IS HARDCODED HERE (`scripts/check_no_home_paths.sh` scans every committed
# `*.py`, this file included). This lane is told to "read runs/ and scores/parquet by absolute
# path from the main checkout" -- the absolute value belongs in the CALLER's environment, not
# in source, exactly like `PI_CACHE_ROOT`/`PI_RUNS_ROOT` already work everywhere else in this
# repository (`pi_run/cache.py::cache_root`). A caller in a worktree (this lane's own operating
# mode) exports `PI_RUNS_ROOT`/`PI_ARTIFACTS_ROOT` to the shared checkout's absolute paths; a
# caller running from the main checkout itself needs neither and gets the relative default.
RUNS_ROOT = Path(os.environ.get("PI_RUNS_ROOT") or "runs")
ARTIFACTS = Path(os.environ.get("PI_ARTIFACTS_ROOT") or "artifacts")

# ------------------------------------------------------------- published fork populations

RETAIL_CODE_VERSIONS = ("828a720", "922060c")
RETAIL_ARMS = ("inquirer_prompted", "self_ask")
RETAIL_RUN_IDS_SHA = "1a060380806fff3e"

AIRLINE_CODE_VERSIONS = ("401b8fe", "922060c")
AIRLINE_ARMS = ("inquirer_prompted", "self_ask")
AIRLINE_RUN_IDS_SHA = "4fc108b375d5b0cb"

# ------------------------------------------------------------------------ banking campaign

BANKING_ARM_FILES = {
    "drafter_only": "run_ids.drafter_only.txt",
    "inquirer_prompted": "run_ids.inquirer_prompted.txt",
    "self_inquire": "run_ids.self_inquire.txt",
}

# ------------------------------------------------------------- the degraded phase4 campaign

# `4_base_mayask` in `phase4_shard.sh`: the ONLY one of the six configured arm-shards that
# reached full completion (34 fork points x 3 seeds x 2 domains = 204) before the campaign was
# killed at 02:12 -- see the module-level docstring of `build_taxonomy.py` for the manifest
# census this was read off (641 matched manifests total at this code_version; only this
# arm/pin/variant triple carries exactly 102 retail + 102 airline).
DEGRADED_CODE_VERSION = "132e3e813d3fcd24c28332689abb310bac025588"
DEGRADED_ARM_ID = "inquirer_may_ask_user"
DEGRADED_INQUIRER_MODEL = "qwen3-8b-base"
DEGRADED_PROMPT_VARIANT = "v1"


def _run_ids_sha(run_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(run_ids)).encode()).hexdigest()[:16]


def _fork_population(
    suite_id: str, code_versions: Sequence[str], arms: Sequence[str], runs_root: Path = RUNS_ROOT
) -> list[RunRecord]:
    out = []
    for run_dir, m in _iter_cached_manifests(runs_root):
        if m.get("suite_id") != suite_id:
            continue
        cv = str(m.get("code_version") or "")
        if not any(cv.startswith(p) for p in code_versions):
            continue
        if m.get("arm_id") not in arms:
            continue
        if str(m.get("split")) != "test":
            continue
        if not m.get("foreign_trace_sha"):
            continue
        r = load_run(run_dir, m)
        if not r.ok:
            continue
        out.append(r)
    return out


@dataclass(frozen=True)
class Population:
    name: str
    label: str
    runs: list[RunRecord]
    run_ids_sha: str | None = None
    expected_sha: str | None = None

    @property
    def sha_ok(self) -> bool | None:
        if self.expected_sha is None:
            return None
        return self.run_ids_sha == self.expected_sha


def published_retail(runs_root: Path = RUNS_ROOT) -> Population:
    runs = _fork_population("tau2_retail", RETAIL_CODE_VERSIONS, RETAIL_ARMS, runs_root)
    sha = _run_ids_sha([r.run_id for r in runs])
    return Population("published_retail", "published retail forks", runs, sha, RETAIL_RUN_IDS_SHA)


def published_airline(runs_root: Path = RUNS_ROOT) -> Population:
    runs = _fork_population("tau2_airline", AIRLINE_CODE_VERSIONS, AIRLINE_ARMS, runs_root)
    sha = _run_ids_sha([r.run_id for r in runs])
    return Population(
        "published_airline", "published airline forks", runs, sha, AIRLINE_RUN_IDS_SHA
    )


def banking(runs_root: Path = RUNS_ROOT, artifacts_dir: Path = ARTIFACTS / "banking") -> Population:
    runs: list[RunRecord] = []
    for arm, fname in BANKING_ARM_FILES.items():
        ids = read_run_id_list(artifacts_dir / fname)
        runs.extend(load_runs_by_id(runs_root, ids))
    return Population("banking", "banking (tier2_confirmatory)", runs)


def degraded(runs_root: Path = RUNS_ROOT) -> Population:
    """The 204-unit population: `4_base_mayask` (`inquirer_may_ask_user`, `qwen3-8b-base`,
    prompt_variant `v1`) at `code_version` `132e3e8...`, both domains, all three seeds -- the
    one arm-shard of the six-arm, session-killed campaign that reached full completion.

    ALL 204 MANIFESTS ARE KEPT, not only the 190 marked `status: ok`. The session limit that
    stopped the campaign at 02:12 caught 12 of the 204 mid-write -- a directory and a manifest
    exist but `status.json` is empty (no `finished_at` was ever recorded) -- plus 1 `timeout`
    and 1 `error`. Dropping the 12 would silently shrink "the 204 completed units" this lane was
    handed to 190 and lose exactly the units a session-limit kill produces; instead every
    caller here sees all 204 and reads `RunRecord.run_status`/`.status` to know which of them
    carry a scorable outcome (`outcome_classifier.classify_outcome` already treats an absent
    `native.tau_reward` as "not scored", never as a fail, so the 12 cannot be miscounted as
    database failures).
    """
    out = []
    for run_dir, m in _iter_cached_manifests(runs_root):
        if m.get("suite_id") not in ("tau2_retail", "tau2_airline"):
            continue
        if not str(m.get("code_version") or "").startswith(DEGRADED_CODE_VERSION):
            continue
        if m.get("arm_id") != DEGRADED_ARM_ID:
            continue
        if str(m.get("prompt_variant_id") or "") != DEGRADED_PROMPT_VARIANT:
            continue
        pins = m.get("pins") or {}
        if str((pins.get("inquirer") or {}).get("model_id") or "") != DEGRADED_INQUIRER_MODEL:
            continue
        r = load_run(run_dir, m)
        out.append(r)
    return Population("degraded", "degraded (phase4, 4_base_mayask)", out)


def domain_of(suite_id: str) -> str:
    return {"tau2_retail": "retail", "tau2_airline": "airline", "tau2": "banking"}.get(
        suite_id, suite_id
    )


def all_populations(runs_root: Path = RUNS_ROOT) -> list[Population]:
    return [
        published_retail(runs_root),
        published_airline(runs_root),
        banking(runs_root),
        degraded(runs_root),
    ]
