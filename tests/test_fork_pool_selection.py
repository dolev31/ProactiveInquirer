"""`conf/forks/tau2_retail_dev.selected34.json` must be a pure function of the pool file beside
it (plus the recorded TEST benchmark's own k-shape), or it stops being a rule and becomes a
list someone typed once.

WHY 34, AND WHY K-STRATIFIED. The retail dev fork pool (`tau2_retail_dev.pool.json`, 500
legal-and-informative points across 18 of 19 dev tasks) is ~14.7x the size of the recorded
34-point retail TEST fork benchmark (`docs/reports/forks_tau2_retail_test.json`, run_ids_sha
1a060380806fff3e). 34 mirrors that benchmark's size and therefore its statistical power, at
2 arms x 3 seeds x 34 = 204 units.

An EARLIER version of this file sorted the pool by (task_id, trace_sha, k) and round-robinned
by task, which always drew each task's shallowest available point first: sample k in
{2: 18, 4: 16}, median 2, mean 2.94, against the pool's own median 8 / mean 10.67 and the TEST
set's median 12.5 / mean 12.32 (measured straight from runs/*/manifest.json, since
docs/reports/forks_tau2_retail_test.json does not carry k). That sample tested a checkpoint
only on prefixes where almost all the work remained -- not what the test set itself looks like,
which defeats the point of a set meant to predict test behaviour. Task-diversity round-robin is
still part of the rule, but only as the tie-break WITHIN a k stratum, after the strata
themselves are sized to match the test set's k histogram.

THE RULE, restated here so a drift between prose and code fails a test rather than a reader's
attention span (also stated in `selection_rule` in the sample file and in docs/HPC_FORKS.md):

  1. Take the recorded TEST benchmark's 34 k values (frozen in this file's own
     `k_histograms.test_benchmark_34_points`, and re-measured live against runs/*/manifest.json
     where that tree is available -- see `test_the_frozen_test_histogram_still_matches_runs`
     below) as 34 ascending targets.
  2. Draw one pool point per target, in ascending order. For each target k, use the pool's
     exact k if any not-yet-drawn pool point has it; otherwise the available k closest to the
     target (ties toward the smaller k) -- the only such case today is target k=34, which has
     no pool point (the pool's own max is 31) and resolves to k=31.
  3. Within a stratum (every draw that resolves to the same k), break ties by task-diversity:
     prefer the task_id that has contributed the FEWEST points to the sample so far, summed
     across every stratum drawn before it (not reset per stratum), then ascending int(task_id),
     then ascending trace_sha.

No randomness. Deterministic given the pool file and the 34 target k values.
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

import pytest

POOL = Path("conf/forks/tau2_retail_dev.pool.json")
SAMPLE = Path("conf/forks/tau2_retail_dev.selected34.json")


def nearest_available_k(target: int, available_ks: set[int]) -> int:
    """The available k closest to `target`; ties toward the smaller k."""
    return min(available_ks, key=lambda k: (abs(k - target), k))


def select_stratified(pool_points: list[dict], targets: list[int]) -> list[dict]:
    """THE RULE, executably. Must stay in step with the module docstring, with
    `selection_rule` in the sample file, and with docs/HPC_FORKS.md -- this is the one place a
    change to any of those three would be caught."""
    remaining = [dict(p) for p in pool_points]
    usage: dict[str, int] = collections.defaultdict(int)
    out: list[dict] = []

    for target_k in targets:
        available_ks = {int(p["k"]) for p in remaining}
        k_star = (
            target_k if target_k in available_ks else nearest_available_k(target_k, available_ks)
        )
        candidates = [p for p in remaining if int(p["k"]) == k_star]
        candidates.sort(key=lambda p: (usage[str(p["task_id"])], int(p["task_id"]), p["trace_sha"]))
        chosen = candidates[0]
        out.append(
            {"task_id": chosen["task_id"], "trace_sha": chosen["trace_sha"], "k": chosen["k"]}
        )
        usage[str(chosen["task_id"])] += 1
        remaining.remove(chosen)
    return out


def _targets_from_histogram(hist: dict[str, int]) -> list[int]:
    """Expand a {k: count} histogram back into the sorted-ascending target list. A histogram
    plus 'process ascending' fully determines the list `select_stratified` needs -- no ordering
    information is lost by storing counts instead of the raw 34-long list."""
    targets: list[int] = []
    for k_str, count in hist.items():
        targets.extend([int(k_str)] * count)
    return sorted(targets)


def _pool() -> dict:
    return json.loads(POOL.read_text())


def _sample() -> dict:
    return json.loads(SAMPLE.read_text())


def test_the_pool_has_the_shape_its_own_metadata_claims():
    """Catches a hand-edit that changes the list but not the counts beside it, or vice versa."""
    pool = _pool()
    pts = pool["fork_points"]
    assert len(pts) == pool["counts"]["n_fork_points"] == 500
    assert {p["task_id"] for p in pts} == set(
        pool["counts"]["dev_task_ids_with_a_recorded_as_star_success"]
    )
    assert len(pool["counts"]["dev_task_ids_with_a_recorded_as_star_success"]) == 18
    assert pool["counts"]["split_census"] == {"train": 69, "dev": 19, "test": 26}
    assert "20" in pool["counts"]["dev_task_ids_with_zero_recorded_as_star_success"]
    # every point is legible enough to sort and replay: non-empty task_id/trace_sha, integer k
    for p in pts:
        assert p["task_id"] and p["trace_sha"] and isinstance(p["k"], int) and p["k"] >= 0
    # (task_id, trace_sha, k) is the unit's identity; no duplicate fork point twice in the pool
    triples = {(p["task_id"], p["trace_sha"], p["k"]) for p in pts}
    assert len(triples) == len(pts)


def test_the_sample_is_exactly_34_points_drawn_from_the_pool():
    pool_triples = {(p["task_id"], p["trace_sha"], p["k"]) for p in _pool()["fork_points"]}
    sample = _sample()
    pts = sample["fork_points"]
    assert sample["n_points"] == len(pts) == 34
    for p in pts:
        assert (p["task_id"], p["trace_sha"], p["k"]) in pool_triples
    # the pool has no duplicate triples (checked above), so 34 distinct draws is 34 distinct
    # triples -- confirms the sample never lists one point twice.
    assert len({(p["task_id"], p["trace_sha"], p["k"]) for p in pts}) == 34


def test_the_sample_k_histogram_matches_the_test_benchmarks_except_the_one_point_out_of_range():
    """THE PROPERTY THE FIX EXISTS FOR. Every stratum the pool could match exactly, it does;
    only k=34 (the pool's max is 31) is substituted, and it is exactly one point."""
    sample = _sample()
    hist = sample["k_histograms"]
    test_hist = {int(k): c for k, c in hist["test_benchmark_34_points"].items()}
    sample_hist = {int(k): c for k, c in hist["this_sample_34_points"].items()}
    assert sum(test_hist.values()) == sum(sample_hist.values()) == 34

    diff_ks = set(test_hist) ^ set(sample_hist)
    shared_ks = set(test_hist) & set(sample_hist)
    mismatched_shared = {k for k in shared_ks if test_hist[k] != sample_hist[k]}
    assert not mismatched_shared, f"counts differ at shared k values: {mismatched_shared}"
    assert diff_ks == {34, 31}, f"expected only the k=34->31 substitution to differ, got {diff_ks}"
    assert test_hist[34] == 1 and sample_hist[31] == 1

    # medians must match exactly; means differ only by the one substituted point's distance
    targets = _targets_from_histogram(hist["test_benchmark_34_points"])
    sample_ks = _targets_from_histogram(hist["this_sample_34_points"])
    assert statistics.median(targets) == statistics.median(sample_ks) == hist["test_median"]
    assert abs(statistics.mean(targets) - statistics.mean(sample_ks)) == pytest.approx(3 / 34)


def test_every_task_in_the_sample_appears_and_none_is_starved_or_overused():
    per_task: dict[str, int] = {}
    for p in _sample()["fork_points"]:
        per_task[p["task_id"]] = per_task.get(p["task_id"], 0) + 1
    assert sum(per_task.values()) == 34
    assert len(per_task) == 18, "every one of the 18 pointed-to dev tasks must appear"
    assert set(per_task.values()) <= {1, 2, 3}, (
        "k-stratification can unbalance counts versus a pure round-robin, but a task taking "
        "more than 3 of 34 points would mean the diversity tie-break stopped doing its job"
    )


def test_the_sample_reproduces_byte_for_byte_from_the_pool_and_the_frozen_target():
    """THE TEST THE FILE PAIR EXISTS FOR. `select_stratified` is re-run here, independently of
    whatever process wrote `tau2_retail_dev.selected34.json`, against nothing but
    `tau2_retail_dev.pool.json`'s own `fork_points` list and this file's own frozen
    `k_histograms.test_benchmark_34_points` (the two inputs `selection_rule` names). If a
    future edit touches the pool, the sample, or the rule and they drift apart, this catches
    it."""
    sample_doc = _sample()
    targets = _targets_from_histogram(sample_doc["k_histograms"]["test_benchmark_34_points"])
    assert len(targets) == 34

    recomputed = select_stratified(_pool()["fork_points"], targets)
    stored = sample_doc["fork_points"]
    assert recomputed == stored, (
        "the stored 34-point sample no longer matches what the documented k-stratified rule "
        "derives from the pool file and the frozen target histogram -- regenerate it or fix "
        "whichever of the two drifted"
    )

    # Not just structurally equal: rebuild the whole document (every other field taken
    # verbatim from the file, only `fork_points` replaced by the recomputation) and
    # re-serialise it with the exact parameters it was written with. That reproduces the
    # file's bytes exactly if and only if the stored list truly IS this rule's output.
    doc = _sample()
    doc["fork_points"] = recomputed
    rendered = json.dumps(doc, indent=2) + "\n"
    assert rendered == SAMPLE.read_text(), "the sample file is not byte-identical to a fresh render"


def test_the_sample_starts_with_the_lowest_numbered_task_first():
    """Ties the stratified rule's first stratum (k=2, the test set's own smallest and most
    populous) to something a reader can check without re-deriving the whole list: within that
    stratum every task_id starts at zero usage, so the tie-break falls through to ascending
    int(task_id) then trace_sha."""
    pts = _sample()["fork_points"]
    assert pts[0]["task_id"] == "0" and pts[0]["k"] == 2
    assert pts[1]["task_id"] == "1" and pts[1]["k"] == 2
    assert pts[2]["task_id"] == "6" and pts[2]["k"] == 2


def test_the_frozen_test_histogram_still_matches_runs():
    """Provenance check, not the core rule: the 34 target k values frozen in
    `k_histograms.test_benchmark_34_points` should still be exactly what
    `runs/*/manifest.json` gives for the recorded tau2_retail TEST fork benchmark today.
    `runs/` is a large, shared, ever-changing tree (see CONTRIBUTING.md's runs-dir-is-shared note),
    not something a committed test should hard-depend on, so this skips rather than fails when
    it is not the expected shape here -- the byte-for-byte reproduction test above is the one
    that always runs and always must pass."""
    root = Path("runs")
    if not root.is_dir() or not any(root.glob("*/manifest.json")):
        pytest.skip("runs/ is not present in this checkout")

    sys.path.insert(0, "scripts")
    import run_tau2_forks as launcher

    pts = launcher.points_from_runs(
        root, suite="tau2_retail", split="test", arms=["inquirer_prompted", "self_ask"]
    )
    if len(pts) != 34:
        pytest.skip(
            f"runs/ here holds {len(pts)} tau2_retail test fork points, not the 34 this pins"
        )

    live_hist = collections.Counter(int(p["k"]) for p in pts)
    frozen_hist = {
        int(k): c for k, c in _sample()["k_histograms"]["test_benchmark_34_points"].items()
    }
    assert dict(live_hist) == frozen_hist, (
        "the recorded TEST benchmark's k histogram has changed since the dev sample was "
        "derived from it -- re-run the derivation and update both conf/forks files"
    )
