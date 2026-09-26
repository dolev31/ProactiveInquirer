"""`pi cache verify` must fail on a NEW divergence, not on closed history.

MEASURED: 41 diverged calls across 33 runs. Every one predates 95d6524, the commit that
made a losing racer adopt the canonical response. They are unrepairable by construction:

  - the recorded response_sha is what that run actually received, and rewriting it would
    falsify provenance;
  - re-rolling cannot overwrite them, because `code_version` is itself in SEMANTIC_FIELDS, so
    a re-run computes a DIFFERENT run_id and lands in a new directory;
  - materialising those re-rolls anyway would put two admissible runs in each P3 grid cell
    and corrupt the aggregate.

So the honest options are to delete real experimental records, or to record the closed set
and fail on anything outside it. This is the second. The baseline is version-controlled and
reviewable, and the check reports how many of its entries it actually used -- a baseline
that silently covers a live bug is the failure mode this has to avoid, so a shrinking one
is visible and an unused entry is named.
"""

from __future__ import annotations

import json

from pi_run.cli import partition_divergences


def _d(run_id: str, request_sha: str) -> dict:
    return {"run_id": run_id, "request_sha": request_sha, "run_recorded": "a", "cache_holds": "b"}


BASELINE = {
    "known": [{"run_id": "r1", "request_sha": "sha1"}, {"run_id": "r2", "request_sha": "sha2"}]
}


def test_a_baselined_divergence_does_not_fail() -> None:
    new, known, unused = partition_divergences([_d("r1", "sha1")], BASELINE)
    assert new == [] and len(known) == 1


def test_a_new_divergence_fails_even_with_a_baseline() -> None:
    """The whole point. A baseline must not become a blanket."""
    new, known, unused = partition_divergences([_d("r9", "sha9")], BASELINE)
    assert len(new) == 1 and new[0]["run_id"] == "r9"


def test_the_same_run_with_a_different_request_is_new() -> None:
    """Keyed on (run_id, request_sha), not run_id: one bad call must not excuse the rest."""
    new, _, _ = partition_divergences([_d("r1", "OTHER")], BASELINE)
    assert len(new) == 1


def test_unused_baseline_entries_are_reported() -> None:
    """A baseline that has stopped matching is stale and should shrink, visibly."""
    new, known, unused = partition_divergences([_d("r1", "sha1")], BASELINE)
    assert unused == [{"run_id": "r2", "request_sha": "sha2"}]


def test_no_baseline_means_everything_is_new() -> None:
    """Absent baseline must not silently pass anything."""
    new, known, unused = partition_divergences([_d("r1", "sha1")], None)
    assert len(new) == 1 and known == [] and unused == []


def test_an_empty_baseline_is_not_a_wildcard() -> None:
    new, known, unused = partition_divergences([_d("r1", "sha1")], {"known": []})
    assert len(new) == 1


def test_the_shipped_baseline_parses_and_is_documented() -> None:
    """The file itself is part of the claim, so it must exist and say why."""
    from pathlib import Path

    p = Path("docs/known_cache_divergences.json")
    assert p.exists(), "the baseline the Makefile/gate points at must be in the repo"
    raw = json.loads(p.read_text())
    assert raw.get("why"), "a baseline without a stated reason is a silenced check"
    assert isinstance(raw.get("known"), list) and raw["known"], "baseline is empty"
    for e in raw["known"]:
        assert set(e) >= {"run_id", "request_sha"}
