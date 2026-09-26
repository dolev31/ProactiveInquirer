"""scripts/tau2_failure_taxonomy/populations.py's SELECTION rule, on a tiny fixture tree.

The real populations (204/204/291/204 runs) are read from the shared `runs/` root and are
re-verified against the published `run_ids_sha` digests directly in
`artifacts/tau2_failure_taxonomy_20260919/RESULT.md` ss0 -- that check needs the actual shared
checkout and is not something a hermetic test suite should depend on. What belongs here is the
FILTER itself: given a handful of manifests that each violate exactly one clause of the
selection rule, does `_fork_population` keep only the one that violates none of them. A rule
that quietly drops the `split` check, or starts accepting non-fork runs, would still "work" on
the real 204-run population today (every published run happens to satisfy every clause) and
only show up the next time someone adds an adjacent campaign to `runs/` -- exactly the
"`arm_id` alone is poisoned" failure mode `artifacts/banking/RESULT.md` documents.
"""

from __future__ import annotations

import json

from scripts.tau2_failure_taxonomy import populations as P


def _write_run(root, run_id, *, manifest_over=None, status_over=None):
    d = root / run_id
    d.mkdir()
    manifest = {
        "run_id": run_id,
        "suite_id": "tau2_retail",
        "split": "test",
        "arm_id": "inquirer_prompted",
        "code_version": "828a720aaaa",
        "foreign_trace_sha": "trace1",
        "foreign_prefix_k": 3,
        "task_id": "22",
        "seed": 0,
    }
    manifest.update(manifest_over or {})
    (d / "manifest.json").write_text(json.dumps(manifest))
    status = {"status": "ok", "n_user_turns": 5, "n_prefix_user_turns": 2}
    status.update(status_over or {})
    (d / "status.json").write_text(json.dumps(status))


def test_fork_population_keeps_only_the_run_matching_every_clause(tmp_path):
    _write_run(tmp_path, "keep_me")
    _write_run(tmp_path, "wrong_arm", manifest_over={"arm_id": "drafter_only"})
    _write_run(tmp_path, "not_a_fork", manifest_over={"foreign_trace_sha": None})
    _write_run(tmp_path, "wrong_split", manifest_over={"split": "train"})
    _write_run(tmp_path, "wrong_code_version", manifest_over={"code_version": "ffffffff"})
    _write_run(tmp_path, "errored", status_over={"status": "error"})

    runs = P._fork_population(
        "tau2_retail", P.RETAIL_CODE_VERSIONS, P.RETAIL_ARMS, runs_root=tmp_path
    )

    assert [r.run_id for r in runs] == ["keep_me"]


def test_fork_population_accepts_either_pinned_code_version():
    # RETAIL_CODE_VERSIONS names two commits (the launch commit and a mid-campaign fix); the
    # selection is an OR over the set, not just the first entry.
    assert len(P.RETAIL_CODE_VERSIONS) == 2


def test_run_ids_sha_is_order_independent():
    assert (
        P._run_ids_sha(["b", "a", "c"])
        == P._run_ids_sha(["a", "b", "c"])
        == P._run_ids_sha(["c", "b", "a"])
    )


def test_run_ids_sha_matches_the_published_digest_format():
    # `paper/appendix_instruments.tex`'s digests are 16 lowercase hex characters
    # (sha256(...).hexdigest()[:16]); a change to the truncation length would silently break
    # every "sha_ok" comparison this lane makes against the paper without failing loudly.
    digest = P._run_ids_sha(["x"])
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)
