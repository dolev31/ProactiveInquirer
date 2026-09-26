"""Gold and adapter must mint the SAME evidence uid, independently.

`EvidenceUnit.uid` is `(corpus_id, doc_id, span)` and nothing else. The gold builder computes
it on the gold side and the adapter computes it on the rollout side, and the two never
communicate -- that separation is what keeps the firewall intact. It also means a disagreement
about the span convention is invisible: no error, no missing file, just `evidence_coverage`
collapsing to 0 for a plumbing reason that is indistinguishable, in a table, from a policy that
found nothing.

The check is cheap and has no model in it, which is the point: it belongs before any spend.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest

# Measured 2026-08-24 on the built corpora. Not a target -- a floor, so a regression that
# silently drops the span convention fails here rather than in a table.
FLOOR = {"musique": 0.99, "strategyqa": 0.95, "wiki2": 0.98}
K = {"musique": 5, "strategyqa": 2, "wiki2": 3}


def precondition(n_corpora: int, gold_exists: bool, suite: str) -> tuple[str, str]:
    """Decide skip vs fail vs ok, because those were one branch and must not be.

    The original condition was `len(corpora) != 1 or not gold.exists(): skip`, which treats
    NOTHING BUILT and AMBIGUOUS identically. They are opposites. Nothing built is a fact about
    the checkout -- a fresh clone has no corpora and this test has no business failing there.
    Two or more built corpora is a configuration error, the same one `_resolve_corpus` refuses
    to guess at because two corpus hashes are two different frozen corpora, and skipping it
    leaves the FLOOR below UNGUARDED while the suite still reports green. A regression that
    drops the span convention would then fail in a table instead of here, which is the exact
    outcome this file's docstring exists to prevent.

    That state is reachable and not hypothetical: data/corpora/synth has held two corpora since
    2026-09-19. synth is not in FLOOR, so this does not fire today -- which is precisely why it
    needed fixing before it mattered rather than after.
    """
    if n_corpora > 1:
        return "fail", (
            f"{suite}: {n_corpora} built corpora, so the uid floor cannot be checked and would "
            "be silently unguarded. Two corpus hashes are two different frozen corpora: remove "
            "or relocate the stale one rather than letting this test skip."
        )
    if n_corpora == 0 or not gold_exists:
        return "skip", f"{suite}: expected one built corpus and its gold"
    return "ok", ""


def _one(suite: str, n: int = 60):
    from pi_eval.build.common import read_graphs
    from pi_run.worker import load_suite

    corpora = sorted(Path(f"data/corpora/{suite}").glob("*/tasks.jsonl"))
    gold = Path(f"data/gold/graphs/{suite}/v1.jsonl")
    verdict, msg = precondition(len(corpora), gold.exists(), suite)
    if verdict == "fail":
        pytest.fail(msg)
    if verdict == "skip":
        pytest.skip(msg)

    adapter = load_suite(suite, str(corpora[0].parent))
    graphs = {g.gold_task_key: g for g in read_graphs(gold)}
    present, retrieved = [], []
    for tid in [t for t in list(adapter.task_ids())[:n] if t in graphs]:
        g = graphs[tid]
        want = {
            u
            for node in g.gold_nodes
            if node.gold_partition == "required"
            for u in node.gold_ev_uids
        }
        if not want:
            continue
        r = adapter.retriever(tid)
        question = adapter.view(tid).question
        # A deliberately enormous k: this asks whether the uid EXISTS in the corpus the
        # adapter reads, which is the plumbing question, separate from whether retrieval
        # happens to rank it highly.
        universe = {u.uid for u in r.search(question, 10_000)}
        present.append(len(want & universe) / len(want))
        got = {u.uid for u in r.search(question, K[suite])}
        retrieved.append(len(want & got) / len(want))
    if not present:
        pytest.skip(f"{suite}: no task carries required gold evidence uids")
    return statistics.mean(present), statistics.mean(retrieved)


@pytest.mark.integration
@pytest.mark.parametrize("suite", sorted(FLOOR))
def test_gold_evidence_uids_exist_in_the_corpus_the_adapter_reads(suite):
    """THE CATASTROPHIC CASE THIS PREVENTS: the two sides disagree on the span convention, every
    match misses, and evidence_coverage is 0 for every arm on every task. Nothing errors."""
    present, _ = _one(suite)
    assert present >= FLOOR[suite], (
        f"{suite}: only {present:.3f} of required gold uids exist in the adapter's corpus. "
        "Gold and adapter have diverged on (corpus_id, doc_id, span)."
    )


@pytest.mark.integration
@pytest.mark.parametrize("suite", sorted(FLOOR))
def test_the_full_question_leaves_headroom_for_inquiry(suite):
    """The complement of the check above, and the reason the paper has a subject. If the task's
    own question already retrieved everything, there would be nothing for a sub-question to
    add. Measured at each suite's pinned k: musique 0.499, strategyqa 0.725, wiki2 0.673."""
    present, retrieved = _one(suite)
    assert retrieved < present, (
        f"{suite}: the full question already retrieves {retrieved:.3f} of the gold evidence "
        f"that exists ({present:.3f}); there is no headroom for inquiry to exploit"
    )
    assert retrieved > 0.1, (
        f"{suite}: the full question retrieves almost nothing ({retrieved:.3f}), which is a "
        "retrieval failure rather than headroom"
    )


def test_ambiguity_fails_rather_than_skipping_a_floor() -> None:
    """The three conditions were one branch, and one of them un-guarded a floor.

    A peer found this while checking whether the two synth corpora could be pooled anywhere. They
    cannot, but this test SKIPS on exactly the state that directory has been in since
    2026-09-19 -- and a skip on a floor-guarding test is green. Same shape as a gate guarded by a
    precondition that is never met.
    """
    assert precondition(2, True, "musique")[0] == "fail"
    assert precondition(3, True, "wiki2")[0] == "fail"
    # Nothing built is a fact about the checkout, not a defect: a fresh clone must still pass.
    assert precondition(0, False, "musique")[0] == "skip"
    assert precondition(1, False, "musique")[0] == "skip"
    assert precondition(1, True, "musique")[0] == "ok"
