"""The registry must agree with the code it describes, or it is documentation.

Every test here is a drift guard: it fails when a suite is added, retired or moved between
roles and one of the four places that knows about suites is not updated with it. That is the
failure this file exists for -- the registry itself is only declarations, so nothing in it
can be "wrong" at runtime; it can only fall out of step with `worker.load_suite`,
`pinq.splitting.EVAL_ONLY_SUITES` and `pi_eval.prereg`.

`test_a_mutated_registry_is_caught` is here because of rule 2: without it, every other test
in this file would pass against an `audit()` that returned `[]` unconditionally, and a guard
that cannot fail is not a guard.
"""

from __future__ import annotations

import dataclasses

import pytest

from pi_run import suites
from pi_run.worker import CORPUS_BACKED, SELF_SOURCED
from pinq.splitting import EVAL_ONLY_SUITES, split_of

LOADABLE = frozenset(CORPUS_BACKED) | frozenset(SELF_SOURCED)


def _prereg_named() -> frozenset[str]:
    """Every suite the preregistration says may carry a p-value.

    Through `prereg.confirmatory_suites()` rather than by re-deriving it from the endpoint
    tuples here: the ROLE is declared (`SUITE_ROLES`) and the endpoint set is only its
    default, so a local re-derivation would be a second answer to the same question -- which
    is the bug the two-declarations work in `pi_eval.prereg` exists to remove.

    Imported inside the function: `pi_eval` is the gold package, and a module-level import
    would make this test file unimportable in any context where the firewall is armed.
    """
    from pi_eval import prereg

    return prereg.confirmatory_suites()


# ------------------------------------------------------------------ the registry vs the code


def test_registry_audit_is_clean():
    assert suites.audit(LOADABLE) == []


def test_registry_matches_the_preregistration():
    """EXACTLY EMPTY, and the bare `pi suites audit` prints `registry audit: clean`.

    This briefly asserted a three-element disagreement instead, because the registry had moved
    to the trained programme's roles (tau2_retail confirmatory, tau2 and drgym exploratory)
    while `default_stage1()` still derived its roles from the PROMPTED era's endpoint tuples.
    An asserted disagreement is better than an ignored one and worse than none: a reader of a
    red audit cannot tell a stale registry from a deliberate one.

    What closed it is `prereg.SUITE_ROLES` -- the roles DECLARED rather than derived, which is
    what a preregistration does with a role in the first place. The endpoints, primaries,
    margins and seeds are untouched: `default_stage1()`'s tau2 endpoint is still declared and
    still DIRECTIONAL, and what changed is only that the document now says out loud that its
    rows carry no p-value, which its own rationale had already said in prose.
    """
    assert suites.audit_against_prereg(_prereg_named()) == []


def test_every_loadable_suite_has_a_declared_role():
    """The direction that actually happens: an adapter lands, a grid runs it, and no one ever
    writes down whether it is measured or mined."""
    assert LOADABLE <= set(suites.REGISTRY)


def test_no_training_source_is_eval_only():
    """The training-set leak, as an assertion.

    A suite that is both mined and eval-only means either the exporter is about to train on a
    transfer target, or the transfer claim is decoration. Both are silent.
    """
    for spec in suites.REGISTRY.values():
        assert not (spec.mines_training_data and spec.eval_only), spec.suite_id


def test_eval_only_suites_are_all_registered():
    """`EVAL_ONLY_SUITES` is the definition; nothing may be eval-only and unknown here."""
    assert set(EVAL_ONLY_SUITES) <= set(suites.REGISTRY)


def test_split_of_sends_every_eval_only_task_to_test():
    """The property the whole transfer claim rests on, checked through the real function
    rather than by re-reading the constant."""
    for sid in EVAL_ONLY_SUITES:
        assert split_of(sid, "any-task-id") == "test"


# --------------------------------------------------------------------------- the derived views


def test_drgym_is_eval_only_and_the_registry_says_so_in_both_directions():
    """The registry side of the 2026-09-15 decision, which the split module cannot state.

    TWO INDEPENDENT FACTS, and neither implies the other. `eval_only` is derived from
    `EVAL_ONLY_SUITES`, so the first assertion would go on passing with the registry's
    `mines_training_data` flipped back to True -- `audit()` would then report the conflict,
    but only if somebody ran it. The second assertion is what makes the flag itself pinned,
    and the pair together is what `test_no_training_source_is_eval_only` checks generically
    and cannot name.

    The note is asserted for its DATE, not its prose: this suite's role changed by a decision
    rather than by a measurement, and a role change whose reason is not written next to it is
    the thing `docs/SUITES.md` and this registry both exist to prevent.
    """
    spec = suites.REGISTRY["drgym"]
    assert spec.eval_only, "drgym must be in EVAL_ONLY_SUITES"
    assert not spec.mines_training_data
    assert spec.carries_endpoints, "eval-only means eval, not cut"
    assert "drgym" in suites.transfer_suites()
    assert "drgym" not in suites.training_sources()
    assert "2026-09-15" in spec.note, "the decision that moved the role must be dated in it"


def test_tau2_is_a_transfer_suite_and_retail_is_not():
    """The pair that a single `tau2` suite_id would have collapsed. Retail exists to be
    trainable precisely so banking can stay untouched."""
    assert "tau2" in suites.transfer_suites()
    assert "tau2_retail" not in suites.transfer_suites()
    assert "tau2_retail" in suites.training_sources()


def test_the_vertical_claim_has_a_suite_carrying_it():
    """If this ever empties, the paper's central claim has no evidence and the failure should
    be loud rather than a missing row in a table."""
    assert suites.suites_for_axis("vertical")
    assert "musique" in suites.suites_for_axis("vertical")


def test_wiki2_is_load_bearing_despite_carrying_no_claim_of_its_own():
    """wiki2 is in the pool of six declared secondary endpoints. Dropping it from the paper
    is an amendment, not an editorial choice, and this test is what makes that visible to
    whoever tries."""
    assert suites.REGISTRY["wiki2"].carries_endpoints
    assert "mechanism" in suites.REGISTRY["wiki2"].axis
    assert "wiki2" in _prereg_named()


def test_suites_built_after_the_prereg_are_marked_exploratory():
    """convlog and userbench were built after the design was written, so neither may carry a
    p-value until a preregistration names them.

    `userbench` was added to this tuple when it went from `planned` to `live`. It is the case
    the check matters most for: an EXTERNAL benchmark is the most quotable number in the
    paper and therefore the one most likely to be written up as if it had been declared.

    `tau2_retail` LEFT this tuple 2026-09-15 (T20) and the departure is the point: the
    stage-1 draft declares its fork pairs a confirmatory family, which is precisely "the
    preregistration is amended" -- by a NEW stage file rather than by an edit to the old one.
    That is the only way a suite may leave this list, so the rule is unchanged and the
    membership is not.
    """
    for sid in ("convlog", "userbench"):
        assert suites.REGISTRY[sid].endpoint_status == "exploratory", sid
        assert sid not in _prereg_named(), sid
    assert suites.REGISTRY["tau2_retail"].endpoint_status == "confirmatory"
    assert "tau2_retail" in _prereg_named(), (
        "it left this list by being DECLARED confirmatory, which is what amending means"
    )


# ------------------------------------------------------------------------ rule 2: it can fail


@pytest.mark.parametrize(
    "field,value,expect",
    [
        ("mines_training_data", True, "EVAL_ONLY_SUITES"),
        ("status", "cut", "still declares a role"),
    ],
)
def test_a_mutated_registry_is_caught(monkeypatch, field, value, expect):
    """Prove the guard bites. Mutating one field of one suite must produce a named problem.

    `tau2` is the subject because it is the transfer target: making it a training source is
    the exact mistake that voids the strongest claim in the paper, and it would leave no
    other trace.
    """
    broken = dict(suites.REGISTRY)
    broken["tau2"] = dataclasses.replace(broken["tau2"], **{field: value})
    monkeypatch.setattr(suites, "REGISTRY", broken)

    problems = suites.audit(LOADABLE)
    assert any(expect in p for p in problems), problems


# ------------------------------------------------------------------- documented gold gaps


def test_every_gold_gap_is_declared_by_name_and_nowhere_else():
    """A gap is declared by NAME, never as a count.

    RE-PINNED when tau2_airline was registered. The old name was
    `test_only_retail_declares_a_gold_gap...`, and "only retail" was a fact about the day it
    was written rather than about the design. Airline has seven, and they are the same KIND of
    gap: upstream ships `"actions": []` for tasks whose correct outcome is a database that does
    not change, this builder's nodes are the argument values gold actions required, so zero
    actions means zero nodes.

        tau2_retail   24, 57      -- 24's customer keeps the grill, 57's item already shipped
        tau2_airline  0, 10, 26, 28, 31, 34, 46

    Both rows MEASURED off upstream's own tasks.json on this checkout: the ids whose
    `evaluation_criteria.actions` is empty are exactly ('24', '57') of retail's 114 and
    ('0', '10', '26', '28', '31', '34', '46') of airline's 50. `pi verify tau2 --replay-gold
    --suite tau2_retail` skips the same two and reproduces 112 of 112.

    What the test still enforces is the property that matters, and it enforces it more tightly
    than the version it replaces: the gaps are ENUMERATED for EVERY suite, so a suite that
    grows a new one -- or silently loses a declared one -- fails here until someone writes down
    which task and why. A count would let the next gap hide behind these, and `validate` would
    keep printing the same reassuring number while gold quietly stopped covering a task that
    matters.
    """
    expected = {
        "tau2_retail": ("24", "57"),
        "tau2_airline": ("0", "10", "26", "28", "31", "34", "46"),
    }
    declared = {sid: s.gold_gap_tasks for sid, s in suites.REGISTRY.items() if s.gold_gap_tasks}
    assert declared == expected, (
        f"a gold gap was added or removed without updating this test's reasoning: {declared}"
    )
