"""Which suites carry a p-value, and which document decides that.

`pi_run.suites.REGISTRY` declares an `endpoint_status` per suite and `audit_against_prereg`
checks it against the preregistration. Until now the only preregistration it could check
against was `pi_eval.prereg`'s module-level PRIMARY / SECONDARY / CALIBRATION -- the PROMPTED
era's programme. The trained programme seals its own stage file with different roles, so the
registry and the audit need to be able to name WHICH document they are agreeing with.

THE THREE DISAGREEMENTS, from the stage-1 draft:

  tau2_retail   registry `exploratory`, draft CONFIRMATORY. Its fork follow-ups paired with
                tau_reward is the draft's second confirmatory family (34 fork points, 102
                pairs per domain, an exact two-sided sign test).
  tau2          registry `confirmatory`, draft EXPLORATORY. Banking's success sits at the
                floor, it has zero legal fork points, and its follow-ups are a transfer row.
  drgym         registry `confirmatory`, draft EXPLORATORY/dropped, per decisions D2 and D8
                and its eval-only status.

Each is a claim about what may carry a p-value, so getting one wrong is a table that either
over-claims or silently under-claims.
"""

from __future__ import annotations

import json

import pytest

from pi_eval.prereg import confirmatory_suites, default_stage1
from pi_run.suites import REGISTRY, audit_against_prereg

# The draft's roles, as the set of suites its endpoints name.
DRAFT_CONFIRMATORY = frozenset({"musique", "strategyqa", "wiki2", "tau2_retail", "synth"})


def test_the_registry_agrees_with_the_drafts_roles():
    """THE CHECK THE COORDINATOR ASKED FOR. Against a stage 1 carrying the draft's roles,
    the registry must have nothing to report."""
    assert audit_against_prereg(DRAFT_CONFIRMATORY) == []


def test_the_three_roles_are_what_the_draft_says():
    assert REGISTRY["tau2_retail"].endpoint_status == "confirmatory"
    assert REGISTRY["tau2"].endpoint_status == "exploratory"
    assert REGISTRY["drgym"].endpoint_status == "exploratory"


def test_each_flipped_note_says_when_and_why():
    """A role that changes without a dated reason is a role that changed by accident."""
    for sid in ("tau2_retail", "tau2", "drgym"):
        note = REGISTRY[sid].note
        assert "2026-09-15" in note, sid
        assert "ROLE" in note, sid


def test_the_audit_reports_a_registry_that_over_promises():
    """The direction that matters most: a suite declared confirmatory that the governing
    document does not name is a promise of a p-value nobody authorised."""
    problems = audit_against_prereg(DRAFT_CONFIRMATORY - {"tau2_retail"})
    assert len(problems) == 1
    assert "tau2_retail" in problems[0] and "may not carry a p-value" in problems[0]


def test_the_audit_reports_a_registry_that_under_claims():
    """And the other direction: a suite the document DOES name while the registry calls it
    exploratory means either the registry is stale or a suite entered a declared pool after
    the design was written."""
    problems = audit_against_prereg(DRAFT_CONFIRMATORY | {"tau2"})
    assert len(problems) == 1
    assert "tau2" in problems[0] and "DOES name it" in problems[0]


# ------------------------------------------------- naming the document the audit checks against


def test_confirmatory_suites_reads_a_sealed_payload(tmp_path):
    """A stage 1 on disk is a mapping of plain dicts, not `Endpoint` objects, and the audit
    must be able to read one: the trained programme's roles live in its stage file and
    nowhere else."""
    payload = {
        "primary": [{"suite_id": "tau2_retail", "metric": "n_user_followups", "pool_suites": []}],
        "secondary": [
            {"suite_id": "*", "metric": "evidence_coverage", "pool_suites": ["musique", "wiki2"]}
        ],
        "calibration": [{"suite_id": "synth", "metric": "evidence_coverage", "pool_suites": []}],
    }
    assert confirmatory_suites(payload) == frozenset(
        {"tau2_retail", "musique", "wiki2", "synth"}
    ), "the POOLED wildcard must resolve to its pool and never to the literal '*'"

    p = tmp_path / "stage1.json"
    p.write_text(json.dumps(payload))
    assert confirmatory_suites(json.loads(p.read_text())) == confirmatory_suites(payload)


def test_confirmatory_suites_defaults_to_the_modules_own_declarations():
    """With no document named, the answer is the programme in code -- which is the prompted
    era's, and is exactly why the audit can now be pointed at another one."""
    assert confirmatory_suites() == confirmatory_suites(default_stage1())
    assert "*" not in confirmatory_suites()
    assert {"musique", "strategyqa", "wiki2"} <= confirmatory_suites()


def test_the_audit_command_can_be_pointed_at_a_stage_file(tmp_path, capsys):
    """`pi suites audit --stage1 <path>`: the registry now describes the TRAINED programme,
    and the module's endpoints are the PROMPTED one, so the command must be able to say which
    it is checking."""
    import argparse

    from pi_run.cmd_suites import cmd_suites_audit

    payload = {
        "primary": [{"suite_id": s, "metric": "m", "pool_suites": []} for s in ("musique",)],
        "secondary": [
            {
                "suite_id": "*",
                "metric": "m",
                "pool_suites": ["strategyqa", "wiki2", "tau2_retail"],
            }
        ],
        "calibration": [{"suite_id": "synth", "metric": "m", "pool_suites": []}],
    }
    p = tmp_path / "stage1.json"
    p.write_text(json.dumps(payload))

    code = cmd_suites_audit(argparse.Namespace(root=None, stage1=str(p)))
    out = capsys.readouterr().out
    assert "registry audit: clean" in out, out
    assert code == 0


def test_a_named_stage_file_that_does_not_exist_is_an_error_not_a_fallback(tmp_path, capsys):
    """Falling back to the module's declarations after being asked for a document would check
    a different programme and print `clean` about it."""
    import argparse

    from pi_run.cmd_suites import cmd_suites_audit

    code = cmd_suites_audit(argparse.Namespace(root=None, stage1=str(tmp_path / "nope.json")))
    assert code != 0
    assert "nope.json" in capsys.readouterr().out


def test_a_suite_may_never_be_both_a_training_source_and_eval_only():
    """Unrelated to the flip and asserted here because tau2_retail's role just moved: it
    MINES training data, so it must not be eval-only, and the fork endpoint is measured on
    its test split."""
    s = REGISTRY["tau2_retail"]
    assert s.mines_training_data and not s.eval_only
    assert s.carries_endpoints


@pytest.mark.parametrize("sid", ["tau2", "drgym"])
def test_a_demoted_suite_still_carries_endpoints(sid):
    """Exploratory is not "not reported". These suites still produce rows; what changes is
    that the rows carry a point estimate and a CI and no p-value."""
    assert REGISTRY[sid].carries_endpoints


# ------------------------------------------------ the roles, DECLARED rather than derived


def test_the_default_stage_ones_roles_are_the_drafts_roles():
    """The bare `pi suites audit` must be clean on this branch, and that is a statement about
    `default_stage1()`, not about a file somebody remembered to pass."""
    from pi_eval.prereg import confirmatory_suites, default_stage1

    assert confirmatory_suites(default_stage1()) == DRAFT_CONFIRMATORY
    assert confirmatory_suites() == DRAFT_CONFIRMATORY


def test_the_roles_are_declared_and_every_endpoint_suite_has_one():
    """A suite an endpoint names and `SUITE_ROLES` does not is a suite whose rows have no
    declared status -- which is exactly the hole the whole role field exists to close."""
    from pi_eval.prereg import SUITE_ROLES, default_stage1, endpoint_suites

    s1 = default_stage1()
    missing = endpoint_suites(s1) - set(SUITE_ROLES)
    assert missing == set(), f"endpoint suites with no declared role: {sorted(missing)}"
    assert set(SUITE_ROLES.values()) <= {"confirmatory", "exploratory", "calibration"}


def test_the_endpoints_themselves_did_not_move():
    """THE CONSTRAINT ON THIS CHANGE, asserted. Only the ROLE fields moved: the endpoint
    tuples, the primaries, the margins and the seeds are the prompted programme's still, and
    the trained programme seals its own stage file rather than editing this one."""
    from pi_eval.prereg import KILL_SWITCH_MARGINS, default_stage1, endpoint_suites

    s1 = default_stage1()
    # tau2 still HAS a primary endpoint; what changed is that its role says no p-value.
    assert "tau2" in endpoint_suites(s1)
    assert any(e.suite_id == "tau2" for e in s1.primary)
    assert next(e for e in s1.primary if e.suite_id == "tau2").claim == "directional"
    # drgym is still in a declared pool.
    assert "drgym" in endpoint_suites(s1)
    # and the numbers nobody was allowed to touch
    assert KILL_SWITCH_MARGINS["verbosity"] == 0.03
    assert all(m == 0.05 for k, m in KILL_SWITCH_MARGINS.items() if k != "verbosity")
    assert s1.ci_resamples == 1000


def test_a_confirmatory_role_with_no_endpoint_in_this_document_is_named():
    """tau2_retail is the one, and it is asserted rather than tolerated: its fork-pair
    endpoints are sealed in the TRAINED stage file, not here. A second such suite appearing
    silently would be a p-value promised by nothing."""
    from pi_eval.prereg import SUITE_ROLES, default_stage1, endpoint_suites

    s1 = default_stage1()
    declared = {s for s, role in SUITE_ROLES.items() if role == "confirmatory"}
    assert declared - endpoint_suites(s1) == {"tau2_retail"}


def test_a_sealed_payload_without_roles_still_reads(tmp_path):
    """BACKWARD-READABLE. `artifacts/prereg_draft/stage1.draft.json` carries no `suite_roles`
    key, and a checker that could not read a document written before the field existed would
    be unable to audit the very document this field was added to agree with."""
    import json as _json

    from pi_eval.prereg import confirmatory_suites

    payload = {
        "primary": [{"suite_id": "musique", "metric": "m", "pool_suites": []}],
        "secondary": [{"suite_id": "*", "metric": "m", "pool_suites": ["wiki2"]}],
        "calibration": [],
    }
    assert confirmatory_suites(payload) == frozenset({"musique", "wiki2"})
    p = tmp_path / "stage1.json"
    p.write_text(_json.dumps(payload))
    assert confirmatory_suites(_json.loads(p.read_text())) == frozenset({"musique", "wiki2"})


def test_a_sealed_payload_with_roles_uses_them(tmp_path):
    """And when the document DOES declare roles, they win over what its endpoints imply --
    otherwise the field would be decoration."""
    from pi_eval.prereg import confirmatory_suites

    payload = {
        "primary": [{"suite_id": "tau2", "metric": "m", "pool_suites": []}],
        "secondary": [],
        "calibration": [],
        "suite_roles": {"tau2": "exploratory", "tau2_retail": "confirmatory"},
    }
    assert confirmatory_suites(payload) == frozenset({"tau2_retail"})


def test_the_bare_audit_is_clean(capsys):
    """THE GATE. `pi suites audit` with no arguments, through the command itself."""
    import argparse

    from pi_run.cmd_suites import cmd_suites_audit

    code = cmd_suites_audit(argparse.Namespace(root=None, stage1=None))
    out = capsys.readouterr().out
    assert "registry audit: clean" in out, out
    assert code == 0


# ------------------------------------------------- the phantom metric is no longer a phantom
#
# WHAT THE OLD TEST BELIEVED, AND WHY THE BELIEF CHANGED.
#
# `test_no_endpoint_names_a_metric_that_exists_nowhere` asserted that
# `user_private_ask_precision` was "named in datasets_description/18_*.md and in NO code on any
# branch -- not in `pi_eval.score`'s register, not in `report`'s derivers, not in a test", and
# guarded the consequence: a preregistered endpoint naming it would be a committed test that
# could never run. Both halves were TRUE when it was written.
#
# The first half is now false BY CONSTRUCTION, not by weakening: the metric is implemented in
# `pi_eval.metrics.human.user_private_ask` and registered in `pi_eval.score.METRICS`
# (tests/test_user_private_ask.py). The second half is still exactly the property worth
# guarding, so it is what these two tests assert -- from both directions, which the old
# one-sided assertion could not do:
#
#   * a preregistered endpoint may not name a metric the scorer cannot compute; and
#   * this metric is deliberately NOT preregistered, because the population that can exhibit
#     it is 8 target='user' asks over the 78 eligible held-out tau2_airline rows of
#     `inquirer_may_ask_user`, on an EXPLORATORY suite, while the only confirmatory tau2
#     domain (tau2_retail) has zero eligible held-out rows of that arm. Sealing an endpoint on
#     that would be the same "committed test that can never run", one implementation later.


def test_the_metric_the_plan_names_now_exists_in_the_scorer():
    """The premise of the old assertion, retired in the open rather than deleted quietly."""
    from pi_eval.score import BY_NAME

    assert "user_private_ask_precision" in BY_NAME
    assert "user_private_ask_recall" in BY_NAME


def test_no_endpoint_names_a_metric_the_scorer_cannot_compute():
    """The property the old test existed for, now asserted over EVERY endpoint rather than one
    name. `cad_ge2` and `frontier_auc` are computed by `report` from indexed families and are
    not scorer metrics, so they are named here rather than left to a stale allowlist."""
    from pi_eval.prereg import all_endpoints
    from pi_eval.score import BY_NAME, family_of

    derived = {"cad_ge2", "frontier_auc"}
    unbacked = sorted(
        e.metric
        for e in all_endpoints()
        if e.metric not in derived and family_of(e.metric) not in BY_NAME
    )
    assert not unbacked, unbacked


def test_the_user_channel_metric_is_not_preregistered():
    """Not a phantom any more, and not an endpoint either. See the comment above for the n."""
    from pi_eval.prereg import all_endpoints

    assert not [e for e in all_endpoints() if e.metric.startswith("user_private_ask")]
