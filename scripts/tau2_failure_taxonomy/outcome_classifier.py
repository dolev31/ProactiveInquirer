"""Classify one unit's database outcome: pass, fail (with a class), or not scored.

`tau_reward` (`status.json["native"]["tau_reward"]`) is the endpoint upstream computes by
replaying the executed action log against a gold environment and comparing DB hashes
(`pinq_adapters.tau2.actuator`'s own docstring: "the definition of the number lives
upstream"). ABSENT IS NOT ZERO -- a task whose `reward_basis` excludes DB carries no
`tau_reward` at all (`_compute_reward`: "A missing measurement must look missing"), and this
classifier preserves that distinction rather than collapsing "not scored" into "failed".

THE FAIL CLASSES, in the priority order applied when more than one signal is present (a run
that both ran over cap AND missed a gold action is filed as the cap runaway, because the cap
is a harness-level explanation that would have capped ANY policy, and it is checked first):

  1. `simulator_ended_early`  -- the run itself did not complete cleanly (top-level
     `status != ok`, or a `termination_reason` that is an environment/infra failure rather
     than a dialogue reaching its own end). This is `artifacts/banking/RESULT.md`'s "Defect A"
     shape generalised: the unit's own machinery, not the policy, ended the attempt.
  2. `over_cap_runaway` -- the Inquirer's retrieval spend exceeded its declared `budget_cap`
     (`is_over_cap`, the exact accounting `paper/appendix_instruments.tex`'s budget-regime
     section and this lane's `turn_accounting.py` both use for "over cap").
  3. `policy_violation` -- a mutating tool call SUCCEEDED that names a tool absent from the
     gold action set for this task. Gold actions are read from tau2's own registry
     (`gold_actions.py`); an EMPTY gold action list is a legitimate answer ("the correct
     outcome is a database that does not change" -- `pi_run/suites.py`'s note on retail/
     airline), so any mutation at all is a violation on those tasks. ONLY CHECKED WHEN THE
     `mutating` FIELD IS MEASURED INFORMATIVE FOR THIS SUITE (`mutating_flag_is_informative`,
     `trust_mutating=True` below) -- see that function's docstring for why banking's copy of
     this field cannot be trusted at all.
  4. `info_never_obtained` -- every ask in the run came back with zero retrieved evidence
     (`turns[*].n_retrieved` sums to 0): the Inquirer's channel never surfaced anything to act
     on, independent of what the gold actions were.
  5. `wrong_or_missing_tool_action` -- the gold action set is known and non-empty, and at least
     one gold action's tool name was never executed successfully, but neither of the two more
     specific classes above applies.
  6. `other` -- none of the above matched; `note` carries what IS known (db_match, reward_error)
     so a reader can extend the taxonomy rather than trust an unexplained bucket.

A `note` accompanies every fail class: CONTRIBUTING.md rule 3 ("never state a measurement you did
not take") applies to a LABEL as much as to a number, and a fail class with nothing behind it
is a guess wearing a category's name.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

from scripts.tau2_failure_taxonomy import gold_actions

FailClass = Literal[
    "simulator_ended_early",
    "over_cap_runaway",
    "policy_violation",
    "info_never_obtained",
    "wrong_or_missing_tool_action",
    "other",
]

_EARLY_TERMINATION_MARKERS = (
    "USER_ERROR",
    "INFRASTRUCTURE_ERROR",
    "TIMEOUT",
    "UNEXPECTED_ERROR",
    "CONTEXT_WINDOW_EXCEEDED",
    "TOO_MANY_ERRORS",
)


def mutating_flag_is_informative(env_calls_sample: Sequence[Mapping[str, Any]]) -> bool:
    """False when `mutating` cannot discriminate anything over this sample of `ok` env calls.

    MEASURED, NOT ASSUMED: `pi_run/stages/tau2_runner.py::env_calls_from` -- the function that
    derives the STORED `env_calls` for any call the tau2 ORCHESTRATOR executed (as opposed to
    the Inquirer's own retriever, which stamps its own calls `mutating=False` unconditionally) --
    sets `mutating=True` on every row it writes, with its own comment: "conservatively; the
    orchestrator does not expose the flag". On the banking population this lane reads, that
    reduces to 100% `True` across all 57 distinct tool names actually called, INCLUDING
    unambiguous reads (`get_current_time`, `get_user_information_by_email`, `KB_search`) --
    verified directly against `env._is_mutating_tool`, which by contrast discriminates
    correctly when called on the live environment object. A classifier that trusted the stored
    field anyway would report "policy_violation" on nearly every banking fail, which is exactly
    the shape an unrelated recording default produces and not a measurement of any policy's
    behaviour -- the "instrument error reads as a finding" failure this repository's own
    history warns about. Retail and airline are unaffected (their `env_calls` are dominated by
    the retriever's own, correctly-labelled-False calls; `RESULT.md` ss2 shows both `True` and
    `False` genuinely present there), so this is a per-suite MEASUREMENT, not a blanket
    disabling of the check.
    """
    seen = {bool(c.get("mutating")) for c in env_calls_sample if c.get("ok")}
    return len(seen) > 1


def is_over_cap(manifest: Mapping[str, Any], status: Mapping[str, Any]) -> bool:
    """The published campaign's own over-cap rule: in-loop plus post-dialogue retrieval charges
    together exceeding the run's own declared `budget_cap` (`paper/appendix_instruments.tex`'s
    budget-regime table; reused verbatim here so Task 2's fail class and Task 3's decomposition
    can never quietly define "over cap" two different ways)."""
    budget_cap = manifest.get("budget_cap")
    if budget_cap is None:
        return False
    spent = (status.get("spent") or {}).get("retrieval_calls")
    if spent is None:
        return False
    return float(spent) > float(budget_cap)


def classify_outcome(
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
    outcome: Mapping[str, Any],
    turns: Sequence[Mapping[str, Any]] = (),
    *,
    trust_mutating: bool = True,
) -> dict[str, Any]:
    native = status.get("native") or {}
    tau_reward = native.get("tau_reward")
    result: dict[str, Any] = {
        "scored": tau_reward is not None,
        "passed": None,
        "fail_class": None,
        "note": "",
    }
    if tau_reward is None:
        result["note"] = (
            "tau_reward absent: reward_basis excludes DB, or the evaluator did not cover this task"
        )
        return result

    passed = float(tau_reward) >= 1.0
    result["passed"] = passed
    if passed:
        return result

    suite_id = str(manifest.get("suite_id") or "")
    task_id = str(manifest.get("task_id") or "")
    run_status = status.get("status")
    termination_reason = str(status.get("termination_reason") or "").upper()

    if run_status not in (None, "ok") or any(
        m in termination_reason for m in _EARLY_TERMINATION_MARKERS
    ):
        result["fail_class"] = "simulator_ended_early"
        result["note"] = (
            f"status={run_status!r} termination_reason={status.get('termination_reason')!r}"
        )
        return result

    if is_over_cap(manifest, status):
        spent = (status.get("spent") or {}).get("retrieval_calls")
        result["fail_class"] = "over_cap_runaway"
        result["note"] = f"retrieval_calls={spent} > budget_cap={manifest.get('budget_cap')}"
        return result

    env_calls = outcome.get("env_calls") or []
    ok_calls = [c for c in env_calls if c.get("ok")]
    called_names = {c.get("tool_name") for c in ok_calls}
    mutating_ok_names = {c.get("tool_name") for c in ok_calls if c.get("mutating")}

    gold_names: tuple[str, ...] | None = None
    try:
        gold_names = gold_actions.gold_action_names(suite_id, task_id)
    except Exception:  # noqa: BLE001 - tau2 unavailable, or an unmapped suite; degrade gracefully
        gold_names = None

    if gold_names is not None and trust_mutating:
        gold_set = set(gold_names)
        extra_mutations = mutating_ok_names - gold_set
        if extra_mutations:
            result["fail_class"] = "policy_violation"
            result["note"] = (
                f"mutating call(s) outside the gold action set: {sorted(extra_mutations)}"
            )
            return result

    n_retrieved_total = sum(int(t.get("n_retrieved") or 0) for t in turns) if turns else None
    if n_retrieved_total == 0:
        result["fail_class"] = "info_never_obtained"
        result["note"] = f"{len(turns)} ask(s), 0 total n_retrieved"
        return result

    if gold_names:
        missing = sorted({n for n in gold_names if n not in called_names})
        if missing:
            result["fail_class"] = "wrong_or_missing_tool_action"
            result["note"] = f"gold action(s) never successfully executed: {missing}"
            return result

    result["fail_class"] = "other"
    result["note"] = (
        f"db_match={native.get('db_match')} reward_error={status.get('reward_error')!r}"
    )
    return result
