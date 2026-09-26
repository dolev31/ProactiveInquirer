"""Task 3: the published retail/airline decomposition, reproduced, plus what a follow-up turn
was for -- as far as the stored artifacts can say, and no further.

REPRODUCTION, NOT A NEW STATISTIC. `reproduce_decomposition` calls `pi_eval.fork_report`
directly (`clustered_engagement`, `paired_engagement`) -- the exact functions
`scripts/report_forks.py` used to write `artifacts/forks_test/forks.tau2_*.test.json` -- rather
than re-deriving the pairing or the bootstrap. `over_cap_table` reuses
`outcome_classifier.is_over_cap`, the SAME rule Task 2's `over_cap_runaway` fail class uses, so
"over cap" cannot silently mean two things in one report.

WHAT THIS MODULE DOES NOT CLAIM. The published follow-up count (`n_user_turns -
n_prefix_user_turns`) is real and reproduced exactly (see RESULT.md ss3). What the CONTENT of
those turns was -- did the customer answer the agent's question, repeat or correct something,
or close the conversation -- is a question about the outer tau2 TRANSCRIPT, and this
repository deliberately never writes that transcript's content to disk:
`pi_run/stages/tau2_runner.py::_msg_digest` reduces every message to
`{role, tool names, content_sha}` specifically because "a tau2 transcript is a customer-service
dialogue containing account numbers and balances, and a run artifact is exactly the wrong place
for it." Recovering the text would mean re-simulating the user, which is a new, paid rollout
that this lane is explicitly not authorised to run (and would not reproduce the ORIGINAL
dialogue even if run, since the user simulator is not temperature-0-deterministic and is not
cache-replayable -- its calls go through tau2's own `litellm_utils`, uncached, never through
`pinq`'s content-addressed cache -- see RESULT.md ss3 for the measurement that established
this). What IS measurable without the transcript is named below and nothing else is claimed:

  * `closing_share`: whether the run's OWN termination reason was the simulated user ending the
    dialogue (`TerminationReason.USER_STOP`) -- a structural fact about how the run ended, not
    an inference about what any one turn contained.
"""

from __future__ import annotations

from typing import Any, Sequence

from scripts.tau2_failure_taxonomy.outcome_classifier import is_over_cap
from scripts.tau2_failure_taxonomy.runs_io import RunRecord

from pi_eval.fork_report import clustered_engagement, follow_ups, paired_engagement


def to_fork_report_row(run: RunRecord) -> dict[str, Any]:
    """The six fields `pi_eval.fork_report` pairs and contrasts on, read off one `RunRecord`."""
    return {
        "run_id": run.run_id,
        "arm_id": run.arm_id,
        "seed": run.seed,
        "foreign_trace_sha": run.manifest.get("foreign_trace_sha"),
        "foreign_prefix_k": run.manifest.get("foreign_prefix_k"),
        "task_id": run.task_id,
        "code_version": run.code_version,
        "status": run.run_status,
        "n_user_turns": run.status.get("n_user_turns"),
        "n_prefix_user_turns": run.status.get("n_prefix_user_turns"),
        "tau_reward": (run.status.get("native") or {}).get("tau_reward"),
    }


def reproduce_decomposition(
    runs: Sequence[RunRecord], *, treatment: str = "inquirer_prompted", control: str = "self_ask"
) -> dict[str, Any]:
    rows = [to_fork_report_row(r) for r in runs]
    return {
        "pooled": paired_engagement(rows, treatment=treatment, control=control),
        "clustered": clustered_engagement(rows, treatment=treatment, control=control),
    }


def over_cap_table(runs: Sequence[RunRecord]) -> dict[str, dict[str, Any]]:
    """Per arm: over-cap count/rate and the two inside/over mean-follow-up populations --
    `paper/appendix_instruments.tex`'s budget-regime table, recomputed from the run directories
    rather than copied from it."""
    by_arm: dict[str, list[RunRecord]] = {}
    for r in runs:
        by_arm.setdefault(r.arm_id, []).append(r)

    out: dict[str, dict[str, Any]] = {}
    for arm, arm_runs in by_arm.items():
        inside: list[int] = []
        over: list[int] = []
        for r in arm_runs:
            f = follow_ups(
                {
                    "n_user_turns": r.status.get("n_user_turns"),
                    "n_prefix_user_turns": r.status.get("n_prefix_user_turns"),
                }
            )
            (over if is_over_cap(r.manifest, r.status) else inside).append(f)
        n_total = len(arm_runs)
        n_over = len(over)
        out[arm] = {
            "n_total": n_total,
            "n_over": n_over,
            "rate": (n_over / n_total) if n_total else float("nan"),
            "mean_inside": (sum(inside) / len(inside)) if inside else float("nan"),
            "n_inside": len(inside),
            "mean_over": (sum(over) / len(over)) if over else float("nan"),
        }
    return out


def closing_share(runs: Sequence[RunRecord]) -> dict[str, Any]:
    """Of the runs that needed at least one follow-up turn, the share whose termination reason
    was the simulated customer ending the dialogue -- see the module docstring for why this is
    the one per-unit "what was this turn for" fact recoverable without the transcript text."""
    with_followups = [
        r
        for r in runs
        if follow_ups(
            {
                "n_user_turns": r.status.get("n_user_turns"),
                "n_prefix_user_turns": r.status.get("n_prefix_user_turns"),
            }
        )
        > 0
    ]
    n = len(with_followups)
    n_user_stop = sum(
        1
        for r in with_followups
        if "USER_STOP" in str(r.status.get("termination_reason") or "").upper()
    )
    return {
        "n_with_followups": n,
        "n_closed_by_user_stop": n_user_stop,
        "share": (n_user_stop / n) if n else float("nan"),
    }
