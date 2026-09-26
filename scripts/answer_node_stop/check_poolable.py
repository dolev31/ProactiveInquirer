"""May these arms be read out of ONE isolated store, or must they be scored separately?

`--checkpoint-model-id` disambiguates the SELECTION -- two adapters sharing an `arm_id` and a
`grid_name` in one store are no longer silently averaged into a single number. It does not make
a shared store as safe as separate ones, and it reports none of the other ways two arms in one
store can differ. This is the check for those, run BEFORE the gate rather than discovered in a
verdict that already looks like a result.

WHAT MUST BE EQUAL ACROSS THE ARMS, and why each one is fatal rather than cosmetic:

  scorer_hash    global over the gold set a pass touched, so two of them means two scoring
                 passes were pooled and no cross-arm number in the store is comparable.
  code_version   one grid name has already been observed pooling three commits, with 221 of
                 1,200 baseline cells disagreeing across them.
  grid_sha256    the same grid NAME can carry different bodies; the sha is the body.
  corpus_dir     gold built against corpus A scoring runs rolled against corpus B misses every
                 span silently, and reads as a policy that retrieves nothing.
  frozen roles   Drafter, Answerer and Judge must be ONE model each across the arms. If they
                 differ, the contrast is between two pipelines, not two Inquirers -- and
                 because `base_url_sha` is inside run identity, a per-arm proxy port re-pins
                 the frozen roles too and breaks the contrast while every arm still looks fine.

WHAT MUST DIFFER: the inquirer's own `model_pin_hash`. Two arms with the SAME inquirer pin are
not two arms; that is the pooling defect in its purest form, and a checker that only looked for
equalities would pass it happily.

USAGE
    PYTHONPATH=<worktree>/src <venv>/bin/python -m scripts.answer_node_stop.check_poolable \\
        --parquet <iso>/parquet --grid-name tier1_trained_qa_base \\
        --arm answernode:inquirer_trained:qwen3-8b-sft-headline-answernode \\
        --arm refresh:inquirer_trained:qwen3-8b-sft-headline-refresh \\
        --arm base:inquirer_prompted:qwen3-8b-base

Exit 0 = poolable, 3 = REFUSED with the reason named. Nothing is written anywhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Columns of `runs` that must hold ONE value across every selected run of every arm.
SHARED_RUN_COLUMNS = ("code_version", "grid_sha256", "corpus_dir", "split")
#: Actors in `calls` that are frozen across arms. The Inquirer is deliberately absent: it is the
#: thing under test and MUST differ.
FROZEN_ACTORS = ("drafter", "answerer", "judge")


class NotPoolable(RuntimeError):
    """These arms cannot be read out of one store. The message names which field and its values."""


def _rows(con, sql: str) -> list[dict]:
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _in(ids: Sequence[str]) -> str:
    inner = ", ".join("'" + str(i).replace("'", "''") + "'" for i in ids)
    return f"({inner})" if inner else "('')"


def shared_values(con, run_ids: Sequence[str], column: str) -> list[Any]:
    rows = _rows(con, f"SELECT DISTINCT {column} AS v FROM runs WHERE run_id IN {_in(run_ids)}")
    return sorted({r["v"] for r in rows}, key=lambda x: (x is None, str(x)))


def frozen_role_models(con, run_ids: Sequence[str]) -> dict[str, list[str]]:
    rows = _rows(
        con,
        f"SELECT DISTINCT lower(actor) AS actor, model FROM calls WHERE run_id IN {_in(run_ids)}",
    )
    out: dict[str, set[str]] = {}
    for r in rows:
        actor = str(r["actor"] or "")
        if actor in FROZEN_ACTORS:
            out.setdefault(actor, set()).add(str(r["model"]))
    return {k: sorted(v) for k, v in out.items()}


def check(con, runs_by_label: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Raise `NotPoolable` on the first fatal disagreement; otherwise return what was checked.

    RETURNS THE VALUES IT CHECKED, not a boolean. A checker that answers yes/no cannot be put in
    a RESULT, and "these arms were poolable" is a claim a reader has to be able to audit.
    """
    all_ids = [str(r["run_id"]) for rs in runs_by_label.values() for r in rs]
    if not all_ids:
        raise NotPoolable(
            "no runs selected for any arm. An empty selection agrees with itself about "
            "everything, so it is refused rather than passed."
        )
    empty = sorted(k for k, v in runs_by_label.items() if not v)
    if empty:
        raise NotPoolable(
            f"these arms selected NO runs: {empty}. An arm that selects nothing reads as a "
            "clean zero in every contrast downstream."
        )

    report: dict[str, Any] = {"n_runs": {k: len(v) for k, v in runs_by_label.items()}}

    for col in SHARED_RUN_COLUMNS:
        vals = shared_values(con, all_ids, col)
        if len(vals) != 1:
            raise NotPoolable(
                f"{col} takes {len(vals)} values across these arms: {vals}. Pooling by arm and "
                "grid was only the first confound; score these arms into SEPARATE stores."
            )
        report[col] = vals[0]

    roles = frozen_role_models(con, all_ids)
    for actor, models in sorted(roles.items()):
        if len(models) != 1:
            raise NotPoolable(
                f"the frozen role {actor!r} ran under {len(models)} models across these arms: "
                f"{models}. The contrast would be between two pipelines, not two Inquirers. "
                "`base_url_sha` is inside run identity, so a per-arm proxy port does exactly "
                "this while every arm still looks healthy on its own."
            )
    report["frozen_roles"] = {k: v[0] for k, v in roles.items()}
    if not roles:
        report["frozen_roles"] = {}
        report["frozen_roles_warning"] = (
            "no drafter/answerer/judge rows in `calls` for these runs -- the frozen roles were "
            "NOT verified. Absence is not agreement."
        )

    # ...and the one thing that must DIFFER.
    pins = {}
    for label, rs in runs_by_label.items():
        vals = shared_values(con, [str(r["run_id"]) for r in rs], "model_pin_hash")
        pins[label] = vals
        if len(vals) != 1:
            raise NotPoolable(
                f"arm {label!r} holds {len(vals)} model pins {vals}: the selection is itself "
                "pooling two checkpoints, which is what --checkpoint-model-id exists to stop."
            )
    flat = [v[0] for v in pins.values()]
    if len(set(flat)) != len(flat):
        raise NotPoolable(
            f"two arms share an inquirer model_pin_hash: {pins}. They are not two arms. A check "
            "that only looked for fields that must be EQUAL would pass this happily."
        )
    report["model_pin_hash"] = {k: v[0] for k, v in pins.items()}
    return report


def build_parser() -> argparse.ArgumentParser:
    from scripts.answer_node_stop.read_contrasts import parse_arm

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--parquet", required=True)
    p.add_argument("--grid-name", default="tier1_trained_qa_base")
    p.add_argument("--arm", action="append", type=parse_arm, required=True)
    p.add_argument("--suite", action="append", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    from scripts.answer_node_coverage import lib
    from scripts.answer_node_stop.read_contrasts import scorer_hash_of
    from scripts.stopping_answer_test.lib import load_arm_runs

    con = lib.open_store(Path(a.parquet))
    runs_by_label = {}
    for label, arm_id, model_id in a.arm:
        rs: list = []
        for suite in a.suite or [None]:
            rs.extend(
                load_arm_runs(
                    con, arm_id=arm_id, model_id=model_id, grid_name=a.grid_name, suite=suite
                )
            )
        runs_by_label[label] = rs
    try:
        report = check(con, runs_by_label)
        report["scorer_hash"] = scorer_hash_of(con)
    except NotPoolable as exc:
        print(f"REFUSED -- score these arms separately:\n  {exc}")
        return 3
    print("POOLABLE. Checked, with the values a reader can audit:")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - a driver
    raise SystemExit(main())
