"""Two arms in one isolated store: which disagreements are fatal, and is the check non-vacuous.

`--checkpoint-model-id` stops two adapters sharing an `arm_id` and a `grid_name` from being
averaged into one number. It disambiguates the SELECTION and nothing else, so every other way a
shared store can confound a contrast is still open. These pin each one, and the last two pin the
thing a checker of equalities would miss: that the arms' own inquirer pins must DIFFER, and that
a silent absence is not agreement.
"""

from __future__ import annotations

import duckdb
import pytest
from scripts.answer_node_stop.check_poolable import NotPoolable, check


def _store(runs: list[dict], calls: list[dict] | None = None):
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE runs (run_id VARCHAR, code_version VARCHAR, grid_sha256 VARCHAR, "
        "corpus_dir VARCHAR, split VARCHAR, model_pin_hash VARCHAR)"
    )
    for r in runs:
        con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            [
                r["run_id"],
                r.get("code_version", "c0"),
                r.get("grid_sha256", "g0"),
                r.get("corpus_dir", "corp0"),
                r.get("split", "test"),
                r.get("model_pin_hash", "pin0"),
            ],
        )
    con.execute("CREATE TABLE calls (run_id VARCHAR, actor VARCHAR, model VARCHAR)")
    for c in calls or []:
        con.execute("INSERT INTO calls VALUES (?, ?, ?)", [c["run_id"], c["actor"], c["model"]])
    return con


def _arms(**kw):
    return {label: [{"run_id": r} for r in ids] for label, ids in kw.items()}


FROZEN = [
    {"run_id": r, "actor": a, "model": "gpt-oss-120b"}
    for r in ("a1", "b1")
    for a in ("drafter", "answerer", "judge")
]


def test_two_arms_that_differ_only_in_their_inquirer_pin_are_poolable():
    con = _store(
        [
            {"run_id": "a1", "model_pin_hash": "pinA"},
            {"run_id": "b1", "model_pin_hash": "pinB"},
        ],
        FROZEN,
    )
    rep = check(con, _arms(answernode=["a1"], refresh=["b1"]))
    assert rep["code_version"] == "c0" and rep["corpus_dir"] == "corp0"
    assert rep["model_pin_hash"] == {"answernode": "pinA", "refresh": "pinB"}
    assert rep["frozen_roles"] == {
        "answerer": "gpt-oss-120b",
        "drafter": "gpt-oss-120b",
        "judge": "gpt-oss-120b",
    }


@pytest.mark.parametrize(
    "column,value",
    [
        ("code_version", "c1"),
        ("grid_sha256", "g1"),
        ("corpus_dir", "corp1"),
        ("split", "dev"),
    ],
)
def test_each_shared_column_is_fatal_on_its_own(column, value):
    """One at a time, so a checker that stopped at the first column is caught.

    Each of these has its own history: one grid name pooling three commits (221 of 1,200 cells
    disagreeing), one grid name carrying two bodies, and gold built against a different corpus
    missing every span while reading as a policy that retrieves nothing.
    """
    con = _store(
        [
            {"run_id": "a1", "model_pin_hash": "pinA"},
            {"run_id": "b1", "model_pin_hash": "pinB", column: value},
        ],
        FROZEN,
    )
    with pytest.raises(NotPoolable, match=column):
        check(con, _arms(answernode=["a1"], refresh=["b1"]))


def test_a_frozen_role_served_by_two_models_is_fatal():
    """`base_url_sha` is inside run identity, so a per-arm proxy port re-pins the Drafter and
    Answerer too -- and every arm still looks healthy on its own while the contrast is gone."""
    con = _store(
        [
            {"run_id": "a1", "model_pin_hash": "pinA"},
            {"run_id": "b1", "model_pin_hash": "pinB"},
        ],
        [
            {"run_id": "a1", "actor": "drafter", "model": "gpt-oss-120b"},
            {"run_id": "b1", "actor": "drafter", "model": "some-other-model"},
        ],
    )
    with pytest.raises(NotPoolable, match="drafter"):
        check(con, _arms(answernode=["a1"], refresh=["b1"]))


def test_two_arms_sharing_an_inquirer_pin_are_refused():
    """The pooling defect in its purest form. A checker that only looked for fields which must be
    EQUAL would pass this happily, which is why it is tested from the other direction."""
    con = _store(
        [
            {"run_id": "a1", "model_pin_hash": "same"},
            {"run_id": "b1", "model_pin_hash": "same"},
        ],
        FROZEN,
    )
    with pytest.raises(NotPoolable, match="not two arms"):
        check(con, _arms(answernode=["a1"], refresh=["b1"]))


def test_one_arm_holding_two_pins_is_refused_as_a_pooled_selection():
    """This is exactly what `--checkpoint-model-id` exists to stop, caught at the store instead
    of being trusted to the flag having been passed."""
    con = _store(
        [
            {"run_id": "a1", "model_pin_hash": "pinA"},
            {"run_id": "a2", "model_pin_hash": "pinB"},
            {"run_id": "b1", "model_pin_hash": "pinC"},
        ],
        FROZEN,
    )
    with pytest.raises(NotPoolable, match="is itself pooling"):
        check(con, _arms(answernode=["a1", "a2"], refresh=["b1"]))


def test_an_arm_that_selected_nothing_is_refused_rather_than_agreeing_with_everything():
    con = _store([{"run_id": "a1", "model_pin_hash": "pinA"}], FROZEN)
    with pytest.raises(NotPoolable, match="selected NO runs"):
        check(con, _arms(answernode=["a1"], refresh=[]))
    with pytest.raises(NotPoolable, match="no runs selected"):
        check(con, _arms(answernode=[], refresh=[]))


def test_missing_frozen_role_rows_are_reported_as_unverified_not_as_agreement():
    """An absence in `calls` must not read as "the roles agree". Three lanes have now been
    caught reporting a schema miss in the shape of a finding about the data."""
    con = _store(
        [
            {"run_id": "a1", "model_pin_hash": "pinA"},
            {"run_id": "b1", "model_pin_hash": "pinB"},
        ],
        [],
    )
    rep = check(con, _arms(answernode=["a1"], refresh=["b1"]))
    assert rep["frozen_roles"] == {}
    assert "NOT verified" in rep["frozen_roles_warning"]


def test_the_inquirer_is_not_treated_as_a_frozen_role():
    """It is the thing under test. Freezing it would refuse every real contrast."""
    con = _store(
        [
            {"run_id": "a1", "model_pin_hash": "pinA"},
            {"run_id": "b1", "model_pin_hash": "pinB"},
        ],
        FROZEN
        + [
            {"run_id": "a1", "actor": "inquirer", "model": "qwen3-8b-sft-headline-answernode"},
            {"run_id": "b1", "actor": "inquirer", "model": "qwen3-8b-sft-headline-refresh"},
        ],
    )
    rep = check(con, _arms(answernode=["a1"], refresh=["b1"]))
    assert "inquirer" not in rep["frozen_roles"]
