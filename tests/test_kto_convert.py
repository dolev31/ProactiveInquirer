"""WHAT THE KTO CONVERTER TURNS THE CORPUS INTO, AND WHICH OF ITS CLAIMS ARE CHECKED.

RC4 (`kto_unpaired`) trains on signals the DPO rung throws away. A preference pair is two
rows to KTO, not one: the chosen side is a DESIRABLE (state, action) and the rejected side an
UNDESIRABLE one, and the pairing itself is discarded. That is the whole point -- the rejected
side of an ask_ask pair, the ASK side of a STOP-chosen synthetic pair, and the STOP side of an
ASK-chosen one are three populations the DPO loss can only see through a partner, and KTO
scores each alone against the reference.

FOUR PROPERTIES ARE CHECKED HERE BECAUSE NOTHING DOWNSTREAM COULD NOTICE THEM GOING WRONG:

  * THE DIRECTION. A sign error on one family -- say, the ASK side of a STOP-chosen pair
    emitted as desirable -- trains the policy to ask at states gold says are finished, which is
    the exact failure §4.2 of the execution plan is about. The loss curve is identical either
    way. 28,613 of the 80,766 rows on today's corpus are that one family, so the sign is not a
    detail.
  * THE DEDUPE. The STOP action is ONE constant, so every synthetic pair at a state emits the
    same (state, STOP) row; without a dedupe a state with 12 synthetic pairs contributes that
    row 12 times and the class counts -- and therefore the weights -- describe a dataset nobody
    built. MEASURED on the live corpus: 139,815 rows emitted, 80,766 distinct.
  * THE CONFLICT. A (state, action) that is desirable in one pair and undesirable in another is
    a contradiction KTO cannot represent: it has no partner to relativise against. The corpus
    HAS them (1,475 keys, measured), overwhelmingly from the ask_ask tournament -- B beats C and
    loses to A at one state -- so `on_conflict` is a declared choice in `cfg.sha`, not a default.
  * THE MASSES. KTO's `desirable_weight`/`undesirable_weight` exist to counter unequal class
    counts. Getting them backwards still trains, still reports a falling loss, and silently
    optimises the majority class.

The fixtures here are hand-built and tiny. The one test that reads the real corpus is guarded on
its presence (`data/rl/` is gitignored and the suite text is not redistributable), so it skips
in a checkout that has no data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from pinq.actions import STOP_ACTION_JSON, ask_action_json
from pinq_train.rung1_sft.train import HeldOutDataset
from pinq_train.rung2_kto import (
    ConflictingLabels,
    NonCanonicalStop,
    UnparseableAction,
    balanced_weights,
    build_kto_rows,
)

PROVENANCE = (
    "suite_id",
    "task_id",
    "run_id",
    "turn_idx",
    "scorer_hash",
    "graph_version",
    "split",
    "code_version",
    "arm_id",
    "source_kind",
)


def sft_row(state: str, action: str, **over: Any) -> dict[str, Any]:
    row = {
        "suite_id": "musique",
        "task_id": "t1",
        "run_id": "r1",
        "turn_idx": 1,
        "state_text": state,
        "action_json": action,
        "scorer_hash": "sc",
        "graph_version": "v1",
        "split": "train",
        "code_version": "cv",
        "arm_id": "inquirer_prompted",
    }
    row.update(over)
    return row


def pair_row(state: str, chosen: str, rejected: str, **over: Any) -> dict[str, Any]:
    row = {
        "suite_id": "musique",
        "task_id": "t1",
        "run_id": "r1",
        "turn_idx": 1,
        "state_text": state,
        "chosen_json": chosen,
        "rejected_json": rejected,
        "chosen_run_id": "c1",
        "rejected_run_id": "j1",
        "scorer_hash": "sc",
        "graph_version": "v1",
        "split": "train",
        "code_version": "cv",
        "arm_id": "inquirer_prompted",
        "pair_kind": "ask_ask",
        "stop_source": "",
    }
    row.update(over)
    return row


A = ask_action_json("who founded it?", "need the founder")
B = ask_action_json("when was it founded?", "need the date")
C = ask_action_json("where was it founded?", "need the place")
S = STOP_ACTION_JSON


def by_action(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["prompt"], r["completion"]): r for r in rows}


# --------------------------------------------------------------------------- direction


def test_every_ask_ask_side_is_emitted_once_with_the_right_label() -> None:
    rows, rep = build_kto_rows([], [pair_row("s1", A, B)])

    assert len(rows) == 2
    got = by_action(rows)
    assert got[("s1", A)]["label"] is True
    assert got[("s1", A)]["source_kind"] == "pair_chosen"
    assert got[("s1", B)]["label"] is False
    assert got[("s1", B)]["source_kind"] == "pair_rejected"
    assert rep["n_by_source_kind"] == {"pair_chosen": 1, "pair_rejected": 1}


def test_stop_chosen_synth_pair_makes_the_ask_undesirable_and_the_stop_desirable() -> None:
    """An ASK at a state gold says was DONE. 28,613 rows of the live corpus are this family."""
    rows, rep = build_kto_rows(
        [],
        [pair_row("s1", S, A, pair_kind="ask_stop_synth", stop_source="synthesised")],
    )

    got = by_action(rows)
    assert got[("s1", S)]["label"] is True
    assert got[("s1", S)]["source_kind"] == "synth_stop_chosen"
    assert got[("s1", A)]["label"] is False
    assert got[("s1", A)]["source_kind"] == "synth_ask_rejected"
    assert rep["n_by_stop_source"] == {"synthesised": 2}


def test_ask_chosen_synth_pair_makes_the_stop_undesirable() -> None:
    """A STOP at a state that was NOT done -- the `synthesised_notdone` direction."""
    rows, _ = build_kto_rows(
        [],
        [pair_row("s1", A, S, pair_kind="ask_stop_synth", stop_source="synthesised_notdone")],
    )

    got = by_action(rows)
    assert got[("s1", S)]["label"] is False
    assert got[("s1", S)]["source_kind"] == "synth_stop_rejected"
    assert got[("s1", A)]["label"] is True
    assert got[("s1", A)]["source_kind"] == "synth_ask_chosen"


def test_an_unrecognised_stop_source_is_carried_not_branched_on() -> None:
    """A kind another agent adds must not need a code change here. Direction is the SIDE."""
    rows, rep = build_kto_rows(
        [],
        [pair_row("s1", A, S, pair_kind="ask_stop_synth", stop_source="a_kind_invented_later")],
    )

    assert {r["label"] for r in rows} == {True, False}
    assert rep["n_by_stop_source"] == {"a_kind_invented_later": 2}
    assert by_action(rows)[("s1", S)]["label"] is False


def test_recorded_ask_stop_pairs_take_the_same_direction_rule() -> None:
    rows, rep = build_kto_rows(
        [],
        [pair_row("s1", S, A, pair_kind="ask_stop", stop_source="recorded")],
    )

    got = by_action(rows)
    assert got[("s1", S)]["label"] is True
    assert got[("s1", A)]["label"] is False
    assert rep["n_by_pair_kind"] == {"ask_stop": 2}


def test_sft_rows_are_desirable_whether_they_ask_or_stop() -> None:
    rows, rep = build_kto_rows([sft_row("s1", A), sft_row("s2", S)], [])

    assert [r["label"] for r in rows] == [True, True]
    assert {r["source_kind"] for r in rows} == {"sft_row"}
    assert rep["n_desirable"] == 2
    assert rep["n_undesirable"] == 0


def test_include_pair_kinds_selects_and_counts_the_drop() -> None:
    pairs = [
        pair_row("s1", A, B),
        pair_row("s2", S, A, pair_kind="ask_stop_synth", stop_source="synthesised"),
    ]
    rows, rep = build_kto_rows([], pairs, include_pair_kinds=("ask_ask",))

    assert len(rows) == 2
    assert rep["n_dropped_by_filter"] == {"pair_kind": 1}


# --------------------------------------------------------------------------- dedupe


def test_the_same_state_action_from_two_sources_is_kept_once_and_counted() -> None:
    """The SFT row wins: it is the actual training target, the pair side only relative to one."""
    rows, rep = build_kto_rows([sft_row("s1", A)], [pair_row("s1", A, B)])

    assert len(rows) == 2  # the SFT row and the pair's chosen side are ONE (state, action)
    assert by_action(rows)[("s1", A)]["source_kind"] == "sft_row"
    assert rep["n_emitted_before_dedupe"] == 3
    assert rep["n_deduped"] == 1
    assert rep["n_deduped_by_source_kind"] == {"pair_chosen": 1}


def test_one_stop_constant_at_one_state_collapses_however_many_pairs_name_it() -> None:
    pairs = [
        pair_row("s1", S, A, pair_kind="ask_stop_synth", stop_source="synthesised"),
        pair_row("s1", S, B, pair_kind="ask_stop_synth", stop_source="synthesised"),
        pair_row("s1", S, C, pair_kind="ask_stop_synth", stop_source="synthesised"),
    ]
    rows, rep = build_kto_rows([], pairs)

    assert rep["n_emitted_before_dedupe"] == 6
    assert rep["n_rows"] == 4  # one STOP, three distinct ASKs
    assert rep["n_deduped"] == 2
    assert sum(1 for r in rows if r["completion"] == S) == 1


def test_the_same_action_at_a_different_state_is_a_different_row() -> None:
    rows, rep = build_kto_rows([sft_row("s1", A), sft_row("s2", A)], [])

    assert rep["n_rows"] == 2
    assert rep["n_deduped"] == 0


# --------------------------------------------------------------------------- conflicts


def test_a_state_action_with_both_labels_is_refused_and_counted() -> None:
    """B beats C at s1 and loses to A at s1. 1,475 keys on the live corpus do this."""
    pairs = [pair_row("s1", B, C), pair_row("s1", A, B)]

    with pytest.raises(ConflictingLabels) as exc:
        build_kto_rows([], pairs)

    assert exc.value.n_conflicts == 1
    assert exc.value.report["n_conflicts"] == 1
    assert exc.value.report["n_conflict_rows"] == 2
    assert "pair_chosen" in str(exc.value) and "pair_rejected" in str(exc.value)


def test_an_sft_row_contradicted_by_a_rejected_side_is_a_conflict_too() -> None:
    with pytest.raises(ConflictingLabels):
        build_kto_rows([sft_row("s1", B)], [pair_row("s1", A, B)])


def test_on_conflict_drop_removes_both_sides_and_reports_the_count() -> None:
    """Both sides. Keeping one would pick a side of a contradiction the data does not settle."""
    pairs = [pair_row("s1", B, C), pair_row("s1", A, B)]
    rows, rep = build_kto_rows([], pairs, on_conflict="drop")

    assert {r["completion"] for r in rows} == {A, C}
    assert rep["n_conflicts"] == 1
    assert rep["n_conflict_rows_dropped"] == 2
    assert rep["on_conflict"] == "drop"


def test_an_unknown_on_conflict_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError, match="on_conflict"):
        build_kto_rows([], [pair_row("s1", A, B)], on_conflict="keep_first")


# --------------------------------------------------------------------------- the masses


def test_balanced_weights_equalise_the_two_masses_at_mean_weight_one() -> None:
    n_d, n_u = 47_285, 33_481  # the live corpus, measured
    w_d, w_u = balanced_weights(n_d, n_u)

    assert n_d * w_d == pytest.approx(n_u * w_u)
    assert (n_d * w_d + n_u * w_u) / (n_d + n_u) == pytest.approx(1.0)
    assert w_d < 1.0 < w_u  # the minority class is the one that gets lifted


def test_balanced_weights_are_one_and_one_when_the_classes_already_match() -> None:
    assert balanced_weights(10, 10) == (1.0, 1.0)


def test_balanced_weights_refuse_an_empty_class() -> None:
    """Refused, not clamped: a class with no rows has no weight that balances anything."""
    with pytest.raises(ValueError, match="0 undesirable"):
        balanced_weights(10, 0)
    with pytest.raises(ValueError, match="0 desirable"):
        balanced_weights(0, 10)


def test_the_report_carries_the_weights_that_would_balance_the_rows_it_built() -> None:
    sfts = [sft_row(f"s{i}", A) for i in range(6)]
    pairs = [pair_row(f"p{i}", A, B) for i in range(2)]
    rows, rep = build_kto_rows(sfts, pairs)

    n_d, n_u = rep["n_desirable"], rep["n_undesirable"]
    assert (n_d, n_u) == (8, 2)
    w_d = rep["balanced_desirable_weight"]
    w_u = rep["balanced_undesirable_weight"]
    assert n_d * w_d == pytest.approx(n_u * w_u)
    assert rep["n_rows"] == len(rows) == n_d + n_u


# --------------------------------------------------------------------------- the wall


def test_a_pair_stamped_dev_is_refused() -> None:
    with pytest.raises(HeldOutDataset, match="dev"):
        build_kto_rows([], [pair_row("s1", A, B, split="dev")])


def test_an_sft_row_stamped_dev_is_refused() -> None:
    with pytest.raises(HeldOutDataset, match="dev"):
        build_kto_rows([sft_row("s1", A, split="dev")], [])


def test_an_empty_split_is_present_and_wrong() -> None:
    with pytest.raises(HeldOutDataset):
        build_kto_rows([], [pair_row("s1", A, B, split="")])


def test_a_row_with_no_split_at_all_is_waved_through_and_counted() -> None:
    row = sft_row("s1", A)
    del row["split"]
    rows, rep = build_kto_rows([row], [])

    assert len(rows) == 1
    assert rows[0]["split"] == ""
    assert rep["n_rows_without_split"] == 1


# --------------------------------------------------------------------------- the bytes


def test_a_stop_side_whose_bytes_are_not_the_one_constant_is_refused() -> None:
    """Chosen and rejected are SEPARATE rows in KTO, so two STOP spellings are two completions."""
    odd = json.dumps({"action": "STOP"}, separators=(",", ":"))
    assert odd != STOP_ACTION_JSON

    with pytest.raises(NonCanonicalStop):
        build_kto_rows([], [pair_row("s1", odd, A, pair_kind="ask_stop", stop_source="recorded")])


def test_an_action_the_policy_parser_cannot_read_is_refused() -> None:
    with pytest.raises(UnparseableAction):
        build_kto_rows([sft_row("s1", '{"action": "ASK", "question": ""}')], [])


# --------------------------------------------------------------------------- provenance


def test_every_emitted_row_carries_the_full_provenance_block() -> None:
    sfts = [sft_row("s1", A)]
    pairs = [
        pair_row("s2", A, B),
        pair_row("s3", S, C, pair_kind="ask_stop_synth", stop_source="synthesised"),
    ]
    rows, _ = build_kto_rows(sfts, pairs)

    assert len(rows) == 5
    for r in rows:
        missing = [k for k in PROVENANCE if k not in r]
        assert not missing, f"{r['source_kind']} is missing {missing}"
        assert r["scorer_hash"] == "sc"
        assert r["graph_version"] == "v1"
        assert isinstance(r["label"], bool)
        assert r["prompt"] and r["completion"]


def test_a_pair_side_records_the_candidate_run_it_came_from() -> None:
    rows, _ = build_kto_rows([], [pair_row("s1", A, B)])

    got = by_action(rows)
    assert got[("s1", A)]["run_id"] == "r1"  # the STATE's run, as on the SFT rows
    assert got[("s1", A)]["candidate_run_id"] == "c1"
    assert got[("s1", B)]["candidate_run_id"] == "j1"


def test_a_pair_whose_two_sides_are_at_different_states_is_refused() -> None:
    """KTO does not need the cancellation, but ONE state_text per pair is still assumed here."""
    from pinq_train.rung2_dpo import NotSameState

    row = pair_row("s1", A, B)
    del row["state_text"]
    row["chosen_state_text"] = "s1"
    row["rejected_state_text"] = "s2"
    with pytest.raises(NotSameState):
        build_kto_rows([], [row])


# --------------------------------------------------------------------------- real rows

_RL = Path(os.environ.get("PI_RL_DATA_DIR") or (Path(__file__).resolve().parents[1] / "data/rl"))
_SFT, _PAIRS = _RL / "sft.jsonl", _RL / "pairs.jsonl"
_HAVE = _SFT.is_file() and _PAIRS.is_file()


def _head(path: Path, n: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
            if len(out) == n:
                break
    return out


@pytest.mark.skipif(not _HAVE, reason=f"no corpus at {_RL} (data/rl is gitignored)")
def test_five_real_rows_of_each_file_convert(capsys: pytest.CaptureFixture[str]) -> None:
    """The exported schema, not a fixture's idea of it. Printed so the report lands in the log.

    Under the DEFAULT `on_conflict`, deliberately. MEASURED on today's export: the first 5 rows
    of each file convert cleanly, the first 50 do too, and the first 500 raise on one key -- so
    the strict path is the one worth exercising here, and a failure means the head of the export
    changed shape rather than that this test was optimistic.
    """
    rows, rep = build_kto_rows(_head(_SFT, 5), _head(_PAIRS, 5))

    with capsys.disabled():
        print("\nreal-row conversion report:")
        print(json.dumps(rep, indent=2, sort_keys=True))

    assert rep["n_rows"] == len(rows) == rep["n_desirable"] + rep["n_undesirable"]
    assert rep["n_emitted_before_dedupe"] == 5 + 2 * 5
    for r in rows:
        assert not [k for k in PROVENANCE if k not in r]
        assert r["scorer_hash"] and r["graph_version"]
        assert r["split"] == "train"
    if rep["n_undesirable"] and rep["n_desirable"]:
        w_d, w_u = balanced_weights(rep["n_desirable"], rep["n_undesirable"])
        assert rep["n_desirable"] * w_d == pytest.approx(rep["n_undesirable"] * w_u)
