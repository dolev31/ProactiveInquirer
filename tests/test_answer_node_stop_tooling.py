"""The L6.1 label-shift accounting, and the two-file export driver.

WHAT THESE GUARD. The label-shift table is the lane's headline number and it is produced by
joining two `Example` lists. Three ways that join can lie, each tested below:

  * joining on the wrong key, so relabelled states silently drop out of the 2x2;
  * counting only the states BOTH files hold, so a state that exists in one file because of
    the relabelling reads as zero rather than as a shift;
  * reading `n_stop_done_before_dedupe` as the stop share -- it is incremented before the exact
    dedupe and is 6,505 higher than the row count on the shipped export.

And one that the driver guards rather than reports: `runs/` is shared and live, so two separate
walks can see different populations. The driver takes one walk and hands the same row list to
both exports; `test_both_files_come_from_one_walk` is what pins that.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.answer_node_stop import label_shift as ls

REPO = Path(__file__).resolve().parents[1]


class _Ex:
    """The three `Example` fields the accounting reads. A real Example is frozen and slotted,
    and building one needs twenty provenance arguments that no assertion here looks at."""

    def __init__(self, suite, task, run, turn, is_stop, action_json="{}"):
        self.suite_id, self.task_id, self.run_id, self.turn_idx = suite, task, run, turn
        self.is_stop, self.action_json = is_stop, action_json


def _ask(q: str) -> str:
    return json.dumps({"action": "ASK", "question": q, "rationale": ""})


# --------------------------------------------------------------- the 2x2


def test_the_shift_is_joined_on_the_state_key_not_on_the_task():
    """Two states of ONE task must not collapse into one cell.

    A join on `(suite, task)` would read these two rows as a single state and report one
    transition instead of two -- and on a corpus averaging 35 runs per task, that is most of
    the table.
    """
    v1 = [_Ex("m", "t", "r", 0, False), _Ex("m", "t", "r", 1, False)]
    an = [_Ex("m", "t", "r", 0, True), _Ex("m", "t", "r", 1, False)]
    sh = ls.shift(v1, an)
    assert (sh.ask_to_stop, sh.ask_to_ask) == (1, 1)
    assert sh.n_v1 == sh.n_variant == 2


def test_a_state_only_one_file_holds_is_its_own_bucket_and_not_a_zero():
    """The variant emits a STOP where v1 emitted nothing (no ASK cleared the floor).

    That is a real consequence of the relabelling and the commonest one after ASK -> STOP, so
    it gets a named bucket. Folding it into `ask_to_stop` would overstate the transitions;
    dropping it silently would understate the shift.
    """
    v1 = [_Ex("m", "t1", "r", 0, False)]
    an = [_Ex("m", "t1", "r", 0, False), _Ex("m", "t2", "r", 0, True)]
    sh = ls.shift(v1, an)
    assert sh.only_in_variant == 1 and sh.only_in_v1 == 0
    assert sh.ask_to_stop == 0, "a state v1 never labelled did not TRANSITION"
    assert sh.n_stop_variant == 1 and sh.n_stop_v1 == 0


def test_the_stop_share_is_recomputed_from_rows_not_read_off_a_counter():
    rows = [_Ex("m", "t", "r", i, i % 4 == 0) for i in range(8)]
    n_stop, n_rows, share = ls.stop_share(rows)
    assert (n_stop, n_rows) == (2, 8) and share == pytest.approx(0.25)


def test_within_task_distinct3_ignores_stop_rows():
    """STOP rows parse to an empty question. Counting them would let a corpus improve its
    diversity score by replacing questions with STOPs -- the exact direction this lane moves."""
    asks = [
        _Ex("m", "t", "r", i, False, _ask(f"who was the {i}th mayor of town {i}?"))
        for i in range(6)
    ]
    stops = [_Ex("m", "t", "r", 100 + i, True) for i in range(6)]
    assert ls.within_task_distinct3(asks + stops) == ls.within_task_distinct3(asks)


# --------------------------------------------------------------- the invariance claim


class _Man:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_a_moved_invariant_counter_raises_rather_than_printing():
    """ "Everything else identical" is a claim, and this is where it is checked.

    A population difference between the two files makes every number in the shift table a
    mixture of a relabelling and a missing run. Printing that and carrying on is how it would
    reach a paper.
    """
    a = _Man(n_cohort_refused=10, n_answer_leak_dropped=3)
    b = _Man(n_cohort_refused=11, n_answer_leak_dropped=3)
    with pytest.raises(AssertionError, match="label-invariant"):
        ls.unchanged_counters(a, b, ["n_cohort_refused", "n_answer_leak_dropped"])
    assert ls.unchanged_counters(a, a, ["n_cohort_refused"]) == {"n_cohort_refused": 10}


def test_manifest_deltas_reports_only_integers_that_moved_and_skips_bools():
    a = _Man(n_x=1, n_y=2, turn0_any_cohort=True, name="v1")
    b = _Man(n_x=1, n_y=5, turn0_any_cohort=False, name="variant")
    assert ls.manifest_deltas(a, b) == {"n_y": (2, 5)}


# --------------------------------------------------------------- the driver, end to end


@pytest.fixture
def synth_runs(tmp_path, monkeypatch):
    """A six-run synth corpus, written the way `rows_from_run`'s own fixtures write one."""
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
    from pi_eval.build.synth_build import build
    from pinq.types import Evidence
    from pinq_adapters.synth.suite import SynthSuite

    corpus, *_ = build(n_tasks=6, n_facets=2, depth=2, root=tmp_path)
    suite = SynthSuite(Path(corpus).parent)
    runs = tmp_path / "runs"
    for k, tid in enumerate(map(str, suite.task_ids())):
        units = list(suite.retriever(tid)._units)[: 2 + (k % 3)]
        held: list = []
        turns = []
        for i, u in enumerate(units):
            before = Evidence.of(tuple(held)).subset_hash
            held.append(u)
            turns.append(
                {
                    "turn_idx": i,
                    "action_kind": "ask",
                    "question": f"Retrieve record {i} for {tid}.",
                    "rationale": f"because {i}",
                    "response_text": f"answer {i}",
                    "retrieved_uids": [u.uid],
                    "new_uids": [u.uid],
                    "subset_hash_before": before,
                }
            )
        turns[-1]["subset_hash_after"] = Evidence.of(tuple(units)).subset_hash
        d = runs / f"run{k}"
        d.mkdir(parents=True)
        d.joinpath("manifest.json").write_text(
            json.dumps(
                {
                    "run_id": f"run{k}",
                    "suite_id": "synth",
                    "task_id": tid,
                    "arm_id": "inquirer_prompted",
                    "split": "train",
                    "branch_of_run_id": "parent",
                    "branch_turn_idx": len(turns) - 1,
                    "budget_cap": 8,
                    "max_turns": 16,
                }
            )
        )
        d.joinpath("turns.jsonl").write_text("\n".join(json.dumps(t) for t in turns))
        d.joinpath("ledger.jsonl").write_text(
            json.dumps({"currency": "retrieval_calls", "cumulative": float(len(turns))})
        )
        d.joinpath("status.json").write_text(
            json.dumps({"stop_reason": "policy_stop", "wall_ms": 5, "usage": {"tok_total": 50}})
        )
        d.joinpath("outcome.json").write_text(json.dumps({"answer": {"text": "some answer"}}))
    return tmp_path


def test_both_files_come_from_one_walk_and_differ_only_in_the_stop_label(synth_runs, tmp_path):
    """The driver's whole reason to exist, asserted on the artefacts it writes.

    Same rows, same states, same provenance; the STOP label is the only thing that moved, and
    every transition runs ASK -> STOP because an answer node is a required node.
    """
    from scripts.answer_node_stop.export_variants import main

    out = tmp_path / "out"
    rc = main(
        [
            "--root",
            str(synth_runs),
            "--gold-root",
            str(synth_runs / "data" / "gold"),
            "--out",
            str(out),
            "--tau",
            "0.0",
            "--sigma-j",
            "0.0",
        ]
    )
    assert rc == 0
    v1 = [json.loads(x) for x in (out / "sft.jsonl").read_text().splitlines() if x.strip()]
    an = [
        json.loads(x) for x in (out / "sft.answer_node.jsonl").read_text().splitlines() if x.strip()
    ]
    m1 = json.loads((out / "sft.manifest.json").read_text())
    m2 = json.loads((out / "sft.answer_node.manifest.json").read_text())

    assert m1["sft_stop_label"] == "done_before"
    assert m2["sft_stop_label"] == "answer_node_covered_before"
    # SAME STATES on both sides, which is what makes the shift a shift.
    key = lambda r: (r["suite_id"], r["task_id"], r["run_id"], r["turn_idx"])  # noqa: E731
    assert {key(r) for r in v1} == {key(r) for r in an}
    # ...and at least one really moved, or this asserts nothing.
    moved = [k for k in {key(r) for r in v1} if _stop(v1, k) != _stop(an, k)]
    assert moved, "the fixture must contain a state the variant relabels"
    for k in moved:
        assert _stop(v1, k) is False and _stop(an, k) is True, "ASK -> STOP is the only direction"

    report = json.loads((out / "label_shift.json").read_text())
    assert report["shift"]["stop_to_ask"] == 0
    assert report["shift"]["ask_to_stop"] == len(moved)
    assert report["v1"]["train_id_set_hash"] and report["variant"]["train_id_set_hash"]


def _stop(rows, k) -> bool:
    return next(
        bool(r["is_stop"])
        for r in rows
        if (r["suite_id"], r["task_id"], r["run_id"], r["turn_idx"]) == k
    )


def test_the_driver_runs_as_a_module_from_the_repo_root(synth_runs, tmp_path):
    """It is launched detached, by path, from a shell -- so `python -m` must work, not just an
    in-process import. A driver that only runs under pytest is a driver nobody can relaunch."""
    out = tmp_path / "out2"
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.answer_node_stop.export_variants",
            "--root",
            str(synth_runs),
            "--gold-root",
            str(synth_runs / "data" / "gold"),
            "--out",
            str(out),
            "--tau",
            "0.0",
            "--sigma-j",
            "0.0",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO / "src"), "HOME": str(tmp_path)},
    )
    assert p.returncode == 0, p.stderr[-3000:]
    assert "label shift (state level" in p.stdout
    assert (out / "sft.answer_node.jsonl").exists()


# --------------------------------------------------------------- the reading driver


def test_an_arm_spec_without_a_model_id_is_refused():
    """`arm_id` alone cannot name a checkpoint.

    Every trained checkpoint runs under `arm_id == "inquirer_trained"`, so selecting on it
    alone pools two checkpoints into one arm and reports their MEAN as a result -- the exact
    failure `artifacts/gate/README.md` documents for a shared parquet.
    """
    import argparse

    from scripts.answer_node_stop.read_contrasts import parse_arm

    assert parse_arm("base:inquirer_prompted:qwen3-8b-base") == (
        "base",
        "inquirer_prompted",
        "qwen3-8b-base",
    )
    for bad in ("inquirer_trained", "a:b", "a::c", ":b:c", "a:b:"):
        with pytest.raises(argparse.ArgumentTypeError, match="label:arm_id:model_id"):
            parse_arm(bad)


def test_a_bound_within_a_hundredth_of_zero_is_flagged_for_the_50k_reread():
    """The campaign's own rule, applied by the instrument instead of by hand.

    A bound at -0.004 and a bound at -0.40 are both "excludes zero"; only the first is a
    number whose sign can move under a different resampling seed.
    """
    from scripts.answer_node_stop.read_contrasts import _near_zero

    assert _near_zero({"lo": -0.004, "hi": 0.21}) is True
    assert _near_zero({"lo": 0.0, "hi": 0.9}) is True
    assert _near_zero({"lo": -0.40, "hi": -0.05}) is False
    assert _near_zero({"lo": None, "hi": None}) is False


def test_the_reading_driver_imports_and_parses_its_own_usage_line():
    """It is run once, by hand, hours after the sweep finishes. An import error or a typo in a
    flag name discovered THEN costs the whole reading; discovered here it costs nothing."""
    from scripts.answer_node_stop.read_contrasts import build_parser

    a = build_parser().parse_args(
        [
            "--parquet",
            "/tmp/iso",
            "--corpora-root",
            "/tmp/corpora",
            "--arm",
            "answernode:inquirer_trained:qwen3-8b-sft-headline-answernode",
            "--arm",
            "headline:inquirer_trained:qwen3-8b-sft-headline",
            "--arm",
            "base:inquirer_prompted:qwen3-8b-base",
            "--focus",
            "answernode",
            "--out",
            "/tmp/out.json",
        ]
    )
    assert a.grid_name == "tier1_trained_qa_base" and a.n_boot == 10000
    assert [x[0] for x in a.arm] == ["answernode", "headline", "base"]
