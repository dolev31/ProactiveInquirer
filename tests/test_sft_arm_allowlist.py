"""Only the prompted policy's own samples train the headline arm.

MEASURED (2026-09-11, `data/rl/sft.jsonl` joined to `scores/parquet/runs.parquet`): 35,833 SFT
rows come from `inquirer_prompted` and 8,399 from `self_inquire`, with ~250 more from the other
comparator arms; on the pairs side 7,286 ask_ask pairs are prompted-vs-prompted and 3,145 are
self_inquire-vs-self_inquire. So a quarter of the ASK supervision was a DIFFERENT policy's, and a
checkpoint trained on it has partly distilled the comparator it is later measured against.

`rows_from_run` refused only the GOLD-EXPOSED arms (`GOLD_EXPOSED_ARMS`). Scope was never a
refusal at all, and nothing on the artifact said which policies had contributed -- so the defect
was invisible in the export and in every table drawn from it.

THE ALLOWLIST IS A DEFAULT, NOT A LAW. `--include-arm` re-admits an arm by name and the manifest
records which arms were admitted and how many runs were refused, so "+ancestor rows" is one flag
and is legible from the artifact rather than from a shell history.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_run import cmd_train
from pi_run.cmd_train import DEFAULT_TRAINABLE_ARMS, ArmNotAllowed, SuiteCache, collect_rows


def _synth(tmp_path: Path):
    from pi_eval.build.synth_build import build
    from pinq_adapters.synth.suite import SynthSuite

    corpus, *_ = build(n_tasks=4, n_facets=2, depth=2, root=tmp_path)
    suite = SynthSuite(Path(corpus).parent)
    tid = str(suite.task_ids()[0])
    return suite, tid, list(suite.retriever(tid)._units)


def _turns(units):
    from pinq.types import Evidence

    rows = []
    held: list = []
    for i, u in enumerate(units):
        before = Evidence.of(tuple(held)).subset_hash
        held.append(u)
        rows.append(
            {
                "turn_idx": i,
                "action_kind": "ask",
                "question": f"Retrieve record {i}.",
                "rationale": f"because {i}",
                "response_text": f"answer {i}",
                "retrieved_uids": [u.uid],
                "new_uids": [u.uid],
                "subset_hash_before": before,
            }
        )
    return rows


def _write_run(runs: Path, run_id: str, *, task: str, turns: list, arm: str) -> Path:
    d = runs / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "suite_id": "synth",
                "task_id": task,
                "arm_id": arm,
                "split": "train",
                "template_id": None,
                "code_version": "c0ffee",
            }
        )
    )
    (d / "turns.jsonl").write_text("\n".join(json.dumps(t) for t in turns))
    (d / "ledger.jsonl").write_text(
        json.dumps({"currency": "retrieval_calls", "cumulative": float(len(turns))})
    )
    (d / "status.json").write_text(
        json.dumps({"stop_reason": "budget", "wall_ms": 5, "usage": {"tok_total": 50}})
    )
    return d


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
    _suite, tid, units = _synth(tmp_path)
    return tmp_path, tid, units, cmd_train._train("reward").RewardWeights()


def test_the_default_allowlist_is_the_prompted_policy_alone():
    """Stated as a frozenset beside GOLD_EXPOSED_ARMS, so the headline's population is one
    readable literal rather than an argparse default nobody looks at."""
    assert DEFAULT_TRAINABLE_ARMS == frozenset({"inquirer_prompted"})


def test_a_comparator_arm_yields_no_rows_and_is_counted(env):
    """`self_inquire` is 19% of the shipped SFT file. It must leave, and the export must SAY it
    left -- a silently smaller dataset reads exactly like a smaller corpus."""
    root, tid, units, weights = env
    runs = root / "runs"
    _write_run(runs, "comparator", task=tid, turns=_turns(units[:2]), arm="self_inquire")

    rows, skipped = collect_rows(runs, root, graph_version="v1", weights=weights)

    assert rows == []
    assert skipped["arm_not_allowed"] == 1


def test_the_prompted_policy_is_admitted_by_the_same_default(env):
    """The other half of the same rule: the default must not empty the dataset it defines."""
    root, tid, units, weights = env
    runs = root / "runs"
    _write_run(runs, "prompted", task=tid, turns=_turns(units[:2]), arm="inquirer_prompted")

    rows, skipped = collect_rows(runs, root, graph_version="v1", weights=weights)

    assert [r["run_id"] for r in rows] == ["prompted", "prompted"]
    assert "arm_not_allowed" not in skipped


def test_include_arms_readmits_the_comparator(env):
    """The ablation is one argument, and it reaches `rows_from_run` rather than being filtered
    afterwards -- a post-hoc filter cannot restore a row the walk never built."""
    root, tid, units, weights = env
    runs = root / "runs"
    _write_run(runs, "comparator", task=tid, turns=_turns(units[:2]), arm="self_inquire")

    rows, skipped = collect_rows(
        runs,
        root,
        graph_version="v1",
        weights=weights,
        include_arms=frozenset({"inquirer_prompted", "self_inquire"}),
    )

    assert [r["run_id"] for r in rows] == ["comparator", "comparator"]
    assert "arm_not_allowed" not in skipped


def test_rows_from_run_raises_arm_not_allowed_rather_than_returning_empty(env):
    """A refusal, not an empty list. `collect_rows` counts exceptions by class; a function that
    returned [] would be indistinguishable from a run with no decision points, and the skip table
    -- the only place the loss is visible -- would stay silent."""
    from pi_run.cmd_train import rows_from_run

    root, tid, units, weights = env
    runs = root / "runs"
    d = _write_run(runs, "comparator", task=tid, turns=_turns(units[:2]), arm="self_inquire")

    with pytest.raises(ArmNotAllowed, match="self_inquire"):
        rows_from_run(d, SuiteCache(root), graph_version="v1", weights=weights)


def test_the_gold_exposed_refusal_still_wins_over_the_allowlist(env):
    """ORDER IS LOAD-BEARING. `gold_evidence` is outside the allowlist too, so whichever guard
    runs first names the reason in the skip table. Gold exposure is a CONTAMINATION refusal and
    scope is a design choice; collapsing the first into the second would quietly downgrade
    'oracle distillation' to 'not in this run's arm list', and `tests/test_train_cli.py` pins the
    gold-exposed arm as `state_mismatch`."""
    root, tid, units, weights = env
    runs = root / "runs"
    _write_run(runs, "ceiling", task=tid, turns=_turns(units[:2]), arm="gold_evidence")

    rows, skipped = collect_rows(runs, root, graph_version="v1", weights=weights)

    assert rows == []
    assert skipped == {"state_mismatch": 1}, "gold exposure must not be reported as scope"


def _export(tmp_path: Path, *extra: str) -> dict:
    from pi_run.cli import build_parser

    args = build_parser().parse_args(
        [
            "train",
            "export",
            "--kind",
            "sft",
            "--root",
            str(tmp_path),
            "--runs-root",
            str(tmp_path / "runs"),
            "--out",
            str(tmp_path / "rl"),
            "--gold-root",
            str(tmp_path / "data" / "gold"),
            *extra,
        ]
    )
    assert args.fn(args) == 0
    return json.loads((tmp_path / "rl" / "sft.manifest.json").read_text())


def test_the_manifest_records_which_arms_were_admitted_and_what_that_cost(env):
    """Rule 1: a dataset that cannot name the population it was drawn from is not a dataset. The
    count of refused runs rides beside the allowlist so the cost of the decision is on the
    artifact, not in a terminal that has scrolled away."""
    root, tid, units, _ = env
    runs = root / "runs"
    _write_run(runs, "prompted", task=tid, turns=_turns(units[:2]), arm="inquirer_prompted")
    _write_run(runs, "comparator", task=tid, turns=_turns(units[:2]), arm="self_inquire")

    man = _export(root)

    assert man["included_arms"] == ["inquirer_prompted"]
    assert man["n_arm_refused"] == 1


def test_include_arm_is_recorded_on_the_manifest_too(env):
    """The ablation's artifact must be distinguishable from the headline's by reading it."""
    root, tid, units, _ = env
    runs = root / "runs"
    _write_run(runs, "prompted", task=tid, turns=_turns(units[:2]), arm="inquirer_prompted")
    _write_run(runs, "comparator", task=tid, turns=_turns(units[:2]), arm="self_inquire")

    man = _export(root, "--include-arm", "inquirer_prompted", "--include-arm", "self_inquire")

    assert man["included_arms"] == ["inquirer_prompted", "self_inquire"]
    assert man["n_arm_refused"] == 0
