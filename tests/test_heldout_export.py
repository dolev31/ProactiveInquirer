"""The dev door: a gold-side entry point that only accepts held-out runs and is not wired up,
and the wall between what trains a checkpoint and what measures it.

THE PROBLEM. 1,130 dev-split rollouts of the prompted policy and 816 dev forks are on disk, and
`pi train export` cannot see them: the scorer refuses anything that is not `train`, by design and
correctly. Without a dev export there is no dev NLL, no STOP 2x2 and no dev pair accuracy, so
model selection would have to happen on the test split -- which voids every confirmatory table.

WHY NOT A REQUEST FIELD. `ScoreRequest.split_assert` exists and is typed `Literal["train"]`, and
that was once the knob: `assert_scorable` compared the run's split against the value the CLIENT
sent, so `from_dict(ScoreRequest, {"split_assert": "test"})` was accepted end to end. The fix
fixed the threshold at "train" and the docstring states the only acceptable remedy -- "a
server-side configuration change with a reviewer attached, not a request field". This module is
that remedy, taken literally:

  * `score_heldout` is a separate FUNCTION with a distinct name, reachable only by import;
  * it is NOT registered in `pi_run.serve.app`, so no HTTP body can reach it from any trainer;
  * it shares ONE measurement body with `handle_score`, so the dev numbers are produced by the
    same instrument as the train ones rather than by a second, independently-driftable copy;
  * the trainer-facing refusal is untouched and is re-pinned here, because a change that routes
    around a firewall must prove it did not open it.

AND THE OTHER END OF THE SAME WALL: THE TRAINERS. A checkpoint that saw dev is a checkpoint whose
dev number is a training number. Nothing downstream can unlearn it: not a filter, not a re-score,
not a re-run of the eval. So the refusal lives at the last point that still knows what the rows
are -- `rung1.preflight`, before a single gradient step -- and it is a raised exception rather
than a warning, because a warning in a training log is a warning nobody reads.

WHAT THE REFUSAL DOES NOT DO, on purpose: a row carrying NO `split` key at all is waved through
and counted as `n_rows_without_split`. Rows written before the exporter stamped splits could not
have had the field, and refusing them would reject every historical dataset over a fact they never
claimed. An EMPTY split is a different thing and IS refused: it means the exporter looked and
could not tell, which is not the same as nobody having asked.

THE SAME WALL, ONE RUNG UP. The moment a dev export exists on disk it is one
`--pairs data/rl/dev/pairs.jsonl` away from being trained on, and a contaminated checkpoint is
not recoverable after the fact: every confirmatory number computed from it is void, and nothing
in the output says it happened. `split.assert_trainable` guards the EXPORTER; the rung-2 half of
this file guards the LOADER, which is the last thing that touches the file before the optimiser
does. A row without a `split` is not a violation there either: every pair in the shipped corpus
predates the field, and reading absence as "not train" would refuse the entire training set. The
dev exporter stamps `split: "dev"`, so the guard fires on exactly the file it exists for.

RUNG 1 AND RUNG 2 EACH RAISE THEIR OWN `HeldOutDataset`. They are two classes, not one, so the
two loader halves below import them under distinct names; an `except` in one rung does not
swallow the other rung's refusal, and neither test can pass on the wrong module's exception.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from pi_run import cmd_train
from pi_run.serve.score import SplitRefused, assert_scorable, handle_score, score_heldout
from pinq.actions import ask_action_json
from pinq.wire import ScoreRequest
from pinq_train.rung1_sft.train import HeldOutDataset as Rung1HeldOutDataset
from pinq_train.rung1_sft.train import SFTConfig, assert_train_split, preflight
from pinq_train.rung2_dpo import HeldOutDataset as Rung2HeldOutDataset
from pinq_train.rung2_dpo import load_pairs
from pinq_train.split import SplitViolation, assert_evaluable, assert_trainable

APP = Path(__file__).parent.parent / "src" / "pi_run" / "serve" / "app.py"
DEV_TASK = "s14"  # buckets to dev under pinq.splitting; s0..s13 do not
STOP = '{"action": "STOP"}'


# ------------------------------------------------------- the dev door (track C)


def _synth(tmp_path: Path):
    from pi_eval.build.synth_build import build
    from pinq_adapters.synth.suite import SynthSuite

    corpus, *_ = build(n_tasks=15, n_facets=2, depth=2, root=tmp_path)
    return SynthSuite(Path(corpus).parent)


def _gold_units(suite, task: str) -> list:
    """The pool units this task's gold graph marks REQUIRED, in pool order.

    Retrieving THESE is what makes phi positive, so the exported rows clear the noise floor the
    way a real rollout's do. The first few units of the pool are not required for every task --
    for s14 the required ones sit at positions 3-6 -- and a fixture built on `units[:3]` scores
    phi 0.0, exports nothing, and would prove nothing about a dev export.
    """
    from pi_eval.gold import load_graphs

    gold = {u for n in load_graphs("synth", "v1")[task].required() for u in n.gold_ev_uids}
    return [u for u in suite.retriever(task)._units if u.uid in gold]


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


def _write_run(runs: Path, run_id: str, *, task: str, split: str, turns: list, **over) -> Path:
    d = runs / run_id
    d.mkdir(parents=True)
    man = {
        "run_id": run_id,
        "suite_id": "synth",
        "task_id": task,
        "arm_id": "inquirer_prompted",
        "split": split,
        "template_id": None,
        "code_version": "3ffa972",
    }
    man.update(over)
    (d / "manifest.json").write_text(json.dumps(man))
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
    return tmp_path, _synth(tmp_path)


# --------------------------------------------------------------- the door that stays shut


def test_the_wire_still_refuses_a_client_chosen_split():
    """MUST PASS BEFORE AND AFTER. The held-out path exists precisely so this one never has to
    bend; if adding it loosened `assert_scorable`, the firewall was traded for a convenience."""
    held_out = {"suite_id": "synth", "task_id": "t", "split": "dev"}
    for asked in ("dev", "test", "all"):
        with pytest.raises(SplitRefused, match="not a value a client may ask for"):
            assert_scorable(held_out, split_assert=asked)
    with pytest.raises(SplitRefused, match="not 'train'"):
        assert_scorable(held_out, split_assert="train")


def test_the_app_never_references_the_heldout_scorer():
    """NOT REACHABLE OVER THE WIRE. `pi_run.serve.app` registers exactly four routes; a fifth
    that reached `score_heldout` would hand any trainer with a socket the held-out set, and the
    refusal that protects the experiment would be one decorator wide.

    Checked by AST rather than by starting a server: an import that only happens inside a handler
    is still an import, and a grep for the name in a string is not a reference."""
    tree = ast.parse(APP.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {a.name for a in node.names} | {a.asname or "" for a in node.names}
    assert "score_heldout" not in names
    assert "assert_heldout" not in names
    # and the check is not vacuous: the name it is looking for does exist
    assert callable(score_heldout)
    assert "handle_score" in names, "the guard only means something while /score is still wired"


def test_the_request_dataclass_offers_no_held_out_mode():
    """The remedy is a FUNCTION, not a field. A `split` on ScoreRequest would be exactly the
    design that failed: `from_dict` deserialises whatever the body contains, so a Literal is a
    type annotation and not a runtime check."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(ScoreRequest)}
    assert fields == {"episode_id", "mode", "split_assert"}


# --------------------------------------------------------------- what score_heldout refuses


def test_score_heldout_refuses_a_train_run(env):
    root, suite = env
    runs = root / "runs"
    _write_run(runs, "trainrun", task="s0", split="train", turns=_turns(_gold_units(suite, "s0")))
    with pytest.raises(SplitRefused, match="not 'dev'"):
        score_heldout(
            ScoreRequest(episode_id="trainrun", mode="prefix_ladder"),
            split="dev",
            runs_root=runs,
            graph_version="v1",
        )


def test_score_heldout_refuses_a_test_run(env):
    """The dev door must not become a test door. Selecting a checkpoint on test and then
    reporting on test is the one contamination no ablation can recover from."""
    root, suite = env
    runs = root / "runs"
    _write_run(runs, "testrun", task="s2", split="test", turns=_turns(_gold_units(suite, "s2")))
    with pytest.raises(SplitRefused, match="not 'dev'"):
        score_heldout(
            ScoreRequest(episode_id="testrun", mode="prefix_ladder"),
            split="dev",
            runs_root=runs,
            graph_version="v1",
        )


def test_score_heldout_refuses_a_split_nobody_may_ask_for(env):
    """`split` is the SERVER's parameter, but "server-side" is not "unbounded": only the two
    held-out splits exist, and `train` belongs to `handle_score` where the trainer-facing
    refusal lives."""
    root, suite = env
    runs = root / "runs"
    _write_run(
        runs, "devrun", task=DEV_TASK, split="dev", turns=_turns(_gold_units(suite, DEV_TASK))
    )
    for bad in ("train", "all", ""):
        with pytest.raises(SplitRefused, match="held-out"):
            score_heldout(
                ScoreRequest(episode_id="devrun", mode="prefix_ladder"),
                split=bad,
                runs_root=runs,
                graph_version="v1",
            )


def test_score_heldout_refuses_eval_only_gold_exposed_and_canary(env):
    """The same three refusals `assert_scorable` makes, because the reasons are unchanged by the
    split: tau2 is the transfer target whatever bucket it hashed into, and a ceiling arm's
    questions were chosen where gold is readable whether or not the task is held out."""
    from pi_run.serve.score import assert_heldout

    with pytest.raises(SplitRefused, match="eval-only"):
        assert_heldout({"suite_id": "tau2", "task_id": "x", "split": "dev"}, split="dev")
    with pytest.raises(SplitRefused, match="gold-exposed"):
        assert_heldout(
            {"suite_id": "synth", "task_id": "x", "split": "dev", "gold_exposed": True},
            split="dev",
        )
    with pytest.raises(SplitRefused, match="gold-exposed"):
        assert_heldout(
            {"suite_id": "synth", "task_id": "x", "split": "dev", "canary_hit": True}, split="dev"
        )


# --------------------------------------------------------------- one measurement, two doors


def test_the_dev_and_train_doors_share_one_measurement(env):
    """ONE instrument. Two copies of the measurement would drift, and the dev number that selects
    a checkpoint would stop being comparable with the train number that produced it -- silently,
    since both would still be floats in the right range.

    Measured behaviourally rather than by patching: the same episode, stamped `train` and stamped
    `dev`, must score identically through the two entry points on every field but the split."""
    import dataclasses

    root, suite = env
    runs = root / "runs"
    turns = _turns(_gold_units(suite, DEV_TASK))
    _write_run(runs, "asdev", task=DEV_TASK, split="dev", turns=turns)
    _write_run(runs, "astrain", task=DEV_TASK, split="train", turns=turns)

    dev = score_heldout(
        ScoreRequest(episode_id="asdev", mode="prefix_ladder"),
        split="dev",
        runs_root=runs,
        graph_version="v1",
    )
    train = handle_score(
        ScoreRequest(episode_id="astrain", mode="prefix_ladder"),
        runs_root=runs,
        graph_version="v1",
    )

    a = dataclasses.asdict(dev)
    b = dataclasses.asdict(train)
    for k in ("episode_id", "split"):
        a.pop(k), b.pop(k)
    assert a == b
    assert dev.split == "dev" and train.split == "train"
    assert dev.scorer_hash == train.scorer_hash, "one scorer_hash or the numbers are not poolable"


# --------------------------------------------------------------- assert_evaluable


def test_assert_trainable_still_refuses_a_dev_task():
    """UNCHANGED. It is layer 2 of the five the split firewall is built from, and widening it
    would be the leak this whole apparatus exists to prevent."""
    with pytest.raises(SplitViolation, match="'dev'"):
        assert_trainable("synth", DEV_TASK)


def test_assert_evaluable_asserts_membership_in_the_split_it_is_told():
    """Not "anything but train": naming the split is what stops a test task riding into a dev
    export on the grounds that it, too, is held out."""
    assert_evaluable("synth", DEV_TASK, split="dev")  # must not raise
    with pytest.raises(SplitViolation, match="not 'test'"):
        assert_evaluable("synth", DEV_TASK, split="test")
    with pytest.raises(SplitViolation, match="not 'dev'"):
        assert_evaluable("synth", "s2", split="dev")  # s2 is test


def test_assert_evaluable_refuses_eval_only_suites_in_every_split():
    """tau2, pare, userbench and drgym are the zero-shot transfer targets. `split_of` maps them
    to `test` by name, so a split check alone would ADMIT them to a test-split export; the
    refusal has to be by name, in every split, exactly as `assert_trainable` does it.

    `drgym` joined the list on 2026-09-15 by the user's decision. It is the case that most
    needs stating here rather than being left to the loop below: unlike the other three, drgym
    has runs already on disk from before the decision -- MEASURED, 201 over 57 tasks stamped
    `split=train` and 51 more over 17 tasks stamped `dev` -- so it is the only one of the five
    where a held-out export could find rows to take. Its id is therefore a REAL train-bucket
    task and not `task_001`, for the reason the frames test below spells out: a task that
    hashes out of the split being asked for is refused with or without the by-name guard."""
    from pinq.splitting import bucket

    assert bucket("drgym", "112903") == 36, (
        "drgym/112903 was chosen because it hashes into train; if that moved, this case stops "
        "proving the refusal is by name"
    )
    for suite, task in (
        ("tau2", "task_001"),
        ("pare", "task_001"),
        ("userbench", "task_001"),
        ("drgym", "112903"),
    ):
        for split in ("dev", "test"):
            with pytest.raises(SplitViolation, match="eval-only"):
                assert_evaluable(suite, task, split=split)


def test_both_held_out_doors_refuse_frames_by_name_in_every_split():
    """FRAMES IS THE FOURTH EVAL-ONLY SUITE AND THE TEST ABOVE NAMES ONLY THREE.

    The two doors landed on one branch and frames on another, so nothing had yet asked the
    question at the seam. Both of them read `pinq.splitting.EVAL_ONLY_SUITES`, so frames is
    already refused -- this pins that it stays refused, which the list above cannot do: it would
    go on passing with frames struck from the set, and the next dev export would mine the one
    suite whose entire value is that the policy never saw it.

    AND THE REFUSAL HAS TO BE BY NAME, which is the whole reason a membership check exists next
    to a split check. Every frames task hashes to `test`, so `assert_evaluable("frames", t,
    split="test")` is a task asking to be exported into the split it is already in -- a
    split-only guard says yes. Asserting that bucket first is what makes the raise below mean
    "because it is frames" rather than "because it is in the wrong split".
    """
    from pi_run.serve.score import assert_heldout
    from pinq.splitting import split_of

    task = "frames_0000"
    assert split_of("frames", task) == "test", (
        "if frames stopped bucketing to test, the test-split case below would raise for the "
        "wrong reason and this test would keep passing while the by-name guard was gone"
    )

    for split in ("dev", "test"):
        # the EXPORTER's door
        with pytest.raises(SplitViolation, match="eval-only"):
            assert_evaluable("frames", task, split=split)
        # the SCORER's door, which a dev export reaches through `score_heldout`
        with pytest.raises(SplitRefused, match="eval-only"):
            assert_heldout({"suite_id": "frames", "task_id": task, "split": split}, split=split)


# --------------------------------------------------------------- the export


def _export(root: Path, *extra: str) -> int:
    from pi_run.cli import build_parser

    args = build_parser().parse_args(
        [
            "train",
            "export",
            "--kind",
            "both",
            "--root",
            str(root),
            "--runs-root",
            str(root / "runs"),
            "--out",
            str(root / "rl"),
            "--gold-root",
            str(root / "data" / "gold"),
            "--tau",
            "0.001",
            # Skip reasons on stdout. An export that refuses every run still exits 0 and writes a
            # manifest, so a failure here is otherwise an empty file with no stated cause -- which
            # is how the stale-graph-cache bug in _graphs stayed invisible.
            "--verbose",
            *extra,
        ]
    )
    return args.fn(args)


def test_a_dev_export_writes_its_own_files_and_stamps_dev_on_every_row(env):
    """SEPARATE FILENAMES, not a flag inside one file. A training run and a validation run
    reading different bytes under one path is the failure `pairs.jsonl` vs
    `pairs.anticipation.jsonl` already exists to prevent."""
    root, suite = env
    runs = root / "runs"
    _write_run(runs, "dev1", task=DEV_TASK, split="dev", turns=_turns(_gold_units(suite, DEV_TASK)))
    _write_run(runs, "trainrun", task="s0", split="train", turns=_turns(_gold_units(suite, "s0")))

    assert _export(root, "--split", "dev") == 0

    sft = root / "rl" / "dev" / "sft.dev.jsonl"
    man = json.loads((root / "rl" / "dev" / "sft.dev.manifest.json").read_text())
    rows = [json.loads(x) for x in sft.read_text().splitlines() if x.strip()]

    assert rows, "the dev export must not be silently empty"
    assert man["split"] == "dev"
    assert all(r["split"] == "dev" for r in rows)
    assert {r["task_id"] for r in rows} == {DEV_TASK}, "the train run must not be in a dev file"
    assert (root / "rl" / "dev" / "pairs.dev.manifest.json").exists()
    assert not (root / "rl" / "sft.jsonl").exists(), "a dev export must not touch the train path"


def test_a_train_export_is_unchanged_by_the_new_flag(env):
    """The default path must keep its filenames and its refusals: every consumer on disk reads
    `data/rl/sft.jsonl`, and `ExportManifest.split` defaults to train so an old manifest and a
    new one say the same thing about the same file."""
    root, suite = env
    runs = root / "runs"
    _write_run(runs, "dev1", task=DEV_TASK, split="dev", turns=_turns(_gold_units(suite, DEV_TASK)))
    _write_run(runs, "trainrun", task="s0", split="train", turns=_turns(_gold_units(suite, "s0")))

    assert _export(root) == 0

    man = json.loads((root / "rl" / "sft.manifest.json").read_text())
    rows = [
        json.loads(x) for x in (root / "rl" / "sft.jsonl").read_text().splitlines() if x.strip()
    ]
    assert man["split"] == "train"
    assert {r["task_id"] for r in rows} == {"s0"}
    assert not (root / "rl" / "dev").exists()


def test_the_cli_offers_no_test_split_door(env):
    """`--split test` is not a choice. Checkpoint selection happens on dev; the test split is not
    touched until one checkpoint per arm has been chosen, and a flag that makes the wrong thing
    possible is a flag someone eventually passes."""
    from pi_run.cli import build_parser

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["train", "export", "--split", "test"])
    assert exc.value.code == 2


def test_the_dev_export_refuses_a_run_whose_stamp_disagrees_with_its_bucket(env):
    """The stamp is what the scorer reads and the bucket is what the exporter re-derives; a run
    stamped `dev` on a task that buckets `train` is admitted by neither, and the loss is counted
    rather than silently exported."""
    root, suite = env
    runs = root / "runs"
    _write_run(runs, "liar", task="s0", split="dev", turns=_turns(_gold_units(suite, "s0")))

    weights = cmd_train._train("reward").RewardWeights()
    rows, _skipped = cmd_train.collect_rows(
        runs, root, graph_version="v1", weights=weights, split="dev"
    )
    # the SCORER admits it (it reads the stamp, by design) ...
    assert rows and all(r["task_id"] == "s0" for r in rows)
    # ... and the EXPORTER refuses it, because s0 does not bucket to dev.
    dataset = cmd_train._train("export.dataset")
    items, man = dataset.export_sft(rows, margin_threshold=0.001, split="dev")
    assert items == [] and sum(man.refused.values()) >= 1


# --------------------------------------------------------------------------- rung 1 (track A)


def _rows(split: str | None, n: int = 40) -> list[dict]:
    """Enough varied questions to clear the diversity gate, so the ONLY thing that can fail is the
    split check. A fixture that trips two gates cannot tell you which one fired."""
    stems = ["who founded {}", "when did {} open", "where is {} located", "which firm bought {}"]
    subjects = "Arclight Bellhaven Corvid Dunmore Everly Fairwood Glenmoor Hollis".split()
    rows = []
    for i in range(n):
        row = {
            "suite_id": "musique",
            "task_id": f"t{i // 4}",
            "is_stop": False,
            "action_json": '{"action": "ASK", "question": "%s in %d?"}'
            % (stems[i % 4].format(subjects[(i // 4) % 8]), 1900 + i),
        }
        if split is not None:
            row["split"] = split
        rows.append(row)
    return rows


def test_the_sft_trainer_refuses_a_dev_dataset():
    cfg = SFTConfig(base_model="m", tau=0.05, sigma_j=0.01)

    with pytest.raises(Rung1HeldOutDataset, match="not on the train side"):
        preflight(cfg, _rows("dev"))
    with pytest.raises(Rung1HeldOutDataset, match="test"):
        preflight(cfg, _rows("test"))
    with pytest.raises(Rung1HeldOutDataset, match="<empty>"):
        preflight(cfg, _rows(""))

    # And one held-out row among many train ones is still a refusal: contamination is not a
    # proportion, it is a yes or a no.
    mixed = _rows("train") + _rows("dev", n=1)
    with pytest.raises(Rung1HeldOutDataset, match="1 of 41"):
        preflight(cfg, mixed)

    rep = preflight(cfg, _rows("train"))
    assert rep["n_examples"] == 40 and rep["n_rows_without_split"] == 0


def test_a_row_with_no_split_key_is_legacy_and_is_counted_not_refused():
    cfg = SFTConfig(base_model="m", tau=0.05, sigma_j=0.01)
    assert_train_split(_rows(None))  # does not raise
    rep = preflight(cfg, _rows(None))
    assert rep["n_rows_without_split"] == 40


# --------------------------------------------------------------------------- rung 2 (track B)


def _p(**over):
    r = {
        "suite_id": "musique",
        "task_id": "t1",
        "run_id": "p",
        "turn_idx": 1,
        "state_text": "S",
        "chosen_json": ask_action_json("who?"),
        "rejected_json": ask_action_json("when?"),
        "margin": 0.5,
        "pair_kind": "ask_ask",
    }
    r.update(over)
    return r


def _write(tmp_path, rows):
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_the_dpo_loader_refuses_a_dev_pairs_file(tmp_path):
    """Fatal, not filtered: a file whose rows are held out is not a training file with some
    bad rows in it, it is the wrong file, and dropping the offending rows would train on
    whatever remained while reporting a clean load."""
    p = _write(tmp_path, [_p(), _p(split="dev")])
    with pytest.raises(Rung2HeldOutDataset, match="dev"):
        load_pairs(p)


def test_the_dpo_loader_refuses_a_test_split_row_too(tmp_path):
    p = _write(tmp_path, [_p(split="test")])
    with pytest.raises(Rung2HeldOutDataset, match="test"):
        load_pairs(p)


def test_the_dpo_loader_accepts_rows_that_make_no_split_claim(tmp_path):
    """The entire shipped corpus predates the field."""
    rows, drops = load_pairs(_write(tmp_path, [_p(), _p(split="train")]))
    assert len(rows) == 2 and drops == {}
