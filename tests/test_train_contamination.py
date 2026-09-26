"""A trained arm must never be scored on a task it could have trained on.

THE STATE THIS FILE WAS WRITTEN IN. `train_id_set_hash` was computed by the exporter, written
into the ExportManifest, carried through `pi compact` into `runs.parquet`, printed by
`pi train status` -- and READ BY NOTHING. A repo-wide grep found it on the evaluation side in
exactly one place: the schema column definition. It was also never STAMPED on a run, so the
column it occupied was NULL in every row there has ever been.

That is the same failure the preregistration hit and named: `excluded_pairs`' docstring opens
"THE SEALED DOCUMENT HAD NO READER" -- a guarantee recorded, digested, frozen, and consulted
by no analysis. `pinq_train.split` lists five layers that keep the split honest and calls the
manifest hash layer 3, "so a later run can prove which ids a checkpoint saw". Nothing ever
asked it to prove anything, and nothing could have: the id list was hashed and dropped, so
there was no preimage on disk to prove anything against.

WHAT IS HERE NOW, in two independent predicates over the same rows (D7, 2026-09-15):

  * the SPLIT check, exactly as it was coded -- recomputed from `pinq.splitting` and compared
    against the stamp, `test` demanded on both sides, `TRAINED_ARMS` as the subject set;
  * the ID SET check beside it -- the exporter's `train_ids.<hash16>.txt` resolved from the
    hash a run was stamped with, and a trained-arm row whose task is in that set flagged
    whatever its split says today.

Neither subsumes the other. The split predicate is a statement about the split FUNCTION, so it
cannot see a task that was exported and has since moved to `test`; the id set cannot see a
train-split task that no export happened to contain. `reason` says which one fired.

The protection before either was entirely grid-side: `tier1_trained` sets `split: test`, and
`tests/test_grids.py` asserts every selected id is test. That is real, and it is ONE mechanism
-- which is exactly what `assert_no_gold_exposed` exists NOT to be, because "the whole point is
to survive someone widening that predicate".

The contamination it guards against is unrecoverable after the fact: if a task the policy
trained on reaches a confirmatory table, every number in that table is void and nothing in
the output says so.

Recomputed, not trusted: the run's STAMPED split is compared too, so a manifest written under
a different bucket function is caught rather than believed.
"""

from __future__ import annotations

import pytest

from pi_eval.report import TRAINED_ARMS, train_split_violations


class _Agg:
    """Just enough Agg to drive the query: it is one SQL call over `runs`."""

    def __init__(self, rows):
        self._rows = rows

    def sql(self, _q):  # the function under test filters in python, not in SQL
        return list(self._rows)


def _run(run_id, arm, task, suite="musique", split="test", tid=None, train_id_set_hash="abc123"):
    return {
        "run_id": run_id,
        "arm_id": arm,
        "suite_id": suite,
        "task_id": task,
        "template_id": tid,
        "split": split,
        "train_id_set_hash": train_id_set_hash,
    }


def test_a_trained_arm_on_a_train_task_is_a_violation() -> None:
    """The failure the check exists for."""
    # bucket() is deterministic; find a musique task id that really is train-split.
    from pinq.splitting import split_of

    train_task = next(f"t{i}" for i in range(9999) if split_of("musique", f"t{i}") == "train")
    bad = _run("r1", "inquirer_trained", train_task, split="train")
    assert train_split_violations(_Agg([bad]), ["r1"])


def test_a_trained_arm_on_a_test_task_is_clean() -> None:
    from pinq.splitting import split_of

    test_task = next(f"t{i}" for i in range(9999) if split_of("musique", f"t{i}") == "test")
    ok = _run("r1", "inquirer_trained", test_task, split="test")
    assert train_split_violations(_Agg([ok]), ["r1"]) == []


def test_an_untrained_arm_on_a_train_task_is_not_a_violation() -> None:
    """`inquirer_prompted` never trained on anything; scoring it on train tasks is legal.

    Reporting it would make the check fire on every pilot and get itself disabled.
    """
    from pinq.splitting import split_of

    train_task = next(f"t{i}" for i in range(9999) if split_of("musique", f"t{i}") == "train")
    ok = _run("r1", "inquirer_prompted", train_task, split="train")
    assert train_split_violations(_Agg([ok]), ["r1"]) == []


def test_a_stamped_split_that_disagrees_with_the_recomputed_one_is_caught() -> None:
    """Recomputed, not trusted. A manifest stamped `test` under a different bucket function
    would otherwise walk straight past a check that believed it."""
    from pinq.splitting import split_of

    train_task = next(f"t{i}" for i in range(9999) if split_of("musique", f"t{i}") == "train")
    lying = _run("r1", "inquirer_trained", train_task, split="test")  # stamp says test
    assert train_split_violations(_Agg([lying]), ["r1"])


def test_the_template_id_is_used_when_present() -> None:
    """`split_of` hashes on template_id where a suite mints one, so the check must too --
    or two instantiations of one scenario land on opposite sides of the wall.

    Uses musique, not tau2: tau2 is in EVAL_ONLY_SUITES, so `split_of` returns "test" for it
    unconditionally and no template of it can ever be train-split.
    """
    from pinq.splitting import split_of

    # A template that IS train, paired with a task id that on its own would be test -- so a
    # check that ignored template_id would pass this row.
    tid = next(f"tpl{i}" for i in range(9999) if split_of("musique", f"tpl{i}") == "train")
    task = next(f"z{i}" for i in range(9999) if split_of("musique", f"z{i}") == "test")
    bad = _run("r1", "inquirer_trained", task, suite="musique", split="train", tid=tid)
    assert train_split_violations(_Agg([bad]), ["r1"])


def test_no_runs_is_no_violations() -> None:
    assert train_split_violations(_Agg([]), []) == []


def test_the_trained_arm_list_is_not_empty() -> None:
    """A check whose subject set is empty passes by finding nothing to find."""
    assert "inquirer_trained" in TRAINED_ARMS


def test_agg_render_refuses_when_a_violation_is_present() -> None:
    """It must be a REFUSAL, not a column. Contamination is unrecoverable after publication."""
    from pi_eval.report import TrainSplitContamination

    with pytest.raises(TrainSplitContamination):
        raise TrainSplitContamination("x")


# --------------------------------------------------------------------------- widened to dev
#
# DEV IS NOT A HELD-OUT SPLIT FOR A CHECKPOINT THAT WAS SELECTED ON IT. The dev rollouts are
# what `pi train gate` reads to choose which checkpoint to promote, so a dev number for a
# trained arm is a number the selection already optimised. Publishing it is the same failure as
# publishing a train number, one step further along: the tasks were not fitted, but the choice
# of model was. The predicate therefore demands `test` on BOTH the recomputed and the stamped
# split rather than merely refusing `train`.


def test_a_dev_split_trained_arm_row_is_a_violation() -> None:
    """The gap this widening closes. A trained arm on a DEV task passed the old predicate --
    `recomputed == "train" or stamped == "train"` is false for a dev row -- so the dev gate's
    own runs could have reached a reported table with nothing saying so."""
    from pinq.splitting import split_of

    dev_task = next(f"t{i}" for i in range(9999) if split_of("musique", f"t{i}") == "dev")
    bad = _run("r1", "inquirer_trained", dev_task, split="dev")
    assert train_split_violations(_Agg([bad]), ["r1"])


def test_a_trained_arm_stamped_test_but_recomputed_dev_is_a_violation() -> None:
    """Recomputed, not trusted -- now on the dev boundary too. A manifest written under a
    different bucket function could stamp `test` on what this code calls dev."""
    from pinq.splitting import split_of

    dev_task = next(f"t{i}" for i in range(9999) if split_of("musique", f"t{i}") == "dev")
    lying = _run("r1", "inquirer_trained", dev_task, split="test")
    assert train_split_violations(_Agg([lying]), ["r1"])


def test_a_trained_arm_stamped_dev_but_recomputed_test_is_a_violation() -> None:
    """The other direction, and it must also fail: if the two disagree, one of them is wrong,
    and 'which one' is not a question a contamination check may guess at."""
    from pinq.splitting import split_of

    test_task = next(f"t{i}" for i in range(9999) if split_of("musique", f"t{i}") == "test")
    lying = _run("r1", "inquirer_trained", test_task, split="dev")
    assert train_split_violations(_Agg([lying]), ["r1"])


def test_an_untrained_arm_on_a_dev_task_is_still_not_a_violation() -> None:
    """The guard against over-widening. `inquirer_prompted` is the DEV BASELINE -- it ran 658
    dev tasks on purpose -- so making dev a violation for every arm would fire on the baseline
    the gate is built around, and a check that fires on correct behaviour gets disabled."""
    from pinq.splitting import split_of

    dev_task = next(f"t{i}" for i in range(9999) if split_of("musique", f"t{i}") == "dev")
    ok = _run("r1", "inquirer_prompted", dev_task, split="dev")
    assert train_split_violations(_Agg([ok]), ["r1"]) == []


def test_a_fixed_split_suite_is_read_by_task_id_even_when_the_row_carries_a_template_id(
    monkeypatch,
) -> None:
    """The recompute must ask `split_of` the question `split_of` answers.

    `bucket()` hashes `template_id or task_id`, so for a HASHED suite it makes no difference
    whether the template id arrives in the `task_id` position or the `template_id` one. For a
    FIXED-SPLIT suite it makes all the difference: `split_of` looks the id up in
    `FIXED_SPLITS[suite]`, which is keyed by TASK id, so a template id in that position is not
    in the table and raises `UnknownFixedSplitTask` -- the contamination check dies on a row it
    was meant to classify, and a refusal that cannot be cleared is a refusal people route
    around.

    NOT YET REACHABLE, AND THAT IS WHY IT IS A TEST AND NOT A BUG REPORT. Measured
    2026-09-15: tau2_airline is the only suite in `FIXED_SPLITS` and 0 of its 2,233 rows carry
    a `template_id`, so nothing on disk takes this path today. The next fixed-split suite, or
    the first templated tau2_airline row, takes it silently.
    """
    import pinq.splitting as splitting

    monkeypatch.setitem(splitting.FIXED_SPLITS, "fixedsuite", {"task-7": "test"})
    row = _run("r1", "inquirer_trained", "task-7", suite="fixedsuite", split="test", tid="tpl-7")
    out = train_split_violations(_Agg([row]), ["r1"])
    assert out == [], (
        "the fixed table assigns task-7 to test and the row is stamped test, so this is a "
        f"clean row; got {out}"
    )


def test_every_arm_named_in_the_checkpoint_registry_is_a_trained_arm() -> None:
    """A guard for the moment the registry is populated, not a live check today.

    `conf/checkpoints.json` keys are served MODEL IDS, not arm ids, so this intersection is
    empty while the registry is empty and the test proves nothing yet. It is here because the
    failure it catches -- someone registering a checkpoint under an arm name that
    `TRAINED_ARMS` does not list, so the contamination check skips it -- is silent, and the
    cost of the guard is one line.
    """
    from pinq_adapters.llm.checkpoints import registry
    from pinq_expt import arms

    named_arms = set(registry()) & set(arms.arm_ids())
    assert named_arms <= set(TRAINED_ARMS), (
        f"{sorted(named_arms - set(TRAINED_ARMS))} is registered as a checkpoint but is not in "
        "TRAINED_ARMS, so train_split_violations skips it entirely"
    )


# ----------------------------------------------------------- the id set, not merely the split
#
# THE SECOND LINE OF DEFENCE, AND WHY THE FIRST IS NOT ENOUGH. Everything above re-derives the
# split from `pinq.splitting` and compares it with the stamp. That catches a task that IS train
# or dev today. It cannot catch a task that was EXPORTED into a training file and has since
# moved -- a re-bucketed suite, a template_id introduced after the export was taken, an export
# taken before `EVAL_ONLY_SUITES` grew -- because afterwards `split_of` answers "test" and it is
# RIGHT: the wall is where it says it is, and the checkpoint still saw the task.
#
# The id set is the record of what was actually exported, so it answers the question the split
# function cannot: not "is this task trainable?" but "did these weights see it?". D7
# (2026-09-15) keeps the split predicate above EXACTLY as coded and adds this beside it.


@pytest.fixture(autouse=True)
def _ids_dir(monkeypatch, tmp_path):
    """The id set the rows above declare. Every fixture row carries `train_id_set_hash:
    "abc123"`, and a stamped hash whose id file cannot be found is a REFUSAL -- so the file has
    to exist for the split-side tests to be testing the split rather than the lookup.

    It lists a task none of them uses, which is the point: those tests assert on the SPLIT, and
    an id set that happened to contain their tasks would make them pass for the other reason.
    """
    d = tmp_path / "ids"
    d.mkdir()
    (d / "train_ids.abc123.txt").write_text("musique/never-used-by-any-test\n")
    monkeypatch.setenv("PI_TRAIN_IDS_DIR", str(d))
    return d


def _ids(tmp_path, stamp, ids):
    d = tmp_path / f"ids-{stamp}"
    d.mkdir(exist_ok=True)
    (d / f"train_ids.{stamp[:16]}.txt").write_text("".join(f"{i}\n" for i in sorted(set(ids))))
    return d


def _test_task(split="test"):
    from pinq.splitting import split_of

    return next(f"t{i}" for i in range(9999) if split_of("musique", f"t{i}") == split)


def test_a_trained_arm_on_an_EXPORTED_task_is_a_violation_even_when_the_split_says_test(
    tmp_path,
) -> None:
    """The gap. The split check passes -- the task really is test-split today -- and the run is
    still scored on a task the checkpoint was fitted on."""
    task = _test_task()
    row = _run("r1", "inquirer_trained", task, split="test")
    out = train_split_violations(
        _Agg([row]), ["r1"], ids_dir=_ids(tmp_path, "abc123", [f"musique/{task}"])
    )
    assert out and out[0]["reason"] == "train_id_set"


def test_an_untrained_arm_on_an_exported_task_is_not_a_violation(tmp_path) -> None:
    """Same guard as the split side: `inquirer_prompted` fitted nothing, so the id set says
    nothing about it. A check that fires on the dev baseline gets disabled."""
    task = _test_task()
    row = _run("r1", "inquirer_prompted", task, split="test")
    d = _ids(tmp_path, "abc123", [f"musique/{task}"])
    assert train_split_violations(_Agg([row]), ["r1"], ids_dir=d) == []


def test_a_trained_arm_on_a_task_the_id_set_does_not_list_is_clean(tmp_path) -> None:
    task = _test_task()
    row = _run("r1", "inquirer_trained", task, split="test")
    d = _ids(tmp_path, "abc123", ["musique/something-else"])
    assert train_split_violations(_Agg([row]), ["r1"], ids_dir=d) == []


def test_a_stamped_hash_with_no_id_file_REFUSES(tmp_path) -> None:
    """A hash without its id set cannot be checked, and an unanswerable question must not be
    answered "clean". That is precisely the failure this whole layer exists to remove: a proof
    recorded, digested, and consulted by nothing."""
    from pi_eval.report import TrainIdSetUnresolved

    row = _run("r1", "inquirer_trained", _test_task(), split="test")
    with pytest.raises(TrainIdSetUnresolved, match="abc123"):
        train_split_violations(_Agg([row]), ["r1"], ids_dir=tmp_path / "empty")


def test_a_null_hash_is_not_a_violation_and_needs_no_file(tmp_path) -> None:
    """Every run made before the stamp existed carries NULL here, and so does every run whose
    Inquirer is not a registered checkpoint. Refusing on those refuses the entire corpus."""
    row = _run("r1", "inquirer_trained", _test_task(), split="test", train_id_set_hash=None)
    assert train_split_violations(_Agg([row]), ["r1"], ids_dir=tmp_path / "empty") == []


def test_both_reasons_are_reported_on_one_row(tmp_path) -> None:
    """A trained arm on a train-split task that is ALSO in the id set fails both checks, and the
    row must say so: "which check caught it" is how anyone decides what to re-run."""
    task = _test_task("train")
    row = _run("r1", "inquirer_trained", task, split="train")
    d = _ids(tmp_path, "abc123", [f"musique/{task}"])
    out = train_split_violations(_Agg([row]), ["r1"], ids_dir=d)
    assert len(out) == 1 and out[0]["reason"] == "split+train_id_set"


def test_a_split_only_violation_is_labelled_split() -> None:
    """The existing line keeps its own name, so widening one never silently relabels the other."""
    row = _run("r1", "inquirer_trained", _test_task("train"), split="train")
    out = train_split_violations(_Agg([row]), ["r1"])
    assert out and out[0]["reason"] == "split"


def test_the_id_file_is_found_by_hash_without_being_told_the_directory(monkeypatch, tmp_path):
    """`ids_dir` is an override, not the mechanism. The exporter writes the sidecar beside the
    manifest somewhere under `data/rl/`, and the reader has to find it there from the hash alone
    -- otherwise every `pi agg` needs a flag that nobody will remember to pass."""
    task = _test_task()
    nested = tmp_path / "rl" / "headline"
    nested.mkdir(parents=True)
    (nested / "train_ids.abc123.txt").write_text(f"musique/{task}\n")
    monkeypatch.setenv("PI_TRAIN_IDS_DIR", str(tmp_path / "rl"))
    assert train_split_violations(_Agg([_run("r1", "inquirer_trained", task)]), ["r1"])


# --------------------------------------------------------------- and it reaches the operator


def test_the_agg_refusal_names_the_reason() -> None:
    """`pi agg` runs this check unconditionally and prints the offenders. A new reason that does
    not reach the printed line is a refusal the operator cannot act on."""
    from pi_run.cli import contamination_report

    txt = contamination_report(
        [
            {
                "table": "T1_primary",
                "run_id": "abc",
                "arm_id": "inquirer_trained",
                "suite_id": "musique",
                "task_id": "t7",
                "split": "test",
                "recomputed_split": "test",
                "reason": "train_id_set",
                "train_id_set_hash": "abc123",
            }
        ]
    )
    assert "ASSERTION FAILED" in txt
    assert "reason=train_id_set" in txt
    assert "abc" in txt and "musique/t7" in txt


# ------------------------------------------------- and it reaches the operator THROUGH RENDER
#
# `pi agg` ran this check on every table it built. `pi render` -- which builds THE SAME TABLES
# and is the command that writes the .tex files the paper includes -- ran nothing: `cmd_agg`
# was the only caller of `train_split_violations` in the repository. So the refusal was one
# command away from being skipped by anyone who rendered without aggregating first, and the
# artifact that actually reaches the paper was the unchecked one.
#
# The split predicate and the id-set predicate are BOTH exercised below, because after
# `ELIGIBLE` gained `AND r.split = 'test'` (2026-09-15) a row stamped `train` cannot reach a
# table at all -- so what remains for this check to catch is exactly the two things the split
# COLUMN cannot express: a stamp that disagrees with the bucket function, and a task the
# exporter shipped that the bucket function now calls test.


def _render_parquet(tmp_path, *, recomputed: str, name: str = "pq", hash_stamp: str = ""):
    """One paired suite, stamped `test` so the rows are eligible, on tasks whose RECOMPUTED
    split is `recomputed`. The C@d rows are there so `pi render` has a figure it can draw:
    without one it refuses for an unrelated reason and the exit code stops meaning anything.
    """
    from tests.test_report_render import run_row, score_row, write_parquet

    from pinq.splitting import split_of

    tasks = [f"t{i}" for i in range(9999) if split_of("musique", f"t{i}") == recomputed][:3]
    runs, scores = [], []
    for arm, val in (("inquirer_trained", 0.8), ("self_inquire", 0.4)):
        for task in tasks:
            rid = f"{arm}-{task}"
            runs.append(
                run_row(
                    rid,
                    suite_id="musique",
                    task_id=task,
                    arm_id=arm,
                    split="test",
                    train_id_set_hash=hash_stamp,
                )
            )
            scores += [
                score_row(rid, "answer_token_f1", val),
                score_row(rid, "cad#0", val),
                score_row(rid, "cad_n#0", 3.0),
            ]
    return write_parquet(tmp_path / name, runs=runs, scores=scores), tasks


def _render_argv(parquet, tmp_path, *, sub="render"):
    argv = [
        sub,
        "--parquet",
        str(parquet),
        "--tables",
        str(tmp_path / "tables"),
        "--n-boot",
        "200",
        "--n-perm",
        "200",
    ]
    if sub == "render":
        return argv + [
            "--figures",
            str(tmp_path / "figs"),
            "--table",
            "T2_arm_ladder",
            "--figure",
            "F2_coverage_at_depth",
        ]
    return argv + ["--arms"]


def _refusal(err: str) -> str:
    """The contamination block only, so two commands can be compared on what they SAY."""
    i = err.find("ASSERTION FAILED")
    return err[i:] if i >= 0 else ""


def test_render_refuses_a_trained_arm_whose_stamp_disagrees_with_the_bucket_function(
    tmp_path, capsys
):
    """THE GAP. `render_all` built the same tables `pi agg` builds and checked nothing."""
    from pi_run.cli import main

    d, _ = _render_parquet(tmp_path, recomputed="train")
    assert main(_render_argv(d, tmp_path)) == 2
    err = capsys.readouterr().err
    assert "ASSERTION FAILED: a TRAINED arm was scored on tasks it may have been fitted on." in err
    assert "reason=split" in err and "arm=inquirer_trained" in err


def test_render_refuses_a_task_the_export_shipped_even_when_the_split_says_test(
    tmp_path, capsys, monkeypatch
):
    """The id-set predicate, through render. This is the one the split column CANNOT express:
    the task is test-split today, and the checkpoint still saw it."""
    from pi_run.cli import main

    d, tasks = _render_parquet(tmp_path, recomputed="test", hash_stamp="abc123")
    ids = tmp_path / "ids"
    ids.mkdir(exist_ok=True)
    (ids / "train_ids.abc123.txt").write_text("".join(f"musique/{t}\n" for t in tasks))
    monkeypatch.setenv("PI_TRAIN_IDS_DIR", str(ids))
    assert main(_render_argv(d, tmp_path)) == 2
    assert "reason=train_id_set" in capsys.readouterr().err


def test_render_and_agg_refuse_with_the_SAME_message(tmp_path, capsys):
    """One helper, one refusal text. A second copy of this message in render would drift, and
    the operator would be told to re-run a grid in one command and something else in the other.
    """
    from pi_run.cli import main

    d, _ = _render_parquet(tmp_path, recomputed="train")
    assert main(_render_argv(d, tmp_path)) == 2
    rendered = _refusal(capsys.readouterr().err)
    assert main(_render_argv(d, tmp_path, sub="agg")) == 2
    assert _refusal(capsys.readouterr().err) == rendered != ""


def test_render_refuses_rather_than_skipping_a_hash_it_cannot_resolve(tmp_path, capsys):
    """ "I could not check" is not "I checked and it was clean", exactly as in `pi agg`: a
    stamped hash whose id file is missing stops the render instead of rendering unchecked."""
    from pi_run.cli import main

    d, _ = _render_parquet(tmp_path, recomputed="test", hash_stamp="deadbeef")
    assert main(_render_argv(d, tmp_path) + ["--ids-dir", str(tmp_path / "empty")]) == 2
    assert "CANNOT CHECK CONTAMINATION" in capsys.readouterr().err


def test_a_clean_population_still_renders(tmp_path):
    """The guard against over-refusing. A check that fires on correct behaviour is a check
    somebody disables, so the same fixture on held-out tasks has to render and exit 0."""
    from pi_run.cli import main

    d, _ = _render_parquet(tmp_path, recomputed="test")
    assert main(_render_argv(d, tmp_path)) == 0
    assert (tmp_path / "tables" / "T2_arm_ladder.tex").stat().st_size > 0


def test_render_all_reports_the_violations_it_found_rather_than_printing_them(tmp_path):
    """The rows come back on the result so ONE function owns the wording -- `cli.contamination
    _report`, which `pi agg` already uses. `render_all` is a library call; a library that
    prints to stderr cannot be composed, and a copy of the message would be a second thing to
    keep in step."""
    from pi_eval import report as rp
    from pi_run.render import render_all

    d, _ = _render_parquet(tmp_path, recomputed="train")
    agg = rp.open_agg(d, n_boot=200, n_perm=200)
    res = render_all(
        agg,
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figs",
        table_ids=["T2_arm_ladder"],
        figure_ids=[],
    )
    assert [v["reason"] for v in res.contaminated] == ["split"] * 3
    assert {v["table"] for v in res.contaminated} == {"T2_arm_ladder"}
    assert res.as_dict()["ok"] is False
