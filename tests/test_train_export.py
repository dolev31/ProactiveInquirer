"""The split firewall and the RL/SFT exporter.

Contamination here is unrecoverable and invisible in the output, so these tests are about
refusal, not about happy paths.
"""

import collections
import json
import pathlib

import pytest

from pinq.actions import STOP_ACTION_JSON
from pinq_train.export.dataset import export_pairs, export_sft, write_jsonl
from pinq_train.split import (
    EVAL_ONLY_SUITES,
    SplitViolation,
    assert_trainable,
    bucket,
    id_set_hash,
    split_of,
)


def _row(
    suite="musique", task="t1", turn=0, value=1.0, action='{"q":"a"}', state="S", run="r", **over
):
    # `scorer_hash`/`graph_version` are NOT decoration here. `export_sft` refuses a row that
    # cannot name the instrument that scored it (AGENTS.md rule 1), because the alternative --
    # defaulting them to "" -- produces a row that LOOKS provenanced to every consumer
    # downstream. A fixture without them is not a smaller real row, it is an invalid one.
    r = {
        "suite_id": suite,
        "task_id": task,
        "run_id": run,
        "turn_idx": turn,
        "state_text": state,
        "action_json": action,
        "value": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
    }
    # `**over` so a test can state the GOLD side of a row -- `done_before` above all, which
    # decides the STOP label and cannot be expressed by any of the positional knobs.
    r.update(over)
    return r


def _train_task(suite="musique"):
    """Find a task id that genuinely lands in the train bucket."""
    for i in range(500):
        t = f"t{i}"
        if split_of(suite, t) == "train":
            return t
    raise AssertionError("no train bucket found")


# ------------------------------------------------------------------ split


def test_the_eval_only_suites_are_the_five_transfer_targets():
    """They are the zero-shot transfer targets; training on them destroys the claim.

    RENAMED AND RE-PINNED. The old name (`test_tau2_and_pare_are_eval_only`) and the old
    literal both encoded "there are exactly these two", which was a fact about the day it was
    written rather than about the design. `userbench` is the third: it is the only benchmark
    in this project that its authors and not we designed, and upstream ships 2,651 TRAIN
    scenarios next to the 471 test ones, so training on it is easy and the resulting number
    would still read as an external validation.

    `frames` is the fourth, and its reason is the plainest of the four: FRAMES ships a single `test` split of 824 questions and there is no train split upstream at all. Declaring it rather than relying on that absence is the point -- "upstream ships no train split" is a fact about today's dataset card, and this is a fact about the code.

    RENAMED AND RE-PINNED AGAIN, from `..._four_...`. `drgym` is the fifth, decided by the
    user on 2026-09-15: DeepResearchGym is an evaluation benchmark and a training row drawn
    from it is contamination of the suite that carries the horizontal endpoint. It is the one
    addition that cost nothing to make, and the reason is worth writing down rather than
    inferring: MEASURED, no shipped training file holds a drgym row -- `data/rl/*.jsonl` grep
    to 0 and `data/rl/sft.manifest.json` names only musique, strategyqa, synth and wiki2 --
    and the registry already declared `mines_training_data=False`, so this changes no dataset
    and only makes the refusal a property of the code. Which is exactly the point: "we did not
    happen to mine it" is a fact about the last export, and this is a fact about every one to
    come.
    """
    # Literal updated 2026-09-23 for `musique_x2` (composed pairs of held-out MuSiQue TEST
    # tasks): the belief that was wrong is "there are five", nothing this test is for moved.
    assert EVAL_ONLY_SUITES == {"tau2", "pare", "userbench", "frames", "drgym", "musique_x2"}
    assert split_of("tau2", "task_001") == "test"
    with pytest.raises(SplitViolation, match="eval-only"):
        assert_trainable("tau2", "task_001")


# Real drgym task ids -- the benchmark's own numeric query ids -- that runs.parquet stamps
# `split=train`, with the bucket the hash gives each. MEASURED 2026-09-15 over the 57 drgym
# tasks carrying train-stamped runs: all 57 bucket <= 59, so the stamp was the hash's own
# answer and not a grid override. These three are named because a task id that hashes OUT of
# train would make the test below pass for the wrong reason -- it would be refused as
# "not 'train'" whether or not the by-name guard existed.
DRGYM_WAS_TRAIN = {"112903": 36, "170195": 10, "252876": 0}


def test_drgym_is_test_in_every_bucket_and_untrainable_by_name():
    """The fifth eval-only suite, through the real functions rather than the constant.

    NOT FOLDED INTO THE TEST ABOVE. That one pins the set's CONTENTS; a set literal goes on
    being true if `split_of` stops short-circuiting on membership, which is the failure that
    would actually leak.

    THE BUCKET IS ASSERTED FIRST, and that is the whole design of this test. `assert_trainable`
    has two refusals and they are indistinguishable from the outside: by name, and by split.
    Pinning that each id still hashes into the train bucket is what makes the raise below mean
    "because it is drgym" rather than "because this id happens to hash to test" -- the second
    is an accident of one hash and would survive the by-name guard being deleted.
    """
    for task, want in sorted(DRGYM_WAS_TRAIN.items()):
        assert bucket("drgym", task) == want, (
            f"drgym/{task} no longer hashes to {want}; pick another train-bucket id or this "
            f"test stops proving the refusal is by name"
        )
        assert split_of("drgym", task) == "test", task
        with pytest.raises(SplitViolation, match="eval-only"):
            assert_trainable("drgym", task)


def test_template_id_binds_near_duplicates_to_one_side():
    """Two instantiations of one template must not straddle the boundary."""
    a = split_of("musique", "inst_a", template_id="tpl7")
    b = split_of("musique", "inst_b", template_id="tpl7")
    assert a == b
    assert bucket("musique", "x", "tpl7") == bucket("musique", "y", "tpl7")


def test_bucketing_is_deterministic_and_spread():
    buckets = [bucket("musique", f"t{i}") for i in range(400)]
    assert buckets == [bucket("musique", f"t{i}") for i in range(400)], "must be deterministic"
    assert len(set(buckets)) > 50, "hash must actually spread across buckets"
    splits = {split_of("musique", f"t{i}") for i in range(400)}
    assert splits == {"train", "dev", "test"}


def test_id_set_hash_is_order_independent():
    assert id_set_hash(["b", "a"]) == id_set_hash(["a", "b", "a"])
    assert id_set_hash(["a"]) != id_set_hash(["a", "b"])


# ------------------------------------------------------------------ SFT export


def test_exporter_refuses_eval_only_suites_and_counts_the_refusals():
    """Refusals are COUNTED, not swallowed: a silently empty export looks like success.

    `drgym` added 2026-09-15, covered the same way tau2 is: by NAME in the refusal key, not
    merely by an empty export. An empty export is what a row that never arrived looks like
    too, and the two must not read alike in the manifest.

    AND THE DRGYM ROW USES A TRAIN-BUCKET ID ON PURPOSE. The manifest key is
    `SplitViolation:<suite>` for BOTH of `assert_trainable`'s refusals, so a drgym id that
    hashed to test would put "drgym" in `refused` with or without the by-name guard -- this
    test passed at the pre-change sha when it was first written that way, which is the exact
    shape of a test that cannot fail. `112903` is a real drgym task carrying train-stamped
    runs and still hashes to bucket 36, so the only thing that can refuse it is the name.
    """
    rows = [
        _row(suite="tau2", task="task_001"),
        _row(suite="pare", task="s1"),
        _row(suite="drgym", task="112903"),
    ]
    ex, man = export_sft(rows, margin_threshold=0.0)
    assert ex == []
    assert sum(man.refused.values()) == 3
    assert any("tau2" in k for k in man.refused)
    assert any("drgym" in k for k in man.refused)


def test_exporter_refuses_dev_and_test_rows():
    dev = next(f"t{i}" for i in range(500) if split_of("musique", f"t{i}") != "train")
    ex, man = export_sft([_row(task=dev)], margin_threshold=0.0)
    assert ex == [] and sum(man.refused.values()) == 1


def test_a_below_margin_winner_on_an_UNFINISHED_task_has_no_target():
    """WAS `test_a_below_margin_winner_is_labelled_stop`, whose docstring read: "'Not
    measurably better than not asking' is exactly what the policy should learn."

    THE BELIEF THAT WAS WRONG, not the code. A tie among the SAMPLES is evidence about the
    SAMPLER, not about the state. Required evidence is still missing here; the right action is
    some question none of these candidates asked, and there is no target to teach. Labelling it
    STOP taught the policy to stop on unfinished tasks, and `stop_share` reached 0.668 that
    way. The state is dropped and counted."""
    t = _train_task()
    ex, man = export_sft([_row(task=t, value=0.05, done_before=False)], margin_threshold=0.20)
    assert ex == [] and man.n_no_target_dropped == 1


def test_a_below_margin_winner_on_a_FINISHED_task_is_labelled_stop():
    """The other half of the same rule: gold says the required evidence was already in hand,
    so STOP is right here whatever the samples scored."""
    t = _train_task()
    ex, man = export_sft([_row(task=t, value=0.05, done_before=True)], margin_threshold=0.20)
    assert len(ex) == 1 and ex[0].is_stop and ex[0].label_rule == "stop_done"
    assert man.n_stop_done_before_dedupe == 1
    # WAS the compact literal. Belief corrected: the STOP target was never coordinated with
    # the ASK serialiser's separators, and a DPO loss can learn that surface difference.
    assert ex[0].action_json == STOP_ACTION_JSON


def test_a_clear_winner_keeps_its_action():
    t = _train_task()
    ex, _ = export_sft([_row(task=t, value=0.9, action='{"q":"good"}')], margin_threshold=0.2)
    assert not ex[0].is_stop and ex[0].action_json == '{"q":"good"}'


def test_argmax_is_chosen_per_state():
    t = _train_task()
    rows = [
        _row(task=t, value=0.3, action='{"q":"weak"}'),
        _row(task=t, value=0.9, action='{"q":"strong"}'),
    ]
    ex, _ = export_sft(rows, margin_threshold=0.1)
    assert len(ex) == 1 and ex[0].action_json == '{"q":"strong"}'


def test_manifest_records_the_id_set_hash():
    """A checkpoint must be provable against the ids it actually saw."""
    t = _train_task()
    _, man = export_sft([_row(task=t, value=1.0)], margin_threshold=0.0)
    assert man.train_id_set_hash and man.n_examples == 1 and man.suites == ["musique"]


# ------------------------------------------------------------------ preference pairs


def test_pairs_come_from_the_same_state_only():
    """Within-state contrast cancels V(s_t); cross-state pairs would reintroduce it."""
    t = _train_task()
    rows = [
        _row(task=t, turn=0, value=0.9, action='{"q":"a"}'),
        _row(task=t, turn=1, value=0.1, action='{"q":"b"}'),
    ]
    pairs, _ = export_pairs(rows, margin_threshold=0.1)
    assert pairs == [], "different turns are different states"


def test_pairs_require_a_margin_above_the_noise_floor():
    t = _train_task()
    rows = [
        _row(task=t, value=0.50, action='{"q":"aa"}'),
        _row(task=t, value=0.49, action='{"q":"bb"}'),
    ]
    assert export_pairs(rows, margin_threshold=0.1)[0] == []


def test_the_length_guard_blocks_a_length_confound():
    """Without it the preference model learns 'longer question wins' -- the same confound
    that makes a naive judge unusable, imported straight into the policy."""
    t = _train_task()
    rows = [
        _row(task=t, value=0.9, action='{"q":"' + "x" * 200 + '"}'),
        _row(task=t, value=0.1, action='{"q":"y"}'),
    ]
    assert export_pairs(rows, margin_threshold=0.1, len_delta_max=40)[0] == []
    pairs, _ = export_pairs(rows, margin_threshold=0.1, len_delta_max=10_000)
    assert len(pairs) == 1 and pairs[0].margin == pytest.approx(0.8)


def test_a_valid_pair_survives():
    t = _train_task()
    rows = [
        _row(task=t, value=0.9, action='{"q":"alpha"}'),
        _row(task=t, value=0.1, action='{"q":"betaa"}'),
    ]
    pairs, man = export_pairs(rows, margin_threshold=0.1)
    assert len(pairs) == 1
    assert pairs[0].chosen_json == '{"q":"alpha"}' and pairs[0].rejected_json == '{"q":"betaa"}'
    assert man.n_pairs == 1


def test_write_jsonl_emits_a_sidecar_manifest(tmp_path):
    t = _train_task()
    ex, man = export_sft([_row(task=t, value=1.0)], margin_threshold=0.0)
    p = write_jsonl(tmp_path / "train.jsonl", ex, man)
    assert p.exists()
    assert (tmp_path / "train.manifest.json").exists()
    assert p.read_text().count("\n") == 1


def test_the_two_split_functions_agree_because_there_is_only_one():
    """Regression. `pi_run.manifest.split_of` hashed h("split", suite, key) while
    `pinq_train.split.bucket` hashed sha256(f"{suite}|{key}") -- so a run was STAMPED train by
    the runner and RE-CHECKED as test by the exporter. Nothing leaked, because the exporter
    intersects, but it silently discarded 66 of 109 rows. A disagreement that only ever loses
    data is still a disagreement about which tasks are trainable, and it had to be settled
    before anything was trained and before the prereg was sealed.
    """
    from pi_run.manifest import split_of as runner_split
    from pinq_train.split import split_of as exporter_split

    disagreements = [
        t
        for t in (f"task_{i}" for i in range(500))
        if runner_split("musique", t) != exporter_split("musique", t)
    ]
    assert disagreements == [], f"{len(disagreements)} tasks classified differently"


def test_both_sides_delegate_rather_than_reimplement():
    """Structure, not behaviour: two functions that agree today can drift tomorrow."""
    import ast
    import inspect
    import textwrap

    from pi_run import manifest
    from pinq_train import split as train_split

    for fn in (manifest.split_of, train_split.split_of, train_split.bucket):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        hashing = [
            n for n in ast.walk(tree) if isinstance(n, ast.Call) and "sha256" in ast.dump(n.func)
        ]
        assert not hashing, f"{fn.__qualname__} hashes on its own instead of delegating"


def test_template_id_still_binds_near_duplicates_after_unification():
    from pinq.splitting import split_of

    assert split_of("musique", "inst_a", "tpl7") == split_of("musique", "inst_b", "tpl7")


def test_eval_only_suites_survive_the_unification():
    from pinq.splitting import EVAL_ONLY_SUITES, split_of

    # Literal updated with the additions of `userbench`, then `frames`, then `drgym`; see
    # test_the_eval_only_suites_are_the_five_transfer_targets for which belief was wrong. And
    # again on 2026-09-23 for `musique_x2`, composed only from held-out MuSiQue test tasks.
    assert EVAL_ONLY_SUITES == {"tau2", "pare", "userbench", "frames", "drgym", "musique_x2"}
    assert split_of("tau2", "task_001") == "test"
    assert split_of("pare", "s1") == "test"
    assert split_of("userbench", "hotel:2-1|flight:2-2") == "test"
    assert split_of("drgym", "201") == "test"


# --------------------------------------------------------------------------- units
#
# `margin_threshold` is `tau + 1.5 * sigma_J * sqrt(2)`, where tau is "the 35th percentile of
# pilot phi_LOO" -- so it is expressed in PHI units. It was compared against `value`, which is
# `w_phi * phi_tilde - w_red * rho - c_ret`: phi rescaled by a weight, minus a redundancy
# penalty, minus a fixed retrieval cost. Two different quantities, one comparison.


def _train_task(suite: str = "musique") -> str:
    from pinq.splitting import split_of

    return next(f"t{i}" for i in range(500) if split_of(suite, f"t{i}") == "train")


def _cand(phi: float, value: float, task: str, **over) -> dict:
    import json as _json

    r = {
        "suite_id": "musique",
        "task_id": task,
        "template_id": None,
        "run_id": "r0",
        "turn_idx": 0,
        "state_text": "s",
        "action_json": _json.dumps({"kind": "ask", "question": "q?"}),
        "value": value,
        "phi_tilde": phi,
        "scorer_hash": "sh",
        "graph_version": "v1",
    }
    r.update(over)
    return r


def _kind(example) -> str:
    import json as _json

    d = _json.loads(example.action_json)
    return str(d.get("kind") or d.get("action"))


def test_acceptance_is_decided_on_phi_not_on_the_cost_adjusted_value():
    """It comes apart in BOTH directions. With w_phi > 1 a question well below the noise floor
    clears a threshold expressed in phi; and a question genuinely above the floor is rejected
    for having also cost a retrieval."""
    from pinq_train.export.dataset import export_sft
    from pinq_train.rung1_sft.mask import margin_threshold

    t = _train_task()
    m = margin_threshold(tau=0.20, sigma_j=0.05)  # 0.3061, in phi units

    # WAS: below-phi/above-value is exported as a STOP. Belief corrected -- on an unfinished
    # task a sample that missed the floor is not a reason to stop, it is a reason to have no
    # target (see test_a_below_margin_winner_on_an_UNFINISHED_task_has_no_target). What this
    # test is ABOUT is unchanged: the accept decision reads phi, never the cost-adjusted value.
    below_phi_above_value, man = export_sft(
        [_cand(0.10, 0.60, t, done_before=False)], margin_threshold=m
    )
    assert below_phi_above_value == [] and man.n_no_target_dropped == 1, (
        "phi is under the noise floor: the question was not measurably better than not asking"
    )

    above_phi_below_value, _ = export_sft(
        [_cand(0.40, 0.05, t, done_before=False)], margin_threshold=m
    )
    assert _kind(above_phi_below_value[0]) == "ask", (
        "phi clears the floor; the retrieval it also cost belongs in the objective, not here"
    )


def test_ranking_still_uses_the_cost_adjusted_value():
    """Two questions, both above the noise floor. The one to imitate is the one whose
    cost-adjusted value is highest -- rank and accept answer different questions."""
    import json as _json

    from pinq_train.export.dataset import export_sft
    from pinq_train.rung1_sft.mask import margin_threshold

    t = _train_task()
    m = margin_threshold(tau=0.20, sigma_j=0.05)
    cheap = _cand(0.40, 0.90, t, action_json=_json.dumps({"kind": "ask", "question": "cheap?"}))
    dear = _cand(0.95, 0.10, t, action_json=_json.dumps({"kind": "ask", "question": "dear?"}))

    exs, _man = export_sft([dear, cheap], margin_threshold=m)
    assert _json.loads(exs[0].action_json)["question"] == "cheap?"


def test_a_row_without_phi_falls_back_to_value():
    """Rows exported before phi_tilde was carried must still export rather than crash. The
    fallback reproduces the OLD behaviour, so it is a compatibility path and not a second
    opinion about which quantity is right."""
    from pinq_train.export.dataset import export_sft
    from pinq_train.rung1_sft.mask import margin_threshold

    t = _train_task()
    m = margin_threshold(tau=0.20, sigma_j=0.05)
    old = _cand(0.0, 0.60, t)
    del old["phi_tilde"]
    exs, _man = export_sft([old], margin_threshold=m)
    assert _kind(exs[0]) == "ask"


def test_the_exporter_carries_phi_from_the_scorer():
    """A threshold in phi units is only meaningful if phi reaches the exporter."""
    import inspect

    from pi_run import cmd_train

    assert '"phi_tilde": float(v.phi_tilde)' in inspect.getsource(cmd_train)


# ------------------------------------------------ near-duplicates, measured on the REAL corpora
#
# `pinq_train/split.py` states a five-layer contamination guarantee whose layer 1 is the bucket
# function, "hashed on template_id where one exists so near-duplicate variants of the same
# template cannot straddle the boundary". Layers 2-5 are all "is this row's split == train"
# checks and cannot see a near-duplicate, so layer 1 is the only one aimed at this failure.
#
# It was inert on every trainable suite: the ONLY adapter implementing `template_id` was tau2,
# which short-circuits to eval-only in `split_of` before the hash is reached. Measured on the
# built 800-task musique corpus: 43 groups of tasks sharing a hop SET but not a hop ORDER --
# genuine paraphrases -- of which 26 straddled the train/dev/test wall.
#
# test_template_id_binds_near_duplicates_to_one_side above hand-supplies template_id="tpl7", so
# it tests the hash and would pass unchanged if template_id_of returned None forever. CLAUDE.md
# rule 2: a test that passes without the fix is not a test.


def _musique_corpus():
    hits = sorted(pathlib.Path("data/corpora/musique").glob("*/tasks.jsonl"))
    if not hits:
        pytest.skip("musique corpus not built")
    return [json.loads(line) for line in hits[0].open()]


def test_a_trainable_suite_actually_produces_a_template_id():
    """Through pi_run.manifest.template_id_of -- the real path a run is stamped by -- and not
    by passing one in."""
    from pi_run.manifest import template_id_of
    from pinq_adapters.musique.suite import MusiqueSuite

    suite = MusiqueSuite.__new__(MusiqueSuite)  # template_id reads only the id string
    got = template_id_of(suite, "4hop2__61674_57411_1952_78767")
    assert got, "the guard's whole mechanism depends on this being non-None"
    assert got == template_id_of(suite, "4hop2__57411_61674_1952_78767"), (
        "the same hops in a different order are the same question in different words"
    )


def test_no_musique_paraphrase_pair_straddles_the_split():
    """The measurement, on the corpus that will actually run."""
    from pinq.splitting import split_of
    from pinq_adapters.musique.suite import MusiqueSuite

    suite = MusiqueSuite.__new__(MusiqueSuite)
    ids = [t["id"] for t in _musique_corpus()]
    groups = collections.defaultdict(list)
    for t in ids:
        head, sep, rest = t.partition("__")
        if sep:
            groups[(head, frozenset(rest.split("_")))].append(t)
    dup = [v for v in groups.values() if len(v) > 1]
    assert dup, "the fixture is only meaningful if the corpus contains paraphrase pairs"

    straddling = [
        v for v in dup if len({split_of("musique", t, suite.template_id(t)) for t in v}) > 1
    ]
    assert not straddling, (
        f"{len(straddling)} of {len(dup)} paraphrase groups straddle: {straddling[:3]}"
    )


def test_the_split_is_still_balanced_after_grouping():
    """Binding duplicates to one side must not collapse a split into nothing."""
    from pinq.splitting import split_of
    from pinq_adapters.musique.suite import MusiqueSuite

    suite = MusiqueSuite.__new__(MusiqueSuite)
    ids = [t["id"] for t in _musique_corpus()]
    n = collections.Counter(split_of("musique", t, suite.template_id(t)) for t in ids)
    assert n["train"] > 300 and n["dev"] > 60 and n["test"] > 120, dict(n)


def test_an_unparseable_id_degrades_to_per_task_splitting():
    """A corpus revision with a different id scheme must not silently group everything into one
    bucket -- which would put the entire suite on one side of the wall."""
    from pinq_adapters.musique.suite import MusiqueSuite

    suite = MusiqueSuite.__new__(MusiqueSuite)
    assert suite.template_id("no-separator-here") is None
    assert suite.template_id("2hop__") is None
    assert suite.template_id("2hop__onlyonehop") is None
