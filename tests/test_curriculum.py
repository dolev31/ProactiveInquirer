"""The rung-1 depth curriculum: what order the supervised rows are shown in, and nothing else.

WHAT THE AXIS IS. The stratum of a row is the TARGET-NEED DEPTH of the ASK it supervises --
S0 = depth 0, S1 = depth 1, S2 = depth >= 2 -- because that is the quantity the ablation is
about: does showing shallow needs before deep ones change what the policy learns to ask. A row
whose `latent_depth` is -1 resolved no required node at all, which is not a depth; those join
S0, where they are the same "nothing deeper was needed here" case as a genuine root.

WHY STOP ROWS ARE DEALT OUT RATHER THAN READ. A STOP row supervises no question, so it has no
target-need depth; its `latent_depth` describes the STATE it stopped in. If it were read as a
depth, every STOP row in the corpus would land in one stratum and epoch 1 would front-load the
STOP share -- and a curriculum arm would then differ from its null in the ordering AND in when
the STOP rows arrive, which is two interventions reported as one. So they are dealt at random,
in proportion to each stratum's ASK mass, which is the assignment under which every stratum
carries the global STOP share exactly. `test_every_stratum_carries_the_global_stop_share`
pins that on an unbalanced corpus, where an equal three-way split does NOT.

NOTHING HERE TOUCHES A GPU. `order_examples` is a pure function over plain dicts, which is the
whole design: the ordering is the intervention, so the intervention is tested on a laptop. The
one thing a laptop cannot run -- `Trainer.train()` -- is reached through the doubles in
`tests/test_resume_from_checkpoint.py`, which is what pins that the trainer is actually handed
the sequence this file computes.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

import pytest

from pinq.actions import STOP_ACTION_JSON, ask_action_json
from pinq_train.rung1_sft.curriculum import (
    CURRICULA,
    SHUFFLED,
    assign_strata,
    order_examples,
)
from pinq_train.rung1_sft.train import SFTConfig, training_argument_kwargs

SUITES = ("musique", "strategyqa", "hotpotqa")


def _ask(i: int, depth: int) -> dict:
    return {
        "suite_id": SUITES[i % len(SUITES)],
        # TWO QUESTIONS PER TASK, like `test_resume_from_checkpoint._dataset`: the diversity
        # gate `preflight` runs measures distinct-3 WITHIN a task, which is undefined at one.
        "task_id": f"t{i // 2}",
        "split": "train",
        "state_text": f"TASK: settle claim {i}. EVIDENCE RETRIEVED SO FAR: none.",
        "action_json": ask_action_json(f"which {i} author revised the {i} charter?"),
        "is_stop": False,
        "latent_depth": depth,
        "sample_weight": 1.0,
    }


def _stop(i: int) -> dict:
    return {
        "suite_id": SUITES[i % len(SUITES)],
        "task_id": f"t{i // 2}",
        "split": "train",
        "state_text": f"TASK: settle claim {i}. EVIDENCE RETRIEVED SO FAR: three documents.",
        "action_json": STOP_ACTION_JSON,
        "is_stop": True,
        # A TRAP, ON PURPOSE. A STOP row's `latent_depth` is a property of the state it stopped
        # in, not of a need it asked for. If the curriculum ever read it, all 1,200 STOP rows
        # below would land in S2 and `test_every_stratum_carries_the_global_stop_share` would
        # fail by ~0.4 rather than by a rounding error.
        "latent_depth": 4,
        "sample_weight": 1.0,
    }


def _corpus() -> list[dict]:
    """3,000 rows: 1,800 ASK deliberately UNBALANCED across the three strata, 1,200 STOP.

    S0 = 1,000 (900 at depth 0 plus 100 at the -1 sentinel), S1 = 500, S2 = 300. The imbalance
    is the point: under an equal three-way deal of the STOP rows S0 would be 400/1,400 = 28.6%
    STOP against a global 40%, so a test on a balanced corpus could not tell the two
    assignments apart.
    """
    rows = [_ask(i, 0) for i in range(900)]
    rows += [_ask(900 + i, -1) for i in range(100)]
    rows += [_ask(1000 + i, 1) for i in range(500)]
    rows += [_ask(1500 + i, 2) for i in range(200)]
    rows += [_ask(1700 + i, 3) for i in range(100)]
    rows += [_stop(1800 + i) for i in range(1200)]
    return rows


# --------------------------------------------------------------------------- exposure


@pytest.mark.parametrize("curriculum", CURRICULA)
def test_every_row_is_shown_exactly_once_per_epoch(curriculum):
    """EXPOSURE-MATCHED, which is what makes the arms comparable at all. A curriculum that
    drops S2 in epoch 1, or that shows S0 twice, is not an ordering intervention -- it is a
    different dataset, and the difference in the checkpoint would be unattributable."""
    rows = _corpus()
    for epochs in (1, 2, 3):
        order = order_examples(rows, curriculum, seed=0, epochs=epochs)
        assert len(order) == epochs * len(rows)
        counts = Counter(order)
        assert set(counts) == set(range(len(rows))), "some row never appears"
        assert set(counts.values()) == {epochs}, f"{Counter(counts.values())} appearances"


# --------------------------------------------------------------------------- the order


def _strata_sequence(rows: list[dict], order: list[int], seed: int = 0) -> list[int]:
    strata = assign_strata(rows, seed)
    return [strata[i] for i in order]


def test_the_first_epoch_climbs_the_strata_and_the_rest_do_not():
    """Epoch 1 of `depth_easy_first` is S0, then S1, then S2, each shuffled within itself;
    epochs 2..n are fully shuffled. The tail matters as much as the head: a curriculum that
    repeated its ordering every epoch would confound "shallow first" with "shallow N times"."""
    rows = _corpus()
    n = len(rows)
    order = order_examples(rows, "depth_easy_first", seed=0, epochs=2)
    seq = _strata_sequence(rows, order)

    first, second = seq[:n], seq[n:]
    assert first == sorted(first), "epoch 1 is not ordered by stratum"
    assert Counter(first) == Counter(assign_strata(rows, 0)), "epoch 1 lost or gained rows"
    assert second != sorted(second), "epoch 2 is ordered by stratum; it must be shuffled"


def test_hard_first_is_the_same_schedule_reversed():
    """The two arms differ in the DIRECTION of the first epoch and in nothing else -- same
    strata, same STOP deal, same exposure -- so a difference between them is the direction."""
    rows = _corpus()
    n = len(rows)
    seq = _strata_sequence(rows, order_examples(rows, "depth_hard_first", seed=0, epochs=2))
    assert seq[:n] == sorted(seq[:n], reverse=True), "epoch 1 is not S2, S1, S0"
    assert seq[n:] != sorted(seq[n:], reverse=True)


def test_the_null_arm_is_shuffled_in_every_epoch_and_is_never_the_identity():
    """`shuffled` is the headline recipe and the null of this ablation. If it were the file's
    own order it would be a curriculum too -- the exporter writes rows grouped by run, so the
    identity permutation shows one task's whole trajectory before the next one's."""
    rows = _corpus()
    n = len(rows)
    order = order_examples(rows, SHUFFLED, seed=0, epochs=2)
    assert order[:n] != list(range(n)), "the first epoch is the file's own order"
    assert order[n:] != list(range(n))
    assert order[:n] != order[n:], "the same permutation twice is one shuffle, not two epochs"
    seq = _strata_sequence(rows, order)
    assert seq[:n] != sorted(seq[:n]), "the null arm is ordered by depth: it is a curriculum"


# --------------------------------------------------------------------------- the confounds


@pytest.mark.parametrize("curriculum", ["depth_easy_first", "depth_hard_first"])
def test_every_stratum_carries_the_global_stop_share(curriculum):
    """THE CONFOUND THIS DESIGN EXISTS TO REMOVE. A stratum whose STOP share is not the file's
    is an epoch-1 block that trains at a different STOP share from the one that follows it, and
    the arm would then be reported as an ordering effect while being a STOP-share schedule.

    +/-0.02 of the global share. The deal is proportional to each stratum's ASK mass, so the
    only error is the integer remainder; an equal three-way deal would put S0 at 0.286 and S2
    at 0.571 on this corpus, which is what the tolerance is set to catch.
    """
    rows = _corpus()
    strata = assign_strata(rows, 0)
    global_share = sum(1 for r in rows if r["is_stop"]) / len(rows)
    assert global_share == pytest.approx(0.4)
    for s in (0, 1, 2):
        block = [r for r, st in zip(rows, strata) if st == s]
        share = sum(1 for r in block if r["is_stop"]) / len(block)
        assert share == pytest.approx(global_share, abs=0.02), (
            f"stratum {s}: {share:.4f} STOP against the file's {global_share:.4f}"
        )
    # And the ordering really is the assignment above, not a second one computed inside.
    order = order_examples(rows, curriculum, seed=0, epochs=1)
    assert Counter(strata[i] for i in order) == Counter(strata)


def test_the_suites_interleave_inside_a_stratum():
    """DOMAIN IS NOT AN ORDERING KEY. Rows of every suite are shuffled together inside a
    stratum, so a run of one suite is what a uniform shuffle gives and not a block.

    MEASURED at seed 0 over epoch 1 of `depth_easy_first` (3,000 rows, three suites dealt
    round-robin): the longest single-suite run is 7, against the log_3(3000) ~= 7.3 a uniform
    shuffle predicts. The bound is 15 -- a suite-ordered epoch would show runs in the hundreds.
    """
    rows = _corpus()
    n = len(rows)
    order = order_examples(rows, "depth_easy_first", seed=0, epochs=1)
    suites = [rows[i]["suite_id"] for i in order[:n]]

    longest, run = 1, 1
    for a, b in zip(suites, suites[1:]):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    assert longest <= 15, f"the longest single-suite run is {longest}: the suites are blocked"

    # Every stratum still holds every suite -- an interleaving that lost a suite from a block
    # would pass the run-length check above by having nothing to run.
    strata = assign_strata(rows, 0)
    for s in (0, 1, 2):
        assert {r["suite_id"] for r, st in zip(rows, strata) if st == s} == set(SUITES)


def test_an_unknown_depth_joins_the_shallow_stratum():
    """`latent_depth == -1` means no required node was resolved at this turn. It is a SENTINEL
    and not a depth, so it cannot be ordered against 0..4; it joins S0, where it is the same
    "nothing deeper was needed" case as a root. A missing key is the same case."""
    rows = [_ask(0, -1), _ask(1, 0), _ask(2, 1), _ask(3, 2), _ask(4, 4)]
    del rows[0]["latent_depth"]
    assert assign_strata(rows, 0) == [0, 0, 1, 2, 2]


# --------------------------------------------------------------------------- determinism


def test_the_order_is_a_function_of_the_seed():
    """The training seed, and nothing else. An ordering that moved between two runs of one
    recipe would make the arm unreproducible while every recorded field still matched."""
    rows = _corpus()
    a = order_examples(rows, "depth_easy_first", seed=0, epochs=2)
    assert a == order_examples(rows, "depth_easy_first", seed=0, epochs=2)
    assert a != order_examples(rows, "depth_easy_first", seed=1, epochs=2)
    assert assign_strata(rows, 0) == assign_strata(rows, 0)
    assert assign_strata(rows, 0) != assign_strata(rows, 1)


def test_an_unknown_curriculum_is_refused_rather_than_silently_shuffled():
    """A typo would otherwise train the null arm under a `cfg.sha` that names an ablation."""
    with pytest.raises(ValueError, match="curriculum"):
        order_examples(_corpus(), "easy_first", seed=0, epochs=1)
    with pytest.raises(ValueError, match="curriculum"):
        order_examples(_corpus(), "", seed=0, epochs=1)


# --------------------------------------------------------------- what the trainer is handed


def _cfg(**kw) -> SFTConfig:
    return SFTConfig(base_model="Qwen/Qwen3-0.6B", tau=0.05, sigma_j=0.0, **kw)


# The `TrainingArguments` field set of the installed transformers (5.17.0 in the training venv)
# plus the two names 4.x still declares, exactly as `test_trainer_arguments_across_transformers`
# spells them -- so this file tests the mapping without importing the library either.
_ALL_FIELDS = frozenset(
    {
        "output_dir",
        "learning_rate",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "warmup_ratio",
        "warmup_steps",
        "seed",
        "logging_steps",
        "save_strategy",
        "save_steps",
        "save_total_limit",
        "report_to",
        "bf16",
        "gradient_checkpointing",
        "group_by_length",
        "remove_unused_columns",
    }
)


def test_the_curriculum_moves_the_epoch_count_into_the_dataset_and_nothing_else():
    """HOW THE EPOCH BOUNDARY MAPS ONTO THE SEQUENCE. `Trainer` reshuffles its sampler every
    epoch, so a curriculum cannot survive as an epoch loop. The sequence is materialised as one
    dataset of `epochs x N` rows and the trainer makes ONE pass over it, which is why
    `num_train_epochs` is 1 under a curriculum and `cfg.epochs` without one.

    THE LR SCHEDULE IS THE SAME SCHEDULE. `max_steps = num_update_steps_per_epoch *
    num_train_epochs` with `num_update_steps_per_epoch = len(dataloader) // grad_accum`, so
    2 x floor(N/16) against floor(2N/16): equal whenever floor(N/16) is exact and otherwise
    within `epochs - 1` steps of it. Warmup is a RATIO of that total and is untouched, so the
    two arms warm up over the same fraction of the same run.
    """
    fields = set(training_argument_kwargs(_cfg(), _ALL_FIELDS))
    null = training_argument_kwargs(_cfg(epochs=2), _ALL_FIELDS)
    assert null["num_train_epochs"] == 2

    for curriculum in ("depth_easy_first", "depth_hard_first"):
        kw = training_argument_kwargs(_cfg(epochs=2, curriculum=curriculum), _ALL_FIELDS)
        assert kw["num_train_epochs"] == 1
        assert set(kw) == fields, "a curriculum changed WHICH kwargs are passed"
        # Everything the schedule is made of is byte-identical to the null arm's.
        for key in ("learning_rate", "warmup_ratio", "seed", "save_steps", "save_strategy"):
            assert kw[key] == null[key], f"{key} moved with the curriculum"


def test_group_by_length_is_still_off_under_a_curriculum():
    """The one curriculum nobody chose stays off. `group_by_length` orders an epoch by
    EVIDENCE SIZE, which correlates with task difficulty -- a second, unrecorded ordering
    intervention layered on top of the one being measured."""
    for curriculum in CURRICULA:
        kw = training_argument_kwargs(_cfg(curriculum=curriculum), _ALL_FIELDS)
        assert kw["group_by_length"] is False


def _run_rung1(tmp_path, monkeypatch, rows, **over):
    """The real `train()` over the fake torch/peft/transformers, as in test_sft_weighting."""
    import json

    from pinq_train.rung1_sft.train import train as rung1_train
    from test_resume_from_checkpoint import _FakeTokenizer, _install_fake_train_libraries

    tmp_path.mkdir(parents=True, exist_ok=True)
    ds = tmp_path / "sft.jsonl"
    ds.write_text("".join(json.dumps(r) + "\n" for r in rows))
    seen = _install_fake_train_libraries(monkeypatch)
    out = rung1_train(
        _cfg(
            dataset=str(ds),
            out_dir=str(tmp_path / "rung1"),
            chat_template=False,
            **over,
        ),
        acknowledge_untested=True,
        tokenizer=_FakeTokenizer(),
    )
    return out, seen


def _small_corpus() -> list[dict]:
    """60 rows that the REAL `preflight` accepts: 36 ASK (30 is its floor to measure diversity
    at all) and 24 STOP, which is 40% and under the mode-collapse ceiling."""
    rows = [_ask(i, i % 3) for i in range(36)]
    return rows + [_stop(100 + i) for i in range(24)]


def test_the_trainer_is_handed_the_curriculum_sequence_and_reads_it_in_order(tmp_path, monkeypatch):
    """THE WIRING, which is the half of this that a pure function cannot pin. The dataset the
    Trainer is constructed with must BE the ordered sequence, and the sampler must walk it in
    order: HF's default `RandomSampler` would reshuffle the sequence and the arm would be its
    own null with a different `cfg.sha`."""
    rows = _small_corpus()
    _, null = _run_rung1(tmp_path / "null", monkeypatch, rows, epochs=2)
    out, seen = _run_rung1(
        tmp_path / "easy", monkeypatch, rows, epochs=2, curriculum="depth_easy_first"
    )

    assert seen["n_rows"] == 2 * len(rows), "the trainer's one pass is not epochs x N rows"
    assert out["n_packed"] == len(rows), "n_packed stopped counting the file's rows"
    assert out["n_train_sequences"] == 2 * len(rows)

    sampler = list(seen["trainer"]._get_train_sampler())
    assert sampler == list(range(2 * len(rows))), (
        "the sampler does not walk the dataset in order, so the ordering is undone by the "
        "dataloader after being computed"
    )

    # WHICH ROW EACH POSITION HOLDS, recovered rather than asserted. The null run's dataset is
    # the file in file order, so its tokens identify a row; looking the curriculum run's
    # positions up in that map reconstructs the permutation the trainer was actually given,
    # which is the only way to see that the ordering survived `build_examples`.
    by_tokens = {tuple(null["train_dataset"][i]["input_ids"]): i for i in range(len(rows))}
    assert len(by_tokens) == len(rows), "two rows tokenize identically; the map is ambiguous"
    ds = seen["train_dataset"]
    assert len(ds) == 2 * len(rows)
    recovered = [by_tokens[tuple(ds[j]["input_ids"])] for j in range(len(ds))]
    assert recovered == order_examples(rows, "depth_easy_first", seed=0, epochs=2)


def test_without_a_curriculum_the_trainer_sees_exactly_what_it_saw_before(tmp_path, monkeypatch):
    """The null arm is the UNCHANGED path: N rows, `num_train_epochs = cfg.epochs`, and the
    library's own sampler. A default that quietly took the new route would re-describe every
    rung-1 run of record."""
    rows = _small_corpus()
    out, seen = _run_rung1(tmp_path, monkeypatch, rows, epochs=2)
    assert seen["n_rows"] == len(rows)
    assert seen["args"].num_train_epochs == 2
    assert out["n_train_sequences"] == len(rows)
    assert not hasattr(seen["trainer"], "_get_train_sampler"), (
        "the null arm was given the curriculum's sequential sampler"
    )


def test_the_preflight_names_the_arm_it_is_about_to_train(tmp_path):
    """`pi train rung1` without `--train` prints this report and stops. The curriculum leaves
    no trace in any other number it carries -- same rows, same weights, same STOP share -- so
    without this key a preflight of the ablation and a preflight of the null are the same
    screen, and the only thing that separates them is a sha nobody reads by eye."""
    from pinq_train.rung1_sft.train import preflight

    rows = _small_corpus()
    assert preflight(_cfg(), rows)["curriculum"] == "shuffled"
    rep = preflight(_cfg(curriculum="depth_hard_first"), rows)
    assert rep["curriculum"] == "depth_hard_first"
    # And the rest of the report is the null's, which is the point: nothing else moves.
    assert rep["stop_share"] == preflight(_cfg(), rows)["stop_share"]


# --------------------------------------------------------------------------- the identity


def test_the_curriculum_is_part_of_the_run_identity_but_its_absent_default_is_not():
    """`shuffled` is BOTH the absent default and the null arm's name -- there is no third
    state to record -- so at `shuffled` the config renders the bytes it rendered before the
    field existed and every rung-1 run already trained keeps its `config_sha`. Any other value
    changes what the checkpoint saw, in what order, and enters the sha like any other field."""
    base = _cfg()
    assert base.curriculum == SHUFFLED
    assert dataclasses.replace(base, curriculum=SHUFFLED).sha == base.sha
    easy = dataclasses.replace(base, curriculum="depth_easy_first")
    hard = dataclasses.replace(base, curriculum="depth_hard_first")
    assert easy.sha != base.sha
    assert hard.sha != base.sha
    assert easy.sha != hard.sha
