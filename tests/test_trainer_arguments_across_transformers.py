"""The `TrainingArguments` kwargs, against the transformers that is actually installed.

WHY THIS FILE EXISTS. `_build_trainer` is the one line of rung 1 no test on a laptop could
reach -- `transformers` lives behind the `[train]` extra -- so the kwargs it passes were pinned
by nothing at all. The `[train]` extra allows `transformers>=4.42`, which spans the 5.0 break,
and transformers 5.0 removed two of the names this trainer passed:

  * `warmup_ratio` -- folded into `warmup_steps`, which is now a float: an integer is a step
    count and a value in [0, 1) is a ratio of total steps;
  * `group_by_length` -- removed outright.

MEASURED on the Mac smoke run, transformers 5.17.0, after the 0.6B model had already loaded:

    TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'

That is a rented-GPU failure: it costs a model load and lands after preflight has printed a
clean report. The remedy is not to rename the key -- `warmup_steps=0.03` on transformers 4.x is
read as a step COUNT, so a straight rename silently removes the warmup on half the supported
range. So the mapping branches on the field set the installed class declares, and this file
tests that mapping against both field sets WITHOUT importing transformers, which is the only
way the check survives in a gate venv that does not have it.
"""

from __future__ import annotations

import pytest

from pinq_train.rung1_sft.train import SFTConfig, training_argument_kwargs

# The names each side of the break declares. Trimmed to the ones this trainer passes plus the
# two that moved -- a full field list would be a copy of the library that rots on its own.
FIELDS_4X = frozenset(
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
# transformers 5.17.0, verified by `dataclasses.fields(TrainingArguments)` in the training venv.
FIELDS_5X = FIELDS_4X - {"warmup_ratio", "group_by_length"}


def _cfg(**kw) -> SFTConfig:
    return SFTConfig(base_model="Qwen/Qwen3-0.6B", tau=0.05, sigma_j=0.0, **kw)


def test_no_kwarg_is_emitted_that_the_installed_class_does_not_declare():
    """The failure this file is named for. A kwarg the dataclass has no field for is a
    TypeError at construction time, i.e. after the model load and after preflight printed a
    clean report."""
    for fields in (FIELDS_4X, FIELDS_5X):
        extra = set(training_argument_kwargs(_cfg(), fields)) - set(fields)
        assert not extra, f"would pass {sorted(extra)} to a class that does not declare it"


def test_the_warmup_is_the_same_fraction_on_both_sides_of_the_rename():
    """0.03 of the run, spelled two ways. NOT a rename: `warmup_steps=0.03` on 4.x is an
    integer step count that truncates to zero, so a straight rename would quietly train the
    4.x half of the supported range with no warmup at all."""
    cfg = _cfg()
    assert cfg.warmup_ratio == 0.03

    four = training_argument_kwargs(cfg, FIELDS_4X)
    assert four["warmup_ratio"] == 0.03
    assert "warmup_steps" not in four, "two warmup knobs at once; which one wins is the library's"

    five = training_argument_kwargs(cfg, FIELDS_5X)
    assert five["warmup_steps"] == 0.03
    assert "warmup_ratio" not in five


def test_group_by_length_is_passed_only_where_it_exists_and_only_as_off():
    """ "OFF, always" -- a length-ordered epoch is a curriculum nobody chose. 5.x has no length
    grouping to turn off, so the absent key and the False flag are the same behaviour; passing
    it anyway is a TypeError."""
    assert training_argument_kwargs(_cfg(), FIELDS_4X)["group_by_length"] is False
    assert "group_by_length" not in training_argument_kwargs(_cfg(), FIELDS_5X)


@pytest.mark.parametrize("fields", [FIELDS_4X, FIELDS_5X], ids=["transformers4", "transformers5"])
def test_every_hyperparameter_the_config_states_reaches_the_trainer(fields):
    """The mapping is where a hyperparameter goes missing without a trace: a dropped key does
    not raise, it trains at the library's default and reports the config's value."""
    cfg = _cfg(epochs=3, max_seq_len=4608, bf16=False)
    kw = training_argument_kwargs(cfg, fields)
    assert kw["learning_rate"] == cfg.learning_rate
    assert kw["num_train_epochs"] == cfg.epochs
    assert kw["per_device_train_batch_size"] == cfg.per_device_batch
    assert kw["gradient_accumulation_steps"] == cfg.grad_accum
    assert kw["seed"] == cfg.seed
    assert kw["output_dir"] == cfg.out_dir
    assert kw["bf16"] is False
    assert kw["gradient_checkpointing"] == cfg.gradient_checkpointing
    assert kw["report_to"] == []


def test_bf16_is_carried_rather_than_hard_coded():
    """`bf16` is a CONFIG field and MPS supports no bf16 autocast, so the smoke run has to be
    able to turn it off. A hard-coded True is a trainer that cannot run anywhere but CUDA."""
    assert training_argument_kwargs(_cfg(bf16=True), FIELDS_5X)["bf16"] is True
    assert training_argument_kwargs(_cfg(bf16=False), FIELDS_5X)["bf16"] is False


def test_the_sample_weight_column_survives_the_trainers_column_filter():
    """`remove_unused_columns` STRIPS `sample_weight` before `compute_loss` can read it.

    THE FAILURE, measured on the Mac smoke run (transformers 5.17.0, Qwen3-0.6B, step 0 of 19):

        File ".../rung1_sft/train.py", line 654, in compute_loss
          w = inputs.pop("sample_weight")
        KeyError: 'sample_weight'

    WHY. `_Rows` is a plain torch `Dataset`, not a `datasets.Dataset`, so `Trainer._get_dataloader`
    takes the `else` branch and wraps the collator in a `RemoveColumnsCollator` built from
    `inspect.signature(model.forward)`. `Qwen3ForCausalLM.forward` has no `sample_weight`
    parameter, so the column the weighted loss exists to read is dropped between the dataset and
    the trainer. `remove_unused_columns` defaults to True.

    NOT A transformers 5 REGRESSION. That branch and that default are years old, so
    `use_sample_weight=True` -- the shipped default, and the thing `accum_scale`, `weighted_loss`
    and `test_the_weights_do_not_cancel_at_a_micro_batch_of_one` were all written for -- could
    never have completed one optimiser step on any version. It would have surfaced as a KeyError
    on the rented card, after the model load, on the first micro-batch.

    OFF UNCONDITIONALLY, not just under `use_sample_weight`. `_Rows` yields exactly the keys this
    trainer put there; there is no such thing as an unused column in a hand-built dataset, so a
    filter over it can only ever subtract something deliberate.
    """
    for fields in (FIELDS_4X, FIELDS_5X):
        for weighted in (True, False):
            kw = training_argument_kwargs(_cfg(use_sample_weight=weighted), fields)
            assert kw["remove_unused_columns"] is False, (
                "the trainer would filter the dataset's columns against the model's forward "
                "signature, which does not mention sample_weight"
            )


def test_the_checkpoint_cadence_is_steps_and_comes_from_the_config():
    """`save_strategy="epoch"` wrote one checkpoint per epoch, and rung 1 is 2 epochs over
    ~44k rows at effective batch 16: a job preempted at 95% of epoch 1 had nothing on disk to
    resume from. A step cadence bounds the loss to `save_steps` steps -- 200 by default, which
    is ~3,200 rows -- and `save_total_limit` keeps the directory from growing without bound.

    IN `cfg.sha`, unlike `resume`: the cadence changes what the output directory contains.
    """
    for fields in (FIELDS_4X, FIELDS_5X):
        kw = training_argument_kwargs(_cfg(), fields)
        assert kw["save_strategy"] == "steps"
        assert kw["save_steps"] == 200
        assert kw["save_total_limit"] == 2
        assert training_argument_kwargs(_cfg(save_steps=50), fields)["save_steps"] == 50


def test_a_library_with_no_step_cadence_keeps_the_epoch_strategy_rather_than_a_bad_kwarg():
    """The probing style this file exists for, applied to the new keys. A `TrainingArguments`
    that declares no `save_steps` gets no `save_steps` -- and, since "steps" would then name a
    cadence it cannot express, no `save_strategy="steps"` either. Epoch checkpoints are worse
    than step checkpoints; a TypeError after the model has loaded is worse than both.
    """
    bare = FIELDS_5X - {"save_steps", "save_total_limit"}
    kw = training_argument_kwargs(_cfg(), bare)
    assert kw["save_strategy"] == "epoch"
    assert "save_steps" not in kw and "save_total_limit" not in kw
