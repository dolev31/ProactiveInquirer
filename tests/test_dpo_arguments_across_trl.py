"""The `trl.DPOConfig` kwargs, against the TRL that is actually installed.

THE SAME HOLE AS RUNG 1'S, ON THE OTHER TRAINER. `rung2_dpo.train()` needs `trl`, so what it
passes to `DPOConfig` was pinned by nothing; `[train]` allows `trl>=0.9`, which spans TRL 1.0.

MEASURED, trl 1.13.0: `DPOConfig` has 137 fields and `max_prompt_length` is not one of them --
TRL 1.x dropped the separate prompt budget and kept only `max_length` plus `truncation_mode`.
`train()` passes `max_prompt_length=cfg.max_prompt_len` unconditionally, so rung 2 raised a
TypeError at config construction on the installed library.

The file already had the right instinct for `chat_template_kwargs` -- probe
`TRLDPOConfig.__dataclass_fields__` and RECORD what was found rather than assume -- and this
extends it to the second field that moved, and puts the probe somewhere a test can reach.
`chat_template_kwargs` is likewise absent on 1.13.0, so the recorded hook is `tokenizer_default`
and the tokenizer's own default governs: for Qwen3 that default EMITS thinking, which is why
serving this checkpoint must pass enable_thinking=False explicitly.
"""

from __future__ import annotations

import pytest

from pinq_train.rung2_dpo.train import DPOConfig, UnsupportedObjective, dpo_argument_kwargs

SHA = "0" * 64

# TRL 0.9-era: a separate prompt budget, a chat-template hook, and the two adapter names.
FIELDS_TRL0 = frozenset(
    {
        "output_dir",
        "beta",
        "bf16",
        "model_adapter_name",
        "ref_adapter_name",
        "loss_type",
        "label_smoothing",
        "learning_rate",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "max_prompt_length",
        "seed",
        "report_to",
        "chat_template_kwargs",
        # `trl.DPOConfig` subclasses `transformers.TrainingArguments`, so `__dataclass_fields__`
        # carries the checkpointing names on both sides of TRL 1.0. Probed anyway, like every
        # other name here: a field set is a claim about the installed library, not a guarantee.
        "save_strategy",
        "save_steps",
        "save_total_limit",
    }
)
# trl 1.13.0, verified by `dataclasses.fields(trl.DPOConfig)` in the training venv: 137 fields,
# NONE of these four among them, and two that the 0.9-era class did not have -- `loss_weights`
# and `ld_alpha` arrived WITH the list-valued `loss_type` (dpo_config.py:241-262) and are the
# multi-loss surface. A TRL that declares no `loss_weights` cannot combine losses at all, so
# that name is what says which spelling `loss_type` takes.
FIELDS_TRL1 = (
    FIELDS_TRL0
    - {
        "max_prompt_length",
        "chat_template_kwargs",
        "model_adapter_name",
        "ref_adapter_name",
    }
) | {"loss_weights", "ld_alpha"}
# The reference forward of trl 1.13.0, as `reference_hook` has to recognise it.
TRL1_SOURCE = (
    '    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):\n'
)
# The loss ladder of the same module, as `installed_loss_types` has to read it. Three arms is
# enough to test the mapping; the full set is checked against the real install in
# tests/test_rung2_objective.py.
TRL1_LOSS_LADDER = (
    '            if loss_type == "sigmoid":\n'
    '            elif loss_type == "ipo":\n'
    '            elif loss_type == "sft":\n'
)


def _cfg(**kw) -> DPOConfig:
    """A MERGED-path config. The adapter path emits two more kwargs and is covered below."""
    base = dict(
        base_model="artifacts/merged",
        adapter="artifacts/rung1",
        reference="merged",
        merge_adapter=True,
        merged_base_sha=SHA,
        max_seq_len=5120,
        max_prompt_len=4608,
    )
    return DPOConfig(**{**base, **kw})


def _adapter_cfg(**kw) -> DPOConfig:
    base = dict(
        base_model="Qwen/Qwen3-8B",
        adapter="artifacts/rung1",
        max_seq_len=5120,
        max_prompt_len=4608,
    )
    return DPOConfig(**{**base, **kw})


@pytest.mark.parametrize("fields", [FIELDS_TRL0, FIELDS_TRL1], ids=["trl0", "trl1"])
def test_no_kwarg_is_emitted_that_the_installed_dpo_config_does_not_declare(fields):
    """A kwarg TRL has no field for is a TypeError raised AFTER the merged base has been loaded
    and the pairs preflighted -- the most expensive place to discover a rename."""
    kw, _ = dpo_argument_kwargs(_cfg(), fields, trl_source=TRL1_SOURCE)
    assert not set(kw) - set(fields), f"would pass {sorted(set(kw) - set(fields))}"


@pytest.mark.parametrize("fields", [FIELDS_TRL0, FIELDS_TRL1], ids=["trl0", "trl1"])
def test_the_adapter_path_emits_no_kwarg_the_installed_dpo_config_does_not_declare(fields):
    """The same property on the default path, where two MORE version-dependent names are in
    play: trl 1.13.0 removed `model_adapter_name`/`ref_adapter_name`, so passing them is the
    `warmup_ratio` failure again -- a TypeError after the base and the adapters have loaded."""
    kw, _ = dpo_argument_kwargs(_adapter_cfg(), fields, trl_source=TRL1_SOURCE)
    assert not set(kw) - set(fields), f"would pass {sorted(set(kw) - set(fields))}"


def test_the_reference_adapter_is_named_explicitly_where_trl_takes_the_name():
    kw, hooks = dpo_argument_kwargs(_adapter_cfg(), FIELDS_TRL0, trl_source="")
    assert kw["model_adapter_name"] == "default" and kw["ref_adapter_name"] == "ref"
    assert hooks["reference_adapter_hook"] == "dpo_config_fields"


def test_where_trl_no_longer_takes_the_name_it_is_the_adapters_own_name_that_carries_it():
    """trl 1.13.0 selects `"ref"` by convention. Nothing is passed, so nothing can be passed
    wrongly -- but WHICH convention applied has to reach the manifest, because the two are not
    the same guarantee."""
    kw, hooks = dpo_argument_kwargs(_adapter_cfg(), FIELDS_TRL1, trl_source=TRL1_SOURCE)
    assert "model_adapter_name" not in kw and "ref_adapter_name" not in kw
    assert hooks["reference_adapter_hook"] == "trl_named_ref_adapter"
    assert hooks["reference_adapter"] == "ref"


def test_the_prompt_budget_is_passed_only_where_trl_still_has_one():
    """TRL 1.x has no separate prompt budget: `max_length` plus `truncation_mode` is the whole
    surface. Reporting WHICH applied is the point -- the two truncate differently, and the
    difference decides whether an over-long pair loses the head of its state or its action."""
    kw0, hooks0 = dpo_argument_kwargs(_cfg(), FIELDS_TRL0)
    assert kw0["max_prompt_length"] == 4608
    assert kw0["max_length"] == 5120
    assert hooks0["prompt_length_hook"] == "max_prompt_length"

    kw1, hooks1 = dpo_argument_kwargs(_cfg(), FIELDS_TRL1)
    assert "max_prompt_length" not in kw1
    assert kw1["max_length"] == 5120
    assert hooks1["prompt_length_hook"] == "max_length_only"


def test_the_chat_template_hook_is_probed_and_recorded_not_assumed():
    """Unchanged behaviour, moved somewhere a test can see it. On trl 1.13.0 there is no hook,
    so the tokenizer's default governs -- and Qwen3's default is to emit thinking."""
    kw0, hooks0 = dpo_argument_kwargs(_cfg(enable_thinking=False), FIELDS_TRL0)
    assert kw0["chat_template_kwargs"] == {"enable_thinking": False}
    assert hooks0["chat_template_hook"] == "chat_template_kwargs"

    kw1, hooks1 = dpo_argument_kwargs(_cfg(), FIELDS_TRL1)
    assert "chat_template_kwargs" not in kw1
    assert hooks1["chat_template_hook"] == "tokenizer_default"


@pytest.mark.parametrize("fields", [FIELDS_TRL0, FIELDS_TRL1], ids=["trl0", "trl1"])
def test_every_hyperparameter_the_config_states_reaches_trl(fields):
    """A dropped key does not raise; it trains at TRL's default under a `cfg.sha` that names
    ours. `beta` and `label_smoothing` are the two that define the objective itself."""
    cfg = _cfg(beta=0.1, label_smoothing=0.2, epochs=1, learning_rate=5e-6)
    kw, _ = dpo_argument_kwargs(cfg, fields)
    assert kw["beta"] == 0.1
    assert kw["label_smoothing"] == 0.2
    # The SPELLING of `loss_type` is version-dependent and has its own test below; what this
    # one pins is that the objective the config names is the objective TRL is asked for.
    assert "sigmoid" in kw["loss_type"]
    assert kw["learning_rate"] == 5e-6
    assert kw["num_train_epochs"] == 1
    assert kw["per_device_train_batch_size"] == cfg.per_device_batch
    assert kw["gradient_accumulation_steps"] == cfg.grad_accum
    assert kw["seed"] == cfg.seed
    assert kw["output_dir"] == cfg.out_dir
    assert kw["report_to"] == []


@pytest.mark.parametrize("fields", [FIELDS_TRL0, FIELDS_TRL1], ids=["trl0", "trl1"])
def test_rung2_asks_trl_to_checkpoint_on_a_cadence_it_chose(fields):
    """Rung 2 passed NO save strategy at all, so how often it wrote `checkpoint-N/` -- and
    therefore how much a preemption cost -- was whatever the installed TRL happened to default
    to. A run with nothing on disk cannot be resumed however the resume call is spelled.

    `save_steps` is in `cfg.sha` (it changes what the output directory contains); `resume` is
    not (it is a property of the invocation, not of the checkpoint).
    """
    kw, _ = dpo_argument_kwargs(_cfg(), fields)
    assert kw["save_strategy"] == "steps"
    assert kw["save_steps"] == 200
    assert kw["save_total_limit"] == 2
    assert dpo_argument_kwargs(_cfg(save_steps=50), fields)[0]["save_steps"] == 50


def test_a_trl_with_no_step_cadence_is_left_at_its_own_default():
    """The probing style, applied to the new keys: a name the installed class does not declare
    is a TypeError after the base and both adapters have loaded, so it is not emitted -- and
    `save_strategy="steps"` is withheld with it, since it would then name a cadence TRL cannot
    express."""
    bare = FIELDS_TRL1 - {"save_steps", "save_total_limit"}
    kw, _ = dpo_argument_kwargs(_cfg(), bare)
    assert "save_steps" not in kw and "save_total_limit" not in kw and "save_strategy" not in kw


# ---- the objective
#
# THE FIELD THAT MOVED THIS TIME. `loss_type` was a single string on both sides of this file
# until trl 1.x made it `list[str]` (dpo_config.py:241) and added `loss_weights` and `ld_alpha`
# beside it. The three failures those bring are the ones every probe here exists for:
#
#   * a name the installed TRL does not dispatch (`kto_pair`) raises INSIDE the loss, i.e.
#     after the base and both adapters are resident;
#   * `ld_alpha` on a TRL that has no such field is a TypeError at config construction;
#   * and the one that does not raise at all -- silently dropping either of them trains plain
#     DPO under a `cfg.sha` that names LD-DPO or a mixture. That is the only one of the three
#     that could reach a table.


def test_the_objective_is_spelled_the_way_the_installed_trl_declares_it():
    """`list[str]` where the multi-loss surface exists, a bare string where it does not."""
    kw1, hooks1 = dpo_argument_kwargs(_cfg(), FIELDS_TRL1, trl_source=TRL1_LOSS_LADDER)
    assert kw1["loss_type"] == ["sigmoid"]
    assert hooks1["loss_type_hook"] == "list"

    kw0, hooks0 = dpo_argument_kwargs(_cfg(), FIELDS_TRL0, trl_source=TRL1_LOSS_LADDER)
    assert kw0["loss_type"] == "sigmoid"
    assert hooks0["loss_type_hook"] == "str"


def test_a_trl_whose_loss_ladder_cannot_be_read_says_so_rather_than_claiming_a_check():
    """`trl_source` is how every probe in this file reaches the installed library. When it
    yields no ladder the names are UNVERIFIED -- which is a different fact from verified, and
    the manifest has to be able to tell them apart."""
    _, hooks = dpo_argument_kwargs(_cfg(), FIELDS_TRL1)
    assert hooks["loss_type_hook"] == "list_unchecked"


def test_a_mixture_and_its_weights_reach_trl_where_the_field_exists():
    """MPO-style: a preference term plus an `sft` NLL anchor on the chosen side, weighted."""
    cfg = _cfg(loss_type=("sigmoid", "sft"), loss_weights=(1.0, 0.5))
    kw, hooks = dpo_argument_kwargs(cfg, FIELDS_TRL1, trl_source=TRL1_LOSS_LADDER)
    assert kw["loss_type"] == ["sigmoid", "sft"]
    assert kw["loss_weights"] == [1.0, 0.5]
    assert hooks["objective_hooks"] == "loss_type,loss_weights"


def test_ld_alpha_reaches_trl_where_the_field_exists():
    kw, hooks = dpo_argument_kwargs(_cfg(ld_alpha=0.5), FIELDS_TRL1, trl_source=TRL1_LOSS_LADDER)
    assert kw["ld_alpha"] == 0.5
    assert hooks["objective_hooks"] == "ld_alpha,loss_type"


def test_an_unset_objective_knob_is_not_passed_at_all():
    """`None` is ABSENT, not a value: passing `loss_weights=None` and `ld_alpha=None` would be
    harmless on trl 1.13.0 and is still wrong to record as "this run set them"."""
    kw, hooks = dpo_argument_kwargs(_cfg(), FIELDS_TRL1, trl_source=TRL1_LOSS_LADDER)
    assert "loss_weights" not in kw and "ld_alpha" not in kw
    assert hooks["objective_hooks"] == "loss_type"


@pytest.mark.parametrize(
    "over, missing",
    [({"loss_weights": (1.0,)}, "loss_weights"), ({"ld_alpha": 0.5}, "ld_alpha")],
    ids=["loss_weights", "ld_alpha"],
)
def test_an_objective_field_the_installed_trl_lacks_is_refused_not_dropped(over, missing):
    """The silent one. Dropping the kwarg trains plain DPO and writes a manifest whose config
    says otherwise -- and every curve looks healthy. The other probes in this file may drop a
    name (`max_prompt_length`, `chat_template_kwargs`) because the run is still the run the
    config describes without it; these two change the objective, so they raise."""
    with pytest.raises(UnsupportedObjective, match=missing):
        dpo_argument_kwargs(_cfg(**over), FIELDS_TRL0, trl_source=TRL1_LOSS_LADDER)


def test_a_mixture_is_refused_on_a_trl_that_cannot_combine_losses():
    """A 0.9-era `loss_type: str` takes one name. Handing it two -- as a list it would compare
    against nothing, as a joined string a name it does not dispatch -- is a run that is not the
    mixture its `cfg.sha` claims."""
    with pytest.raises(UnsupportedObjective, match="loss_weights"):
        dpo_argument_kwargs(
            _cfg(loss_type=("sigmoid", "sft")), FIELDS_TRL0, trl_source=TRL1_LOSS_LADDER
        )


def test_a_loss_the_installed_trl_does_not_dispatch_is_refused():
    """Read off the ladder, not off a list in this repository: the point is what the library on
    the rented box will actually run."""
    with pytest.raises(UnsupportedObjective, match="discopop"):
        dpo_argument_kwargs(_cfg(loss_type=("discopop",)), FIELDS_TRL1, trl_source=TRL1_LOSS_LADDER)


def test_an_unreadable_ladder_refuses_nothing():
    """A wrong refusal is as bad as a wrong acceptance: it stops a legitimate run on a library
    this code could not parse. The hook records that nothing was checked."""
    kw, hooks = dpo_argument_kwargs(_cfg(loss_type=("discopop",)), FIELDS_TRL1)
    assert kw["loss_type"] == ["discopop"]
    assert hooks["loss_type_hook"] == "list_unchecked"
