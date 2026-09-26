"""THE OBJECTIVE IS A CONFIG FIELD, AND AN OBJECTIVE TRL WILL NOT RUN MUST NOT REACH THE GPU.

`DPOConfig.loss_type` was a single string validated against `("sigmoid", "kto_pair")`. Both
halves of that were wrong on the library rung 2 actually runs:

  * MEASURED on the installed trl 1.13.0 -- the loss ladder in `trl/trainer/dpo_trainer.py`,
    lines 1472-1589 -- `kto_pair` IS NOT DISPATCHED. It was removed. A run configured with it
    passed validation, preflighted, loaded an 8B base and two adapters, and only then hit
    `raise ValueError(f"Unknown loss type: ...")` inside the loss. That is the
    `max_prompt_length` failure again: a rename discovered in the most expensive place.
  * the same TRL takes a LIST (`loss_type: list[str]`, `dpo_config.py:241`) plus `loss_weights`,
    which is how MPO-style mixtures -- a preference term with an `sft` NLL anchor -- are
    spelled. A single string cannot express the arm at all.

WHAT THIS FILE PINS. The accepted names are the ones the installed TRL dispatches; the three
new fields are in `cfg.sha`; and the SINGLE-ELEMENT DEFAULT HASHES WHAT THE OLD STRING HASHED,
so no rung-2 run already trained is renamed by this change. That last one is the reason the sha
below is a literal: it was measured at the pre-change commit (f3f3f0f) and pasted here, which is
the only form of that assertion that can fail.
"""

from __future__ import annotations

import pytest

from pinq_train.rung2_dpo import (
    DEFAULT_LOSS_TYPE,
    LOSS_TYPES,
    DPOConfig,
    installed_loss_types,
)

# MEASURED at f3f3f0f, before `loss_type` became a tuple:
#   DPOConfig(base_model="Qwen/Qwen3-8B", adapter="artifacts/rung1").sha
# with `loss_type == "sigmoid"` (a str). Every rung-2 manifest written so far carries a
# `config_sha` computed by that code, so this is the identity the new default must reproduce.
PRE_CHANGE_DEFAULT_SHA = "63dcf89f421d0284772c20270d92103e2438a3dde6e6b25525dd9a7b814e1048"


def _cfg(**over) -> DPOConfig:
    """The same config whose sha was measured pre-change. The default reference (rung 1's
    frozen adapter) needs no merge, so nothing here re-tests the merge path."""
    return DPOConfig(base_model="Qwen/Qwen3-8B", adapter="artifacts/rung1", **over)


# ---- identity


def test_the_default_objective_hashes_exactly_what_the_old_string_hashed():
    """A knob added later at its absent default must render the bytes the older code rendered.
    Otherwise this commit renames every finished rung-2 run, and the tables that cite those
    `config_sha`s stop joining -- the failure `SHA_OMIT_WHEN_NONE` exists for in rung 1."""
    cfg = _cfg()
    assert cfg.loss_type == DEFAULT_LOSS_TYPE == ("sigmoid",)
    assert cfg.loss_weights is None and cfg.ld_alpha is None
    assert cfg.sha == PRE_CHANGE_DEFAULT_SHA


def test_one_loss_named_two_ways_is_one_run():
    """`loss_type="sigmoid"` and `loss_type=("sigmoid",)` are the same objective, so they are
    the same identity. This is what licenses the rendering rule above; it is not a special case
    for the default value."""
    assert _cfg(loss_type=("sigmoid",)).sha == _cfg(loss_type="sigmoid").sha


@pytest.mark.parametrize(
    "over",
    [
        {"loss_type": ("ipo",)},
        {"loss_type": ("sigmoid", "sft")},
        {"loss_type": ("sigmoid", "sft"), "loss_weights": (1.0, 0.5)},
        {"ld_alpha": 0.5},
        {"ld_alpha": 1.0},
    ],
    ids=["ipo", "mixture", "weights", "ld_alpha", "ld_alpha_noop"],
)
def test_every_objective_knob_moves_the_config_sha(over):
    """Two arms that train different objectives must not claim one identity. `ld_alpha=1.0` is
    in the list on purpose: it is a numerical no-op, and "the knob was considered and set to a
    no-op" is still a different run from "the knob did not exist"."""
    assert _cfg(**over).sha != PRE_CHANGE_DEFAULT_SHA


def test_the_weights_are_part_of_the_identity_not_a_presentation_detail():
    a = _cfg(loss_type=("sigmoid", "sft"), loss_weights=(1.0, 0.5))
    b = _cfg(loss_type=("sigmoid", "sft"), loss_weights=(1.0, 0.25))
    assert a.sha != b.sha


def test_the_order_of_a_mixture_is_part_of_the_identity():
    """`loss_weights` is zipped with `loss_type` positionally by TRL, so the two orderings are
    two different weightings of the same two terms."""
    assert _cfg(loss_type=("sigmoid", "sft")).sha != _cfg(loss_type=("sft", "sigmoid")).sha


# ---- what the installed library will actually run


def test_the_accepted_names_are_the_ones_the_installed_trl_dispatches():
    """The set is not a guess and not a copy of the docstring: it is read off the loss ladder.

    Skipped where trl is not installed (the laptop venv runs every other test in this file);
    run it in the training venv, which is the one whose library rung 2 would use.
    """
    trl = pytest.importorskip("trl", reason="trl lives in .venv-train, not the laptop venv")
    import inspect

    from trl.trainer.dpo_trainer import DPOTrainer

    dispatched = installed_loss_types(inspect.getsource(inspect.getmodule(DPOTrainer)))
    assert dispatched, f"could not read a loss dispatch out of trl {trl.__version__}"
    assert set(LOSS_TYPES) == dispatched, (
        "the names this config accepts and the names the installed trl dispatches have "
        f"diverged: accepted-only {sorted(set(LOSS_TYPES) - dispatched)}, "
        f"dispatched-only {sorted(dispatched - set(LOSS_TYPES))}"
    )


def test_the_names_are_read_from_the_ladder_and_not_from_the_error_message():
    """The parser, without trl. The `else:` arm of the same ladder lists every name inside one
    f-string in SINGLE quotes -- a regex that caught those would report a set that is right by
    coincidence and would keep being right after the ladder itself changed."""
    src = (
        '            if loss_type == "sigmoid":\n'
        "                pass\n"
        '            elif loss_type == "sft":\n'
        "                pass\n"
        "            else:\n"
        '                raise ValueError(f"Unknown loss type: {loss_type}. Should be one of '
        "['sigmoid', 'hinge', 'ipo']\")\n"
    )
    assert installed_loss_types(src) == {"sigmoid", "sft"}


def test_a_source_with_no_ladder_reads_as_no_claim_rather_than_as_an_empty_set():
    """An unreadable source must not be mistaken for "this TRL supports nothing": the caller
    distinguishes the two, and refusing every name on a library we could not parse would be a
    wrong refusal, which is as bad as a wrong acceptance."""
    assert installed_loss_types("") == frozenset()


# ---- refusals


def test_kto_pair_is_refused_because_the_installed_trl_no_longer_dispatches_it():
    """The old validator accepted exactly this value. On trl 1.13.0 it is an 8B load, two
    adapter loads and a preflight before `Unknown loss type` is raised inside the loss."""
    assert "kto_pair" not in LOSS_TYPES
    with pytest.raises(ValueError, match="kto_pair"):
        _cfg(loss_type=("kto_pair",)).validate()


def test_an_unknown_loss_name_is_refused_before_anything_loads():
    with pytest.raises(ValueError, match="sigmiod"):
        _cfg(loss_type=("sigmoid", "sigmiod")).validate()


def test_an_empty_loss_type_is_refused():
    """Empty would hand TRL a zero-term sum: a loss that is the constant 0.0, a flat curve, and
    a checkpoint identical to its input under a config that names an objective."""
    with pytest.raises(ValueError, match="loss_type"):
        _cfg(loss_type=()).validate()


def test_a_repeated_loss_name_is_refused():
    """TRL zips names with weights positionally, so a duplicate is a weight silently split in
    two -- indistinguishable, in the manifest, from the mixture the operator meant to write."""
    with pytest.raises(ValueError, match="sigmoid"):
        _cfg(loss_type=("sigmoid", "sigmoid")).validate()


def test_loss_weights_must_have_one_weight_per_loss():
    """TRL raises the same refusal in `DPOConfig.__post_init__` (dpo_config.py:356). Raising it
    here means the operator sees it before the trainer imports torch, not after."""
    with pytest.raises(ValueError, match="loss_weights"):
        _cfg(loss_type=("sigmoid", "sft"), loss_weights=(1.0,)).validate()
    with pytest.raises(ValueError, match="loss_weights"):
        _cfg(loss_type=("sigmoid",), loss_weights=(1.0, 0.5)).validate()
    _cfg(loss_type=("sigmoid", "sft"), loss_weights=(1.0, 0.5)).validate()


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_ld_alpha_must_be_a_share(bad):
    """LD-DPO's alpha weights the log-probabilities of the tokens past the shared prefix: 1.0
    applies no weighting, 0.0 masks them entirely. Outside [0, 1] it is neither, and trl 1.13.0
    does not check it -- it multiplies and trains."""
    with pytest.raises(ValueError, match="ld_alpha"):
        _cfg(ld_alpha=bad).validate()
    _cfg(ld_alpha=0.0).validate()
    _cfg(ld_alpha=1.0).validate()


def test_exo_pair_without_label_smoothing_is_refused_here_not_after_the_base_loads():
    """trl 1.13.0 raises `Label smoothing must be greater than 0.0 when using 'exo_pair' loss`
    in `DPOTrainer.__init__` (dpo_trainer.py:794) -- i.e. after the merged base and both
    adapters are in memory. The same refusal costs nothing here."""
    with pytest.raises(ValueError, match="exo_pair"):
        _cfg(loss_type=("exo_pair",)).validate()
    _cfg(loss_type=("exo_pair",), label_smoothing=1e-3).validate()
