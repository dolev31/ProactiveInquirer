"""WHAT `KTOConfig` REFUSES, WHAT ITS SHA COVERS, AND WHAT IT HANDS THE INSTALLED TRL.

Everything here runs without trl, torch or a GPU, which is the point: `train()` is the one
function in the rung that cannot be executed on this machine, so what it passes to the library
is pinned by a PURE probe (`kto_argument_kwargs`) rather than by the first run on a rented box.
`dpo_argument_kwargs` earned that treatment by raising `TypeError` on `max_prompt_length` after
the base had loaded; KTO's field list is a different one and is probed the same way.

MEASURED on the installed trl 1.13.0 (`.venv-train`, 2026-09-15): `KTOConfig` declares 129
fields; `beta`, `desirable_weight` and `undesirable_weight` are among them
(`trl/trainer/kto_config.py:201,208,215`); `max_prompt_length`, `label_smoothing`,
`chat_template_kwargs`, `model_adapter_name` and `ref_adapter_name` are NOT. The fake field
sets below stand for both sides of that line.

THE TWO WEIGHTS ARE IN `cfg.sha` AND THEY ARE NOT DERIVED AT TRAIN TIME. KTO's class weights
are what stops the majority class from owning the gradient, so a run whose weights were computed
from whatever rows happened to load is a run whose identity does not name its own objective.
`balanced_weights()` computes the pair, a caller pins it on the config, and `preflight` REFUSES
a config whose weights do not equalise the masses of the rows it was handed -- unless the run
declares `require_balanced_masses=False`, which is then in the sha like any other choice.
"""

from __future__ import annotations

import pytest

from pinq_train.rung1_sft.train import NotValidatedOnHardware
from pinq_train.rung2_kto import (
    ImbalancedMasses,
    KTOConfig,
    NoKTORows,
    TooFewUndesirable,
    balanced_weights,
    kto_argument_kwargs,
    preflight,
    train,
)
from pinq_train.rung2_kto.train import MASS_RATIO_TOL

# trl 1.13.0's KTOConfig, reduced to the names this module reads.
TRL_1_13 = frozenset(
    {
        "output_dir",
        "beta",
        "loss_type",
        "desirable_weight",
        "undesirable_weight",
        "learning_rate",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "bf16",
        "seed",
        "report_to",
        "save_steps",
        "save_total_limit",
    }
)
# The 0.9-era shape: a separate prompt budget, the template kwargs, and the adapter names.
TRL_0_9 = TRL_1_13 | {
    "max_prompt_length",
    "chat_template_kwargs",
    "model_adapter_name",
    "ref_adapter_name",
}
# `reference_hook` falls back to reading the trainer source when the config has no adapter
# fields. This is the line it looks for, copied from trl/trainer/kto_trainer.py:1280.
KTO_SOURCE = (
    'with use_adapter(unwrapped_model, adapter_name="ref" if "ref" in '
    "unwrapped_model.peft_config else None):"
)


def cfg(**over: object) -> KTOConfig:
    base = {"base_model": "Qwen/Qwen3-4B", "adapter": "artifacts/rung1"}
    base.update(over)
    return KTOConfig(**base)  # type: ignore[arg-type]


def kto_row(label: bool, state: str = "s1") -> dict[str, object]:
    return {
        "prompt": state,
        "completion": '{"action": "STOP"}',
        "label": label,
        "suite_id": "musique",
        "task_id": "t1",
        "run_id": "r1",
        "turn_idx": 1,
        "scorer_hash": "sc",
        "graph_version": "v1",
        "split": "train",
        "code_version": "cv",
        "arm_id": "inquirer_prompted",
        "source_kind": "sft_row" if label else "pair_rejected",
        "pair_kind": "",
        "stop_source": "",
        "candidate_run_id": "",
    }


# --------------------------------------------------------------------------- validate


def test_an_empty_base_model_is_refused() -> None:
    with pytest.raises(ValueError, match="base_model"):
        KTOConfig(adapter="artifacts/rung1").validate()


def test_an_empty_adapter_is_refused_because_the_reference_is_rung_one() -> None:
    with pytest.raises(ValueError, match="adapter"):
        KTOConfig(base_model="Qwen/Qwen3-4B").validate()


@pytest.mark.parametrize("beta", [0.0, -0.1, 1.5])
def test_beta_outside_zero_to_one_is_refused(beta: float) -> None:
    with pytest.raises(ValueError, match="beta"):
        cfg(beta=beta).validate()


@pytest.mark.parametrize("w", [0.0, -1.0])
def test_a_non_positive_class_weight_is_refused(w: float) -> None:
    with pytest.raises(ValueError, match="desirable_weight"):
        cfg(desirable_weight=w).validate()
    with pytest.raises(ValueError, match="undesirable_weight"):
        cfg(undesirable_weight=w).validate()


def test_an_empty_pair_kind_selection_is_refused() -> None:
    with pytest.raises(ValueError, match="include_pair_kinds"):
        cfg(include_pair_kinds=()).validate()


def test_an_unknown_pair_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="include_pair_kinds"):
        cfg(include_pair_kinds=("ask_ask", "ask_shrug")).validate()


def test_an_unknown_on_conflict_is_refused() -> None:
    with pytest.raises(ValueError, match="on_conflict"):
        cfg(on_conflict="sft_wins").validate()


def test_an_unknown_loss_type_is_refused() -> None:
    with pytest.raises(ValueError, match="loss_type"):
        cfg(loss_type="sigmoid").validate()


@pytest.mark.parametrize("share", [-0.1, 1.1])
def test_an_out_of_range_undesirable_floor_is_refused(share: float) -> None:
    with pytest.raises(ValueError, match="min_undesirable_share"):
        cfg(min_undesirable_share=share).validate()


def test_a_non_positive_save_cadence_is_refused() -> None:
    with pytest.raises(ValueError, match="save_steps"):
        cfg(save_steps=0).validate()


def test_the_shipped_defaults_validate() -> None:
    cfg().validate()


# --------------------------------------------------------------------------- the sha


def test_the_sha_moves_with_beta() -> None:
    assert cfg(beta=0.1).sha != cfg(beta=0.2).sha


def test_the_sha_moves_with_either_class_weight() -> None:
    base = cfg().sha
    assert cfg(desirable_weight=0.86).sha != base
    assert cfg(undesirable_weight=1.21).sha != base
    assert cfg(desirable_weight=0.86).sha != cfg(undesirable_weight=0.86).sha


def test_the_sha_moves_with_the_conflict_rule_and_the_pair_kinds() -> None:
    assert cfg(on_conflict="drop").sha != cfg(on_conflict="refuse").sha
    assert cfg(include_pair_kinds=("ask_ask",)).sha != cfg().sha


def test_the_sha_moves_with_the_balance_requirement() -> None:
    """Two runs over one file, one balanced and one not, are two experiments."""
    assert cfg(require_balanced_masses=False).sha != cfg().sha


def test_resume_is_provenance_not_identity() -> None:
    assert cfg(resume=False).sha == cfg(resume=True).sha


def test_the_sha_is_namespaced_away_from_the_dpo_rung() -> None:
    """A rung prefix, not just a field digest: two objectives must not collide in one table."""
    from dataclasses import asdict

    from pinq.ids import canon, h
    from pinq_train.rung1_sft.train import SHA_EXCLUDED

    fields = {k: v for k, v in asdict(cfg()).items() if k not in SHA_EXCLUDED}
    assert cfg().sha == h("rung2_kto", canon(fields))
    assert cfg().sha != h("rung2", canon(fields))


def test_the_sha_moves_with_the_loss_type() -> None:
    assert cfg(loss_type="kto").sha != cfg(loss_type="apo_zero_unpaired").sha


# --------------------------------------------------------------------------- the probe


def test_the_installed_trl_gets_the_three_kto_fields() -> None:
    kw, hooks = kto_argument_kwargs(
        cfg(beta=0.1, desirable_weight=0.86, undesirable_weight=1.21),
        TRL_1_13,
        trl_source=KTO_SOURCE,
    )

    assert kw["beta"] == 0.1
    assert kw["desirable_weight"] == 0.86
    assert kw["undesirable_weight"] == 1.21
    assert kw["loss_type"] == "kto"
    assert kw["max_length"] == 9_216
    assert kw["report_to"] == []


def test_the_installed_trl_has_no_prompt_budget_so_the_action_is_the_truncated_end() -> None:
    kw, hooks = kto_argument_kwargs(cfg(), TRL_1_13, trl_source=KTO_SOURCE)

    assert "max_prompt_length" not in kw
    assert hooks["truncation_hook"] == "max_length_right"
    assert hooks["chat_template_hook"] == "tokenizer_default"


def test_an_older_trl_gets_the_prompt_budget_and_the_template_kwargs() -> None:
    kw, hooks = kto_argument_kwargs(cfg(enable_thinking=False), TRL_0_9)

    assert kw["max_prompt_length"] < kw["max_length"]
    assert kw["chat_template_kwargs"] == {"enable_thinking": False}
    assert hooks["truncation_hook"] == "max_prompt_length"
    assert hooks["chat_template_hook"] == "chat_template_kwargs"


def test_an_older_trl_is_told_the_two_adapter_names_explicitly() -> None:
    kw, hooks = kto_argument_kwargs(cfg(), TRL_0_9)

    assert kw["model_adapter_name"] == "default"
    assert kw["ref_adapter_name"] == "ref"
    assert hooks["reference_adapter_hook"] == "dpo_config_fields"


def test_the_installed_trl_is_recognised_by_the_adapter_it_selects() -> None:
    _, hooks = kto_argument_kwargs(cfg(), TRL_1_13, trl_source=KTO_SOURCE)

    assert hooks["reference_adapter_hook"] == "trl_named_ref_adapter"
    assert hooks["policy_adapter"] == "default"
    assert hooks["reference_adapter"] == "ref"


def test_a_trl_with_neither_mechanism_is_refused_rather_than_run_against_the_bare_base() -> None:
    from pinq_train.rung2_dpo import UnknownReferenceHook

    with pytest.raises(UnknownReferenceHook):
        kto_argument_kwargs(cfg(), TRL_1_13, trl_source="nothing selects an adapter here")


def test_a_trl_that_declares_no_save_cadence_is_given_none() -> None:
    kw, _ = kto_argument_kwargs(cfg(), TRL_1_13 - {"save_steps"}, trl_source=KTO_SOURCE)

    assert "save_strategy" not in kw and "save_steps" not in kw


# --------------------------------------------------------------------------- preflight


def test_an_empty_dataset_is_not_a_passing_preflight() -> None:
    with pytest.raises(NoKTORows):
        preflight(cfg(), [])


def test_weights_that_do_not_equalise_the_masses_are_refused_by_default() -> None:
    rows = [kto_row(True, f"d{i}") for i in range(8)] + [kto_row(False, "u1"), kto_row(False, "u2")]

    with pytest.raises(ImbalancedMasses) as exc:
        preflight(cfg(), rows)  # 1.0 / 1.0 against an 8:2 split

    assert "0.625" in str(exc.value) and "2.5" in str(exc.value)


def test_the_balanced_pair_passes_the_same_check() -> None:
    rows = [kto_row(True, f"d{i}") for i in range(8)] + [kto_row(False, "u1"), kto_row(False, "u2")]
    w_d, w_u = balanced_weights(8, 2)
    rep = preflight(cfg(desirable_weight=w_d, undesirable_weight=w_u), rows)

    assert rep["mass_ratio"] == pytest.approx(1.0, abs=MASS_RATIO_TOL)
    assert rep["mass_desirable"] == pytest.approx(rep["mass_undesirable"])
    assert rep["mean_row_weight"] == pytest.approx(1.0)
    assert rep["config_sha"]
    assert rep["n_desirable"] == 8 and rep["n_undesirable"] == 2


def test_an_arm_may_declare_the_imbalance_and_it_lands_in_the_sha() -> None:
    rows = [kto_row(True, f"d{i}") for i in range(8)] + [kto_row(False, "u1")]
    rep = preflight(cfg(require_balanced_masses=False), rows)

    assert rep["mass_ratio"] == pytest.approx(8.0)
    assert rep["require_balanced_masses"] is False


def test_a_declared_undesirable_floor_refuses_a_dataset_that_cannot_meet_it() -> None:
    rows = [kto_row(True, f"d{i}") for i in range(9)] + [kto_row(False, "u1")]
    w_d, w_u = balanced_weights(9, 1)

    with pytest.raises(TooFewUndesirable):
        preflight(
            cfg(desirable_weight=w_d, undesirable_weight=w_u, min_undesirable_share=0.2), rows
        )


def test_the_floor_refuses_nothing_at_its_default() -> None:
    rows = [kto_row(True, f"d{i}") for i in range(9)] + [kto_row(False, "u1")]
    w_d, w_u = balanced_weights(9, 1)
    rep = preflight(cfg(desirable_weight=w_d, undesirable_weight=w_u), rows)

    assert rep["undesirable_share"] == pytest.approx(0.1)


# --------------------------------------------------------------------------- the GPU line


def test_train_refuses_without_an_acknowledgement_and_before_importing_trl() -> None:
    """The refusal comes FIRST: this suite runs in a venv that has no trl at all."""
    with pytest.raises(NotValidatedOnHardware):
        train(cfg(), rows=[kto_row(True)])
