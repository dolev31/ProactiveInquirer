"""WHAT RUNG 2'S REFERENCE POLICY ACTUALLY IS, AND WHY IT HAD TO BE MADE UNAMBIGUOUS.

The paper's rung-2 claim is "DPO moved the policy from the SFT checkpoint". That sentence is
only meaningful if the reference distribution `pi_ref` IS rung 1. The trainer used to do two
things at once: it called `model.load_adapter(cfg.adapter, adapter_name="default")` AND passed a
fresh `peft_config` to `DPOTrainer`. TRL computes the reference by DISABLING the adapter -- and
with two adapters loaded under one name, which one "disabled" refers to is a property of the
installed TRL and peft versions, not of anything this repository states. The docstring asserted
the favourable reading ("the reference policy is rung 1's checkpoint, loaded as the ADAPTER on
the same base"). Nothing checked it, and no loss curve could have told anyone it was wrong: a
run whose reference is the BARE BASE still trains, still reports a falling loss and a rising
reward margin, and its headline sentence is then false in a way no artifact records.

So the ambiguity is removed rather than documented. There are three ways to remove it and this
file covers all three, because `DPOConfig.reference` selects between them.

`reference="adapter"` (THE DEFAULT) is the standard two-adapter recipe: ONE base in memory, rung
1's LoRA loaded TWICE under two EXPLICIT names -- a trainable copy that DPO optimises and a
frozen copy that is the reference. Nothing is merged, so nothing can be rounded away, and
"which adapter is the reference" is answered by a name this repository writes down rather than
by a library default. It is the default because merging is LOSSY at the checkpoint's own dtype:
44.7% of rung 1's update did not survive a bf16 fold (see `test_merge_precision.py`), and the
fp32 fix costs ~32 GB for an 8B base that rung 2 then has to load.

`reference="merged"` is the older path and is kept: rung 1 is folded into the weights, the only
adapter in the process is the fresh one DPO trains, and `merged_base_sha` names the weights. It
is what a SERVING deployment wants (one fused checkpoint, no adapter at inference time).

`reference="base"` is the ABLATION's mode, and it is the only one of the three in which rung 1 is
absent: a fresh LoRA on the UNTRAINED base, with `pi_ref` that same untrained base. It is what
the A14/E39 arm (`dpo_from_base`) measures -- the preference label alone, with no imitation in
front of it -- and its checks are the adapter mode's INVERTED, which is why they are here rather
than in a file of their own: the property that makes one path right makes the other wrong.

TWO NAMES IN THE NEW PATH ARE DECIDED BY THE INSTALLED LIBRARIES, NOT BY TASTE, and both are
measured here rather than assumed:

  * the POLICY adapter is `"default"` because `PeftModel.save_pretrained` writes any other
    adapter name into a SUBDIRECTORY of the output (`peft_model.py`: `output_dir = join(
    save_directory, adapter_name) if adapter_name != "default"`). Naming it "train" would put
    rung 2's checkpoint in `artifacts/rung2/train/`, where `pi train eval-offline`, `pi train
    merge` and the checkpoint registry do not look for it.
  * the REFERENCE adapter is `"ref"` because that is the name TRL 1.x SELECTS. MEASURED on trl
    1.13.0: `DPOConfig` declares 137 fields and `model_adapter_name`/`ref_adapter_name` are NOT
    among them -- they were removed -- and `dpo_trainer.py` instead does `use_adapter(model,
    adapter_name="ref" if "ref" in model.peft_config else None)` in three places. A reference
    adapter named anything else is therefore not found, all adapters are disabled, and `pi_ref`
    is THE BARE BASE: the exact silent-wrong-reference failure this module exists to prevent.
    So the name is probed and RECORDED, and a TRL that offers neither mechanism is refused.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

from pinq_train import merge
from pinq_train.rung2_dpo import AmbiguousReference, DPOConfig, dpo_argument_kwargs

# NOT `from pinq_train.rung2_dpo import train`: the package rebinds that name to the FUNCTION,
# so the attribute path would hand back `train()` rather than the module the constants live in.
rung2_train = importlib.import_module("pinq_train.rung2_dpo.train")

SHA = "0" * 64  # a merge output sha, in the shape `merge_adapter` returns
# What the trainer double records when `train()` was called with no checkpoint argument at all.
NOT_PASSED = "<train() was called with no checkpoint argument>"
# The older path, spelled out. It is no longer the default, so a test about it has to say so --
# which is the point: `reference` is in `cfg.sha`, and a run must name which policy it started
# from rather than inherit one from a dataclass default that changed underneath it.
MERGED = {"reference": "merged", "merge_adapter": True}


# The trl 1.13.0 reference-forward line, verbatim, as the probe has to see it. Three copies of
# it exist in `trl/trainer/dpo_trainer.py`; one is enough to establish the convention.
_TRL1_SOURCE = (
    '    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):\n'
    "        ref_outputs = self.model(**ref_model_kwargs)\n"
)
# The TRL that still declares the two names on its config (0.9-era through early 1.x).
_TRL0_FIELDS = frozenset({"output_dir", "beta", "model_adapter_name", "ref_adapter_name"})


# Two LoRA tensors per adapter, as plain floats: `verify_frozen_reference` compares them and
# reads `requires_grad`, and nothing it does needs a real tensor. `lora_B` is non-zero on
# purpose -- a reference whose B is zero IS the bare base, and the check has to notice.
_LORA_TENSORS = {"q_proj.lora_A.{}.weight": 0.5, "q_proj.lora_B.{}.weight": 0.25}
# THE SAME TENSORS AS PEFT INITIALISES THEM -- `lora_B` ZERO, so the adapter is the identity and
# the model IS the base. Under `reference="adapter"` that is a BUG (rung 1 would be the bare
# base); under `reference="base"` it is the REQUIREMENT, because the policy there is a fresh
# LoRA that has not trained yet. One double has to be able to produce both.
_FRESH_LORA_TENSORS = {"q_proj.lora_A.{}.weight": 0.5, "q_proj.lora_B.{}.weight": 0.0}


class _FakePeftModel:
    """A PeftModel that records how its adapters were loaded, and under which names."""

    def __init__(
        self,
        seen: dict,
        base: object,
        path: str,
        *,
        tensors: dict = _LORA_TENSORS,
        record: bool = True,
        **kw,
    ) -> None:
        self._seen = seen
        self._tensors = tensors
        self.peft_config: dict = {}
        self._params: dict = {}
        self._add(kw.get("adapter_name", "default"))
        # NOT recorded when TRL built it: `seen["calls"]` is the list of loads THIS repository
        # made, and an assertion that the base path loads nothing has to keep meaning that.
        if record:
            seen["calls"].append(("peft_from_pretrained", base, path, kw))

    def _add(self, name: str) -> None:
        self.peft_config[name] = object()
        for tmpl, v in self._tensors.items():
            self._params[f"base_model.model.{tmpl.format(name)}"] = v

    def load_adapter(self, path: str, **kw) -> None:
        self._seen["calls"].append(("load_adapter", path, kw))
        self._add(kw.get("adapter_name", "default"))

    def named_parameters(self):
        return list(self._params.items())

    def save_pretrained(self, out: str, **kw) -> None:
        self._seen["saved"] = kw
        Path(out).mkdir(parents=True, exist_ok=True)


def _install_fake_train_libraries(monkeypatch, *, trl_fields=_TRL0_FIELDS) -> dict:
    """`datasets`, `peft`, `transformers` and `trl` as doubles that record every call.

    The gate venv has none of the four, and `train()` is the one function in the rung that a
    laptop cannot execute -- which is exactly why what it PASSES was pinned by nothing. These
    record the calls; the arithmetic they stand in for is checked by the integration test.
    """
    seen: dict = {"calls": [], "trl_config": None, "trainer": None, "saved": None}

    th = types.ModuleType("torch")
    th.equal = staticmethod(lambda a, b: a == b)

    ds = types.ModuleType("datasets")
    ds.Dataset = type("Dataset", (), {"from_list": staticmethod(lambda rows: list(rows))})

    tr = types.ModuleType("transformers")
    tr.AutoTokenizer = type("AutoTokenizer", (), {"from_pretrained": staticmethod(lambda n: n)})

    def _load_base(name: str):
        seen["calls"].append(("load_base", name))
        return "BASE-MODEL-OBJECT"

    tr.AutoModelForCausalLM = type(
        "AutoModelForCausalLM", (), {"from_pretrained": staticmethod(_load_base)}
    )

    pf = types.ModuleType("peft")
    pf.LoraConfig = type("LoraConfig", (), {"__init__": lambda self, **kw: None})
    pf.PeftModel = type(
        "PeftModel",
        (),
        {
            "from_pretrained": staticmethod(
                lambda base, path, **kw: _FakePeftModel(seen, base, path, **kw)
            )
        },
    )

    # WHAT TRL DOES WITH A `peft_config`, MEASURED on trl 1.13.0 (`DPOTrainer.__init__`):
    # `model = get_peft_model(model, peft_config)`. The trainer therefore holds an object the
    # caller never saw, carrying ONE adapter named "default" whose `lora_B` is zero and NO
    # "ref" -- so a check that read the caller's own object would read a model with no adapter
    # at all, and `reference="base"` is exactly the mode whose check runs on that object.
    def _get_peft_model(base, config, **kw):
        seen["peft_wrapped"] = base
        return _FakePeftModel(seen, base, "", tensors=_FRESH_LORA_TENSORS, record=False)

    pf.get_peft_model = _get_peft_model

    class _TRLConfig:
        __dataclass_fields__ = dict.fromkeys(trl_fields)

        def __init__(self, **kw) -> None:
            seen["trl_config"] = kw

    class _TRLTrainer:
        def __init__(self, **kw) -> None:
            seen["trainer"] = kw
            # The wrap is TRL's, not the caller's -- see `_get_peft_model`. `self.model` is
            # what `train()` must verify under `reference="base"`.
            model = kw["model"]
            if kw.get("peft_config") is not None:
                model = pf.get_peft_model(model, kw["peft_config"])
            self.model = model

        def train(self, resume_from_checkpoint=NOT_PASSED) -> None:
            seen["trained"] = True
            # RECORDED THROUGH A SENTINEL, so that "told to start fresh" (None) and "never told
            # anything" (the default) are different observations. `trainer.train()` with no
            # argument restarts a preempted job from step 0, and a double whose default were
            # None could not tell that apart from a deliberate fresh start.
            seen["resume_from_checkpoint"] = resume_from_checkpoint

        def save_model(self, out: str) -> None:
            seen["saved"] = {"save_model": out}
            Path(out).mkdir(parents=True, exist_ok=True)

    trl = types.ModuleType("trl")
    trl.DPOConfig = _TRLConfig
    trl.DPOTrainer = _TRLTrainer
    for name, mod in (
        ("datasets", ds),
        ("transformers", tr),
        ("peft", pf),
        ("trl", trl),
        ("torch", th),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return seen


def _pair() -> dict:
    from pinq.actions import ask_action_json

    return {
        "suite_id": "musique",
        "task_id": "t1",
        "state_text": "S",
        "chosen_json": ask_action_json("who?"),
        "rejected_json": ask_action_json("when?"),
        "margin": 0.5,
        "pair_kind": "ask_ask",
    }


def _run_train(tmp_path: Path, adapter: Path, **over) -> str:
    """`train()` over one pair, with the doubles installed. Returns the output directory."""
    cfg = DPOConfig(
        base_model=over.pop("base_model", "Qwen/Qwen3-8B"),
        adapter=str(adapter),
        out_dir=str(tmp_path / "rung2"),
        **over,
    )
    return rung2_train.train(
        cfg, acknowledge_untested=True, tokenizer=object(), rows=[_pair()], drops={}
    )["out_dir"]


# --------------------------------------------------------------- the two config refusals


def test_a_merged_run_must_name_the_weights_it_started_from():
    """`merge_adapter=True` says base_model is the OUTPUT of `pi train merge`. Without the sha
    of that output the run cannot say which weights its reference policy was, and "DPO moved
    the policy from the SFT checkpoint" is a sentence nothing can check."""
    with pytest.raises(ValueError, match="merged_base_sha"):
        DPOConfig(base_model="artifacts/merged", adapter="artifacts/rung1", **MERGED).validate()
    DPOConfig(
        base_model="artifacts/merged",
        adapter="artifacts/rung1",
        merged_base_sha=SHA,
        **MERGED,
    ).validate()


def test_not_merging_while_naming_an_adapter_is_the_two_adapter_ambiguity():
    """The old code path, refused by name. An adapter to load PLUS the fresh LoRA DPO trains is
    two adapters, and which one TRL's reference pass disables is a property of the installed
    versions rather than of anything this repository states."""
    with pytest.raises(AmbiguousReference, match="two adapters"):
        DPOConfig(
            base_model="Qwen/Qwen3-8B",
            adapter="artifacts/rung1",
            reference="merged",
            merge_adapter=False,
        ).validate()
    # merge_adapter=False with NO adapter is the one unambiguous reading: base_model is already
    # the policy, and the reference is base_model with the fresh LoRA disabled.
    DPOConfig(
        base_model="artifacts/already_merged",
        adapter="",
        reference="merged",
        merge_adapter=False,
    ).validate()


def test_the_reference_decision_is_in_the_config_sha():
    """Two runs whose reference policies are different models must not claim one identity."""
    merged = DPOConfig(base_model="m", adapter="a", merged_base_sha=SHA, **MERGED)
    assert (
        merged.sha
        != DPOConfig(base_model="m", adapter="", reference="merged", merge_adapter=False).sha
    )
    assert (
        merged.sha != DPOConfig(base_model="m", adapter="a", merged_base_sha="1" * 64, **MERGED).sha
    )


# --------------------------------------- the reference without a merge (THE DEFAULT PATH)


def test_the_default_reference_is_a_frozen_copy_of_rung_1_not_a_merge():
    """Nothing has to be folded, so nothing can be rounded away. One base, two named adapters."""
    cfg = DPOConfig(base_model="Qwen/Qwen3-8B", adapter="artifacts/rung1")
    assert cfg.reference == "adapter"
    assert cfg.merge_adapter is False
    cfg.validate()


def test_the_adapter_reference_needs_the_adapter_it_freezes():
    """`reference="adapter"` with no adapter is DPO against the bare base wearing rung 2's name:
    it trains, it reports a falling loss, and the paper's sentence is false."""
    with pytest.raises(ValueError, match="reference='adapter'"):
        DPOConfig(base_model="Qwen/Qwen3-8B", adapter="").validate()


def test_the_adapter_reference_refuses_a_merge_in_the_same_config():
    """Two reference policies in one config is not a merge plus a fallback -- it is a run that
    cannot say which distribution `pi_ref` was."""
    with pytest.raises(ValueError, match="merge_adapter"):
        DPOConfig(base_model="b", adapter="a", merge_adapter=True).validate()
    with pytest.raises(ValueError, match="merged_base_sha"):
        DPOConfig(base_model="b", adapter="a", merged_base_sha=SHA).validate()


def test_an_unknown_reference_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="reference="):
        DPOConfig(base_model="b", adapter="a", reference="frozen_copy").validate()


def test_which_reference_the_run_used_is_in_the_config_sha():
    """A merged run and an adapter run over the same pairs are two experiments."""
    adapter = DPOConfig(base_model="b", adapter="a")
    merged = DPOConfig(base_model="b", adapter="a", merged_base_sha=SHA, **MERGED)
    assert adapter.sha != merged.sha


# ------------------------------------- the reference that IS the base (the dpo_from_base arm)
#
# WHAT THIS THIRD MODE MEASURES, and why it is not a fourth spelling of "merged". The A14/E39
# ablation asks whether the SFT rung is NECESSARY before preference learning: it runs DPO with
# no rung 1 anywhere, so the policy is a fresh LoRA on the UNTRAINED base and `pi_ref` is that
# same untrained base. The preference label is then the only supervision in the run -- no
# imitation of any demonstration precedes it -- which is precisely the claim the arm exists to
# test. "merged" cannot express it: `merge_adapter=False` with no adapter says base_model is
# ALREADY the policy (merged elsewhere, or a full-weight fine-tune), which is a different
# sentence about a different checkpoint, and `validate()` there still refuses the raw base
# under a merge. Two runs whose pi_ref is a different model must not share one cfg.sha.


def test_the_base_is_a_third_named_reference_and_not_an_unknown_value():
    """`reference` is refused unless it is one of `REFERENCES`, so the mode has to be named
    there before anything can select it -- a run configured with an unknown value would have a
    pi_ref that is whatever the installed libraries happened to do."""
    assert rung2_train.REFERENCES == ("adapter", "merged", "base")
    DPOConfig(base_model="Qwen/Qwen3-8B", adapter="", reference="base").validate()


def test_the_base_reference_refuses_an_adapter_because_which_one_is_the_policy():
    """A base reference with an adapter named is the two-adapter ambiguity again, from the
    other end: the run would hold rung 1's LoRA AND the fresh LoRA DPO trains, and nothing says
    which of them is the policy the arm's name claims it has no rung 1 in at all."""
    with pytest.raises(AmbiguousReference, match="reference='base'"):
        DPOConfig(
            base_model="Qwen/Qwen3-8B", adapter="artifacts/rung1", reference="base"
        ).validate()


def test_the_base_reference_refuses_a_merge_in_the_same_config():
    """Mirrors the adapter mode's two refusals. Nothing is folded into the weights under
    'base' -- there is nothing to fold -- so a merge flag or a merge sha here describes a step
    this run does not take, and a provenance field naming a step that did not happen
    reconciles against a directory nobody loaded."""
    with pytest.raises(ValueError, match="merge_adapter"):
        DPOConfig(base_model="b", adapter="", reference="base", merge_adapter=True).validate()
    with pytest.raises(ValueError, match="merged_base_sha"):
        DPOConfig(base_model="b", adapter="", reference="base", merged_base_sha=SHA).validate()


def test_the_base_reference_is_its_own_identity_in_the_config_sha():
    """dpo_from_base and the SFT-then-DPO arm are the comparison the ablation IS. If they hash
    alike the two rows in the table are one run reported twice."""
    base = DPOConfig(base_model="Qwen/Qwen3-8B", adapter="", reference="base")
    assert base.sha != DPOConfig(base_model="Qwen/Qwen3-8B", adapter="artifacts/rung1").sha
    assert (
        base.sha
        != DPOConfig(
            base_model="Qwen/Qwen3-8B", adapter="", reference="merged", merge_adapter=False
        ).sha
    ), "the same weights reached two ways are still two claims about what pi_ref is"


def test_adding_the_base_mode_did_not_rename_a_single_finished_adapter_run():
    """MEASURED before this commit and pinned here. `REFERENCES` is a module constant, not a
    field, so widening it must not touch `cfg.sha` -- and if it did, every rung-2 run already
    on disk would stop joining to the tables that cite its `config_sha`."""
    assert (
        DPOConfig(base_model="Qwen/Qwen3-8B", adapter="artifacts/rung1").sha
        == "63dcf89f421d0284772c20270d92103e2438a3dde6e6b25525dd9a7b814e1048"
    )


# ------------------------------- the two adapter names are measured, not chosen for readability


def test_the_policy_adapter_is_named_default_because_peft_subdirectories_anything_else():
    """`PeftModel.save_pretrained` writes any adapter whose name is not "default" into a
    SUBDIRECTORY of the output. A policy adapter named "train" would land in
    `artifacts/rung2/train/`, where eval-offline, `pi train merge` and the checkpoint registry
    do not look -- and the failure is a missing file, discovered on the rented box."""
    assert rung2_train.POLICY_ADAPTER == "default"


def test_the_reference_adapter_is_named_ref_because_that_is_the_name_trl_selects():
    """MEASURED on trl 1.13.0: the reference forward is `use_adapter(model, adapter_name="ref"
    if "ref" in model.peft_config else None)`. Under any other name TRL finds nothing, disables
    ALL adapters, and `pi_ref` is the bare base -- silently."""
    assert rung2_train.REFERENCE_ADAPTER == "ref"
    assert rung2_train.reference_hook(frozenset(), _TRL1_SOURCE) == "trl_named_ref_adapter"


def test_a_trl_that_still_declares_the_two_fields_is_told_explicitly():
    """Older TRL takes the names on its config. Then BOTH mechanisms agree, because the name
    passed is the same "ref" the newer library picks by convention."""
    assert rung2_train.reference_hook(_TRL0_FIELDS, "") == "dpo_config_fields"
    kw, hooks = dpo_argument_kwargs(
        DPOConfig(base_model="b", adapter="a"), _TRL0_FIELDS, trl_source=""
    )
    assert kw["model_adapter_name"] == "default"
    assert kw["ref_adapter_name"] == "ref"
    assert hooks["reference_adapter_hook"] == "dpo_config_fields"
    assert hooks["policy_adapter"] == "default" and hooks["reference_adapter"] == "ref"


def test_a_trl_that_offers_neither_mechanism_is_refused_rather_than_guessed():
    """The whole point of the refusal: a TRL that neither takes the names nor looks for "ref"
    would train against the bare base and report a perfectly healthy loss curve."""
    with pytest.raises(rung2_train.UnknownReferenceHook, match="ref"):
        rung2_train.reference_hook(frozenset(), "def compute_loss(self): pass")
    with pytest.raises(rung2_train.UnknownReferenceHook):
        dpo_argument_kwargs(DPOConfig(base_model="b", adapter="a"), frozenset(), trl_source="")


def test_the_merged_path_emits_no_adapter_names_at_all():
    """Under a merge there is one adapter and it is the one being trained; naming a reference
    adapter that does not exist would make TRL look for it and find nothing."""
    cfg = DPOConfig(base_model="m", adapter="a", merged_base_sha=SHA, **MERGED)
    kw, hooks = dpo_argument_kwargs(cfg, _TRL0_FIELDS, trl_source=_TRL1_SOURCE)
    assert "model_adapter_name" not in kw and "ref_adapter_name" not in kw
    assert hooks["reference_adapter_hook"] == "merged_weights"


def test_the_base_path_names_no_adapter_either_and_records_which_mechanism_disables_them():
    """Under 'base' the reference is reached by TRL DISABLING the only adapter in the process,
    which is the `None` branch of the same line the adapter mode relies on. The hook is
    RECORDED rather than assumed for the same reason it is there: "the reference was the
    untrained base" has to be checkable from the artifact, and it is a different fact from
    "rung 1 was in the weights" even though both disable adapters."""
    cfg = DPOConfig(base_model="Qwen/Qwen3-8B", adapter="", reference="base")
    kw, hooks = dpo_argument_kwargs(cfg, _TRL0_FIELDS, trl_source=_TRL1_SOURCE)
    assert "model_adapter_name" not in kw and "ref_adapter_name" not in kw
    assert hooks["reference_adapter_hook"] == "trl_disable_adapters"
    assert "reference_adapter" not in hooks, "there is no reference adapter to name"


def test_the_base_path_refuses_a_trl_whose_reference_forward_cannot_be_read():
    """The `None` branch is the whole mechanism here, so a TRL that does not show it is a TRL
    on which pi_ref is unknown -- and refused BEFORE the weights load, which is why the probe
    lives in `dpo_argument_kwargs` and not only in the post-construction check."""
    cfg = DPOConfig(base_model="Qwen/Qwen3-8B", adapter="", reference="base")
    with pytest.raises(rung2_train.UnknownReferenceHook, match="DISABLES"):
        dpo_argument_kwargs(cfg, _TRL0_FIELDS, trl_source="def compute_loss(self): pass")


# ------------------------------------------------- what `train()` actually does, with doubles


def test_train_loads_one_base_and_the_same_adapter_twice_under_two_names(tmp_path, monkeypatch):
    """THE RECIPE, as calls. One base in memory; rung 1's LoRA loaded twice; the trainable copy
    is the policy and the frozen copy is `pi_ref`. No merge, so no rounding."""
    seen = _install_fake_train_libraries(monkeypatch)
    adapter = _adapter_dir(tmp_path)
    _run_train(tmp_path, adapter)

    assert [c[0] for c in seen["calls"]] == ["load_base", "peft_from_pretrained", "load_adapter"]
    assert seen["calls"][0][1] == "Qwen/Qwen3-8B"
    _, base_obj, path, kw = seen["calls"][1]
    assert base_obj == "BASE-MODEL-OBJECT", "the policy adapter goes onto the base just loaded"
    assert path == str(adapter)
    assert kw == {"adapter_name": "default", "is_trainable": True}
    _, ref_path, ref_kw = seen["calls"][2]
    assert ref_path == str(adapter), "the reference is the SAME adapter, not another checkpoint"
    assert ref_kw == {"adapter_name": "ref", "is_trainable": False}


def test_the_trainer_is_given_the_peft_model_itself_and_no_fresh_lora(tmp_path, monkeypatch):
    """`peft_config` would make TRL add a THIRD adapter and train that instead, leaving rung 1's
    trainable copy untouched. `ref_model=None` is what routes the reference through the adapter
    rather than through a second full model in memory."""
    seen = _install_fake_train_libraries(monkeypatch)
    _run_train(tmp_path, _adapter_dir(tmp_path))

    trainer = seen["trainer"]
    assert trainer["peft_config"] is None
    assert trainer["ref_model"] is None
    assert trainer["model"].__class__.__name__ == "_FakePeftModel"
    assert seen["trl_config"]["model_adapter_name"] == "default"
    assert seen["trl_config"]["ref_adapter_name"] == "ref"


def test_the_manifest_names_the_reference_policy_its_sha_and_the_lora_it_inherits(
    tmp_path, monkeypatch
):
    """ "The reference was rung 1" has to be checkable from the artifact alone. The LoRA shape is
    rung 1's too -- the policy adapter IS rung 1's adapter -- so `cfg.lora` does not describe
    this run and the manifest has to say what does."""
    _install_fake_train_libraries(monkeypatch)
    adapter = _adapter_dir(tmp_path)
    out = _run_train(tmp_path, adapter)

    man = json.loads((Path(out) / "rung2.manifest.json").read_text())
    ref = man["pairs"]["reference_policy"]
    assert ref["reference"] == "adapter"
    assert ref["adapter"] == str(adapter)
    assert ref["adapter_sha"] == merge.adapter_sha(adapter)
    assert ref["policy_adapter"] == "default" and ref["reference_adapter"] == "ref"
    assert ref["reference_adapter_hook"] == "dpo_config_fields"
    assert ref["lora_source"] == "rung1 adapter_config.json"
    assert ref["lora"] == {"r": 16, "lora_alpha": 32}
    assert ref["verified"] == {
        "n_lora_pairs": 2,
        "equal": True,
        "frozen": True,
        "is_the_base": False,
    }


def test_only_the_policy_adapter_is_written_to_the_output_directory(tmp_path, monkeypatch):
    """The frozen copy is rung 1's adapter byte for byte. Saving it beside rung 2's would put a
    second `adapter_model.safetensors` in the directory that names rung 2."""
    seen = _install_fake_train_libraries(monkeypatch)
    _run_train(tmp_path, _adapter_dir(tmp_path))
    assert seen["saved"] == {"selected_adapters": ["default"]}


def test_no_adapter_is_loaded_onto_a_merged_base(tmp_path, monkeypatch):
    """THE ORIGINAL DEFECT, as a property of the calls rather than of the source text.

    This used to be `assert "load_adapter" not in inspect.getsource(train)`. That belief is now
    wrong -- the default path loads two adapters deliberately, under two names -- but what it
    was guarding is still true: under `reference="merged"` rung 1 is in the WEIGHTS, so an
    adapter loaded on top of them plus the fresh LoRA is the old ambiguity all over again.
    """
    seen = _install_fake_train_libraries(monkeypatch)
    _run_train(
        tmp_path,
        _adapter_dir(tmp_path),
        reference="merged",
        merge_adapter=True,
        merged_base_sha=SHA,
    )
    assert [c[0] for c in seen["calls"]] == ["load_base"]
    assert seen["trainer"]["peft_config"] is not None, "the merged path trains a FRESH LoRA"
    assert "ref_adapter_name" not in seen["trl_config"]


def test_train_from_the_base_loads_nothing_but_the_base_and_trains_a_fresh_lora(
    tmp_path, monkeypatch
):
    """dpo_from_base AS CALLS. No adapter is loaded at all -- that is the arm: there is no rung
    1 in this run -- and the fresh LoRA goes to TRL as `peft_config`, exactly as the merged
    path does, so the only difference between the two is which weights are underneath."""
    seen = _install_fake_train_libraries(monkeypatch)
    _run_train(tmp_path, "", reference="base")

    assert [c[0] for c in seen["calls"]] == ["load_base"]
    assert seen["calls"][0][1] == "Qwen/Qwen3-8B", "the RAW base, not a merge output"
    assert seen["trainer"]["peft_config"] is not None
    assert seen["trainer"]["ref_model"] is None, "pi_ref is this model with adapters disabled"
    assert seen["peft_wrapped"] == "BASE-MODEL-OBJECT"


def test_the_manifest_says_the_reference_was_the_untrained_base_and_names_it(tmp_path, monkeypatch):
    """ "DPO with no SFT rung" has to be checkable from the artifact alone: which weights
    pi_ref was, that no adapter was involved, and by which mechanism TRL reached them. The
    LoRA here IS this run's own -- unlike the adapter mode, where it is rung 1's."""
    _install_fake_train_libraries(monkeypatch)
    out = _run_train(tmp_path, "", reference="base")

    man = json.loads((Path(out) / "rung2.manifest.json").read_text())
    ref = man["pairs"]["reference_policy"]
    assert ref["reference"] == "base"
    assert ref["base_model"] == "Qwen/Qwen3-8B"
    # None, not "": under 'base' there is no adapter in the process at all, and an empty string
    # beside the two modes that DO name one reads as a field nobody filled in.
    assert ref["adapter"] is None
    assert ref["adapter_sha"] == ""
    assert ref["reference_adapter_hook"] == "trl_disable_adapters"
    assert ref["lora_source"] == "DPOConfig.lora"
    assert ref["lora"] == {"r": 32, "lora_alpha": 64}
    assert ref["verified"] == {
        "n_policy_lora": 2,
        "ref_adapter_absent": True,
        "disables_adapters": True,
        "is_the_base": True,
    }


def test_the_base_run_verifies_the_model_trl_built_not_the_one_it_was_handed(tmp_path, monkeypatch):
    """MEASURED on trl 1.13.0: a `peft_config` makes the trainer REPLACE the model with
    `get_peft_model(model, peft_config)`. Checking the object `train()` loaded would check a
    bare `AutoModelForCausalLM` -- no LoRA parameters, no `peft_config` -- and the check would
    pass by finding nothing, which is the shape of every failure this file exists to stop."""
    seen = _install_fake_train_libraries(monkeypatch)
    _run_train(tmp_path, "", reference="base")
    assert seen["peft_wrapped"] == "BASE-MODEL-OBJECT", "the handed model is NOT the checked one"


def test_the_base_run_writes_the_trainer_s_checkpoint_not_a_named_adapter(tmp_path, monkeypatch):
    """There is one adapter and TRL owns it, so `save_model` is the call -- `save_pretrained`
    with `selected_adapters` is the two-adapter path's, and the object that has it is not the
    object this run holds."""
    seen = _install_fake_train_libraries(monkeypatch)
    out = _run_train(tmp_path, "", reference="base")
    assert seen["saved"] == {"save_model": out}


# --------------------------------------------------- what a preempted rung 2 resumes from
#
# The rung-1 half of this, the pure `latest_checkpoint` helper and the reason `resume` is not in
# `cfg.sha` are in `tests/test_resume_from_checkpoint.py`. These live here because the TRL
# doubles do: a second copy of them would be a second thing to keep in step with trl 1.x.


def _checkpoint(out_dir: Path, step: int) -> Path:
    d = out_dir / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "trainer_state.json").write_text(json.dumps({"global_step": step}) + "\n")
    return d


def test_rung2_resumes_from_the_checkpoint_on_disk(tmp_path, monkeypatch):
    """Rung 2 is ~25 GPU-h per arm and three arms are packed on one node, so a preemption that
    restarts from step 0 costs all three. TRL's trainer is an HF `Trainer`, so the checkpoint
    argument is the same one -- it just had to be passed."""
    seen = _install_fake_train_libraries(monkeypatch)
    ckpt = _checkpoint(tmp_path / "rung2", 600)
    out = _run_train(tmp_path, _adapter_dir(tmp_path))
    assert seen["resume_from_checkpoint"] == str(ckpt)
    man = json.loads((Path(out) / "rung2.manifest.json").read_text())
    assert man["resumed_from"] == str(ckpt)


def test_rung2_starts_fresh_when_resume_is_off(tmp_path, monkeypatch):
    """`None` explicitly, not by omission: this run DECIDED to ignore what is on disk."""
    seen = _install_fake_train_libraries(monkeypatch)
    _checkpoint(tmp_path / "rung2", 600)
    out = _run_train(tmp_path, _adapter_dir(tmp_path), resume=False)
    assert seen["resume_from_checkpoint"] is None
    assert json.loads((Path(out) / "rung2.manifest.json").read_text())["resumed_from"] is None


def test_rung2_without_a_checkpoint_resumes_nothing_and_says_so(tmp_path, monkeypatch):
    seen = _install_fake_train_libraries(monkeypatch)
    out = _run_train(tmp_path, _adapter_dir(tmp_path))
    assert seen["resume_from_checkpoint"] is None
    assert json.loads((Path(out) / "rung2.manifest.json").read_text())["resumed_from"] is None


def test_rung2_asks_trl_to_write_checkpoints_at_all(tmp_path, monkeypatch):
    """Rung 2 passed NO save strategy, so the cadence was whatever the installed TRL defaulted
    to -- and a run with nothing on disk cannot be resumed however the call is spelled."""
    seen = _install_fake_train_libraries(
        monkeypatch, trl_fields=_TRL0_FIELDS | {"save_strategy", "save_steps", "save_total_limit"}
    )
    _run_train(tmp_path, _adapter_dir(tmp_path), save_steps=50)
    assert seen["trl_config"]["save_strategy"] == "steps"
    assert seen["trl_config"]["save_steps"] == 50
    assert seen["trl_config"]["save_total_limit"] == 2


# ------------------------------------------------------------------------ merge plumbing


def _fake_loaders(calls: list, *, weights: bytes = b"merged-weights"):
    """transformers/peft as three callables, so the plumbing is testable without torch.

    The doubles record the ORDER and the ARGUMENTS of every call: the adapter must be applied
    to the model that was loaded from `base_model`, and the tokenizer must come from the base
    (a LoRA does not change the tokenizer, and a merged directory with no tokenizer cannot be
    served).
    """

    class Merged:
        def save_pretrained(self, path):
            calls.append(("save_model", str(path)))
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "model-00001-of-00001.safetensors").write_bytes(weights)
            (p / "config.json").write_text("{}\n")

    class Peft:
        def merge_and_unload(self):
            calls.append(("merge_and_unload",))
            return Merged()

    class Tok:
        def save_pretrained(self, path):
            calls.append(("save_tokenizer", str(path)))
            (Path(path) / "tokenizer.json").write_text("{}\n")

    def model(p):
        calls.append(("load_base", p))
        return "BASE-MODEL-OBJECT"

    def tokenizer(p):
        calls.append(("load_tokenizer", p))
        return Tok()

    def adapter(m, p):
        calls.append(("load_adapter", m, p))
        return Peft()

    return merge.Loaders(model=model, tokenizer=tokenizer, adapter=adapter)


def _adapter_dir(tmp_path: Path) -> Path:
    d = tmp_path / "rung1"
    d.mkdir(exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(b"lora")
    (d / "adapter_config.json").write_text('{"r": 16, "lora_alpha": 32}\n')
    return d


def test_merge_loads_the_adapter_onto_the_base_and_saves_a_tokenizer(tmp_path):
    calls: list = []
    out = tmp_path / "merged"
    merge.merge_adapter(
        "Qwen/Qwen3-8B", str(_adapter_dir(tmp_path)), str(out), _loaders=_fake_loaders(calls)
    )
    assert [c[0] for c in calls] == [
        "load_base",
        "load_adapter",
        "merge_and_unload",
        "save_model",
        "load_tokenizer",
        "save_tokenizer",
    ]
    assert calls[0] == ("load_base", "Qwen/Qwen3-8B")
    assert calls[1][1] == "BASE-MODEL-OBJECT", "the adapter must go onto the model just loaded"
    assert calls[4] == ("load_tokenizer", "Qwen/Qwen3-8B"), "the tokenizer comes from the BASE"
    assert (out / "tokenizer.json").exists() and (out / "config.json").exists()


def test_merge_returns_the_sha_of_the_weights_it_wrote_and_records_it(tmp_path):
    """The returned sha is `merged_base_sha`: the run's claim about which weights it started
    from. It must be a function of the weight FILES on disk, so a manifest and a directory
    can be reconciled by anyone who has both."""
    out = tmp_path / "merged"
    adapter = _adapter_dir(tmp_path)
    sha = merge.merge_adapter("base", str(adapter), str(out), _loaders=_fake_loaders([]))

    assert len(sha) == 64 and sha == merge.dir_sha(out)
    man = json.loads((out / "merge.manifest.json").read_text())
    assert man["base_model"] == "base"
    assert man["adapter"] == str(adapter)
    assert man["output_sha"] == sha
    assert man["adapter_sha"] == merge.adapter_sha(adapter)
    # the manifest is not a weight file, so writing it cannot change the sha it records
    assert merge.dir_sha(out) == sha


def test_a_different_merge_is_a_different_sha(tmp_path):
    a = merge.merge_adapter(
        "base", str(_adapter_dir(tmp_path)), str(tmp_path / "a"), _loaders=_fake_loaders([])
    )
    b = merge.merge_adapter(
        "base",
        str(_adapter_dir(tmp_path)),
        str(tmp_path / "b"),
        _loaders=_fake_loaders([], weights=b"other-weights"),
    )
    assert a != b


def test_a_merge_that_wrote_no_weights_is_refused_rather_than_hashed(tmp_path):
    """The sha of an empty file set is a perfectly good hex string, and it would land in a
    manifest as provenance for a merge that never happened."""

    class Empty:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    loaders = merge.Loaders(
        model=lambda p: None,
        tokenizer=lambda p: Empty(),
        adapter=lambda m, p: type("P", (), {"merge_and_unload": lambda s: Empty()})(),
    )
    with pytest.raises(merge.NothingMerged, match="no .*safetensors"):
        merge.merge_adapter(
            "base", str(_adapter_dir(tmp_path)), str(tmp_path / "o"), _loaders=loaders
        )


# ---------------------------------------- what an adapter's identity is, and what it is not


def test_an_intermediate_checkpoint_is_not_part_of_the_adapters_identity(tmp_path):
    """MEASURED on the smoke run: `save_strategy="epoch"` leaves `checkpoint-6/` beside the
    adapter, and hashing the directory tree made DELETING it change `adapter_sha` while the
    adapter itself did not change a byte. A provenance field that moves when nothing about the
    thing it names moved is a field people learn to ignore -- and `adapter_sha` sits inside
    ModelPin.key -> semantic_hash -> run_id, so "learn to ignore" is not available."""
    adapter = _adapter_dir(tmp_path)
    before = merge.adapter_sha(adapter)
    (adapter / "checkpoint-6").mkdir()
    (adapter / "checkpoint-6" / "adapter_model.safetensors").write_bytes(b"step-6")
    (adapter / "checkpoint-6" / "optimizer.pt").write_bytes(b"states")
    assert merge.adapter_sha(adapter) == before


def test_the_trainer_s_own_bookkeeping_is_not_part_of_it_either(tmp_path):
    """`training_args.bin` pickles the output_dir and the seed; `rung1.manifest.json` records
    the sha's own inputs. Neither changes what the adapter computes."""
    adapter = _adapter_dir(tmp_path)
    before = merge.adapter_sha(adapter)
    (adapter / "training_args.bin").write_bytes(b"pickled TrainingArguments")
    (adapter / "rung1.manifest.json").write_text("{}\n")
    (adapter / "README.md").write_text("# card\n")
    assert merge.adapter_sha(adapter) == before


def test_a_changed_adapter_weight_is_a_different_sha(tmp_path):
    """The only thing the sha has to track: the tensors the policy is a function of."""
    adapter = _adapter_dir(tmp_path)
    before = merge.adapter_sha(adapter)
    (adapter / "adapter_model.safetensors").write_bytes(b"lorb")
    assert merge.adapter_sha(adapter) != before


def test_the_adapter_config_is_part_of_it(tmp_path):
    """Same tensors under a different r or a different target_modules set is a different
    adapter -- peft reads the config to decide where the weights are applied."""
    adapter = _adapter_dir(tmp_path)
    before = merge.adapter_sha(adapter)
    (adapter / "adapter_config.json").write_text('{"r": 32, "lora_alpha": 64}\n')
    assert merge.adapter_sha(adapter) != before


def test_a_directory_with_no_adapter_weights_is_refused_rather_than_hashed(tmp_path):
    """The sha of an empty file set is a perfectly good hex string; it would name an adapter
    that does not exist. A checkpoint subdirectory does not rescue it."""
    d = tmp_path / "empty"
    (d / "checkpoint-6").mkdir(parents=True)
    (d / "checkpoint-6" / "adapter_model.safetensors").write_bytes(b"x")
    with pytest.raises(merge.NothingMerged, match="adapter_model"):
        merge.adapter_sha(d)


# ------------------------------------------------------- the sha is verified, not believed


def test_a_merged_base_sha_that_disagrees_with_the_directory_is_fatal(tmp_path):
    """Otherwise `merged_base_sha` is a string someone typed, which is not provenance."""
    out = tmp_path / "merged"
    sha = merge.merge_adapter(
        "base", str(_adapter_dir(tmp_path)), str(out), _loaders=_fake_loaders([])
    )

    ok = merge.verify_merged_base(str(out), sha)
    assert ok["verified"] is True and ok["output_sha"] == sha
    with pytest.raises(merge.MergeMismatch, match="merged_base_sha"):
        merge.verify_merged_base(str(out), "f" * 64)


def test_a_base_with_no_merge_manifest_is_recorded_as_unverified_not_asserted(tmp_path):
    """A hub id or a directory merged by some other tool cannot be checked here. Saying so in
    the manifest is honest; pretending the sha was verified is not."""
    assert merge.verify_merged_base("Qwen/Qwen3-8B", SHA) == {
        "verified": False,
        "output_sha": "",
        "why": "no merge.manifest.json",
    }


# ------------------------------------------ the reference is checked on the assembled model


class _Params:
    """A model that is only its named parameters and its adapter names.

    `peft_config` is the second thing the base-mode check reads, and it is not redundant with
    the parameters: TRL selects the reference by `"ref" in model.peft_config`, so an adapter
    registered there with no LoRA tensor of its own (a prompt-learning adapter, or one added
    and not yet populated) would still capture the reference forward.
    """

    def __init__(self, params: dict, peft_config: dict | None = None) -> None:
        self._params = params
        self.peft_config = {"default": object()} if peft_config is None else peft_config

    def named_parameters(self):
        return list(self._params.items())


def _eq(a, b) -> bool:
    return a == b


def _both(policy=(0.5, 0.25), ref=(0.5, 0.25)) -> dict:
    out = {}
    for name, vals in (("default", policy), ("ref", ref)):
        out[f"m.q_proj.lora_A.{name}.weight"] = vals[0]
        out[f"m.q_proj.lora_B.{name}.weight"] = vals[1]
    return out


def test_a_reference_adapter_equal_to_the_policy_and_frozen_is_what_passes():
    rep = rung2_train.verify_frozen_reference(_Params(_both()), equal=_eq)
    assert rep == {"n_lora_pairs": 2, "equal": True, "frozen": True, "is_the_base": False}


def test_a_missing_reference_adapter_is_the_bare_base_and_is_refused():
    """The exact failure the name probe exists to prevent, seen from the other end: TRL with no
    adapter to switch to disables all of them, and pi_ref becomes the base."""
    params = {k: v for k, v in _both().items() if ".ref." not in k}
    with pytest.raises(AmbiguousReference, match="BARE BASE"):
        rung2_train.verify_frozen_reference(_Params(params), equal=_eq)


def test_a_reference_that_is_not_rung_1_at_step_0_is_refused():
    with pytest.raises(AmbiguousReference, match="differ from the policy"):
        rung2_train.verify_frozen_reference(_Params(_both(ref=(0.5, 0.9))), equal=_eq)


def test_a_reference_that_would_move_with_the_policy_is_refused():
    """A reference that trains alongside the policy makes the log-ratio identically zero and
    the objective independent of the data -- a flat loss that looks like convergence."""

    class _Trainable(float):
        requires_grad = True

    params = _both()
    params["m.q_proj.lora_B.ref.weight"] = _Trainable(0.25)
    with pytest.raises(AmbiguousReference, match="carry gradients"):
        rung2_train.verify_frozen_reference(_Params(params), equal=_eq)


def test_a_zero_reference_adapter_is_recorded_as_being_the_base():
    """LoRA initialises B to zero, so an untrained rung 1 IS the base. Every other check passes
    on it, which is exactly why the run has to say so rather than stay silent."""
    rep = rung2_train.verify_frozen_reference(
        _Params(_both(policy=(0.5, 0.0), ref=(0.5, 0.0))), equal=_eq
    )
    assert rep["is_the_base"] is True


# ------------------------------- and the base reference is checked for the OPPOSITE property
#
# Under `reference="adapter"` a reference adapter must EXIST and must equal the policy. Under
# `reference="base"` it must NOT exist -- if it did, TRL would select it and pi_ref would be a
# copy of the fresh LoRA rather than the untrained base -- and the policy must be the identity
# at step 0, which is the same `is_the_base` fact read as a requirement instead of as a bug.


def _fresh(policy=(0.5, 0.0)) -> dict:
    """ONE adapter, the fresh LoRA, `lora_B` zero as peft initialises it."""
    return {
        "m.q_proj.lora_A.default.weight": policy[0],
        "m.q_proj.lora_B.default.weight": policy[1],
    }


def test_a_fresh_zero_lora_and_no_ref_adapter_is_what_passes_under_the_base_reference():
    rep = rung2_train.verify_base_reference(_Params(_fresh()), trl_source=_TRL1_SOURCE, equal=_eq)
    assert rep == {
        "n_policy_lora": 2,
        "ref_adapter_absent": True,
        "disables_adapters": True,
        "is_the_base": True,
    }


def test_a_ref_adapter_under_the_base_reference_is_refused():
    """The adapter mode's failure, exactly inverted: with a "ref" adapter present TRL takes the
    NAMED branch, and pi_ref becomes a frozen copy of the fresh LoRA -- which at step 0 is the
    base and after the first update is not. The run would still train and still report a
    falling loss, and the arm's name ("no SFT rung, reference is the untrained base") would be
    false with nothing in the artifact to say so."""
    with pytest.raises(AmbiguousReference, match="'ref'"):
        rung2_train.verify_base_reference(
            _Params(_both(policy=(0.5, 0.0), ref=(0.5, 0.0))),
            trl_source=_TRL1_SOURCE,
            equal=_eq,
        )


def test_a_ref_adapter_named_only_in_the_peft_config_is_refused_too():
    """TRL's selection is `"ref" in model.peft_config`, not a scan of the parameters. An
    adapter registered under that name captures the reference forward whether or not it has a
    LoRA tensor this check can see."""
    with pytest.raises(AmbiguousReference, match="'ref'"):
        rung2_train.verify_base_reference(
            _Params(_fresh(), peft_config={"default": object(), "ref": object()}),
            trl_source=_TRL1_SOURCE,
            equal=_eq,
        )


def test_a_policy_lora_that_is_not_the_identity_at_step_0_is_refused():
    """A non-zero `lora_B` before the first update means something was LOADED into the policy.
    Whatever it is, the run is then not "DPO from the base" -- and because the reference here
    IS the base, the log-ratio at step 0 would be non-zero, which is the arm's own definition
    of having started somewhere else."""
    with pytest.raises(AmbiguousReference, match="not the bare base"):
        rung2_train.verify_base_reference(
            _Params(_fresh(policy=(0.5, 0.25))), trl_source=_TRL1_SOURCE, equal=_eq
        )


def test_the_base_reference_check_refuses_a_trl_it_cannot_read_the_none_branch_in():
    """Symmetric to `test_a_trl_that_offers_neither_mechanism_is_refused_rather_than_guessed`:
    the base mode rests entirely on TRL taking the `None` branch, so a library whose source
    does not show that line leaves pi_ref unknown."""
    with pytest.raises(rung2_train.UnknownReferenceHook, match="DISABLES"):
        rung2_train.verify_base_reference(
            _Params(_fresh()), trl_source="def compute_loss(self): pass", equal=_eq
        )


def test_a_model_with_no_policy_lora_at_all_is_refused_under_the_base_reference():
    """Nothing is being trained, so nothing is the policy -- and the check would otherwise pass
    by finding no counterexample, which is how a vacuous assertion reads as a green run."""
    with pytest.raises(AmbiguousReference, match="no LoRA"):
        rung2_train.verify_base_reference(_Params({}), trl_source=_TRL1_SOURCE, equal=_eq)


# ------------------------------------------------------------------------------------ cli


def _cli_pairs(tmp_path: Path) -> Path:
    f = tmp_path / "pairs.jsonl"
    f.write_text(json.dumps(_pair()) + "\n")
    return f


def _run_cli(monkeypatch, tmp_path, *flags) -> tuple[int, dict]:
    """`pi train rung2 --train` with the trainer replaced by a recorder. Returns (rc, cfg)."""
    import pinq_train.rung2_dpo as rung2
    from pi_run.cli import build_parser

    seen: dict = {}

    def fake_train(cfg, **kw):
        seen["cfg"] = cfg
        return {"out_dir": cfg.out_dir}

    monkeypatch.setattr(rung2, "train", fake_train)
    args = build_parser().parse_args(
        [
            "train",
            "rung2",
            "--pairs",
            str(_cli_pairs(tmp_path)),
            "--out",
            str(tmp_path / "rung2"),
            "--train",
            "--acknowledge-untested",
            *flags,
        ]
    )
    return args.fn(args), seen


def test_the_cli_defaults_to_the_adapter_reference_and_says_so_in_the_config(
    tmp_path, monkeypatch, capsys
):
    """The flag has to reach `cfg.sha`; a default that lives only in argparse is a run whose
    identity does not record which reference policy it used."""
    rc, seen = _run_cli(monkeypatch, tmp_path, "--base-model", "Qwen/Qwen3-8B", "--adapter", "a")
    capsys.readouterr()
    assert rc == 0
    assert seen["cfg"].reference == "adapter"
    assert seen["cfg"].merge_adapter is False and seen["cfg"].merged_base_sha == ""


def test_the_cli_refuses_a_merged_base_under_the_adapter_reference(tmp_path, monkeypatch, capsys):
    """Otherwise the directory is read, ignored, and the operator believes it selected the
    reference policy it named."""
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "merge.manifest.json").write_text(json.dumps({"output_sha": SHA}) + "\n")
    rc, seen = _run_cli(monkeypatch, tmp_path, "--merged-base", str(merged), "--adapter", "a")
    assert rc == 1 and "cfg" not in seen
    assert "--merged-base is only meaningful" in capsys.readouterr().err


def test_the_cli_still_reaches_the_merged_path_when_it_is_asked_for(tmp_path, monkeypatch, capsys):
    """Kept for serving a fused checkpoint. `merge_adapter` follows --merged-base, because a
    merge that did not happen must not be claimed in the manifest."""
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "merge.manifest.json").write_text(json.dumps({"output_sha": SHA}) + "\n")
    rc, seen = _run_cli(
        monkeypatch,
        tmp_path,
        "--reference",
        "merged",
        "--merged-base",
        str(merged),
        "--adapter",
        "artifacts/rung1",
    )
    capsys.readouterr()
    assert rc == 0
    cfg = seen["cfg"]
    assert cfg.reference == "merged" and cfg.merge_adapter is True
    assert cfg.base_model == str(merged) and cfg.merged_base_sha == SHA


def test_the_cli_reaches_the_base_reference_and_puts_it_in_the_config(
    tmp_path, monkeypatch, capsys
):
    """The dpo_from_base arm is a command line, not a python REPL: `reference` is in cfg.sha,
    so an arm that cannot be spelled as flags is an arm whose identity nobody records."""
    rc, seen = _run_cli(
        monkeypatch, tmp_path, "--reference", "base", "--base-model", "Qwen/Qwen3-8B"
    )
    capsys.readouterr()
    assert rc == 0
    cfg = seen["cfg"]
    assert cfg.reference == "base"
    assert cfg.base_model == "Qwen/Qwen3-8B" and cfg.adapter == ""
    assert cfg.merge_adapter is False and cfg.merged_base_sha == ""
    assert cfg.sha != DPOConfig(base_model="Qwen/Qwen3-8B", adapter="artifacts/rung1").sha


def test_the_cli_refuses_an_adapter_under_the_base_reference(tmp_path, monkeypatch, capsys):
    """`--reference base --adapter artifacts/rung1` is a run that both has and has not got an
    SFT rung in it. Refused with a non-zero exit rather than resolved by a precedence rule."""
    rc, seen = _run_cli(
        monkeypatch,
        tmp_path,
        "--reference",
        "base",
        "--base-model",
        "Qwen/Qwen3-8B",
        "--adapter",
        "artifacts/rung1",
    )
    assert rc == 1 and "cfg" not in seen
    err = capsys.readouterr().err
    assert "PREFLIGHT FAILED" in err and "reference='base'" in err


def test_the_cli_refuses_a_merged_base_under_the_base_reference(tmp_path, monkeypatch, capsys):
    """Same rule as under 'adapter', and for the same reason: `--merged-base` RESOLVES into
    base_model, so a directory named under a mode that merges nothing would either be loaded as
    the base (the opposite of this arm) or read and ignored."""
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "merge.manifest.json").write_text(json.dumps({"output_sha": SHA}) + "\n")
    rc, seen = _run_cli(monkeypatch, tmp_path, "--reference", "base", "--merged-base", str(merged))
    assert rc == 1 and "cfg" not in seen
    err = capsys.readouterr().err
    assert "--merged-base is only meaningful" in err
    assert "base" in err


# ---------------------------------------------------------------------------- integration


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("PI_SMOKE_MODEL"),
    reason="needs PI_SMOKE_MODEL (a small HF id) and the [train] extra; no CUDA here",
)
def test_reference_logprobs_equal_a_freshly_loaded_merged_model(tmp_path):
    """THE PROPERTY THE WHOLE RUNG RESTS ON, measured rather than argued.

    `merge_and_unload()` returns a model in memory; DPO trains against the one reloaded from
    disk. If saving and reloading changed the distribution by more than numerical noise, the
    reference policy would not be the SFT checkpoint and the rung's sentence would be false.
    Run on a box with the [train] extra:  PI_SMOKE_MODEL=Qwen/Qwen3-0.6B pytest -m integration
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = os.environ["PI_SMOKE_MODEL"]
    tok = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32)
    peft_model = get_peft_model(
        model,
        LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    # LoRA initialises B to zero, so an unperturbed merge is the identity and the comparison
    # would be vacuous. Perturb deterministically so the merge genuinely moves the weights.
    torch.manual_seed(0)
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                p.add_(torch.randn_like(p) * 0.02)
    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(str(adapter_dir))

    out = tmp_path / "merged"
    sha = merge.merge_adapter(base_id, str(adapter_dir), str(out))
    assert merge.verify_merged_base(str(out), sha)["verified"] is True

    ids = tok("who founded the institute?", return_tensors="pt")
    in_memory = peft_model.merge_and_unload()
    reloaded = AutoModelForCausalLM.from_pretrained(str(out), torch_dtype=torch.float32)
    with torch.no_grad():
        a = torch.log_softmax(in_memory(**ids).logits, dim=-1)
        b = torch.log_softmax(reloaded(**ids).logits, dim=-1)
    assert torch.max(torch.abs(a - b)).item() < 1e-4


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("PI_SMOKE_MODEL"),
    reason="needs PI_SMOKE_MODEL (a small HF id) and the [train] extra; no CUDA here",
)
def test_the_adapter_reference_is_rung_1_and_so_is_the_policy_at_step_0(tmp_path):
    """THE PROPERTY THE DEFAULT PATH RESTS ON, measured instead of argued.

    Two copies of rung 1 on one base. The frozen one must give exactly rung 1's log-probs --
    that is what makes it `pi_ref` -- and so must the trainable one BEFORE the first update,
    because they start as the same weights. If either drifted, the run would be comparing the
    policy against something that is not the SFT checkpoint, which is the whole failure the
    merge path was introduced to avoid and the fp32 merge only half fixed.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = os.environ["PI_SMOKE_MODEL"]
    adapter_dir = _perturbed_adapter(tmp_path, base_id)
    tok = AutoTokenizer.from_pretrained(base_id)
    ids = tok("who founded the institute?", return_tensors="pt")

    # exactly what `train()` builds
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float32),
        str(adapter_dir),
        adapter_name=rung2_train.POLICY_ADAPTER,
        is_trainable=True,
    )
    model.load_adapter(
        str(adapter_dir), adapter_name=rung2_train.REFERENCE_ADAPTER, is_trainable=False
    )
    # and what rung 1 on its own is
    fresh = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float32), str(adapter_dir)
    )

    with torch.no_grad():
        want = torch.log_softmax(fresh(**ids).logits, dim=-1)
        model.set_adapter(rung2_train.REFERENCE_ADAPTER)
        got_ref = torch.log_softmax(model(**ids).logits, dim=-1)
        model.set_adapter(rung2_train.POLICY_ADAPTER)
        got_policy = torch.log_softmax(model(**ids).logits, dim=-1)

    assert torch.max(torch.abs(got_ref - want)).item() < 1e-4
    assert torch.max(torch.abs(got_policy - want)).item() < 1e-4
    # and the frozen copy is the one that carries no gradient
    grads = {name: p.requires_grad for name, p in model.named_parameters() if "lora_" in name}
    assert any(v for k, v in grads.items() if rung2_train.POLICY_ADAPTER in k)
    assert not any(v for k, v in grads.items() if f".{rung2_train.REFERENCE_ADAPTER}." in k)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("PI_SMOKE_MODEL"),
    reason="needs PI_SMOKE_MODEL (a small HF id) and the [train] extra; no CUDA here",
)
def test_the_installed_trl_leaves_the_reference_adapter_equal_to_rung_1(tmp_path):
    """THE ONE THING A DOUBLE CANNOT CHECK: what the INSTALLED TRL does to the model.

    MEASURED on trl 1.13.0, and it is more than the probe assumed: `DPOTrainer.__init__` builds
    the frozen copy ITSELF when it is handed a PeftModel with a pretrained adapter and no
    `ref_model` -- `default_config = model.peft_config["default"]; model.add_adapter("ref",
    default_config)` and then a parameter-wise copy of every `.default.` tensor into `.ref.`.
    That is why `POLICY_ADAPTER` must be exactly "default": the name is indexed, not inferred.

    The assertion is EXACT rather than statistical. A step-0 DPO loss of log 2 would be the
    prettier check -- at identical weights the two log-ratios cancel -- but on MPS the
    reference forward runs under `no_grad` and the policy's does not, and the measured margin
    at step 0 is ~1e-2 rather than 0, which is the same order as the effect a wrong reference
    would produce on a 5-token completion. Parameter equality has no noise floor.
    """
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    base_id = os.environ["PI_SMOKE_MODEL"]
    adapter_dir = _perturbed_adapter(tmp_path, base_id)
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float32),
        str(adapter_dir),
        adapter_name=rung2_train.POLICY_ADAPTER,
        is_trainable=True,
    )
    model.load_adapter(
        str(adapter_dir), adapter_name=rung2_train.REFERENCE_ADAPTER, is_trainable=False
    )
    DPOTrainer(
        model=model,
        args=TRLDPOConfig(output_dir=str(tmp_path / "trl"), max_length=64, report_to=[]),
        train_dataset=Dataset.from_list([{"prompt": "a", "chosen": "b", "rejected": "c"}]),
        ref_model=None,
        peft_config=None,
    )
    rep = rung2_train.verify_frozen_reference(model)
    assert rep == {"n_lora_pairs": 112, "equal": True, "frozen": True, "is_the_base": False}


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("PI_SMOKE_MODEL"),
    reason="needs PI_SMOKE_MODEL (a small HF id) and the [train] extra; no CUDA here",
)
def test_train_writes_rung_2s_adapter_at_the_root_of_its_output_directory(tmp_path):
    """WHY THE POLICY ADAPTER IS CALLED "default", measured on peft rather than read off it.

    `save_pretrained` writes any other adapter name into a subdirectory, so a policy adapter
    named "train" would put rung 2's checkpoint in `artifacts/rung2/train/` -- where
    eval-offline, `pi train merge` and the checkpoint registry do not look. This runs the real
    `train()` end to end on the installed stack and checks where the file landed.
    """
    base_id = os.environ["PI_SMOKE_MODEL"]
    adapter_dir = _perturbed_adapter(tmp_path, base_id)
    out = tmp_path / "rung2"
    cfg = DPOConfig(
        base_model=base_id,
        adapter=str(adapter_dir),
        out_dir=str(out),
        max_seq_len=64,
        max_prompt_len=32,
        grad_accum=1,
    )
    rung2_train.train(cfg, acknowledge_untested=True, rows=[_pair(), _pair()], drops={})

    assert (out / "adapter_model.safetensors").is_file(), sorted(p.name for p in out.iterdir())
    assert not (out / rung2_train.REFERENCE_ADAPTER).exists(), "the frozen copy was saved too"
    man = json.loads((out / "rung2.manifest.json").read_text())
    ref = man["pairs"]["reference_policy"]
    assert ref["reference"] == "adapter"
    assert ref["reference_adapter_hook"] in ("dpo_config_fields", "trl_named_ref_adapter")
    assert ref["adapter_sha"] == merge.adapter_sha(adapter_dir)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("PI_SMOKE_MODEL"),
    reason="needs PI_SMOKE_MODEL (a small HF id) and the [train] extra; no CUDA here",
)
def test_the_installed_trl_builds_no_ref_adapter_when_it_is_handed_a_peft_config(tmp_path):
    """THE ONE THING A DOUBLE CANNOT CHECK, for the base mode: that the INSTALLED TRL really
    does leave pi_ref as the bare base here.

    Read off trl 1.13.0's constructor: a `peft_config` takes the `get_peft_model` branch, and
    the `elif is_peft_model(model) and ref_model is None` branch that ADDS the "ref" adapter is
    then not reached. So no "ref" exists, `use_adapter(model, adapter_name=... else None)`
    takes the None branch, and the reference forward runs with every adapter disabled -- which
    on an untrained base IS the untrained base. This asserts the model state rather than the
    source: parameter names have no noise floor, and a step-0 loss of log 2 does.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    base_id = os.environ["PI_SMOKE_MODEL"]
    trainer = DPOTrainer(
        model=AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float32),
        args=TRLDPOConfig(output_dir=str(tmp_path / "trl"), max_length=64, report_to=[]),
        train_dataset=Dataset.from_list([{"prompt": "a", "chosen": "b", "rejected": "c"}]),
        ref_model=None,
        peft_config=LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
        ),
    )
    rep = rung2_train.verify_base_reference(
        trainer.model, trl_source=rung2_train._module_source(DPOTrainer)
    )
    assert rep["ref_adapter_absent"] is True
    assert rep["is_the_base"] is True, "a fresh LoRA has lora_B == 0 and IS the base"
    assert rep["n_policy_lora"] > 0


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("PI_SMOKE_MODEL"),
    reason="needs PI_SMOKE_MODEL (a small HF id) and the [train] extra; no CUDA here",
)
def test_train_from_the_base_writes_a_checkpoint_and_a_manifest_that_names_the_base(tmp_path):
    """`train()` end to end on the installed stack in the mode the A14/E39 ablation runs."""
    base_id = os.environ["PI_SMOKE_MODEL"]
    out = tmp_path / "rung2-from-base"
    cfg = DPOConfig(
        base_model=base_id,
        adapter="",
        reference="base",
        out_dir=str(out),
        max_seq_len=64,
        max_prompt_len=32,
        grad_accum=1,
    )
    rung2_train.train(cfg, acknowledge_untested=True, rows=[_pair(), _pair()], drops={})

    man = json.loads((out / "rung2.manifest.json").read_text())
    ref = man["pairs"]["reference_policy"]
    assert ref["reference"] == "base" and ref["adapter"] is None
    assert ref["base_model"] == base_id
    assert ref["reference_adapter_hook"] == "trl_disable_adapters"
    assert ref["verified"]["ref_adapter_absent"] is True


def _perturbed_adapter(tmp_path: Path, base_id: str) -> Path:
    """A rung-1 LoRA that actually moves the weights.

    LoRA initialises B to zero, so an unperturbed adapter is the identity and every comparison
    against the bare base would pass for the wrong reason.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    peft_model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float32),
        LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    torch.manual_seed(0)
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                p.add_(torch.randn_like(p) * 0.02)
    out = tmp_path / "rung1-adapter"
    peft_model.save_pretrained(str(out))
    return out
