"""Where the base model's weights go when there is more than one card.

WHY THIS FILE EXISTS. Rung 1 loaded the base with a bare
`AutoModelForCausalLM.from_pretrained(cfg.base_model)` -- no `device_map`, anywhere in the
trainer. That is exactly right for the 8B (16 GB of bf16 weights on one A100-80GB, measured at
2,241.75 tok/s by job 765560) and it cannot work for a 32B: ~64 GB of bf16 weights do not fit
on one 80 GB card once the optimiser state, the activations and the LoRA are on it too, so the
layers have to be split across two. `device_map="auto"` is accelerate's naive model
parallelism, and it is the ONLY thing in this path that makes the HF Trainer leave the model
alone -- see `test_a_device_mapped_model_is_not_wrapped_in_dataparallel`, which transcribes the
rule from the installed library.

EVERYTHING HERE IS PURE. `base_model_load_kwargs` takes an integer and returns a dict, so the
placement decision is testable on a laptop with no CUDA device and no `transformers` -- the
same split that keeps `training_argument_kwargs` honest. What a GPU is needed for is whether
the split FITS, and that is a probe, not a test.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from pinq_train.rung1_sft.train import SFTConfig, base_model_load_kwargs, training_argument_kwargs

# The dtype the trainer asks for under `bf16: true`, spelled as a string so nothing here
# imports torch. `transformers.modeling_utils.str_to_torch_dtype` resolves it, and
# `pinq_train.merge` already passes its dtype the same way.
BF16 = "bfloat16"


def test_more_than_one_visible_gpu_splits_the_layers_across_them():
    """The 32B case. Without `device_map` the whole 64 GB is loaded onto one card and the load
    itself raises CUDA OOM -- after the tokenizer, the dataset build and the preflight report
    have already run."""
    assert base_model_load_kwargs(2, dtype=BF16) == {"device_map": "auto", "dtype": BF16}


def test_the_dtype_is_stated_rather_than_left_to_the_library():
    """`device_map="auto"` computes the split FROM the dtype, so an unstated dtype decides the
    placement. transformers 5.x defaults to the checkpoint's own (bf16 for every Qwen3) and 4.x
    upcast to float32 -- which for a 32B is 128 GB, more than two 80 GB cards hold, and
    accelerate answers that by offloading layers to CPU rather than by raising. The probe would
    then measure a tokens/s that describes a disk, not a GPU."""
    kw = base_model_load_kwargs(2, dtype=BF16)
    assert kw["dtype"] == BF16
    # The value is carried, not hard-coded: `bf16: false` (the Mac smoke) must not silently
    # become bfloat16 on a box that happens to have two cards.
    assert base_model_load_kwargs(2, dtype="float32")["dtype"] == "float32"


@pytest.mark.parametrize("n", [0, 1])
def test_one_card_or_none_loads_exactly_as_it_did_before(n):
    """UNCHANGED, and that is the point: the 8B run of record (job 765560) and the 4B/8B
    headline arms all load on one card, and a placement kwarg that appeared on that path would
    change how a finished arm's weights were loaded without changing anything that names it.
    Zero is the Mac and the gate venv, where `torch.cuda.device_count()` is 0."""
    assert base_model_load_kwargs(n, dtype=BF16) == {}


# --------------------------------------------------------------------------------------------
# THE DATAPARALLEL RULE, TRANSCRIBED FROM THE INSTALLED LIBRARY.
#
# transformers 5.17.0 (the training venv on HPC; `torch 2.14.0`, `peft 0.20.0`,
# `accelerate 1.15.0`), read on the cluster at
# `~/ProactiveInquirer/.venv/lib/python3.*/site-packages/transformers/`:
#
#   trainer.py:449-455   self.is_model_parallel = False
#                        if getattr(model, "hf_device_map", None) is not None:
#                            devices = [d for d in set(model.hf_device_map.values())
#                                       if d not in ["cpu", "disk"]]
#                            if len(devices) > 1: self.is_model_parallel = True
#   trainer.py:489-491   # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
#                        if self.is_model_parallel: self.args._n_gpu = 1
#   trainer.py:2543-2548 if (self.args.n_gpu > 1 and not ...is_loaded_in_8bit...):
#                            model = nn.DataParallel(model)
#   training_args.py:1917  self._n_gpu = torch.cuda.device_count()
#
# `hf_device_map` is set by `from_pretrained` only when it was given a `device_map`. So on two
# visible cards the bare load takes the second branch: `n_gpu == 2`, `nn.DataParallel(model)`,
# and DataParallel REPLICATES -- two full copies of a model that did not fit once. (The same
# `is_model_parallel` also gates `place_model_on_device`, trainer.py:465-477, which is what
# stops the Trainer calling `.to(cuda:0)` on a model whose layers are deliberately not there.)
# --------------------------------------------------------------------------------------------


def _transformers_would_wrap_in_dataparallel(load_kwargs: dict, n_visible_gpus: int) -> bool:
    """The rule above, as a predicate over what rung 1 passes to `from_pretrained`."""
    hf_device_map = load_kwargs.get("device_map") is not None
    is_model_parallel = hf_device_map and n_visible_gpus > 1
    n_gpu = 1 if is_model_parallel else n_visible_gpus
    return n_gpu > 1


@pytest.mark.parametrize("n", [2, 4, 8])
def test_a_device_mapped_model_is_not_wrapped_in_dataparallel(n):
    """The failure the `device_map` exists to prevent, stated as the library states it."""
    assert not _transformers_would_wrap_in_dataparallel(base_model_load_kwargs(n, dtype=BF16), n)


@pytest.mark.parametrize("n", [2, 4, 8])
def test_the_rule_has_teeth_the_bare_load_is_the_one_that_gets_replicated(n):
    """Without this the test above passes for a predicate that can never be true. The bare load
    -- what the trainer did before -- is what `nn.DataParallel` replicates."""
    assert _transformers_would_wrap_in_dataparallel({}, n)


def test_the_trainer_arguments_do_not_countermand_the_placement():
    """A model split across cards is undone by one TrainingArguments field. `device_map` is not
    a TrainingArguments field at all; `place_model_on_device=True` would move the whole model to
    cuda:0, and trainer.py:465-477 honours it VERBATIM whenever it is not None -- leaving it
    unset is what lets the model-parallel branch decide; and `use_cpu`/`no_cuda`/`fsdp`/
    `deepspeed` each pick a different strategy entirely. None of them is emitted today; this
    pins that, because the symptom of emitting one is an OOM inside the Trainer constructor that
    reads like a memory problem rather than a config one."""
    placement_fields = {
        "device_map",
        "place_model_on_device",
        "use_cpu",
        "no_cuda",
        "fsdp",
        "deepspeed",
        "parallel_mode",
        "ddp_backend",
        "local_rank",
        "n_gpu",
        "_n_gpu",
    }
    cfg = SFTConfig(base_model="Qwen/Qwen3-32B", tau=0.05, sigma_j=0.0, max_seq_len=5120)
    # Both sides of the transformers 5.0 break, as in
    # tests/test_trainer_arguments_across_transformers.py: the field set is the caller's.
    for field_names in (placement_fields | {"output_dir"}, {"output_dir"}):
        emitted = set(training_argument_kwargs(cfg, field_names)) & placement_fields
        assert not emitted, f"would hand the Trainer {sorted(emitted)} and override the split"


def test_the_placement_is_not_part_of_the_checkpoints_identity():
    """`cfg.sha` is WHAT was trained -- rows, hyperparameters, base model. How many cards the
    weights were spread over is a property of the MACHINE, like `wall_ms` and `usd`: recorded in
    the manifest, never part of the identity. If placement entered `cfg.sha`, one recipe run on
    one card and on two would be two experiments no table could rejoin -- and the 8B arms of
    record would have to be re-run to be comparable with a 32B arm.

    Stated structurally rather than by pinning a literal digest: `cfg.sha` legitimately moves
    when a HYPERPARAMETER is added, so a pinned digest here would fire on somebody else's
    correct change and say nothing about placement."""
    cfg = SFTConfig(base_model="Qwen/Qwen3-32B", tau=0.05, sigma_j=0.0, max_seq_len=5120)
    before = cfg.sha
    # The placement differs between one card and two ...
    assert base_model_load_kwargs(1, dtype=BF16) != base_model_load_kwargs(2, dtype=BF16)
    # ... and the config that names the checkpoint does not know which happened.
    assert cfg.sha == before
    names = {f.name for f in fields(SFTConfig)}
    machine_fields = names & {"device_map", "device", "n_gpus", "n_visible_gpus", "placement"}
    assert not machine_fields, f"{sorted(machine_fields)} is in cfg.sha: the box renames the run"


# --------------------------------------------------------------------------------------------
# `lora_capacity`: THE TRAINABLE/BASE PARAMETER COUNT AND THE RESOLVED TARGET-MODULE LIST,
# WRITTEN BESIDE `placement`, NOT INTO `config`.
#
# Same reasoning as `test_the_placement_is_not_part_of_the_checkpoints_identity` above, for a
# different discovery: architecture, target_modules_resolved, trainable_params and total_params
# are all learned only AFTER the base model has loaded -- the class name transformers gave it,
# what `resolve_lora_targets` narrowed the CONFIGURED modules to for THAT architecture, and what
# `get_nb_trainable_parameters()` reports on the peft-wrapped result -- never something the run
# was asked for. None of them belongs in `cfg.sha`.
# --------------------------------------------------------------------------------------------


def test_lora_capacity_does_not_move_the_config_sha():
    """Structural check first, exactly as the placement test above does it, and for the same
    reason: `cfg.sha` legitimately moves when a real HYPERPARAMETER is added, so a test that
    only pins a literal digest would fire on somebody else's correct, unrelated change and say
    nothing about capacity leaking into identity.

    `sha` (defined above `SFTConfig`, ~line 267) is `h("rung1", canon(d))` where
    `d = {k: v for k, v in asdict(self).items() if k not in SHA_EXCLUDED}` -- a pure function of
    `asdict(self)`, i.e. of `SFTConfig`'s own dataclass fields and nothing else. architecture,
    target_modules_resolved, trainable_params, total_params and lora_capacity are none of them a
    field of SFTConfig, so there are no bytes belonging to any of them for `asdict` to see in
    the first place -- verified by checking dataclass field membership rather than by reading
    the sha's algorithm, so this stays true even if `sha`'s hash function changes later.

    The live digest below is a second, narrower check: not "capacity can't enter the sha" in
    general, but "this specific config produces this specific sha today." Pinned 2026-09-17,
    immediately after `lora_capacity` was added to `train()` -- confirmed by `git diff -U0 HEAD
    -- src/pinq_train/rung1_sft/train.py`, whose hunks all start at line 583 or later, while
    `SFTConfig`, the `sha` property and the SHA_* constants sit at lines 158-267, entirely
    outside every hunk. Recomputed fresh three times across this change (before touching
    train.py, immediately after, and again just now, in a fresh process each time): identical
    all three times. If a future, real hyperparameter changes this literal value, that is
    expected and only this one assertion needs updating -- the field-set assertion above it
    does not."""
    names = {f.name for f in fields(SFTConfig)}
    capacity_fields = names & {
        "architecture",
        "target_modules_resolved",
        "trainable_params",
        "total_params",
        "lora_capacity",
    }
    assert not capacity_fields, f"{sorted(capacity_fields)} is in cfg.sha: capacity renames the run"
    cfg = SFTConfig(base_model="Qwen/Qwen3-8B", tau=0.05, sigma_j=0.0, max_seq_len=5120)
    assert cfg.sha == "2db1eeaa943c71a077ae2d5ef34a8c778b4f8be7bb87dff25431cca347a99bdd"


# --------------------------------------------------------------------------------------------
# THE TRAINABLE/BASE PARAMETER COUNT ITSELF, CROSS-CHECKED AGAINST A SECOND, INDEPENDENTLY-RUN
# PROBE ON THE TWO REAL BASE MODELS THIS REPO TRAINS.
#
# Gated behind pytest.importorskip, but do not assume that means "normally skipped": as of
# 2026-09-17 both `.venv` (this gate's own venv -- peft 0.21.0) and `.venv-train` (peft 0.20.0)
# have torch/peft/accelerate/transformers installed, and both produce IDENTICAL counts for
# every figure pinned below, so this runs for real in the gate, not only in `.venv-train`.
# Meta-device: no weight bytes are ever materialised, so "real ML libraries" does not mean
# "needs a GPU" here -- confirmed to run in well under 5s on a laptop CPU.
# --------------------------------------------------------------------------------------------


def _cfg_targets_expected() -> tuple[str, ...]:
    """The seven target modules `LoraSpec`'s defaults configure, unchanged by
    `resolve_lora_targets` for a non-GptOss architecture (see `resolve_lora_targets`,
    train.py ~line 128: it only ever narrows, and only for `GptOss*`). Named as a function
    rather than a module-level constant so a reader sees where the seven names come from
    (`SFTConfig().lora.target_modules`) rather than a second, easily-stale copy of them."""
    return SFTConfig(base_model="x", tau=0.0, sigma_j=0.0, max_seq_len=1).lora.target_modules


def _cached_config(model_id: str):
    """`local_files_only=True` makes a network fetch impossible, not merely unlikely -- a cache
    miss is an OSError, never a download -- so this cannot spend anything even by accident.
    Skips (an environment fact) rather than fails (a code defect) on a machine that has never
    loaded this model's config before."""
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(model_id, local_files_only=True)
    except OSError:
        pytest.skip(f"{model_id}: config.json is not in the local Hugging Face cache")


def _lora_capacity_via_meta_device(model_id: str) -> dict:
    """`train()`'s own sequence, verbatim (tie_weights -> name the architecture -> resolve the
    configured target modules for THAT architecture -> get_peft_model -> read
    get_nb_trainable_parameters), against a meta-device skeleton instead of `cfg.base_model`'s
    real weights. Parameter counts are shape-derived (`.numel()` over `named_parameters()`,
    filtered by `requires_grad`), so a meta tensor and a real bf16 tensor of the same shape
    report the identical number -- a weight-free build is a faithful stand-in for train()'s real
    load here, not an approximation of it. Built from `cfg.lora` (the repository's actual
    `LoraSpec` defaults), not a second, hand-written copy of r/alpha/dropout/bias/targets/
    task_type -- so this tracks `train()`'s real LoRA config automatically, and if that config's
    defaults ever change, the literal numbers pinned below should fail loudly rather than
    silently keep testing a stale config."""
    import torch
    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from pinq_train.rung1_sft.train import resolve_lora_targets

    cfg = SFTConfig(base_model=model_id, tau=0.05, sigma_j=0.0, max_seq_len=5120)
    hf_cfg = _cached_config(model_id)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(hf_cfg, dtype=torch.bfloat16)
    model.tie_weights()
    architecture = type(model).__name__
    target_modules_resolved = resolve_lora_targets(architecture, cfg.lora.target_modules)
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=list(target_modules_resolved),
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
        ),
    )
    trainable_params, total_params = model.get_nb_trainable_parameters()
    return {
        "architecture": architecture,
        "target_modules_resolved": target_modules_resolved,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }


def test_qwen3_8b_trainable_and_total_match_the_measured_cross_check():
    """SOURCED, independently, to a second script this repo does not own: a cluster-resident
    probe at `/tmp/pinq-hpc/probe_qwen_lora.py` (login node, read there by the coordinator, not
    by this test suite -- out of reach of the "no cluster" constraint this file is written
    under). Its method, reported verbatim: offline HF cache, `CUDA_VISIBLE_DEVICES` emptied,
    `AutoConfig.from_pretrained("Qwen/Qwen3-8B")`, `init_empty_weights()` +
    `AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)`, `get_peft_model` at
    this same r=32/alpha=64/dropout=0.05/bias=none/CAUSAL_LM/seven-target config, summed over
    `peft_model.parameters()` (not `named_parameters()`, and no `tie_weights()` call) -- on
    transformers 5.17.0, the same version this test runs on. It reports 87,293,952 trainable of
    8,278,029,312 total, 1.0545%: identical to the figure this test pins.

    Tie state cannot be the reason the two methods agree, because there is nothing here for a
    tie to change: Qwen3-8B has `tie_word_embeddings=False`. Measured directly, twice (this
    function, and again with an explicit `tie_weights()` call spliced into
    `_lora_capacity_via_meta_device` for a one-off check): total_params is bit-for-bit identical
    whether or not `tie_weights()` is called, in both `.venv` (peft 0.21.0) and `.venv-train`
    (peft 0.20.0). So `.parameters()` vs. deduplicated `named_parameters()` and tied vs. untied
    construction are both moot for this architecture -- there is only one tensor in the pair
    either way, and both counting routes see it once."""
    pytest.importorskip(
        "peft",
        reason="peft/accelerate/transformers live in .venv-train and, as of 2026-09-17, in "
        ".venv too; either way this must skip cleanly on a machine that has neither",
    )
    result = _lora_capacity_via_meta_device("Qwen/Qwen3-8B")
    assert result["architecture"] == "Qwen3ForCausalLM"
    assert result["target_modules_resolved"] == _cfg_targets_expected()
    assert result["trainable_params"] == 87_293_952
    assert result["total_params"] == 8_278_029_312


def test_granite_3_3_8b_total_requires_the_tie_weights_call_to_be_correct():
    """Granite-3.3-8B has `tie_word_embeddings=True`: `lm_head.weight` and
    `model.embed_tokens.weight` are meant to be the SAME tensor.

    THIS FIGURE'S EARLIER CITATION IS SUPERSEDED. 8,269,824,000 total / 98,959,360 trainable /
    1.1966% had been reported elsewhere as this architecture's count, attributed to the same
    cluster probe that produced the Qwen figure above. It is not: that probe only ever contained
    Qwen (verified by the coordinator directly, who has cluster access and read the one Python
    file at that path; this test suite does not and did not access the cluster to check). No
    second script for Granite exists anywhere reachable. Under rule 1 that made the 1.1966%
    figure, as originally cited, a number without provenance -- the citation is withdrawn, not
    the number.

    THE NUMBER ITSELF IS NOW SOURCED TO THIS TEST, measured directly rather than assumed: built
    Granite fresh under `init_empty_weights()` + `from_config()` (the same documented path the
    Qwen probe uses), checked `lm_head.weight is model.embed_tokens.weight` immediately on
    return -- False, despite `config.tie_word_embeddings` already True and
    `model.all_tied_weights_keys` already holding the correct mapping
    (`{'lm_head.weight': 'model.embed_tokens.weight'}`). That construction, carried through to
    `get_peft_model` and summed, gives total_params=8,471,179,264 -- double-counting
    `model.embed_tokens.weight` (shape (49159, 4096), 201,355,264 elements: the untied/tied
    difference below matches this to the last digit). Calling `model.tie_weights()` explicitly
    before `get_peft_model` -- what `train()` does, and what this test's helper does -- makes
    that identity True and total_params=8,269,824,000, trainable_params=98,959,360 unchanged
    (the LoRA adapter's own parameters never touch the tied pair). 8,269,824,000 is exactly
    8,471,179,264 minus 201,355,264: one embedding matrix, to the parameter. This is the correct
    treatment for a tied architecture, and it is what this test pins.

    A THIRD, UNRELATED NUMBER ALSO EXISTS AND IS NOT THIS ONE, AND IT BELONGS TO THE OTHER BASE:
    a separate probe derived a base total of 8,190,735,360 for QWEN from `model.safetensors.index.json`
    reporting 16,381,470,720 bytes at bf16 (2 bytes/param) -- a byte-accounting route that never
    constructs a model or calls anything named `tie_weights` at all, and the source of an
    earlier, different, since-corrected 1.0658% Qwen share. The arithmetic settles the attribution
    on its own: 8,278,029,312 minus 8,190,735,360 is 87,293,472 short of nothing, it is exactly
    87,293,952, the Qwen adapter itself, so the two denominators are base-plus-adapter and base-only
    for the SAME base rather than totals for two different ones. An earlier version of this docstring
    attributed that total to Granite and was wrong. It is named here only so the next reader
    who finds a THIRD Granite total in this project's history knows where it came from and that
    it answers a different question (bytes on disk) than this test does (parameters peft would
    train and count).

    THE ORDER-OF-CONSTRUCTION FINDING, worth recording because it generalises past this one
    architecture: `transformers.PreTrainedModel.post_init()` calls `self.init_weights()`
    unconditionally at the end of every model's construction, and `init_weights()`
    (modeling_utils.py) reads:

        if get_torch_context_manager_or_global_device() != torch.device("meta"):
            self.initialize_weights()
        self.tie_weights(recompute_mapping=False)

    -- the meta-device guard gates only the random-init half; `tie_weights(recompute_mapping=
    False)` is called regardless. Reading that source predicts an automatic tie even under
    `init_empty_weights()`. Measured behaviour contradicts it: `tied_before_call` is False as
    reported above, and calling the IDENTICAL method with the IDENTICAL argument a second time,
    manually, immediately afterward, changes it to True. So the automatic call exists in source
    but does not take effect for this architecture, on this installed version (transformers
    5.17.0, in both `.venv` and `.venv-train`), under `init_empty_weights()` -- while the same
    call issued again after full construction succeeds. Checked, not assumed, that this is not
    `peft` doing it either: grepped the installed `peft` package directly for `tie_weights`,
    zero matches anywhere in `get_peft_model`/`PeftModel`/`LoraModel`. Anyone counting parameters
    on a meta device for a tied architecture will hit this; call `tie_weights()` explicitly
    after construction and do not rely on `post_init` to have already done it."""
    pytest.importorskip(
        "peft",
        reason="peft/accelerate/transformers live in .venv-train and, as of 2026-09-17, in "
        ".venv too; either way this must skip cleanly on a machine that has neither",
    )
    result = _lora_capacity_via_meta_device("ibm-granite/granite-3.3-8b-instruct")
    assert result["architecture"] == "GraniteForCausalLM"
    assert result["target_modules_resolved"] == _cfg_targets_expected()
    assert result["trainable_params"] == 98_959_360
    assert result["total_params"] == 8_269_824_000
