"""Rung 2 -- DPO/KTO on SAME-STATE preference pairs.

WHY SAME-STATE PAIRING IS THE WHOLE POINT, AND NOT A CONVENIENCE
================================================================
Write the advantage of an action at a state in the usual way:

    A(s_t, a) = Q(s_t, a) - V(s_t)

A terminal-reward policy gradient never observes `V(s_t)`. It must ESTIMATE it -- with a
learned critic, a batch mean, or a leave-one-out baseline -- from returns collected across
DIFFERENT tasks. In this experiment that estimate is hopeless, and it is hopeless for a
measurable reason rather than a stylistic one: the between-task variance of `V(s_t)` is set by
how hard the tasks are, and on these suites that spread is roughly an order of magnitude
larger than the effect the paper is trying to detect. The REML decomposition of the pilot
shows it numerically -- task variance dominates arm variance -- so a gradient signal of the
size we care about arrives buried under a baseline error several times its magnitude. More
rollouts shrink the noise as 1/sqrt(n) and never remove the bias in a mis-specified baseline.

DPO on two candidates drawn at the SAME `s_t` does not estimate `V(s_t)`. It cancels it
EXACTLY. The preference objective depends only on the difference of the two log-ratios:

    log pi(a+|s) - log pi_ref(a+|s)  -  [ log pi(a-|s) - log pi_ref(a-|s) ]

`s` is literally the same token sequence in both terms, so every quantity that is a function
of the state alone -- the task's difficulty, the size of its evidence pool, the drafter's
prior, `V(s_t)` itself -- appears identically in both and subtracts to zero. Nothing has to be
modelled, tuned, or estimated. That is why rung 2 costs ~25 GPU-h and rung 3 costs $2,400 for
a result the design is honest enough to expect might not converge: rung 2 is asking a question
the data can answer with the variance it has.

Two consequences, both enforced below:

  * `assert_same_state` REFUSES a pair whose two states are not byte-identical. A pair whose
    states differ by even a whitespace is not a within-state contrast, and the cancellation
    argument silently stops applying while the loss keeps decreasing.
  * the LENGTH GUARD (`|len(chosen) - len(rejected)| <= 40` characters, applied by
    `export_pairs`) survives into training. Without it the preference model learns "longer
    question wins" -- the same confound that makes a naive LLM judge unusable, imported
    straight into the policy's weights.

WHAT THE REFERENCE POLICY IS. Rung 1's LoRA is MERGED into the weights before this runs
(`pinq_train.merge`), `cfg.base_model` points at the merge output and `cfg.merged_base_sha`
names it. Nothing loads an adapter here, so the only adapter in the process is the fresh one
DPO trains and TRL's reference pass -- which disables the adapter -- disables exactly that one.
The previous arrangement loaded rung 1's adapter AND passed a fresh `peft_config`, which left
"the adapter" naming two things; both readings train and only one makes the rung's sentence
true. See `merge.py`.

WHAT THIS RUNG REUSES. Nothing new is rolled out: rung 1 already scored N candidates at every
state, and the pairs are those same candidates re-read from `data/rl/pairs.jsonl`. That is the
entire reason the rung is cheap.

HARDWARE HONESTY. As with rung 1, everything except `train()` runs and is tested on a laptop.
`train()` has never executed -- there is no local CUDA device -- and it refuses to start
unless the caller acknowledges that.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Container, Iterable, Sequence

from pinq.actions import STOP_ACTION_JSON
from pinq.ids import canon, h
from pinq_train.export.dataset import question_len_delta
from pinq_train.merge import adapter_sha, verify_merged_base
from pinq_train.resume import latest_checkpoint
from pinq_train.rung1_sft.mask import IGNORE_INDEX, Tokenizer, build_masked_example
from pinq_train.rung1_sft.train import SHA_EXCLUDED, LoraSpec, NotValidatedOnHardware


class AmbiguousReference(ValueError):
    """A configuration in which `pi_ref` names more than one distribution.

    Fatal rather than resolved by a default: both readings train to completion and report the
    same metrics, so the run would produce a checkpoint and a manifest under a claim
    ("DPO moved the policy from the SFT checkpoint") that nothing in the artifact can check.
    """


class HeldOutDataset(RuntimeError):
    """A pairs file carrying rows the split says are not trainable.

    Fatal rather than filtered. A file whose rows are held out is not a training file with
    some bad rows in it -- it is the wrong file, and quietly dropping the offending rows
    would train on whatever remained while reporting a clean load. Contamination is not
    recoverable after the fact: every confirmatory number computed from the checkpoint is
    void and nothing in the output says so.
    """


class TooFewStopPairs(RuntimeError):
    """The run declared a STOP-share floor its pairs cannot meet. See `min_stop_chosen_share`."""


class TooFewPairsToSubsample(RuntimeError):
    """`subsample_pairs` asks for more rows than the filters left.

    A RAISE, not "take what there is". The whole point of the field is that this arm's n equals
    another arm's n; an arm that quietly trained on fewer is a matched control that is not
    matched, reporting a `config_sha` that says it is -- and `n_pairs` in the manifest would be
    the only trace, in a number no reader has a reason to compare against anything.
    """


class NotSameState(ValueError):
    """A "preference pair" whose two candidates were not drawn at the same state.

    Fatal rather than filtered: a mixed dataset trains a model whose objective is partly the
    within-state contrast and partly an uncontrolled cross-state comparison, and no metric on
    the training curve can tell you which part moved.
    """


# What a pair can be about. "ask_ask" and "ask_stop" are SAMPLED contrasts -- two candidates
# drawn at one state. "ask_stop_synth" is a STOP the exporter derived from gold at a state
# where no candidate stopped: a true statement (the task was already complete) but not a
# sampled one, and it outnumbered the recorded STOPs 18,227 to 627 on the live corpus. See
# `DPOConfig.include_pair_kinds`, which defaults to the sampled kinds only.
PAIR_KINDS = ("ask_ask", "ask_stop", "ask_stop_synth")
SAMPLED_PAIR_KINDS = ("ask_ask", "ask_stop")
# Every kind with a STOP on one side. The length guard is meaningless for all of them -- an
# 18-byte constant against a 40-120 character question -- and naming the SET rather than one
# member is what stops the next kind from silently losing the exemption, which is exactly how
# splitting `ask_stop_synth` out broke rung 2's preflight on 18,227 pairs.
STOP_PAIR_KINDS = ("ask_stop", "ask_stop_synth")
# The diversity sidecar's ordering, least distinct first. `min_q_distinctness` names a FLOOR.
Q_DISTINCTNESS = ("paraphrase", "related", "different")
# Who ordered the pair. "rule" is the exporter's own precedence; "rater" is a pair the rule
# REFUSED -- below its margin floor, or cut by the per-state cap -- that an A6 rater majority
# could order. They are different evidence and the paper compares them, so a run must be able
# to take one, the other or both. A row without the field predates it and is a rule pair.
LABEL_SOURCES = ("rule", "rater")
# HOW `pi_ref` IS OBTAINED. "adapter" is the default and needs no merge: one base in memory and
# rung 1's LoRA loaded TWICE, a trainable copy that DPO optimises and a frozen copy that IS the
# reference. "merged" is the older path -- rung 1 folded into the weights -- kept because a
# SERVING deployment wants one fused checkpoint with no adapter at inference time, and because
# a run whose LoRA shape must differ from rung 1's cannot reuse rung 1's adapter as its policy.
# "base" is the ABLATION's mode and not a third way to spell either: there is no rung 1 in the
# run at all, so the policy is a fresh LoRA on the UNTRAINED base and `pi_ref` is that same
# untrained base. It answers "is the SFT rung necessary before preference learning" -- what the
# preference label alone buys, with no imitation in front of it. It is not "merged" with the
# merge left out: `merge_adapter=False` under "merged" says base_model is ALREADY the policy
# (merged elsewhere, or a full-weight fine-tune), which is a claim about a checkpoint that was
# trained, and this arm's whole point is that its starting policy was not.
REFERENCES = ("adapter", "merged", "base")
# THE POLICY ADAPTER'S NAME, AND WHY IT IS NOT "train". Two independent measurements, either of
# which alone would settle it:
#
#   * trl 1.13.0 INDEXES the name. `DPOTrainer.__init__`, handed a PeftModel with a pretrained
#     adapter and no `ref_model`, builds the frozen reference copy itself with
#     `default_config = model.peft_config["default"]`, `model.add_adapter("ref", default_config)`
#     and a parameter-wise copy of every tensor whose name contains `".default."`. A policy
#     adapter called "train" raises KeyError there -- or, on a TRL that guards it, leaves no
#     reference copy at all and falls through to the bare base.
#   * `PeftModel.save_pretrained` writes any adapter whose name is not "default" into a
#     SUBDIRECTORY of the output (peft 0.20, `peft_model.py`: `output_dir = join(save_directory,
#     adapter_name) if adapter_name != "default"`). A policy adapter called "train" would put
#     rung 2's checkpoint in `artifacts/rung2/train/`, where `pi train eval-offline`, `pi train
#     merge` and the checkpoint registry do not look.
POLICY_ADAPTER = "default"
# THE REFERENCE ADAPTER'S NAME, AND WHY IT IS NOT "reference". MEASURED on trl 1.13.0: the
# reference forward is `use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else
# None)` -- three occurrences in `trl/trainer/dpo_trainer.py`. Under any other name TRL finds
# nothing, takes the `None` branch, DISABLES ALL ADAPTERS, and `pi_ref` becomes the bare base:
# the run still trains, still reports a falling loss and a rising reward margin, and the
# paper's sentence is false with nothing in the artifact to say so.
#
# THE COPY IS LOADED HERE ANYWAY, even though trl 1.13.0 would build it. Older TRL -- the one
# that takes `ref_adapter_name` on its config -- does NOT, and a rung that only works on the
# version that happens to be installed is a rung that breaks silently on the rented box. When
# both do it the weights are the same weights, which `verify_frozen_reference` checks exactly
# rather than assuming.
REFERENCE_ADAPTER = "ref"
# The same line, as something a probe can recognise. See `reference_hook`.
_TRL_SELECTS_NAMED_REF = re.compile(
    r'adapter_name\s*=\s*(["\'])ref\1\s+if\s+\1ref\1\s+in\s+[\w.]*peft_config'
)


class UnknownReferenceHook(RuntimeError):
    """The installed TRL offers no mechanism by which the frozen adapter can be the reference.

    Refused rather than attempted: a TRL that neither accepts the adapter names on its config
    nor selects one called "ref" computes `pi_ref` with every adapter disabled, which is the
    BARE BASE. That run trains, reports a healthy loss curve, writes a checkpoint, and makes
    rung 2's claim false -- and no artifact it produces records which reference it used.
    """


def reference_hook(field_names: Container[str], trl_source: str) -> str:
    """WHICH mechanism makes the frozen copy the reference, on the TRL that is INSTALLED.

    Probed and returned rather than assumed, for the same reason `max_prompt_length` is: what
    `train()` hands TRL is the one thing in this rung no laptop test can execute, and the two
    spellings are a version apart.

      * `dpo_config_fields` -- `DPOConfig` declares `model_adapter_name`/`ref_adapter_name`
        (TRL 0.9-era through early 1.x). The names are passed EXPLICITLY.
      * `trl_named_ref_adapter` -- those fields are gone (trl 1.13.0 declares 137 fields and
        neither is among them) and the trainer instead looks for an adapter literally called
        "ref". The name is the contract, so `REFERENCE_ADAPTER` is that name and both
        mechanisms agree on it.

    Anything else raises. See `UnknownReferenceHook` for why that is not excessive caution.
    """
    if "model_adapter_name" in field_names and "ref_adapter_name" in field_names:
        return "dpo_config_fields"
    if _TRL_SELECTS_NAMED_REF.search(trl_source):
        return "trl_named_ref_adapter"
    raise UnknownReferenceHook(
        "the installed trl neither declares model_adapter_name/ref_adapter_name on DPOConfig "
        f"nor selects an adapter named {REFERENCE_ADAPTER!r} for its reference forward, so "
        "there is no way to make the frozen rung-1 copy pi_ref. TRL would disable every "
        "adapter instead, making the reference the BARE BASE -- which trains and reports a "
        "healthy loss curve while rung 2's claim is false. Pin a supported trl, or run this "
        "rung with --reference merged (`pi train merge` first)."
    )


def base_reference_hook(trl_source: str) -> str:
    """WHICH mechanism makes the BARE BASE `pi_ref`, on the TRL that is INSTALLED.

    `reference="base"` wants exactly what `reference_hook` exists to prevent -- the reference
    forward run with every adapter DISABLED -- so it is the same line, read for its other
    branch: `use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None)`
    with no adapter called "ref" anywhere takes `None`, and `use_adapter(None)` is
    `model.disable_adapter()`. MEASURED on trl 1.13.0 (`trl/trainer/dpo_trainer.py`, three
    occurrences; `trl/trainer/utils.py:1261`).

    THE FIELD PROBE IS NOT ACCEPTED HERE, and that is deliberate rather than an omission. A TRL
    that still declares `model_adapter_name`/`ref_adapter_name` would reach the base through its
    own null-reference context when neither is passed -- probably; this repository has only trl
    1.13.0 installed and cannot measure it. Refusing a run whose `pi_ref` cannot be shown is the
    same trade `UnknownReferenceHook` already makes: the alternative trains, reports a healthy
    curve, and leaves the ablation's headline sentence unverifiable.
    """
    if _TRL_SELECTS_NAMED_REF.search(trl_source):
        return "trl_disable_adapters"
    raise UnknownReferenceHook(
        "reference='base' needs a trl whose reference forward DISABLES every adapter when none "
        f"is named {REFERENCE_ADAPTER!r}, and the installed one does not show that line "
        '(trl 1.13.0: `use_adapter(model, adapter_name="ref" if "ref" in '
        "model.peft_config else None)`). Without it pi_ref is whatever that library does "
        "instead, which is the one thing this mode is a claim about. Pin a supported trl, or "
        "run the arm you actually meant with --reference adapter."
    )


# ---- THE OBJECTIVE, AND WHICH NAMES THE INSTALLED LIBRARY WILL ACTUALLY RUN
#
# MEASURED on trl 1.13.0 (the training venv): the loss ladder in `trl/trainer/dpo_trainer.py`
# dispatches these fifteen names and no others (lines 1472-1589), and `DPOConfig.loss_type` is
# `list[str]` there (dpo_config.py:241) with `loss_weights` (:250) and `ld_alpha` (:259) beside
# it.
#
# `kto_pair` IS NOT AMONG THEM. This config's previous validator accepted exactly two values,
# "sigmoid" and "kto_pair", and TRL has removed the second. A run configured with it validated,
# preflighted, loaded an 8B base and two adapters, and only then raised `ValueError: Unknown
# loss type` from inside the loss -- the `max_prompt_length` failure again, discovered in the
# most expensive place there is. KTO on these pairs is `trl.KTOTrainer`: a different trainer
# with a different dataset shape, not a value of this field.
LOSS_TYPES = (
    "sigmoid",
    "hinge",
    "ipo",
    "exo_pair",
    "nca_pair",
    "robust",
    "bco_pair",
    "sppo_hard",
    "aot",
    "aot_unpaired",
    "apo_zero",
    "apo_down",
    "discopop",
    "sft",
    "sigmoid_norm",
)
# Plain DPO. A tuple because the field is a tuple; the single-element case renders in `cfg.sha`
# as the bare string the old `loss_type: str` rendered. See `DPOConfig.sha`.
DEFAULT_LOSS_TYPE = ("sigmoid",)
# The ladder itself, as something a probe can read. The `else:` arm of that same ladder repeats
# every name inside one f-string in SINGLE quotes, so this pattern cannot match it -- a set
# scraped from the error message would be right by coincidence and would go on reading as
# "right" after the ladder it claims to describe had changed.
_TRL_LOSS_DISPATCH = re.compile(r'\bloss_type == "([a-z0-9_]+)"')


def installed_loss_types(trl_source: str) -> frozenset[str]:
    """The loss names the installed `DPOTrainer` dispatches, read off its own source.

    EMPTY MEANS "COULD NOT TELL", NOT "SUPPORTS NOTHING", and every caller has to treat it that
    way: refusing every name on a library this pattern failed to parse would stop a legitimate
    run, which is the same size of mistake as letting an unsupported one reach the GPU.
    """
    return frozenset(_TRL_LOSS_DISPATCH.findall(trl_source))


class UnsupportedObjective(ValueError):
    """The installed TRL cannot express the objective this config names.

    Distinct from a plain validation failure because the two are different facts with different
    fixes: `validate()` refuses a config that is wrong on its own terms, this refuses one that
    is coherent and that THIS library cannot run. Only the first is fixed by editing the config.
    """


# OPTIONAL KNOBS AT THEIR ABSENT DEFAULT ARE NOT PART OF THE IDENTITY -- rung 1's
# `SHA_OMIT_WHEN_NONE` argument, applied to rung 2's own optional fields. Rung 1's set is
# deliberately NOT imported: it names rung-1 fields, and a set naming fields this dataclass does
# not have would silently do nothing here while looking like it did something.
SHA_OMIT_WHEN_NONE = frozenset({"loss_weights", "ld_alpha", "subsample_pairs", "subsample_seed"})


@dataclass(frozen=True, slots=True)
class DPOConfig:
    # THE MERGED rung-1 POLICY when `merge_adapter` (the default): the output directory of
    # `pi train merge`, not the raw base. See merge.py for why the merge is not optional.
    base_model: str = ""
    # Rung 1's LoRA. Under `merge_adapter` this is PROVENANCE -- which adapter went into
    # `base_model` -- and nothing loads it here; `merged_base_sha` is what makes the claim
    # checkable. Under `merge_adapter=False` it must be empty, because an adapter to load plus
    # the fresh LoRA DPO trains is two adapters and one ambiguous reference.
    adapter: str = ""
    out_dir: str = "artifacts/rung2"
    pairs: str = "data/rl/pairs.jsonl"
    lora: LoraSpec = field(default_factory=LoraSpec)
    beta: float = 0.1
    learning_rate: float = 5e-6  # an order of magnitude below rung 1: DPO is a nudge, not a fit
    epochs: int = 1
    max_seq_len: int = 12_288
    max_prompt_len: int = 11_264
    per_device_batch: int = 1
    grad_accum: int = 16
    seed: int = 0
    len_delta_max: int = 40  # characters; mirrors export_pairs' guard
    # ---- WHAT SUBSET OF THE PAIRS FILE THIS RUN TRAINS ON. On the CONFIG, not applied by
    # hand before training, because `cfg.sha` covers every field here: two runs over different
    # subsets of one file would otherwise claim one identity, which is what provenance exists
    # to prevent. Every drop is counted and reaches `rung2.manifest.json`.
    #
    # `include_pair_kinds` selects the DECISION being trained. ask_stop pairs carry a ~60-char
    # length asymmetry that is what the two actions ARE (a STOP is one constant), so they are
    # a separate population, not noise inside ask_ask -- and an ablation on this field is the
    # cheapest way to show what the stopping data bought.
    include_pair_kinds: tuple[str, ...] = SAMPLED_PAIR_KINDS
    exclude_suites: tuple[str, ...] = ()
    # The FLOOR on the question-diversity sidecar (`scripts/curate_pairs_with_diversity.py`),
    # ordered paraphrase < related < different. "" applies no filter. Measured on the interim
    # export: 9.0% of pairs are paraphrases -- two renderings of one question, ranked by which
    # retrieval happened to come back better.
    min_q_distinctness: str = ""
    # WHICH ORDERING THIS RUN TRUSTS. On `pairs.rater.jsonl` the rescued rows are pairs our own
    # score could not order (50.2% agreement with the A6 majority, i.e. chance) and a reader
    # could. Training on rule-only vs both is the comparison that says what the rater label
    # bought, and it must be in `cfg.sha` or the two runs claim one identity.
    include_label_sources: tuple[str, ...] = LABEL_SOURCES
    # WHICH DECIDER ORDERED THE PAIR, which is NOT the same question as which file it came
    # from. MEASURED on `data/rl/pairs.rater.jsonl`: 532 rows carry label_source="rule" and
    # decided_by="rater" -- pairs the mechanical rule could have ordered and a rater majority
    # actually did. A run that means to exclude rater judgement and filters `label_source`
    # excludes none of them, under a sha that says it did.
    exclude_decided_by: tuple[str, ...] = ()
    # THE COHORT: which code produced the rows. Empty applies no filter; non-empty is an
    # ALLOWLIST, and a row that carries no `code_version` cannot be shown to belong to it and
    # is dropped (counted separately -- "outside the cohort" and "predates the field" are
    # different facts). Rows exported before and after a loop change are not one measurement.
    include_code_versions: tuple[str, ...] = ()
    # HOW MANY OF THE SURVIVING PAIRS THIS RUN TRAINS ON, AND WHICH DRAW. Not a filter: the
    # matched control of an ablation ("dpo_control subsampled to dpo_outcome_only's n") asks for
    # ANY n of them, which is a property of a draw and not of a row. Both fields are ABSENT by
    # default (`SHA_OMIT_WHEN_NONE`), so every rung-2 run trained before they existed keeps its
    # recorded `config_sha`; set, they enter it -- the full file and a draw from it are two
    # datasets, and two draws at two seeds are two more. `validate()` refuses one without the
    # other, and there is no default seed on purpose: see `load_pairs`.
    subsample_pairs: int | None = None
    subsample_seed: int | None = None
    # CONSERVATIVE DPO. The mechanical rule agrees with the A6 raters about 60% of the time,
    # so a share of the labels is flipped; smoothing trains against that rather than against
    # the fiction that every label is right. 0.0 is plain DPO.
    label_smoothing: float = 0.0
    # HOW `pi_ref` IS OBTAINED, and therefore WHAT "DPO moved the policy from the SFT
    # checkpoint" means for this run. In `cfg.sha`, because a merged run and an adapter run over
    # the same pairs are two experiments. See `REFERENCES`, and docs/TRAINING.md 6.1 for the
    # measurement that made "adapter" the default: folding a LoRA into bf16 base weights lost
    # 44.7% of rung 1's update, and the fp32 fix costs ~32 GB for an 8B base rung 2 must load.
    reference: str = "adapter"
    # WHETHER `base_model` IS THE MERGE OUTPUT. MEANINGFUL ONLY UNDER `reference="merged"`, and
    # `validate()` refuses it under "adapter" -- where nothing is merged and there is nothing
    # for a merge sha to name. See merge.py. False under "merged" says base_model is already
    # the policy DPO starts from (merged elsewhere, or a full-weight fine-tune) and the
    # reference is base_model itself with the fresh LoRA disabled; it is not a way to run DPO
    # on a raw base, which `validate` still refuses by requiring the adapter under a merge.
    merge_adapter: bool = False
    # The sha `merge_adapter()` returned for `base_model`. Required with `merge_adapter`: it
    # is the only thing that makes "the reference policy was rung 1" checkable afterwards.
    merged_base_sha: str = ""
    # HOW THE TWO HALVES ARE RENDERED. Recorded here; the tokenisation at DPO time is TRL's,
    # through `processing_class` -- see `train()`, which reports which hook it found.
    chat_template: bool = True
    enable_thinking: bool = False
    # THE FLOOR ON STOP-WINNING PAIRS. MEASURED on `data/rl/pairs.jsonl`: among ask_stop pairs
    # ASK is chosen 1,356 times and STOP 236. A run that includes ask_stop in order to teach
    # stopping, over a file that is almost all ASK-wins, trains "keep asking" while its config
    # says otherwise -- and `n_pairs` cannot tell the two apart. 0.0 refuses nothing, which is
    # right for the ask_ask-only primary arm where the share is 0 by construction.
    min_stop_chosen_share: float = 0.0
    # THE COMPUTE DTYPE OF THE TRAINING RUN, which is not the dtype anything is STORED in --
    # see merge.py for that distinction and the 44.7% it cost. Rung 1 has had this field since
    # it was written; rung 2 trained at TRL's default under a cfg.sha that said nothing.
    bf16: bool = True
    # ---- THE OBJECTIVE. A TUPLE, because trl 1.x takes a LIST and sums the named terms with
    # `loss_weights`. That is not decoration: `("sigmoid", "sft")` is the RPO/MPO shape -- the
    # preference term plus an NLL anchor on the CHOSEN side -- and an anchor is the direct
    # answer to the thing rung 2 is most likely to do wrong, which is separate the two sides by
    # walking both of them away from the SFT checkpoint. `LOSS_TYPES` is what the installed
    # library dispatches; `kto_pair`, which this field used to accept, is not in it.
    #
    # THE SINGLE-ELEMENT DEFAULT HASHES WHAT THE OLD STRING HASHED -- see `sha`.
    loss_type: tuple[str, ...] = DEFAULT_LOSS_TYPE
    # ONE WEIGHT PER TERM, or None for TRL's own "1.0 each". In `cfg.sha` when set and ABSENT
    # when not (`SHA_OMIT_WHEN_NONE`): equal weights is what every run trained before this field
    # existed used, and those runs must keep their identity.
    loss_weights: tuple[float, ...] | None = None
    # LD-DPO's alpha: the weight on the log-probabilities of the tokens PAST the shared prefix
    # of the two sides. 1.0 applies no weighting (plain DPO), 0.0 masks that divergent tail
    # entirely. It is a second lever on the length confound the |Δlen| <= 40 guard bounds from
    # the other side -- and trl 1.13.0 does NOT range-check it, it multiplies and trains, so
    # `validate()` does. None is ABSENT, not 1.0.
    ld_alpha: float | None = None
    tokens_per_second: float = 1_600.0  # two forward passes per example, so ~half rung 1
    # ---- SURVIVING A PREEMPTION. Rung 2 passed TRL no save strategy at all, so how often it
    # wrote `checkpoint-N/` was whatever the installed library defaulted to -- and three DPO
    # arms are packed on one node, so one preemption costs all three. `save_steps` is in
    # `cfg.sha` (it changes what the output directory contains); `resume` is not, and
    # `SHA_EXCLUDED` in rung 1's module says why for both rungs.
    save_steps: int = 200
    resume: bool = True

    @property
    def sha(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k not in SHA_EXCLUDED}
        for k in SHA_OMIT_WHEN_NONE:
            if d.get(k) is None:
                d.pop(k, None)
        # ONE LOSS NAMED TWO WAYS IS ONE RUN. `loss_type` was a `str` until trl 1.x made it a
        # list; rendering a single-element tuple as that bare string is what keeps every rung-2
        # `config_sha` written before this commit joinable. Same argument as
        # `SHA_OMIT_WHEN_NONE`, for a field that changed SHAPE rather than appeared -- and it
        # is not a special case for the default value: "sigmoid" alone IS the objective
        # ("sigmoid",) names, so the two must not be two identities.
        if len(self.loss_type) == 1:
            d["loss_type"] = self.loss_type[0]
        return h("rung2", canon(d))

    def validate(self) -> None:
        self.lora.validate()
        if not self.include_pair_kinds or set(self.include_pair_kinds) - set(PAIR_KINDS):
            raise ValueError(
                f"include_pair_kinds={self.include_pair_kinds!r}: expected a non-empty subset "
                f"of {PAIR_KINDS}. Empty would train on nothing while reporting a config."
            )
        if not self.include_label_sources or set(self.include_label_sources) - set(LABEL_SOURCES):
            raise ValueError(
                f"include_label_sources={self.include_label_sources!r}: expected a non-empty "
                f"subset of {LABEL_SOURCES}. Empty would train on nothing while reporting a "
                "config that says otherwise."
            )
        if self.min_q_distinctness and self.min_q_distinctness not in Q_DISTINCTNESS:
            raise ValueError(
                f"min_q_distinctness={self.min_q_distinctness!r}: expected one of "
                f"{Q_DISTINCTNESS} or '' for no filter."
            )
        if not self.base_model:
            raise ValueError("base_model is empty; there is no default (see rung 1's reason)")
        if self.reference not in REFERENCES:
            raise ValueError(
                f"reference={self.reference!r}: expected one of {REFERENCES}. There is no "
                "default reading of an unknown value -- pi_ref would be whatever the installed "
                "libraries happened to do, which is the ambiguity this field exists to remove."
            )
        if self.merge_adapter and not self.adapter:
            raise ValueError(
                "adapter is empty. Rung 2 starts from rung 1's checkpoint: DPO applied to the "
                "raw base optimises a preference the base has no policy to express, and the "
                "result is not comparable to the SFT arm it is supposed to improve on. Under "
                "merge_adapter this field is not loaded -- it records WHICH adapter was merged "
                "into base_model, so the manifest can be reconciled with the merge's."
            )
        if not 0.0 < self.beta <= 1.0:
            raise ValueError(f"beta={self.beta}: expected 0 < beta <= 1")
        if self.save_steps < 1:
            raise ValueError(
                f"save_steps={self.save_steps}: must be >= 1. A non-positive cadence is a run "
                "that writes no checkpoint, which on a preemptible queue is a run that cannot "
                "be resumed however the resume call is spelled."
            )
        # ---- THE OBJECTIVE. Refused HERE, on a laptop, rather than by the trainer: every one
        # of these raises inside `DPOTrainer` or inside the loss itself, i.e. after an 8B base
        # and two adapters are resident on a rented card.
        if not self.loss_type:
            raise ValueError(
                "loss_type is empty: a zero-term sum is the constant 0.0, which trains "
                "nothing, reports a perfectly flat curve and writes a checkpoint identical to "
                "its input under a config that names an objective. Expected names from "
                f"{LOSS_TYPES}."
            )
        unknown = [t for t in self.loss_type if t not in LOSS_TYPES]
        if unknown:
            raise ValueError(
                f"loss_type={tuple(self.loss_type)!r}: {unknown!r} is not dispatched by the "
                f"trl this rung runs on. Expected names from {LOSS_TYPES}. 'kto_pair', which "
                "this field used to accept, was REMOVED from trl -- a run configured with it "
                "loads the base and both adapters before raising inside the loss. KTO on "
                "these pairs is trl.KTOTrainer, a different trainer, not a value here."
            )
        if len(set(self.loss_type)) != len(self.loss_type):
            raise ValueError(
                f"loss_type={tuple(self.loss_type)!r} repeats a name. TRL zips the names with "
                "`loss_weights` positionally, so a duplicate is one term's weight silently "
                "split in two -- and nothing in the manifest distinguishes that from the "
                "mixture the operator meant to write."
            )
        if self.loss_weights is not None and len(self.loss_weights) != len(self.loss_type):
            raise ValueError(
                f"loss_weights has {len(self.loss_weights)} weights for "
                f"{len(self.loss_type)} loss types. One weight per term, or None for TRL's own "
                "equal weighting. A mixture whose weights do not line up with its terms is not "
                "the objective its cfg.sha names."
            )
        if self.ld_alpha is not None and not 0.0 <= self.ld_alpha <= 1.0:
            raise ValueError(
                f"ld_alpha={self.ld_alpha}: expected a share in [0, 1] -- 1.0 applies no "
                "weighting, 0.0 masks every token past the shared prefix -- or None for "
                "absent. trl 1.13.0 does not check this: it multiplies the tail "
                "log-probabilities by whatever it is handed and trains."
            )
        if "exo_pair" in self.loss_type and self.label_smoothing == 0.0:
            raise ValueError(
                "loss_type includes 'exo_pair' with label_smoothing=0.0, which trl refuses in "
                "`DPOTrainer.__init__` -- after the base and both adapters are resident. EXO's "
                "epsilon IS that field; the paper's recommended value is 1e-3."
            )
        if self.max_prompt_len >= self.max_seq_len:
            raise ValueError(
                f"max_prompt_len={self.max_prompt_len} >= max_seq_len={self.max_seq_len}: the "
                "action would be truncated away and the pair would supervise nothing."
            )
        if not 0.0 <= self.label_smoothing < 0.5:
            raise ValueError(
                f"label_smoothing={self.label_smoothing}: expected 0 <= eps < 0.5. At 0.5 the "
                "two sides weigh the same and the objective no longer depends on the label."
            )
        if not 0.0 <= self.min_stop_chosen_share <= 1.0:
            raise ValueError(
                f"min_stop_chosen_share={self.min_stop_chosen_share}: expected a share in [0, 1]"
            )
        # ---- THE MATCHED-n DRAW. Refused here as well as in `load_pairs` because a config is
        # the thing a grid file writes and a manifest records: an arm whose target and seed
        # disagree must fail on the laptop that wrote it, not on the node that queued it.
        if self.subsample_pairs is not None and self.subsample_pairs < 1:
            raise ValueError(
                f"subsample_pairs={self.subsample_pairs}: expected a positive target, or None "
                "for 'train on every row the filters kept'."
            )
        if self.subsample_pairs is not None and self.subsample_seed is None:
            raise ValueError(
                "subsample_pairs is set with no subsample_seed. WHICH rows were drawn is half "
                "of what this arm trained on, and an unnamed draw is an input the run cannot "
                "reproduce. There is no default: `cfg.seed` would tie the draw to the training "
                "seed, so three seed replicates would be three datasets and the arm's spread "
                "would no longer be seed variance; 0 would hide the choice in a field nobody "
                "wrote."
            )
        if self.subsample_seed is not None and self.subsample_pairs is None:
            raise ValueError(
                f"subsample_seed={self.subsample_seed} with no subsample_pairs: a seed that "
                "draws nothing still enters `cfg.sha`, so it renames the run without changing "
                "a row of its data."
            )
        # THE REFERENCE-POLICY REFUSALS, LAST so that every filter and hyperparameter refusal
        # above keeps firing on its own message.
        if self.reference == "adapter":
            # The three ways an "adapter" run can be internally inconsistent. Each of them
            # leaves the run unable to say which distribution pi_ref was, which is the only
            # thing that makes rung 2's sentence checkable.
            if not self.adapter:
                raise ValueError(
                    "reference='adapter' with an empty adapter. The reference policy IS rung "
                    "1's LoRA, loaded frozen beside the trainable copy; with no adapter there "
                    "is nothing to freeze and nothing to train, and DPO would optimise a fresh "
                    "LoRA against the BARE BASE while the config claims rung 1."
                )
            if self.merge_adapter:
                raise ValueError(
                    "reference='adapter' with merge_adapter=True: two reference policies in "
                    "one config. Under 'adapter' nothing is folded into the weights -- that is "
                    "the point, a bf16 fold loses 44.7% of the update -- so a merge flag here "
                    "describes a step this run does not take. Choose --reference merged if the "
                    "base really is a merge output."
                )
            if self.merged_base_sha:
                raise ValueError(
                    f"reference='adapter' with merged_base_sha={self.merged_base_sha!r}: that "
                    "sha names the output of a merge, and this run merges nothing. A "
                    "provenance field describing a step that did not happen is worse than an "
                    "empty one, because it reconciles against a directory nobody loaded."
                )
            return
        if self.reference == "base":
            # THE SAME THREE, mirrored onto the mode that has no rung 1 in it. Each one leaves
            # the run unable to say which distribution pi_ref was, and here that is the whole
            # claim: the ablation's sentence is "no SFT rung preceded this policy".
            if self.adapter:
                raise AmbiguousReference(
                    f"reference='base' with adapter={self.adapter!r}: a base reference with an "
                    "adapter named is ambiguous -- WHICH one is the policy? The run would hold "
                    "rung 1's LoRA and the fresh LoRA DPO trains, and 'DPO with no SFT rung' is "
                    "then false under the first reading and unverifiable under the second. "
                    "Under 'base' nothing is loaded onto base_model: the policy is a fresh LoRA "
                    "and pi_ref is the untrained base beneath it. Use --reference adapter if "
                    "rung 1 is meant to be in this run."
                )
            if self.merge_adapter:
                raise ValueError(
                    "reference='base' with merge_adapter=True: two reference policies in one "
                    "config. A merge folds rung 1 INTO the weights, and this mode's reference "
                    "is the base with nothing folded into it -- which is exactly what it "
                    "measures. Choose --reference merged if the base really is a merge output."
                )
            if self.merged_base_sha:
                raise ValueError(
                    f"reference='base' with merged_base_sha={self.merged_base_sha!r}: that sha "
                    "names the output of a merge, and this run merges nothing. A provenance "
                    "field describing a step that did not happen is worse than an empty one, "
                    "because it reconciles against a directory nobody loaded."
                )
            return
        if self.merge_adapter and not self.merged_base_sha:
            raise ValueError(
                "merge_adapter=True but merged_base_sha is empty. base_model is then the "
                "OUTPUT of `pi train merge`, and without the sha that merge returned the run "
                "cannot name the weights its reference policy was -- which makes 'DPO moved "
                "the policy from the SFT checkpoint' a claim no artifact can check. Run "
                "`pi train merge --base-model ... --adapter ... --out ...` and pass its "
                "directory as --merged-base."
            )
        if not self.merge_adapter and self.adapter:
            raise AmbiguousReference(
                f"merge_adapter=False with adapter={self.adapter!r}: that is two adapters in "
                "one process -- the one loaded onto the base and the fresh LoRA DPO trains -- "
                "and TRL computes the reference by disabling 'the adapter'. Which one that "
                "means is a property of the installed trl/peft versions, so pi_ref is either "
                "rung 1 or the bare base and nothing in the run records which. Merge rung 1 "
                "into the weights instead (`pi train merge`)."
            )


# --------------------------------------------------------------------------- pairs


@dataclass(frozen=True, slots=True)
class EncodedPair:
    """One same-state pair, tokenised. The prompt half is IDENTICAL by construction."""

    prompt_ids: tuple[int, ...]
    chosen_ids: tuple[int, ...]
    rejected_ids: tuple[int, ...]
    margin: float

    @property
    def n_shared_prompt(self) -> int:
        return len(self.prompt_ids)


def pair_kind_of(pair: dict[str, Any]) -> str:
    """What decision this pair is about, RE-DERIVED from the two action payloads.

    Not read off the `pair_kind` field. That field is a string on a jsonl file that can be
    filtered, concatenated or hand-edited between the export and the run, and it gates the
    length-guard exemption below -- so trusting it would let a mislabelled ask_ask pair
    disable the very check that is supposed to catch it. `action_kind_of` is the policy
    parser's own normalisation, so this and the loop can never disagree about a row.

    A label that contradicts the payloads is FATAL, not corrected: it means the file was
    edited by something that did not understand it, and nothing else in it can be trusted.
    """
    from pinq.actions import action_kind_of

    kinds = (action_kind_of(pair.get("chosen_json")), action_kind_of(pair.get("rejected_json")))
    derived = "ask_stop" if "stop" in kinds else "ask_ask"
    claimed = str(pair.get("pair_kind") or "")
    # A recorded STOP and a synthesised one are the SAME bytes, so the payloads cannot tell
    # them apart and the label is the only witness. Accept it when it refines the derived
    # kind; still refuse one the bytes contradict, which is what gates the length exemption.
    if claimed == "ask_stop_synth" and derived == "ask_stop":
        return claimed
    if claimed and claimed != derived:
        raise ValueError(
            f"pair_kind={claimed!r} but the payloads are {kinds}: the label and the bytes "
            "disagree. The label gates the length-guard exemption, so a wrong one disables "
            "the check that would have caught it."
        )
    return derived


def assert_same_state(pair: dict[str, Any]) -> None:
    """The invariant the whole rung rests on. See the module docstring."""
    if "state_text" in pair:
        return  # the exporter emits ONE state_text per pair; there is nothing to disagree.
    chosen_state = pair.get("chosen_state_text")
    rejected_state = pair.get("rejected_state_text")
    if chosen_state is None or rejected_state is None:
        raise NotSameState(
            "a pair carries neither `state_text` nor both of "
            "`chosen_state_text`/`rejected_state_text`. Without the state there is no way to "
            "check the property that makes this rung valid."
        )
    if chosen_state != rejected_state:
        raise NotSameState(
            "the two candidates were drawn at DIFFERENT states, so the within-state contrast "
            "does not cancel V(s_t) and the loss is measuring task difficulty. This is fatal, "
            "not filterable: a mixed dataset trains a model whose objective is partly an "
            "uncontrolled cross-task comparison, and the training curve cannot tell you which "
            "part moved."
        )


def assert_length_guard(pair: dict[str, Any], *, len_delta_max: int) -> None:
    """Re-check the exporter's guard at load time.

    Re-checking rather than trusting: the pairs file is a plain jsonl that can be filtered,
    concatenated or hand-edited between the export and the run, and "longer question wins" is
    the single confound most likely to be reintroduced by a well-meaning merge.
    """
    if pair_kind_of(pair) in STOP_PAIR_KINDS:
        # NOT APPLICABLE. A STOP is one 18-byte constant and a question is 40-120 characters,
        # so every ask_stop pair violates any cap this guard could sanely carry. The delta
        # there is what the two ACTIONS are, not a property of two questions, and enforcing it
        # would delete the entire category the rung exists to train on. The asymmetry is made
        # VISIBLE instead: `pair_kind` on every row, a per-kind split in `pairs_report`, and
        # `include_pair_kinds` to train with or without it.
        return
    dl = question_len_delta(str(pair["chosen_json"]), str(pair["rejected_json"]))
    if dl > len_delta_max:
        raise ValueError(
            f"|delta question len| = {dl} > {len_delta_max}. Without the guard the "
            "preference model learns 'longer question wins', which is the judge confound "
            "imported into the policy's weights. Measured on the parsed question via "
            "`export.dataset.question_len_delta` -- the same function the exporter applied."
        )


def assert_trainable_split(pair: dict[str, Any]) -> None:
    """Refuse a row the split says is held out.

    A ROW THAT MAKES NO CLAIM IS NOT A VIOLATION. Every pair exported before the field existed
    carries no `split`, and reading absence as "not train" would refuse the entire training
    corpus. The dev exporter stamps `split: "dev"` on what it writes, so this fires on exactly
    the file it exists for -- the one an operator points `--pairs` at by mistake.
    """
    claimed = str(pair.get("split") or "train")
    if claimed != "train":
        raise HeldOutDataset(
            f"a pair is stamped split={claimed!r}. Held-out rows are the measurement, not the "
            "training set: a checkpoint that saw them makes every confirmatory number "
            "computed from it void, and nothing in the output would say so. This is fatal "
            "rather than a filter -- a file of dev rows is the wrong file, not a training "
            "file with some bad rows in it."
        )


def subsample_rows(rows: Sequence[dict[str, Any]], *, n: int, seed: int) -> list[dict[str, Any]]:
    """`n` of these rows, drawn by `seed`: the mechanism a MATCHED-n arm is made of.

    WHY THIS IS NOT A FILTER, and could not be written as one. Every other selector on
    `DPOConfig` is a predicate over a row's own fields -- this kind, that suite, this decider --
    and the matched control's membership question ("is this pair one of the n?") is not a
    property of the pair at all. It is a property of a DRAW, so the draw itself has to be named
    (`seed`) and has to be in `cfg.sha`, or the arm cannot say which rows it trained on.

    THE KEY IS `pair_id`, NOT POSITION. Two exports of one row set differ in line order and are
    the same training set; a sample taken over positions would differ between them while
    claiming one identity. `pair_id` is the exporter's own name for a pair -- sha256 over
    (suite, task, run, turn, both candidate run ids, kind) -- so the drawn subset is a function
    of (seed, the SET of rows) and of nothing else. MEASURED on data/rl/pairs.jsonl: 47,989
    rows, 47,989 distinct pair_ids, none empty, so the two refusals below cost the real corpus
    nothing and would fire only on a file no subsample could name.

    RETURNS A SUBSET IN THE FILE'S OWN ORDER. Presentation order is the trainer's business
    (`cfg.seed` shuffles), and a loader that also reordered would put two seeds in one arm.
    """
    if n < 1:
        raise ValueError(
            f"subsample_pairs={n}: expected a positive target. A zero-row draw trains nothing "
            "and reports a config that names a dataset."
        )
    ids: list[str] = []
    for i, r in enumerate(rows):
        pid = str(r.get("pair_id") or "")
        if not pid:
            raise ValueError(
                f"row {i} carries no `pair_id`, so a subsample cannot name what it drew. The "
                "only alternative key is the row's position in the file, which makes the draw "
                "depend on export order -- two files holding one row set would train two "
                "different arms under one cfg.sha. Re-export the pairs (every row written by "
                "`pinq_train.export.dataset._pair` carries one)."
            )
        ids.append(pid)
    counts = Counter(ids)
    dupes = sorted(k for k, c in counts.items() if c > 1)
    if dupes:
        raise ValueError(
            f"{len(dupes)} `pair_id`s appear more than once (e.g. {dupes[0]!r}): a draw of k "
            "keys would then return more than k rows, so the arm's n would not be the n its "
            "config names. Deduplicate the file before subsampling it."
        )
    if n > len(rows):
        raise TooFewPairsToSubsample(
            f"subsample_pairs={n} but only {len(rows)} pairs survived this config's filters. "
            "Returning the ones there are would be an arm that reports a matched n it does "
            "not have. Either the target is wrong, or a filter above removed more than the "
            "operator expected -- `n_dropped_by_filter` in the preflight report says which."
        )
    chosen = set(random.Random(seed).sample(sorted(ids), n))
    return [r for r, pid in zip(rows, ids) if pid in chosen]


def load_pairs(
    path: str | Path,
    *,
    len_delta_max: int = 40,
    include_pair_kinds: Sequence[str] = SAMPLED_PAIR_KINDS,
    exclude_suites: Sequence[str] = (),
    min_q_distinctness: str = "",
    include_label_sources: Sequence[str] = LABEL_SOURCES,
    exclude_decided_by: Sequence[str] = (),
    include_code_versions: Sequence[str] = (),
    subsample_pairs: int | None = None,
    subsample_seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read the pairs file, assert the two invariants, apply the config's filters.

    Returns the rows AND the drop counts. A filter that removes rows without saying how many
    is a filter nobody can audit; the counts ride into `rung2.manifest.json` through
    `pairs_report`, so a reader of the artifact can reconcile it against the file on disk.

    ASSERTS FIRST, FILTERS SECOND. `assert_same_state` and the length guard are properties
    every pair in the file must have -- a violation means the file is wrong, and filtering it
    away would hide that. The filters express what this RUN trains on, which is a different
    question and a legitimate choice.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist. Build it with `pi train export --kind pairs`; rung 2 never "
            "invents pairs and never re-rolls -- it re-reads rung 1's scored candidates."
        )
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    for r in rows:
        assert_same_state(r)
        assert_trainable_split(r)
        assert_length_guard(r, len_delta_max=len_delta_max)

    if min_q_distinctness and not any("q_distinctness" in r for r in rows):
        raise ValueError(
            f"min_q_distinctness={min_q_distinctness!r} but no row carries `q_distinctness`. "
            "The filter would silently match nothing and the run would claim -- under a config "
            "sha that says otherwise -- to have excluded paraphrases. Enrich the file with "
            "`scripts/curate_pairs_with_diversity.py` first."
        )
    if include_code_versions and not any(r.get("code_version") for r in rows):
        raise ValueError(
            f"include_code_versions={tuple(include_code_versions)!r} but no row carries "
            "`code_version`. Every row would be dropped and the refusal that followed would "
            "read as 'no pairs were exported', which is a different problem with a different "
            "fix. Export the pairs with the cohort field, or drop the filter."
        )
    floor = Q_DISTINCTNESS.index(min_q_distinctness) if min_q_distinctness else -1
    keep: list[dict[str, Any]] = []
    drops: dict[str, int] = {}

    def drop(why: str) -> None:
        drops[why] = drops.get(why, 0) + 1

    for r in rows:
        if pair_kind_of(r) not in set(include_pair_kinds):
            drop("pair_kind")
            continue
        if str(r.get("suite_id", "")) in set(exclude_suites):
            drop("suite")
            continue
        # A row with no `label_source` was written before the field existed, and every such
        # row was ordered by the rule. Reading it as unknown would silently drop the entire
        # existing corpus the first time anyone filtered on this.
        if str(r.get("label_source") or "rule") not in set(include_label_sources):
            drop("label_source")
            continue
        # WHO ORDERED IT, not which file it came from. A row that claims no decider cannot
        # be excluded by one; the corpus predates the field.
        if str(r.get("decided_by") or "") in set(exclude_decided_by):
            drop("decided_by")
            continue
        if include_code_versions:
            cv = str(r.get("code_version") or "")
            if not cv:
                # Counted apart from `code_version`: a reader of the manifest has to be able
                # to tell "excluded by the allowlist" from "written before the field existed".
                drop("code_version_missing")
                continue
            if cv not in set(include_code_versions):
                drop("code_version")
                continue
        if floor >= 0:
            d = str(r.get("q_distinctness") or "")
            if d not in Q_DISTINCTNESS or Q_DISTINCTNESS.index(d) < floor:
                drop("q_distinctness")
                continue
        keep.append(r)

    # ---- THE DRAW, AFTER EVERY PREDICATE. The matched control is matched to the n ANOTHER arm
    # reported, and that n is itself post-filter, so the target is checked against what the
    # filters left rather than against the file. Counted into `drops` like any other loss: a
    # selection that removes rows without saying how many is a selection nobody can audit, and
    # this one removes rows no predicate can describe.
    if subsample_pairs is None:
        if subsample_seed is not None:
            raise ValueError(
                f"subsample_seed={subsample_seed} with no subsample_pairs: a seed that draws "
                "nothing still enters `cfg.sha`, so it renames the run without changing a row "
                "of its data."
            )
        return keep, drops
    if subsample_seed is None:
        raise ValueError(
            "subsample_pairs is set with no subsample_seed. There is no default: borrowing "
            "`cfg.seed` would make the three seed replicates of a matched arm train on three "
            "different draws, so the arm's spread would mix training-seed variance with subset "
            "choice while the table calls it seed variance."
        )
    before = len(keep)
    keep = subsample_rows(keep, n=subsample_pairs, seed=subsample_seed)
    if before != len(keep):
        drops["subsample"] = before - len(keep)
    return keep, drops


def rows_for(cfg: DPOConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """THE mapping from a config to the rows it trains on. There must be exactly one.

    `cmd_train_rung2` preflighted with one argument list and `train()` loaded with another --
    they differed by `include_label_sources` -- so one `cfg.sha` covered two datasets and the
    printed report described neither reliably. A filter that is on `DPOConfig` is in `cfg.sha`
    by construction; if it does not reach the loader, the run's identity claims a subset it
    never trained on. One function, so a new field is wired in one place.
    """
    return load_pairs(
        cfg.pairs,
        len_delta_max=cfg.len_delta_max,
        include_pair_kinds=cfg.include_pair_kinds,
        exclude_suites=cfg.exclude_suites,
        min_q_distinctness=cfg.min_q_distinctness,
        include_label_sources=cfg.include_label_sources,
        exclude_decided_by=cfg.exclude_decided_by,
        include_code_versions=cfg.include_code_versions,
        subsample_pairs=cfg.subsample_pairs,
        subsample_seed=cfg.subsample_seed,
    )


def encode_pairs(
    tok: Tokenizer, rows: Iterable[dict[str, Any]], *, max_seq_len: int
) -> list[EncodedPair]:
    """Tokenise pairs, sharing the prompt ids between the two halves.

    NOT THE TRAINING TOKENISATION PATH, and this docstring used to imply that it was. `train()`
    hands TRL a `Dataset` of raw strings and TRL tokenises them itself through
    `processing_class`, so nothing here runs on the GPU line. What this is for is the
    laptop-checkable property the pairing argument rests on -- that the two halves of a pair
    share a prompt token for token -- and for offline analysis that needs ids without a
    trainer. `build_masked_example` is used so those ids are rung 1's, not a third rendering.
    """
    out: list[EncodedPair] = []
    for r in rows:
        state = str(r.get("state_text") or r["chosen_state_text"])
        pos = build_masked_example(tok, state, str(r["chosen_json"]), append_eos=True)
        neg = build_masked_example(tok, state, str(r["rejected_json"]), append_eos=True)
        prompt_ids = pos.input_ids[: pos.n_prompt]
        if len(prompt_ids) + max(pos.n_action, neg.n_action) > max_seq_len:
            cut = len(prompt_ids) + max(pos.n_action, neg.n_action) - max_seq_len
            prompt_ids = prompt_ids[cut:]
        out.append(
            EncodedPair(
                prompt_ids=tuple(prompt_ids),
                chosen_ids=tuple(pos.input_ids[pos.n_prompt :]),
                rejected_ids=tuple(neg.input_ids[neg.n_prompt :]),
                margin=float(r.get("margin", 0.0)),
            )
        )
    return out


def pairs_report(
    rows: Sequence[dict[str, Any]], *, drops: dict[str, int] | None = None
) -> dict[str, Any]:
    margins = [float(r.get("margin", 0.0)) for r in rows]
    deltas = [abs(len(str(r["chosen_json"])) - len(str(r["rejected_json"]))) for r in rows]
    n = len(rows)
    return {
        "n_pairs": n,
        "n_states": len(
            {
                (r.get("suite_id"), r.get("task_id"), r.get("run_id"), r.get("turn_idx"))
                for r in rows
            }
        ),
        "margin_mean": (sum(margins) / n) if n else 0.0,
        "margin_min": min(margins) if margins else 0.0,
        "len_delta_max": max(deltas) if deltas else 0,
        "suites": sorted({str(r.get("suite_id", "")) for r in rows}),
        # WHICH DECISION THE LOSS SAW, and in which direction. An ask_stop split of 0 means
        # the run trained no stopping behaviour whatever the config asked for, and that is
        # invisible in `n_pairs` alone.
        "n_by_pair_kind": {
            k: sum(1 for r in rows if pair_kind_of(r) == k)
            for k in PAIR_KINDS
            if any(pair_kind_of(r) == k for r in rows)
        },
        # WHOSE ORDERING THE LOSS SAW. A run that trained on rescued pairs and one that did
        # not are different experiments, and `n_pairs` alone cannot tell them apart.
        "n_by_label_source": {
            k: sum(1 for r in rows if str(r.get("label_source") or "rule") == k)
            for k in LABEL_SOURCES
            if any(str(r.get("label_source") or "rule") == k for r in rows)
        },
        "n_stop_chosen": (
            n_stop_chosen := sum(
                1 for r in rows if str(r.get("chosen_json") or "") == STOP_ACTION_JSON
            )
        ),
        # THE SHARE, not only the count: `min_stop_chosen_share` is a floor on this, and a
        # count without its denominator cannot be compared between two runs over two files.
        "stop_chosen_share": (n_stop_chosen / n) if n else 0.0,
        "n_dropped_by_filter": dict(sorted((drops or {}).items())),
    }


def estimate_gpu_hours(cfg: DPOConfig, n_tokens: int) -> float:
    if cfg.tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be positive")
    return (n_tokens * cfg.epochs) / cfg.tokens_per_second / 3600.0


def adapter_lora_spec(adapter_dir: str | Path) -> dict[str, Any]:
    """The LoRA shape rung 2 INHERITS, read off rung 1's own `adapter_config.json`.

    Under `reference="adapter"` the policy IS rung 1's adapter, so `cfg.lora` describes nothing
    this run does -- r and alpha are whatever rung 1 fitted. Recording the inherited values is
    the only way a reader of `rung2.manifest.json` can tell which LoRA was trained; leaving
    `cfg.lora` in the manifest alone would state a shape the run never used.
    """
    cfgf = Path(adapter_dir) / "adapter_config.json"
    if not cfgf.is_file():
        return {}
    conf = json.loads(cfgf.read_text())
    return {k: conf[k] for k in ("r", "lora_alpha") if k in conf}


def dpo_argument_kwargs(
    cfg: DPOConfig, field_names: Container[str], *, trl_source: str = ""
) -> tuple[dict[str, Any], dict[str, str]]:
    """The `trl.DPOConfig` kwargs, spelled for the TRL that is INSTALLED, plus what was found.

    PURE, so the mapping is tested without trl -- `train()` is the one function here that needs
    a GPU, and what it passed to TRL was consequently pinned by nothing. `[train]` allows
    `trl>=0.9`, i.e. both sides of TRL 1.0.

    TWO FIELDS ARE VERSION DEPENDENT, and BOTH are reported rather than assumed:

      * `max_prompt_length` is gone in TRL 1.x. MEASURED on trl 1.13.0: `DPOConfig` declares
        137 fields and this is not among them, so rung 2 raised `TypeError` at config
        construction -- after the merged base had loaded and the pairs had preflighted. TRL 1.x
        keeps only `max_length` plus `truncation_mode`, and the two truncate DIFFERENTLY: a
        separate prompt budget trimmed the state and kept the action, whereas
        `truncation_mode="keep_start"` over the concatenation trims from the RIGHT, which is
        where the action is. TRL 1.x drops a row whose prompt alone fills `max_length`, so the
        "supervise nothing" case is caught, but a prompt just under it still loses part of its
        action. `prompt_length_hook` says which regime a run was under, because the two are not
        the same experiment.
      * `chat_template_kwargs` is likewise absent on 1.13.0. Unchanged behaviour, moved here so
        a test can reach it: without the hook the tokenizer's own default governs, and Qwen3's
        default EMITS thinking -- which is why serving this checkpoint must pass
        enable_thinking=False explicitly (docs/TRAINING.md; the serving proxy sets it).

    THE OBJECTIVE IS THE THIRD, AND IT IS THE ONE THAT MAY NOT BE DROPPED. The two above are
    passed where they exist and omitted where they do not, because the run is still the run the
    config describes without them. `loss_weights` and `ld_alpha` are not: omitting either
    trains PLAIN DPO under a `cfg.sha` that names a mixture or LD-DPO, with every curve looking
    healthy, so a TRL that lacks them raises `UnsupportedObjective` here instead.
    """
    kw: dict[str, Any] = {
        "output_dir": cfg.out_dir,
        "beta": cfg.beta,
        "label_smoothing": cfg.label_smoothing,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.epochs,
        "per_device_train_batch_size": cfg.per_device_batch,
        "gradient_accumulation_steps": cfg.grad_accum,
        "max_length": cfg.max_seq_len,
        "bf16": cfg.bf16,
        "seed": cfg.seed,
        "report_to": [],
    }
    hooks = {"chat_template_hook": "tokenizer_default", "prompt_length_hook": "max_length_only"}
    # --- the objective ---
    # THE SPELLING IS VERSION DEPENDENT AND THE NAMES ARE LIBRARY DEPENDENT, so both are
    # probed. `loss_weights` is the name that settles the spelling: it arrived WITH the
    # list-valued `loss_type` in trl 1.x, so a class that declares it takes a list and sums
    # terms, and a class that does not takes a bare string and holds exactly one.
    multi = "loss_weights" in field_names
    supported = installed_loss_types(trl_source)
    if supported and not set(cfg.loss_type) <= supported:
        raise UnsupportedObjective(
            f"the installed trl does not dispatch {sorted(set(cfg.loss_type) - supported)}; "
            f"its loss ladder handles {sorted(supported)}. Refused here rather than inside the "
            "loss, which is reached only after the base and both adapters have loaded."
        )
    if not multi and len(cfg.loss_type) > 1:
        raise UnsupportedObjective(
            f"loss_type={tuple(cfg.loss_type)!r} is a mixture and the installed trl declares no "
            "`loss_weights`: that version takes ONE loss name and cannot sum terms at all, so "
            "the run would not be the objective its cfg.sha claims."
        )
    kw["loss_type"] = list(cfg.loss_type) if multi else cfg.loss_type[0]
    hooks["loss_type_hook"] = ("list" if multi else "str") + ("" if supported else "_unchecked")
    for name, value in (("loss_weights", cfg.loss_weights), ("ld_alpha", cfg.ld_alpha)):
        if value is None:  # ABSENT. Passing None would record "this run set it", which is false
            continue
        if name not in field_names:
            raise UnsupportedObjective(
                f"{name}={value!r} but the installed trl's DPOConfig declares no such field. "
                "Dropping it -- which is what this function does for `max_prompt_length` and "
                "`chat_template_kwargs` -- would train plain DPO under a cfg.sha that names "
                f"{name}, and nothing in the loss curve would say so. Pin a trl that has it."
            )
        kw[name] = list(value) if isinstance(value, tuple) else value
    hooks["objective_hooks"] = ",".join(
        sorted(n for n in ("loss_type", "loss_weights", "ld_alpha") if n in kw)
    )
    # --- end the objective ---
    # WHICH ADAPTER IS THE REFERENCE, said out loud wherever the installed TRL lets it be said.
    # Under a merge there is exactly one adapter -- the fresh LoRA -- so naming a reference
    # adapter would send TRL looking for one that does not exist.
    if cfg.reference == "adapter":
        hook = reference_hook(field_names, trl_source)
        hooks["reference_adapter_hook"] = hook
        hooks["policy_adapter"] = POLICY_ADAPTER
        hooks["reference_adapter"] = REFERENCE_ADAPTER
        if hook == "dpo_config_fields":
            kw["model_adapter_name"] = POLICY_ADAPTER
            kw["ref_adapter_name"] = REFERENCE_ADAPTER
    elif cfg.reference == "base":
        # NO NAME IS PASSED, and none may be: naming a reference adapter is what would stop TRL
        # taking the `None` branch, and the `None` branch IS this mode. Probed HERE, before the
        # weights load, for the reason the whole call was moved above the model -- a refusal on
        # the far side of a ~16 GB read is a refusal nobody can afford to test.
        hooks["reference_adapter_hook"] = base_reference_hook(trl_source)
        hooks["policy_adapter"] = POLICY_ADAPTER
    else:
        hooks["reference_adapter_hook"] = "merged_weights"
    if "max_prompt_length" in field_names:
        kw["max_prompt_length"] = cfg.max_prompt_len
        hooks["prompt_length_hook"] = "max_prompt_length"
    if "chat_template_kwargs" in field_names:
        kw["chat_template_kwargs"] = {"enable_thinking": cfg.enable_thinking}
        hooks["chat_template_hook"] = "chat_template_kwargs"
    # --- resume ---
    # THE CADENCE, PROBED LIKE EVERYTHING ELSE HERE. Rung 2 passed no save strategy at all, so
    # a preempted arm had whatever TRL's default had left on disk -- possibly nothing. A TRL
    # that declares no `save_steps` gets neither it nor `save_strategy="steps"`, which would
    # otherwise name a cadence the library cannot express.
    if "save_steps" in field_names:
        kw["save_strategy"] = "steps"
        kw["save_steps"] = cfg.save_steps
    if "save_total_limit" in field_names:
        kw["save_total_limit"] = 2
    # --- end resume ---
    return kw, hooks


def verify_frozen_reference(
    model: Any,
    *,
    policy: str = POLICY_ADAPTER,
    reference: str = REFERENCE_ADAPTER,
    equal: Any = None,
) -> dict[str, Any]:
    """Check, on the assembled model, that `pi_ref` IS rung 1 -- and raise if it is not.

    EVERY LAYER ABOVE THIS IS ABOUT INTENT. The config says which reference the run wants, the
    hook says which mechanism the installed TRL offers, and the calls say what was loaded. None
    of them observes the weights that the reference forward will actually use. This does: it
    pairs each LoRA tensor of the policy adapter with its counterpart under the reference name
    and requires them to be equal and frozen.

    It runs AFTER `DPOTrainer` is constructed, because trl 1.13.0 creates the reference copy in
    its constructor. A failure here is the silent-wrong-reference bug caught before a single
    optimiser step, rather than after a week of GPU time and a table.
    """
    if equal is None:  # torch only on the GPU path; the doubles inject their own comparison
        import torch

        def equal(a: Any, b: Any) -> bool:
            return bool(torch.equal(a, b))

    params = dict(model.named_parameters())
    pairs = [
        (n, n.replace(f".{policy}.", f".{reference}."))
        for n in params
        if f".{policy}." in n and "lora_" in n
    ]
    if not pairs:
        raise AmbiguousReference(
            f"the assembled model carries no LoRA parameter named .{policy}., so there is no "
            "policy adapter to compare a reference against and nothing here can say what "
            "pi_ref is."
        )
    missing = [ref for _, ref in pairs if ref not in params]
    if missing:
        raise AmbiguousReference(
            f"{len(missing)} of {len(pairs)} policy LoRA tensors have no .{reference}. "
            f"counterpart (first: {missing[0]}). TRL computes pi_ref by switching to the "
            f"adapter named {reference!r}; with the adapter absent it disables every adapter "
            "instead and the reference becomes the BARE BASE, which trains and reports a "
            "healthy loss curve while rung 2's claim is false."
        )
    unequal = [ref for pol, ref in pairs if not equal(params[pol], params[ref])]
    if unequal:
        raise AmbiguousReference(
            f"{len(unequal)} of {len(pairs)} reference LoRA tensors differ from the policy's "
            f"at step 0 (first: {unequal[0]}). The two copies are rung 1 twice; if they are "
            "not equal before any update, pi_ref is not the SFT checkpoint."
        )
    thawed = [ref for _, ref in pairs if getattr(params[ref], "requires_grad", False)]
    if thawed:
        raise AmbiguousReference(
            f"{len(thawed)} reference LoRA tensors carry gradients (first: {thawed[0]}). A "
            "reference that moves with the policy makes the log-ratio identically zero and the "
            "objective independent of the data."
        )
    # A reference equal to a ZERO adapter is equal to the bare base, and every check above
    # would still pass. LoRA initialises B to zero, so an untrained rung 1 is exactly that.
    nonzero = any(not equal(params[ref], params[ref] * 0) for _, ref in pairs if "lora_B" in ref)
    return {
        "n_lora_pairs": len(pairs),
        "equal": True,
        "frozen": True,
        "is_the_base": not nonzero,
    }


def verify_base_reference(
    model: Any,
    *,
    policy: str = POLICY_ADAPTER,
    reference: str = REFERENCE_ADAPTER,
    trl_source: str = "",
    equal: Any = None,
) -> dict[str, Any]:
    """Check, on the assembled model, that `pi_ref` IS the untrained base -- and raise if not.

    `verify_frozen_reference`'s three properties, every one of them INVERTED, because the two
    modes are claims about different distributions:

      * a reference adapter must NOT exist. With one present TRL takes the NAMED branch and
        pi_ref becomes a frozen copy of the fresh LoRA -- equal to the base at step 0 and not
        equal to it afterwards, so the run drifts off its own claim while training normally.
      * the installed TRL must take the `None` branch, i.e. disable adapters for the reference
        forward. That is `base_reference_hook`, the same probe read for its other outcome.
      * the policy must BE the base at step 0: `lora_B == 0`, as peft initialises it. Under
        "adapter" that same fact is a bug (`is_the_base` true means rung 1 never trained);
        here it is the requirement, and a non-zero tensor means something was loaded into the
        policy that the arm's name says is not there.

    THE MODEL TO PASS IS `trainer.model`, NOT the one handed to `DPOTrainer`. MEASURED on trl
    1.13.0: a `peft_config` makes the constructor REPLACE it with `get_peft_model(model,
    peft_config)`. The caller's object is a bare `AutoModelForCausalLM` with no `peft_config`
    at all, and every check below would pass on it by finding nothing.
    """
    if equal is None:  # torch only on the GPU path; the doubles inject their own comparison
        import torch

        def equal(a: Any, b: Any) -> bool:
            return bool(torch.equal(a, b))

    params = dict(model.named_parameters())
    policy_lora = [n for n in params if f".{policy}." in n and "lora_" in n]
    if not policy_lora:
        raise AmbiguousReference(
            f"the assembled model carries no LoRA parameter named .{policy}., so nothing in it "
            "is the policy DPO is supposed to train and this check would pass by finding no "
            "counterexample."
        )
    # BOTH spellings, because TRL selects on the CONFIG: `"ref" in model.peft_config`. An
    # adapter registered under that name with no tensor of its own still captures the reference
    # forward, and a parameter scan alone would not see it.
    registered = reference in getattr(model, "peft_config", {})
    named = [n for n in params if f".{reference}." in n]
    if named or registered:
        raise AmbiguousReference(
            f"the assembled model carries an adapter named {reference!r} ({len(named)} "
            f"parameters, registered={registered}). reference='base' means pi_ref is the "
            "UNTRAINED base, which TRL reaches by disabling every adapter -- and it only does "
            f"that when no adapter is called {reference!r}. With one present the reference is a "
            "copy of the policy instead, the run trains, the loss falls, and the arm's claim "
            "is false."
        )
    hook = base_reference_hook(trl_source)
    moved = [n for n in policy_lora if "lora_B" in n and not equal(params[n], params[n] * 0)]
    if moved:
        raise AmbiguousReference(
            f"{len(moved)} of the policy's lora_B tensors are non-zero at step 0 (first: "
            f"{moved[0]}), so the policy is not the bare base. peft initialises lora_B to zero; "
            "a non-zero one means weights were loaded into the adapter this run is supposed to "
            "start fresh. The reference IS the base here, so the log-ratio would be non-zero "
            "before a single update -- which is the arm's own definition of having started "
            "somewhere else."
        )
    return {
        "n_policy_lora": len(policy_lora),
        "ref_adapter_absent": True,
        "disables_adapters": hook == "trl_disable_adapters",
        # TRUE, and CORRECT here: a fresh LoRA is the identity, so the policy at step 0 is the
        # base. The field is spelled the same as the adapter mode's so the two manifests can be
        # read side by side -- there it is a warning, here it is the requirement.
        "is_the_base": True,
    }


def _module_source(obj: Any) -> str:
    """The source of the module `obj` was defined in, or "" when it cannot be read.

    Empty rather than raising: the source probe is one of two ways `reference_hook` can resolve,
    and a library shipped without sources (a zipimport, a frozen build) must fall through to the
    field probe and then to a clear refusal, not to a traceback from `inspect`.
    """
    import inspect

    try:
        return inspect.getsource(inspect.getmodule(obj))
    except (OSError, TypeError):
        return ""


class NoPairs(RuntimeError):
    """Rung 2 was asked to train on zero preference pairs."""


def preflight(
    cfg: DPOConfig, rows: Sequence[dict[str, Any]], *, drops: dict[str, int] | None = None
) -> dict[str, Any]:
    cfg.validate()
    rep = pairs_report(rows, drops=drops)
    rep["config_sha"] = cfg.sha
    rep["ignore_index"] = IGNORE_INDEX
    # AN EMPTY PAIRS FILE IS NOT A PASSING PREFLIGHT.
    #
    # This returned a clean report with `n_pairs: 0` and exited 0. Rung 1 refuses the analogous
    # case loudly -- "an empty sample passes every diversity test ever written, so it is refused
    # rather than scored" -- and rung 2 did not. With `--train --acknowledge-untested` on a GPU
    # box it would have proceeded to `Dataset.from_list([])`, run DPOTrainer over nothing, and
    # written rung2.manifest.json as if a checkpoint had been produced. That is a rented-GPU
    # failure mode, invisible on a laptop where `--train` cannot run at all.
    #
    # `n_pairs` is 0 by construction today: a pair needs TWO candidates at one state, recorded
    # runs hold one action per state, and no candidate sampler exists. So this fires on the
    # actual current state of the repository, not on a hypothetical one.
    if not rep.get("n_pairs"):
        raise NoPairs(
            "0 preference pairs. A pair needs TWO candidates at one state; recorded runs hold "
            "one action per state, so pairs come from candidate sampling, not from a finished "
            "sweep. Training on an empty dataset would still write a manifest as though a "
            "checkpoint existed. Build pairs with `pi train export --kind pairs` once a "
            "candidate sampler exists."
        )
    # AND A RUN THAT ASKED FOR STOPPING DATA AND GOT ALMOST NONE IS NOT A PASSING PREFLIGHT
    # EITHER. Same argument as rung 1's STOP-share ceiling: `n_stop_chosen` was computed,
    # printed, and acted on by nothing, which documents the failure instead of preventing it.
    # The floor is declared per run and is 0.0 by default, so this refuses nothing until a
    # run says what it expects -- an ask_ask-only arm has a share of 0 by construction.
    if cfg.min_stop_chosen_share > 0.0 and rep["stop_chosen_share"] < cfg.min_stop_chosen_share:
        raise TooFewStopPairs(
            f"stop_chosen_share={rep['stop_chosen_share']:.2f} "
            f"({rep['n_stop_chosen']}/{rep['n_pairs']}) is below the declared floor "
            f"{cfg.min_stop_chosen_share:.2f}. The pairs this run selected teach ASK over "
            "STOP almost everywhere, so it would train the opposite of the behaviour its "
            "config asks for while every loss curve looks healthy."
        )
    return rep


# --------------------------------------------------------------------------- the GPU line


def train(
    cfg: DPOConfig,
    *,
    acknowledge_untested: bool = False,
    tokenizer: Tokenizer | None = None,
    rows: Sequence[dict[str, Any]] | None = None,
    drops: dict[str, int] | None = None,
) -> dict[str, Any]:
    """DPO over same-state pairs. THE ONLY function here that needs a GPU.

    THREE SHAPES, chosen by `cfg.reference`, and the difference is what `pi_ref` IS.

      * "adapter" (the default): `cfg.base_model` is the RAW base and `cfg.adapter` is rung 1's
        LoRA, loaded onto it TWICE -- a trainable copy named `POLICY_ADAPTER` that DPO
        optimises and a frozen copy named `REFERENCE_ADAPTER` that TRL selects for the
        reference forward. One set of base weights in memory, no merge, and therefore nothing
        that a dtype can round away. TRL is given NO `peft_config`: a fresh LoRA would be a
        THIRD adapter, and it, not rung 1's copy, is what would get trained.
      * "merged": `cfg.base_model` is the output of `pi train merge`, whose sha is
        `cfg.merged_base_sha`. Nothing loads an adapter, so TRL's reference pass disables the
        one adapter that exists -- the fresh LoRA -- and `pi_ref` is rung 1's merged weights.
      * "base": the SAME assembly as "merged" over DIFFERENT weights -- `cfg.base_model` is the
        raw, untrained base and no rung 1 exists anywhere in the run. TRL disables the fresh
        LoRA for the reference pass, so `pi_ref` is that untrained base. This is the A14/E39
        ablation (`dpo_from_base`): what the preference label buys with no imitation in front
        of it. Its KL is anchored to a different model, so its beta and lr do NOT transfer
        from the adapter arm -- docs/TRAINING.md 6.1.

    `rows`/`drops` arrive from a caller that has already loaded them (`pi train rung2`
    preflights before it trains). Loading them a second time here is how the reported dataset
    and the optimised dataset came to be different sets of rows.
    """
    if not acknowledge_untested:
        raise NotValidatedOnHardware(
            "rung 2 has never been executed: this project has no local CUDA device. Every "
            "property that can be checked without one is covered by tests; this call is not."
        )
    cfg.validate()
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    tok = tokenizer or AutoTokenizer.from_pretrained(cfg.base_model)
    if rows is None:
        rows, drops = rows_for(cfg)
    report = preflight(cfg, rows, drops=drops)
    # WHAT THE REFERENCE WEIGHTS ACTUALLY ARE, reconciled against the merge that produced
    # them. Unverifiable (a hub id, someone else's merge) is recorded as such, not asserted.
    report["merged_base"] = (
        verify_merged_base(cfg.base_model, cfg.merged_base_sha) if cfg.merge_adapter else {}
    )
    # WHAT THE INSTALLED TRL ACTUALLY ACCEPTS, probed rather than assumed. See
    # `dpo_argument_kwargs`; every hook it resolves is recorded in the manifest.
    #
    # BEFORE THE WEIGHTS LOAD. This call used to sit below the model, which put every refusal
    # it can raise -- a loss name this trl does not dispatch, an objective field it does not
    # declare -- on the far side of a ~32 GB read and two adapter loads. It is pure in
    # `(cfg, field names, source)`, so nothing about the model informs it and there is no
    # reason to pay for that first.
    # ONCE, and shared: `dpo_argument_kwargs` probes this text for the reference mechanism and
    # `verify_base_reference` asserts the branch it resolved. Two separate reads could disagree
    # only by accident, and the accident would be silent.
    trl_source = _module_source(DPOTrainer)
    targs, hooks = dpo_argument_kwargs(
        cfg,
        set(getattr(TRLDPOConfig, "__dataclass_fields__", {})),
        trl_source=trl_source,
    )
    report.update(hooks)
    ds = Dataset.from_list(
        [
            {
                "prompt": str(r.get("state_text") or r["chosen_state_text"]),
                "chosen": str(r["chosen_json"]),
                "rejected": str(r["rejected_json"]),
            }
            for r in rows
        ]
    )
    if cfg.reference == "adapter":
        # ONE base, TWO named copies of rung 1. `is_trainable` is passed on BOTH calls rather
        # than left to peft's default: which copy carries gradients is the difference between
        # training rung 2 and training nothing, and a correctness property that rests on a
        # library default is a property this repository has not stated.
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model)
        model = PeftModel.from_pretrained(
            base, cfg.adapter, adapter_name=POLICY_ADAPTER, is_trainable=True
        )
        model.load_adapter(cfg.adapter, adapter_name=REFERENCE_ADAPTER, is_trainable=False)
        peft_config = None
    else:
        # ONE ASSEMBLY, TWO CLAIMS. Under "merged" base_model IS the policy (rung 1 is already
        # in these weights); under "base" base_model is the untrained base and there is no rung
        # 1 to be in them. Either way nothing is loaded on top, so "the adapter" names exactly
        # one thing -- the fresh LoRA -- for the rest of this function, and TRL's reference
        # pass disables it. WHICH of the two weight sets that reference is, is the difference
        # between the arms and is why `reference` is in `cfg.sha`.
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model)
        peft_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.target_modules),
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
        )
    # WHAT THE REFERENCE POLICY IS, in the artifact, beside the checkpoint. Under "adapter" the
    # LoRA hyperparameters are rung 1's -- the policy adapter IS rung 1's adapter -- so
    # `cfg.lora` describes nothing this run did and the inherited shape is recorded instead.
    report["reference_policy"] = {
        "reference": cfg.reference,
        # WHICH WEIGHTS WERE LOADED. Under "merged" and "base" they ARE pi_ref, so naming them
        # is the whole claim; under "adapter" they are half of it and `adapter_sha` is the rest.
        "base_model": cfg.base_model,
        # None, not "", under "base": there is no adapter in the run at all, and an empty
        # string beside the two modes that do name one reads as a field nobody filled in.
        "adapter": None if cfg.reference == "base" else cfg.adapter,
        "adapter_sha": adapter_sha(cfg.adapter) if cfg.reference == "adapter" else "",
        "policy_adapter": hooks.get("policy_adapter", ""),
        "reference_adapter": hooks.get("reference_adapter", ""),
        "reference_adapter_hook": hooks["reference_adapter_hook"],
        "lora_source": (
            "rung1 adapter_config.json" if cfg.reference == "adapter" else "DPOConfig.lora"
        ),
        "lora": (
            adapter_lora_spec(cfg.adapter)
            if cfg.reference == "adapter"
            else {"r": cfg.lora.r, "lora_alpha": cfg.lora.alpha}
        ),
    }
    trainer = DPOTrainer(
        model=model,
        args=TRLDPOConfig(**targs),
        train_dataset=ds,
        processing_class=tok,
        ref_model=None,
        peft_config=peft_config,
    )
    if cfg.reference == "adapter":
        # AFTER the trainer is built: trl 1.13.0 creates the frozen copy in its constructor.
        # A wrong reference is caught here, before the first optimiser step, instead of after
        # the run has produced a checkpoint and a number.
        report["reference_policy"]["verified"] = verify_frozen_reference(model)
    elif cfg.reference == "base":
        # THE TRAINER'S model, not the one built above: a `peft_config` makes trl 1.13.0 REPLACE
        # it with `get_peft_model(...)`, so the local object is a bare base carrying no adapter
        # and no `peft_config` -- on which every check below would pass by finding nothing.
        report["reference_policy"]["verified"] = verify_base_reference(
            trainer.model, trl_source=trl_source
        )
    # WHERE THE PREEMPTED ARM LEFT OFF. Read before the output directory is created, so a first
    # submission resumes nothing. `None` is passed explicitly under `resume=False`: "told to
    # start fresh" and "never told anything" are different facts, and `trainer.train()` with no
    # argument -- what this line used to be -- is the second one.
    resumed_from = latest_checkpoint(cfg.out_dir) if cfg.resume else None
    trainer.train(resume_from_checkpoint=resumed_from)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if cfg.reference == "adapter":
        # ONLY the policy copy. The frozen one is rung 1's adapter byte for byte, and
        # `save_pretrained` with no selection writes every adapter it holds -- which would put
        # a second `adapter_model.safetensors` in the directory that names rung 2.
        model.save_pretrained(str(out), selected_adapters=[POLICY_ADAPTER])
    else:
        trainer.save_model(str(out))
    (out / "rung2.manifest.json").write_text(
        json.dumps(
            # Beside the config, not in it: a resumed run and the run it continues share one
            # `cfg.sha`, so this is the only place that can say which of the two produced this
            # directory.
            {"config": asdict(cfg), "pairs": report, "resumed_from": resumed_from},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {"out_dir": str(out), "resumed_from": resumed_from, **report}
