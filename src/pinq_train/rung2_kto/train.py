"""Rung 2, RC4 -- KTO over the unpaired signals the pairs file discards.

WHY AN UNPAIRED OBJECTIVE AT ALL, GIVEN THE SAME-STATE ARGUMENT. `rung2_dpo` opens with the
reason the headline is a within-state contrast: it cancels `V(s_t)` exactly, and on these suites
the between-task variance of `V` is an order of magnitude larger than the effect the paper is
trying to detect. That argument is not retracted here. It is a statement about which estimator
is LOW-VARIANCE, not about which rows carry information, and RC4 is the arm that asks the second
question: the corpus states facts about single actions that no pairing can express.

  * 28,613 rows that say "an ASK here was wrong" at a state gold says was already DONE. In the
    pairs file these exist only as the losing half of a synthetic pair whose winner is one
    18-byte constant, so DPO spends them teaching a contrast against `{"action": "STOP"}`.
  * 4,688 rejected ask_ask sides -- questions that lost, at states where the winner is already
    an SFT target, so the contrast the pair buys is one the SFT rung has largely already made.
  * 43,577 SFT targets, which DPO sees only through whichever pair happens to name them.

KTO scores one completion at a time against the reference, so each of those is a row. It pays
for that with the variance DPO avoids: the loss depends on `V(s_t)` through the reference's own
log-probability rather than cancelling it. That is the trade the arm exists to measure, and it
is why RC4 is a COMBINATION arm read against `dpo_control` on Tier B, not a replacement for the
headline. See `plans/2026-09-14-training-programme-v3.md` 5b and `docs/TRAINING.md` 6.3.

WHAT `pi_ref` IS, AND WHY THE ANSWER IS IMPORTED RATHER THAN RESTATED. Exactly what it is for
DPO: ONE base in memory with rung 1's LoRA loaded TWICE -- a trainable copy named
`POLICY_ADAPTER` and a frozen copy named `REFERENCE_ADAPTER` that TRL selects for the reference
forward. Every helper that makes that checkable (`reference_hook`, `verify_frozen_reference`,
`adapter_sha`, `adapter_lora_spec`) is imported from `rung2_dpo.train`. A copy would be a second
place for the name "ref" to be written down, and MEASURED on trl 1.13.0 that name is the whole
contract: `kto_trainer.py:1280` reads `use_adapter(unwrapped_model, adapter_name="ref" if "ref"
in unwrapped_model.peft_config else None)` -- three occurrences, the same line DPO's probe
already recognises -- so an adapter under any other name is not found, ALL adapters are
disabled, and `pi_ref` becomes the BARE BASE while the loss curve looks healthy.

THE MERGED REFERENCE IS NOT OFFERED HERE. DPO keeps it for a serving deployment that wants one
fused checkpoint; nothing in RC4 asks for that, and the measurement that made "adapter" DPO's
default (a bf16 fold loses 44.7% of rung 1's update, docs/TRAINING.md 6.1) applies unchanged.
An option nobody needs is an option nobody tests.

HARDWARE HONESTY. As with rungs 1 and 2, everything except `train()` runs and is tested on a
laptop. `train()` has never executed -- there is no local CUDA device, and no KTO run of any
size has been performed -- so it refuses to start unless the caller acknowledges that.

NO CLI IS REGISTERED FROM HERE. `pi train kto` is a separate task; this module is the library it
will call, and `rows_for`/`preflight`/`train` are the three entry points it needs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Container, Sequence

from pinq.ids import canon, h
from pinq_train.merge import adapter_sha
from pinq_train.resume import latest_checkpoint
from pinq_train.rung1_sft.train import SHA_EXCLUDED, NotValidatedOnHardware, read_jsonl
from pinq_train.rung2_dpo.train import (
    PAIR_KINDS,
    POLICY_ADAPTER,
    REFERENCE_ADAPTER,
    _module_source,
    adapter_lora_spec,
    reference_hook,
    verify_frozen_reference,
)
from pinq_train.rung2_kto.convert import (
    ON_CONFLICT,
    PROVENANCE_FIELDS,
    balanced_weights,
    build_kto_rows,
)

# TRL's only two unpaired losses on 1.13.0 (`kto_config.py:192`, choices ["kto",
# "apo_zero_unpaired"]). Named here rather than left to the library's default so a future
# default change does not silently move the objective of a run whose sha says nothing about it.
KTO_LOSS_TYPES = ("kto", "apo_zero_unpaired")
# HOW FAR THE TWO CLASS MASSES MAY BE APART BEFORE `preflight` refuses. Not a hyperparameter:
# it exists so a weight written to three decimals is accepted while a real imbalance is not.
# trl's own recommended band (KTO paper Eq. 8, checked in `KTOTrainer._prepare_dataset`) is a
# mass ratio in [1, 1.33], so 1% is an order of magnitude inside the library's tolerance.
MASS_RATIO_TOL = 0.01


class NoKTORows(RuntimeError):
    """RC4 was asked to train on zero signals.

    Refused for rung 2's reason: `Dataset.from_list([])` trains, writes a checkpoint directory
    and a manifest, and reports a config as though something had been optimised.
    """


class ImbalancedMasses(RuntimeError):
    """The configured class weights do not equalise the two masses of the rows handed in.

    The point of KTO's two weights is that the majority class does not own the gradient. A run
    that leaves them at 1.0 over the live corpus puts 58.5% of the mass on the desirable side
    and trains, silently, mostly the SFT objective it started from. Refused rather than
    auto-corrected: the weights are in `cfg.sha`, and a value the trainer computed for itself
    is a value the run's identity does not name. `balanced_weights()` prints the pair to pin.
    """


class TooFewUndesirable(RuntimeError):
    """The run declared an undesirable-share floor its rows cannot meet. See
    `KTOConfig.min_undesirable_share`. RC4 exists FOR the undesirable half; a file that is
    almost all desirable trains SFT-with-a-reference under a config that claims otherwise."""


@dataclass(frozen=True, slots=True)
class KTOConfig:
    # THE RAW BASE. Nothing is merged in this rung -- see the module docstring -- so this is the
    # base rung 1 was fitted on, not a merge output.
    base_model: str = ""
    # RUNG 1's LoRA, loaded TWICE: the trainable policy and the frozen reference. Unlike DPO's
    # `adapter` under a merge, this one is LOADED, and it is the whole reference claim.
    adapter: str = ""
    out_dir: str = "artifacts/rung2_kto"
    # THE TWO INPUT FILES. Both, because the desirable class is the union of the SFT targets and
    # the winning pair sides; a KTO run over the pairs alone is a different arm with a different
    # dataset, so the two paths are in `cfg.sha` separately.
    dataset: str = "data/rl/sft.jsonl"
    pairs: str = "data/rl/pairs.jsonl"
    beta: float = 0.1
    learning_rate: float = 5e-6  # rung 2's, not rung 1's: this is a nudge from the SFT policy
    epochs: int = 1
    # SHORTER THAN DPO's 12,288 ON PURPOSE. DPO holds two completions per example against one
    # prompt; KTO holds one, but `KTOTrainer` also runs a KL batch beside it, so the memory per
    # step is not the saving the completion count suggests. 9,216 is the declared budget.
    max_seq_len: int = 9_216
    # MEANINGFUL ONLY ON A TRL THAT DECLARES `max_prompt_length`. MEASURED: trl 1.13.0's
    # `KTOConfig` declares 129 fields and that is not among them, so on the installed library
    # the whole sequence is truncated FROM THE RIGHT against `max_length` -- which is where the
    # ACTION is, i.e. the tokens the loss is about. `kto_argument_kwargs` reports which regime a
    # run was under, because the two are not the same experiment. The 1,024-token gap is the
    # same action budget rung 2 uses; the action is the same object in both rungs.
    max_prompt_len: int = 8_192
    per_device_batch: int = 1
    grad_accum: int = 16
    seed: int = 0
    # WHICH PAIR KINDS BECOME SIGNALS. ALL THREE by default, unlike the DPO loader -- RC4 is
    # defined as the arm that uses the synthetic STOP family the headline discards, and an
    # ask_ask-only KTO run is a different (much smaller) experiment.
    include_pair_kinds: tuple[str, ...] = PAIR_KINDS
    # WHAT TO DO WITH A (state, action) THAT CARRIES BOTH LABELS. MEASURED on the live corpus:
    # 1,475 keys, 1,098 of them the ask_ask tournament ordering one candidate two ways. In
    # `cfg.sha` because "refused" and "dropped both sides" are two datasets. See `convert.py`.
    on_conflict: str = "refuse"
    chat_template: bool = True
    enable_thinking: bool = False
    loss_type: str = "kto"
    # ---- THE CLASS WEIGHTS. PLAIN FIELDS, therefore in `cfg.sha`, therefore naming the
    # objective this checkpoint was trained under. They are NOT derived inside `train()` from
    # whatever rows happened to load: a weight the trainer computed for itself is a weight the
    # run's identity cannot state, and two runs over two exports would then share one sha while
    # optimising two different balances. `convert.balanced_weights()` computes the pair to pin,
    # and `preflight` refuses a config whose weights do not balance the rows it was handed.
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
    # THE REFUSAL ABOVE, AS A DECLARED CHOICE. False is a legitimate arm -- "what does the
    # unbalanced objective do" is a question someone may want to ask -- and it is in the sha,
    # so the answer cannot be confused with the balanced arm's.
    require_balanced_masses: bool = True
    # THE FLOOR ON THE UNDESIRABLE SHARE. 0.0 refuses nothing. Set it on any arm whose claim is
    # about the negative half: rung 1's STOP-share ceiling exists for the same reason, and a
    # count that is printed and acted on by nothing documents a failure instead of preventing it.
    min_undesirable_share: float = 0.0
    bf16: bool = True
    # ---- SURVIVING A PREEMPTION. `save_steps` is in `cfg.sha` (it changes what the output
    # directory contains); `resume` is not -- rung 1's `SHA_EXCLUDED` says why for every rung.
    save_steps: int = 200
    resume: bool = True

    @property
    def sha(self) -> str:
        return h(
            "rung2_kto", canon({k: v for k, v in asdict(self).items() if k not in SHA_EXCLUDED})
        )

    def validate(self) -> None:
        if not self.base_model:
            raise ValueError("base_model is empty; there is no default (see rung 1's reason)")
        if not self.adapter:
            raise ValueError(
                "adapter is empty. RC4 starts from the SFT checkpoint and its reference policy "
                "IS that checkpoint, loaded frozen beside the trainable copy. With no adapter "
                "there is nothing to freeze and nothing to train: KTO would optimise the raw "
                "base against the raw base, which is an objective with no gradient and a "
                "config that claims rung 1."
            )
        if not 0.0 < self.beta <= 1.0:
            raise ValueError(f"beta={self.beta}: expected 0 < beta <= 1")
        if self.desirable_weight <= 0:
            raise ValueError(
                f"desirable_weight={self.desirable_weight}: must be > 0. Zero deletes the "
                "desirable class from the gradient while leaving it in every count the manifest "
                "reports; a negative weight trains the policy away from its own SFT targets."
            )
        if self.undesirable_weight <= 0:
            raise ValueError(
                f"undesirable_weight={self.undesirable_weight}: must be > 0. Zero is RC4 with "
                "its own subject matter deleted -- the arm exists for the undesirable half."
            )
        if not self.include_pair_kinds or set(self.include_pair_kinds) - set(PAIR_KINDS):
            raise ValueError(
                f"include_pair_kinds={self.include_pair_kinds!r}: expected a non-empty subset "
                f"of {PAIR_KINDS}."
            )
        if self.on_conflict not in ON_CONFLICT:
            raise ValueError(
                f"on_conflict={self.on_conflict!r}: expected one of {ON_CONFLICT}. There is no "
                "default reading of an unknown value, and the two that exist are two datasets."
            )
        if self.loss_type not in KTO_LOSS_TYPES:
            raise ValueError(f"loss_type={self.loss_type!r}: expected one of {KTO_LOSS_TYPES}")
        if not 0.0 <= self.min_undesirable_share <= 1.0:
            raise ValueError(
                f"min_undesirable_share={self.min_undesirable_share}: expected a share in [0, 1]"
            )
        if self.epochs < 1:
            raise ValueError(f"epochs={self.epochs}: must be >= 1")
        if self.max_prompt_len >= self.max_seq_len:
            raise ValueError(
                f"max_prompt_len={self.max_prompt_len} >= max_seq_len={self.max_seq_len}: the "
                "action would be truncated away and the row would supervise nothing."
            )
        if self.save_steps < 1:
            raise ValueError(
                f"save_steps={self.save_steps}: must be >= 1. A non-positive cadence is a run "
                "that writes no checkpoint, which on a preemptible queue cannot be resumed."
            )


# --------------------------------------------------------------------------- the rows


def rows_for(cfg: KTOConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """THE mapping from a config to the rows it trains on. There must be exactly one.

    Rung 2's lesson, reused: its CLI preflighted with one argument list and trained with
    another, so one `cfg.sha` covered two datasets and the printed report described neither.
    Every field of this config that selects rows is applied here and nowhere else.
    """
    for path, what in ((cfg.dataset, "sft"), (cfg.pairs, "pairs")):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} does not exist. Build it with `pi train export --kind {what}`; this "
                "rung never invents data and never re-rolls -- it re-reads rung 1's rows."
            )
    return build_kto_rows(
        read_jsonl(cfg.dataset),
        read_jsonl(cfg.pairs),
        include_pair_kinds=cfg.include_pair_kinds,
        on_conflict=cfg.on_conflict,
    )


def rows_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """What the converted rows ARE, recomputed from the rows themselves.

    Beside `build_kto_rows`' report rather than instead of it: that one describes the CONVERSION
    (what was dropped, deduped, refused), this one describes the list actually handed to the
    trainer. `pi train kto` will preflight and train from one list, but a caller that filters
    between the two would otherwise report a dataset it did not optimise.
    """
    n = len(rows)
    n_d = sum(1 for r in rows if r["label"])
    n_u = n - n_d
    return {
        "n_rows": n,
        "n_desirable": n_d,
        "n_undesirable": n_u,
        "undesirable_share": (n_u / n) if n else 0.0,
        "n_by_source_kind": _counts(rows, "source_kind"),
        "n_by_action_kind": _counts(rows, "action_kind"),
        "n_by_pair_kind": _counts(rows, "pair_kind"),
        "n_by_stop_source": _counts(rows, "stop_source"),
        "suites": sorted({str(r.get("suite_id", "")) for r in rows}),
        "n_states": len({(r.get("suite_id"), r.get("task_id"), r.get("state_sha")) for r in rows}),
        "n_rows_missing_provenance": sum(
            1 for r in rows if any(k not in r for k in PROVENANCE_FIELDS)
        ),
    }


def _counts(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        v = str(r.get(field) or "")
        if v:
            out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def mass_report(cfg: KTOConfig, n_desirable: int, n_undesirable: int) -> dict[str, Any]:
    """What the two configured weights do to the two classes. Pure.

    `mean_row_weight` is here for rung 1's reason: the accumulation window is normalised by
    `per_device_batch * grad_accum` on the strength of the mean weight being 1, so a weighting
    that moves it scales the gradient uniformly against every arm it is compared with. The
    equal-mass pair from `balanced_weights` holds it at 1 exactly; a hand-set pair may not, and
    the number belongs in the manifest rather than in a reader's head.
    """
    mass_d = n_desirable * cfg.desirable_weight
    mass_u = n_undesirable * cfg.undesirable_weight
    total = n_desirable + n_undesirable
    out: dict[str, Any] = {
        "desirable_weight": cfg.desirable_weight,
        "undesirable_weight": cfg.undesirable_weight,
        "mass_desirable": mass_d,
        "mass_undesirable": mass_u,
        "mass_ratio": (mass_d / mass_u) if mass_u else float("inf"),
        "mean_row_weight": ((mass_d + mass_u) / total) if total else 0.0,
        "require_balanced_masses": cfg.require_balanced_masses,
        "mass_ratio_tol": MASS_RATIO_TOL,
    }
    if n_desirable and n_undesirable:
        w_d, w_u = balanced_weights(n_desirable, n_undesirable)
        out["balanced_desirable_weight"] = w_d
        out["balanced_undesirable_weight"] = w_u
    return out


# --------------------------------------------------------------------------- the probe


def kto_argument_kwargs(
    cfg: KTOConfig, field_names: Container[str], *, trl_source: str = ""
) -> tuple[dict[str, Any], dict[str, str]]:
    """The `trl.KTOConfig` kwargs, spelled for the TRL that is INSTALLED, plus what was found.

    PURE for `dpo_argument_kwargs`' reason: `train()` is the one function in this rung that
    needs a GPU, so what it hands TRL would otherwise be pinned by nothing until the first run
    on a rented box -- which is how rung 2 discovered `max_prompt_length` was gone, after the
    base had loaded and the pairs had preflighted.

    THREE FIELDS ARE VERSION DEPENDENT AND ALL THREE ARE REPORTED RATHER THAN ASSUMED. MEASURED
    on trl 1.13.0 (`.venv-train`, 2026-09-15): `KTOConfig` declares 129 fields; `beta`
    (`kto_config.py:201`), `desirable_weight` (`:208`) and `undesirable_weight` (`:215`) are
    among them, while `max_prompt_length`, `chat_template_kwargs`, `model_adapter_name` and
    `ref_adapter_name` are NOT.

      * `truncation_hook` says whether the prompt had its own budget. Without one, `max_length`
        truncates the concatenation FROM THE RIGHT (`kto_config.py:56`) -- and the right is
        where the action is, i.e. the only tokens the loss is about.
      * `chat_template_hook` says whether `enable_thinking=False` was passed explicitly or left
        to the tokenizer's default. Qwen3's default EMITS thinking, which is why the serving
        proxy sets it too (docs/TRAINING.md).
      * `reference_adapter_hook` says HOW the frozen copy becomes `pi_ref` -- the same probe
        DPO uses, against the same two mechanisms, because trl spells it the same way in both
        trainers (MEASURED: three occurrences of the `"ref"` selection in `kto_trainer.py`).
    """
    kw: dict[str, Any] = {
        "output_dir": cfg.out_dir,
        "beta": cfg.beta,
        "loss_type": cfg.loss_type,
        # THE TWO NUMBERS THAT MAKE THIS ARM WHAT IT IS. They come off the config, never off the
        # dataset: see `KTOConfig.desirable_weight`.
        "desirable_weight": cfg.desirable_weight,
        "undesirable_weight": cfg.undesirable_weight,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.epochs,
        "per_device_train_batch_size": cfg.per_device_batch,
        "gradient_accumulation_steps": cfg.grad_accum,
        "max_length": cfg.max_seq_len,
        "bf16": cfg.bf16,
        "seed": cfg.seed,
        "report_to": [],
    }
    hook = reference_hook(field_names, trl_source)
    hooks = {
        "chat_template_hook": "tokenizer_default",
        "truncation_hook": "max_length_right",
        "reference_adapter_hook": hook,
        "policy_adapter": POLICY_ADAPTER,
        "reference_adapter": REFERENCE_ADAPTER,
        "loss_type": cfg.loss_type,
    }
    if hook == "dpo_config_fields":
        kw["model_adapter_name"] = POLICY_ADAPTER
        kw["ref_adapter_name"] = REFERENCE_ADAPTER
    if "max_prompt_length" in field_names:
        kw["max_prompt_length"] = cfg.max_prompt_len
        hooks["truncation_hook"] = "max_prompt_length"
    if "chat_template_kwargs" in field_names:
        kw["chat_template_kwargs"] = {"enable_thinking": cfg.enable_thinking}
        hooks["chat_template_hook"] = "chat_template_kwargs"
    # A TRL that declares no `save_steps` gets neither it nor `save_strategy="steps"`, which
    # would otherwise name a cadence the library cannot express.
    if "save_steps" in field_names:
        kw["save_strategy"] = "steps"
        kw["save_steps"] = cfg.save_steps
    if "save_total_limit" in field_names:
        kw["save_total_limit"] = 2
    return kw, hooks


# --------------------------------------------------------------------------- preflight


def preflight(
    cfg: KTOConfig, rows: Sequence[dict[str, Any]], *, convert_report: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Everything that can be checked before a device exists. Runs in milliseconds."""
    cfg.validate()
    rep = rows_report(rows)
    rep["config_sha"] = cfg.sha
    rep["conversion"] = dict(convert_report or {})
    n_d, n_u = rep["n_desirable"], rep["n_undesirable"]
    rep.update(mass_report(cfg, n_d, n_u))

    if not rep["n_rows"]:
        raise NoKTORows(
            "0 KTO rows. Training on an empty dataset would still write a manifest as though a "
            "checkpoint existed -- rung 2's `NoPairs` exists for the same reason. Build the "
            "inputs with `pi train export --kind sft` and `--kind pairs`."
        )
    if cfg.min_undesirable_share > 0.0 and rep["undesirable_share"] < cfg.min_undesirable_share:
        raise TooFewUndesirable(
            f"undesirable_share={rep['undesirable_share']:.3f} ({n_u}/{rep['n_rows']}) is below "
            f"the declared floor {cfg.min_undesirable_share:.3f}. RC4 is the arm that trains on "
            "the corpus' negative half; a dataset without one is SFT with a reference model, "
            "under a config that says otherwise."
        )
    if not n_d or not n_u:
        raise NoKTORows(
            f"one class is empty (n_desirable={n_d}, n_undesirable={n_u}). KTO's loss is a "
            "comparison against the reference in BOTH directions; with one class absent there "
            "is no balance to set and no contrast to learn."
        )
    if cfg.require_balanced_masses and abs(rep["mass_ratio"] - 1.0) > MASS_RATIO_TOL:
        w_d, w_u = balanced_weights(n_d, n_u)
        raise ImbalancedMasses(
            f"the configured weights put {rep['mass_desirable']:.6g} of mass on "
            f"{n_d} desirable rows and {rep['mass_undesirable']:.6g} on {n_u} undesirable ones "
            f"(ratio {rep['mass_ratio']:.6g}, tolerance {MASS_RATIO_TOL}). KTO's two weights "
            "exist precisely to counter unequal class counts, so leaving them unbalanced trains "
            "the majority class while every loss curve looks healthy. Pass "
            f"desirable_weight={w_d:.6g} undesirable_weight={w_u:.6g} -- or declare the "
            "imbalance with require_balanced_masses=False, which lands in cfg.sha."
        )
    return rep


# --------------------------------------------------------------------------- the GPU line


def train(
    cfg: KTOConfig,
    *,
    acknowledge_untested: bool = False,
    tokenizer: Any | None = None,
    rows: Sequence[dict[str, Any]] | None = None,
    convert_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """KTO over unpaired signals. THE ONLY function here that needs a GPU.

    `rows`/`convert_report` arrive from a caller that has already loaded them, so the reported
    dataset and the optimised dataset are one list. Loading twice is how rung 2's preflight came
    to describe rows the trainer never saw.
    """
    if not acknowledge_untested:
        raise NotValidatedOnHardware(
            "RC4 has never been executed: this project has no local CUDA device and no KTO run "
            "of any size has been performed. Every property that can be checked without one is "
            "covered by tests; this call is not."
        )
    cfg.validate()
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import KTOConfig as TRLKTOConfig
    from trl import KTOTrainer

    tok = tokenizer or AutoTokenizer.from_pretrained(cfg.base_model)
    if rows is None:
        rows, convert_report = rows_for(cfg)
    report = preflight(cfg, rows, convert_report=convert_report)
    # THE THREE COLUMNS TRL READS, AND NOTHING ELSE. `KTOTrainer` tokenises `prompt` +
    # `completion` itself and reads `label` (trl 1.13.0, `kto_trainer.py:225-226`); the
    # provenance block on each row stays out of the Dataset, where it would be carried through
    # `map()` and dropped by `_set_signature_columns_if_needed` anyway. It reaches the artifact
    # through the manifest's aggregate counts instead, which is where a reader can use it.
    ds = Dataset.from_list(
        [
            {
                "prompt": str(r["prompt"]),
                "completion": str(r["completion"]),
                "label": bool(r["label"]),
            }
            for r in rows
        ]
    )
    # ONE base, TWO named copies of rung 1. `is_trainable` is passed on BOTH calls rather than
    # left to peft's default: which copy carries gradients is the difference between training
    # RC4 and training nothing. TRL is given NO `peft_config` -- a fresh LoRA would be a THIRD
    # adapter, and it, not rung 1's copy, is what would get trained.
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model)
    model = PeftModel.from_pretrained(
        base, cfg.adapter, adapter_name=POLICY_ADAPTER, is_trainable=True
    )
    model.load_adapter(cfg.adapter, adapter_name=REFERENCE_ADAPTER, is_trainable=False)

    targs, hooks = kto_argument_kwargs(
        cfg,
        set(getattr(TRLKTOConfig, "__dataclass_fields__", {})),
        trl_source=_module_source(KTOTrainer),
    )
    report.update(hooks)
    report["reference_policy"] = {
        "reference": "adapter",
        "adapter": cfg.adapter,
        "adapter_sha": adapter_sha(cfg.adapter),
        "policy_adapter": POLICY_ADAPTER,
        "reference_adapter": REFERENCE_ADAPTER,
        "reference_adapter_hook": hooks["reference_adapter_hook"],
        # The policy IS rung 1's adapter, so its LoRA shape is rung 1's and this config has no
        # LoRA field to describe it. Read off the checkpoint, not restated.
        "lora_source": "rung1 adapter_config.json",
        "lora": adapter_lora_spec(cfg.adapter),
    }
    trainer = KTOTrainer(
        model=model,
        args=TRLKTOConfig(**targs),
        train_dataset=ds,
        processing_class=tok,
        ref_model=None,
    )
    # AFTER the trainer is built: trl 1.13.0 creates the frozen copy in its constructor
    # (`kto_trainer.py:724`), exactly as the DPO trainer does. A wrong reference is caught here,
    # before the first optimiser step, instead of after the run has produced a number.
    report["reference_policy"]["verified"] = verify_frozen_reference(model)

    resumed_from = latest_checkpoint(cfg.out_dir) if cfg.resume else None
    trainer.train(resume_from_checkpoint=resumed_from)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # ONLY the policy copy. The frozen one is rung 1's adapter byte for byte, and
    # `save_pretrained` with no selection writes every adapter it holds.
    model.save_pretrained(str(out), selected_adapters=[POLICY_ADAPTER])
    (out / "rung2_kto.manifest.json").write_text(
        json.dumps(
            # Beside the config, not in it: a resumed run and the run it continues share one
            # `cfg.sha`, so this is the only place that can say which produced this directory.
            {"config": asdict(cfg), "dataset": report, "resumed_from": resumed_from},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {"out_dir": str(out), "resumed_from": resumed_from, **report}
