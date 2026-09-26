"""Rung 1 -- rejection-sampling SFT with LoRA, loss masked to the Inquirer's own tokens.

WHAT IS TESTABLE WITHOUT A GPU, WHICH IS EVERYTHING EXCEPT ONE LINE. Config validation, the
dataset build, the loss mask, packing, the acceptance rule and the mode-collapse gate are pure
functions over plain data and are exercised by `tests/test_train_rungs.py`. The only thing
this file cannot check on an Apple M1 is `trainer.train()` itself. That split is deliberate:
the development machine has no CUDA device, so a design in which the interesting logic lives
inside a `Trainer` subclass would be a design in which the interesting logic is untested.

HEAVY IMPORTS ARE LAZY AND STAY LAZY. `torch`, `peft`, `trl` and `transformers` live behind the
`[train]` extra and are imported INSIDE `train()`. `make gate` must never touch them: a gate
that needs a 2 GB wheel to tell you a docstring is misformatted is a gate people stop running.

THE HONEST STATUS OF THIS RUNG. Nothing below has been executed end to end -- there is no
local GPU, and no rented one has been booked. `preflight()` says so out loud and refuses to
start unless the caller acknowledges it. What HAS been verified is every property that can be
verified without a device, plus the arithmetic in `estimate_gpu_hours`.

THE CONFIG IS THE PLAN'S, NOT A GUESS. LoRA r=32 alpha=64, lr 1e-4, 2 epochs, 12k packed
sequences. r=32 rather than r=8 because the policy has to learn a decision rule over long
contexts rather than a surface style; alpha=2r is the standard coupling that keeps the
effective learning rate of the adapter independent of r.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Container, Iterable, Mapping, Sequence

from pinq.actions import action_kind_of
from pinq.ids import canon, h
from pinq_train.resume import latest_checkpoint
from pinq_train.rung1_sft.collapse import (
    DISTINCT3_FLOOR,
    STOP_SHARE_CEILING,
    ModeCollapse,
    assert_diverse,
    within_task_distinct_n,
)
from pinq_train.rung1_sft.curriculum import CURRICULA, SHUFFLED, order_examples
from pinq_train.rung1_sft.mask import (
    IGNORE_INDEX,
    ChatTokenizer,
    MaskedExample,
    OverLength,
    Tokenizer,
    build_chat_masked_example,
    build_masked_example,
    margin_threshold,
    pack,
    truncate_left,
)


class HeldOutDataset(RuntimeError):
    """A dataset carrying rows that are not on the `train` side of the wall.

    Refused in `preflight`, which is the last place before a gradient that still knows what the
    rows are. Contamination here is not recoverable by anything downstream: a checkpoint that saw
    dev is a checkpoint whose dev number is a training number, and no later filter can unlearn it.
    The refusal is a RuntimeError rather than a warning for the same reason `ModeCollapse` is --
    a warning in a training log is a warning nobody reads.
    """


class NotValidatedOnHardware(RuntimeError):
    """`train()` was called without acknowledging that this rung has never been run.

    Not a warning. A rung that has never touched a GPU and reports numbers as if it had is the
    single failure this repository's rules exist to prevent (AGENTS.md rule 3).
    """


@dataclass(frozen=True, slots=True)
class LoraSpec:
    """LoRA hyperparameters. `alpha = 2 * r` is a coupling, not two free knobs."""

    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def validate(self) -> None:
        if self.r < 1:
            raise ValueError(f"r={self.r}: must be >= 1")
        if self.alpha != 2 * self.r:
            raise ValueError(
                f"alpha={self.alpha} != 2*r={2 * self.r}. The 2r coupling keeps the adapter's "
                "effective learning rate independent of r; decoupling them means a rank sweep "
                "also silently sweeps the learning rate."
            )


# WHICH MODULES A MIXTURE-OF-EXPERTS MODEL ACTUALLY HAS. gpt-oss fuses its 32 experts into 3-D
# PARAMETERS on one `GptOssExperts` module -- `mlp.experts.gate_up_proj (32, 2880, 5760)` and
# `mlp.experts.down_proj (32, 2880, 2880)` -- rather than into `nn.Linear` submodules. peft
# matches `target_modules` against MODULE names, and no module in `GptOssForCausalLM` is named
# `gate_proj`, `up_proj` or `down_proj`.
#
# MEASURED (peft 0.20.0, transformers 5.17.0, a 2-layer `GptOssForCausalLM`): `get_peft_model`
# with the Qwen set and with the attention-only set produce THE SAME 8 `lora_A` tensors, on
# q/k/v/o, with no error and no warning. So the checkpoint is unaffected either way -- and that
# is precisely the problem. The recorded `cfg.sha` would name seven target modules of which
# three were never adapted, the manifest would claim the MLP was trained, and an ablation of
# "full LoRA vs attention-only LoRA" on gpt-oss would compare two byte-identical runs and report
# a null result that is an artefact of a silent drop. A config that misdescribes its own
# checkpoint is the one thing this repository's rules exist to prevent.
#
# The checkpoint's MXFP4 quantisation points the same way: its `modules_to_not_convert` is
# exactly `self_attn`, `mlp.router`, `embed_tokens`, `lm_head`, so the attention projections are
# the part that stays in bf16 and takes an adapter cleanly.
GPT_OSS_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def resolve_lora_targets(architecture: str, configured: Sequence[str]) -> tuple[str, ...]:
    """The configured target set, narrowed to what this architecture can actually match. Pure.

    ONLY EVER REMOVES. A caller who asked for q/v on gpt-oss gets q/v, not the full attention
    block: silently widening a deliberately narrow set would be the same class of lie in the
    other direction. `mlp.router` is a real module (`GptOssTopKRouter`) and so peft COULD adapt
    it; it is excluded because LoRA on the router changes which experts fire, which is a
    different intervention from LoRA on the attention, and an arm that does both at once cannot
    be read.
    """
    if not architecture.startswith("GptOss"):
        return tuple(configured)
    kept = tuple(t for t in configured if t in GPT_OSS_LORA_TARGETS)
    if not kept:
        raise ValueError(
            f"target_modules={tuple(configured)} matches no module in {architecture}: its "
            f"experts are fused parameters, not submodules, so peft would attach no adapter at "
            f"all and train nothing while still writing a checkpoint and a loss curve. The "
            f"adaptable set is {GPT_OSS_LORA_TARGETS}."
        )
    return kept


# THE ONE FIELD `cfg.sha` DOES NOT COVER, and the exception is stated here rather than buried
# in the property. `cfg.sha` is the CHECKPOINT'S identity: what was trained, on which rows, at
# which hyperparameters. Whether this particular invocation continued a preempted one is a
# property of the EXECUTION -- the resumed run and the run it continues produce one checkpoint
# and must carry one identity, or a preemption silently splits an experiment into two that no
# table can rejoin. Which checkpoint (if any) was actually picked up is recorded in the
# manifest as `resumed_from`, where it belongs: provenance, not identity.
SHA_EXCLUDED = frozenset({"resume"})
# OPTIONAL KNOBS AT THEIR ABSENT DEFAULT ARE NOT PART OF THE IDENTITY. A field added later with
# default None renders exactly the bytes the older code rendered, so it must not move the sha of
# every run trained before it existed (measured 2026-09-15: the reasoning_effort field would
# have moved SFTConfig(base_model="Qwen/Qwen3-8B", tau=0.05, sigma_j=0.01).sha from
# 44eee2b7... to 9cb0f33f..., unjoining the four rung-1 runs already trained). When such a
# knob IS set it changes the rendered prompt and enters the sha like any other field.
SHA_OMIT_WHEN_NONE = frozenset({"reasoning_effort", "stop_weight"})
# THE SAME RULE FOR A KNOB WHOSE ABSENT STATE HAS A NAME. `curriculum` cannot use the frozenset
# above, because its "off" is not `None`: the arm that shuffles every epoch is a real schedule
# with a spelling, a row in the table and a flag on the command line, and it is also what every
# rung-1 run trained before this field existed did. Unlike `stop_weight`, there is no third
# state to record -- "the knob was considered and set to shuffled" and "the knob did not exist"
# produce the same bytes in the same order from the same seed, so giving them two identities
# would split one experiment rather than distinguish two. At the named default the field is
# dropped from the sha; at any other value it enters it like any other field.
SHA_OMIT_WHEN_DEFAULT: dict[str, Any] = {"curriculum": SHUFFLED}


@dataclass(frozen=True, slots=True)
class SFTConfig:
    base_model: str = ""
    out_dir: str = "artifacts/rung1"
    dataset: str = "data/rl/sft.jsonl"
    lora: LoraSpec = field(default_factory=LoraSpec)
    learning_rate: float = 1e-4
    epochs: int = 2
    max_seq_len: int = 12_288  # 12k, packed
    per_device_batch: int = 1
    grad_accum: int = 16
    warmup_ratio: float = 0.03
    seed: int = 0
    # The acceptance rule's two MEASURED inputs. No defaults: a threshold that was never
    # estimated must not be allowed to decide what the checkpoint imitates.
    tau: float | None = None
    sigma_j: float | None = None
    distinct3_floor: float = DISTINCT3_FLOOR
    # Cost model for `estimate_gpu_hours`, from the plan: ~180 GPU-h for 1,500 tasks x ~6
    # states x 8 candidates on one 80 GB card.
    tokens_per_second: float = 3_200.0
    # ---- HOW THE ROWS ARE RENDERED. `chat_template` is the default because the SERVER renders
    # through the template and training on the raw concatenation measures every dev NLL on a
    # prompt the policy never sees. See `mask.build_chat_masked_example`.
    chat_template: bool = True
    enable_thinking: bool = False
    # HARMONY'S KNOB, NOT QWEN'S. gpt-oss's template has no `enable_thinking`; it has
    # `reasoning_effort`, which sets a "Reasoning: low|medium|high" line in the SYSTEM message
    # (measured: one token either way, `n_prompt` 653 for all three on a dev row). `None` means
    # "the template's own default" -- which is also what the server renders when the gateway
    # sends no `chat_template_kwargs`, so `None` is the only value that is byte-identical to
    # serving by default. It is a plain field, so `cfg.sha` covers it: two checkpoints trained at
    # two efforts must not share one identity. Ignored, and refused, on a ChatML base.
    reasoning_effort: str | None = None
    # Packing lets example i+1 attend to example i's evidence, which destroys the byte identity
    # the template buys -- so the two are mutually exclusive and `validate()` says so.
    pack_sequences: bool = False
    # ---- THE LOSS. Three musique tasks contribute 107 ASK states each; unweighted, the policy
    # learns those tasks' phrasing. The exporter already computed the weight; this decides whether
    # the trainer honours it.
    use_sample_weight: bool = True
    # THE ONE KNOB THE STOP-EARLY ABLATION TURNS. Every rung-1 checkpoint stops far earlier than
    # the prompted base when it is run live (mean asks 3.0-3.5 against 6.25 on musique dev,
    # 1.3-1.5 against 5.6 on strategyqa) and loses evidence coverage doing it. Lowering the STOP
    # share of the FILE from 68% to 39% did not move that, which leaves the other reading --
    # the STOP rows take too much of the GRADIENT -- and this is the direct lever on it: every
    # STOP row's `sample_weight` is multiplied by this before the accumulation window is
    # normalised, so the arm is one number. `None` is ABSENT, not 1.0: at the absent default the
    # weights are byte-for-byte the ones every finished rung-1 run trained on, so it belongs in
    # `SHA_OMIT_WHEN_NONE`; set, it enters `cfg.sha` like any other field, 1.0 included -- "the
    # knob was considered and set to a no-op" is a different run from "the knob did not exist".
    #
    # WHAT IT COSTS, STATED HERE RATHER THAN DISCOVERED LATER. `accum_scale` normalises the
    # window by `per_device_batch * grad_accum` and that substitution is licensed by the
    # dataset's mean weight being 1 (the exporter normalises within kind). A stop weight is
    # exactly what breaks that: at 0.5 over a 68%-STOP file the mean weight is ~0.66, so the
    # window loss -- and the gradient -- is uniformly scaled by ~0.66 against the null arm.
    # `preflight` therefore reports `mean_row_weight` beside `stop_share_of_loss`, so the scale
    # is a recorded number rather than an assumption. The per-example RATIO, which is what this
    # ablation is about, is untouched by it.
    stop_weight: float | None = None
    # ---- IN WHAT ORDER THE ROWS ARE SHOWN. `"shuffled"` is the null and the default: every
    # epoch is a full shuffle, which is what every rung-1 run of record trained under. The two
    # alternatives order the FIRST epoch by the target need's depth (shallow-first or
    # deep-first) and shuffle the rest; see `curriculum.py` for what a stratum is and why the
    # STOP rows are dealt across them rather than read. Exposure is identical in all three --
    # every row exactly `epochs` times -- so the arms differ in order and in nothing else.
    #
    # WHAT IT COSTS ELSEWHERE, stated here rather than discovered in a loss curve. Under a
    # curriculum the sequence is materialised as one dataset of `epochs x N` rows and
    # `num_train_epochs` becomes 1 (`training_argument_kwargs`), because `Trainer` reshuffles
    # its sampler at every epoch boundary and would otherwise undo the ordering between epoch 1
    # and epoch 2. The optimiser step count is `floor(epochs*N / (batch*accum))` against the
    # null's `epochs * floor(N / (batch*accum))` -- equal when the division is exact and within
    # `epochs - 1` steps otherwise -- and the warmup is a RATIO of that total, so both arms
    # warm up over the same fraction of the same schedule.
    curriculum: str = SHUFFLED
    bf16: bool = True
    gradient_checkpointing: bool = True
    # ---- WHICH ARMS THE DATASET MAY CONTAIN. RECORDED HERE, ENFORCED BY THE EXPORTER: this rung
    # sees rows, not runs, so it cannot police provenance itself. Carrying it in `cfg.sha` is what
    # makes "which arms trained this checkpoint" answerable from the manifest alone.
    include_arms: tuple[str, ...] = ("inquirer_prompted",)
    # ---- SURVIVING A PREEMPTION. `save_steps` bounds what a killed job loses: 200 steps is
    # ~3,200 rows at the shipped effective batch (1 x 16). It is in `cfg.sha` because it changes
    # what the output directory contains. `resume` is NOT -- see `SHA_EXCLUDED`.
    save_steps: int = 200
    resume: bool = True

    @property
    def sha(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k not in SHA_EXCLUDED}
        for k in SHA_OMIT_WHEN_NONE:
            if d.get(k) is None:
                d.pop(k, None)
        for k, absent in SHA_OMIT_WHEN_DEFAULT.items():
            if d.get(k) == absent:
                d.pop(k, None)
        return h("rung1", canon(d))

    def validate(self) -> None:
        self.lora.validate()
        if not self.base_model:
            raise ValueError(
                "base_model is empty. There is no default: two rungs trained from two "
                "different bases and compared as one arm is exactly the confound the model "
                "pin exists to make impossible."
            )
        if self.tau is None or self.sigma_j is None:
            raise ValueError(
                "tau and sigma_j must both be MEASURED before the dataset is built: tau is "
                "the 35th percentile of pilot phi_LOO (pinq_train.reward.tau_from_pilot) and "
                "sigma_j is the judge's measured standard deviation. Guessing either one "
                "changes what the checkpoint imitates with no trace in the output."
            )
        if self.epochs < 1:
            raise ValueError(f"epochs={self.epochs}: must be >= 1")
        if self.save_steps < 1:
            raise ValueError(
                f"save_steps={self.save_steps}: must be >= 1. A non-positive cadence is a run "
                "that writes no checkpoint, which on a preemptible queue is a run that cannot "
                "be resumed however the resume call is spelled."
            )
        if self.stop_weight is not None:
            if not self.stop_weight > 0:
                raise ValueError(
                    f"stop_weight={self.stop_weight}: must be > 0. Zero deletes every STOP row "
                    "from the gradient while leaving it in every count the manifest reports, "
                    "and a negative weight trains the policy to do the opposite of what those "
                    "rows say while the loss curve still goes down."
                )
            if not self.use_sample_weight:
                raise ValueError(
                    "stop_weight is set but use_sample_weight is off, so the trainer is the "
                    "stock `Trainer` and is handed no weight column at all: the knob would be "
                    "in cfg.sha, in the manifest and in nothing that reaches a gradient."
                )
            if self.pack_sequences:
                raise ValueError(
                    "stop_weight is set but pack_sequences is on. A packed window is a "
                    "concatenation of rows carrying no single row's weight, so it trains at "
                    "1.0 by construction and the multiplier would silently apply to nothing."
                )
        if self.curriculum not in CURRICULA:
            raise ValueError(
                f"curriculum={self.curriculum!r}: must be one of {list(CURRICULA)}. A name "
                "nothing recognises would train the shuffled null arm while sitting in "
                "cfg.sha, in the manifest and in the preflight report as an ablation -- the "
                "arm would measure a difference of zero and the zero would be believed."
            )
        if self.curriculum != SHUFFLED and self.pack_sequences:
            raise ValueError(
                f"curriculum={self.curriculum!r} is set but pack_sequences is on. A packed "
                "window is a concatenation of whole rows chosen first-fit, so the row is no "
                "longer the unit the trainer steps over and an ordering defined on rows "
                "cannot survive it: the sequence would be re-cut into windows and the "
                "curriculum would apply to nothing."
            )
        if self.pack_sequences and self.chat_template:
            raise ValueError(
                "pack_sequences and chat_template cannot both be on. Packing concatenates "
                "examples into one window, so example i+1 attends to example i's evidence and "
                "the sequence stops being anything the server could render -- which is the one "
                "property the chat template exists to provide. Choose one."
            )

    @property
    def margin(self) -> float:
        self.validate()
        return margin_threshold(float(self.tau), float(self.sigma_j))


# --------------------------------------------------------------------------- the dataset


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist. Build it with `pi train export --kind sft` from recorded "
            "runs; this rung never invents its own data."
        )
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


@dataclass(frozen=True, slots=True)
class BuiltExamples:
    """What `build_examples` produced, AND what it had to throw away.

    The count is returned rather than logged because a dataset that quietly lost its longest rows
    is a dataset whose evidence-heavy states are missing -- exactly the states where deciding is
    hard. `train()` records `n_over_length` in the manifest, so the number lands beside the
    checkpoint rather than in a terminal nobody kept.
    """

    examples: tuple[MaskedExample, ...]
    n_over_length: int = 0
    # WHICH INPUT ROW EACH EXAMPLE CAME FROM. A `MaskedExample` carries tokens and a weight and
    # no identity, so once a row has been dropped for length the two lists no longer line up --
    # and anything that indexes the examples by a row's position (the curriculum does) would
    # then order a row's depth against another row's tokens, silently, on exactly the
    # evidence-heavy rows that were dropped. Empty on a `BuiltExamples` built by hand.
    kept_row_indices: tuple[int, ...] = ()


def is_stop_row(row: Mapping[str, Any]) -> bool:
    """Is this exported row a STOP? Read off the ACTION BYTES, and off nothing else.

    WHICH FIELD IS AUTHORITATIVE AT TRAINING TIME, and why it is this one. A row reaches the
    trainer as `state_text` plus `action_json`, and `build_masked_example` supervises exactly
    the tokens of `action_json` -- so the bytes ARE the target, and a weight keyed on anything
    beside them can scale the loss of a row whose target is an ASK. The two candidates were:

      * `label_rule == "stop_done"` -- the exporter's branch. It is `""` on every row written
        before the stop rule existed (`export.dataset.Example.label_rule`), so keying on it
        applies the ablation to a subset of the file that nothing in the manifest can name.
      * `is_stop` -- a flag set beside the bytes rather than derived from them, which is the
        exact disagreement `export.dataset._is_stop_row` already refuses to trust.

    So this is `_is_stop_row`'s call, made the same way: `pinq.actions.action_kind_of` is the
    policy parser's own normalisation, so the exporter, the loop and the trainer cannot
    disagree about a row. MEASURED on `data/rl/sft.jsonl` (43,837 rows, 2026-09-15): the three
    fields agree on all 43,837 and the 29,742 STOP rows are `STOP_ACTION_JSON` byte for byte.
    The choice is therefore about which field cannot drift, not about today's corpus.
    """
    return action_kind_of(row.get("action_json")) == "stop"


def row_weight(
    row: Mapping[str, Any], *, use_sample_weight: bool = True, stop_weight: float | None = None
) -> float:
    """The weight one exported row enters the loss with. PURE, and the ONLY place it is decided.

    `preflight` reports the share of the loss the STOP rows take and `build_examples` is what
    actually takes it; both call this, so the number recorded beside the checkpoint is the
    number the checkpoint trained on rather than a second implementation of it.

    `.get(k, 1.0)` returns None when the key EXISTS and is null, and `float(None)` raises. An
    explicit null is a row the exporter could not weight, which is the same case as a missing
    key: default it, and let `dataset_report` count it.
    """
    if not use_sample_weight:
        return 1.0
    raw = row.get("sample_weight")
    w = 1.0 if raw is None else float(raw)
    if stop_weight is not None and is_stop_row(row):
        w *= float(stop_weight)
    return w


def stop_loss_share(
    rows: Sequence[Mapping[str, Any]],
    *,
    use_sample_weight: bool = True,
    stop_weight: float | None = None,
) -> dict[str, Any]:
    """The STOP share OF THE LOSS after weighting, which is not the STOP share of the file.

    `dataset_report.stop_share` counts ROWS; the claim this ablation makes is about the
    GRADIENT, and the two are only equal because the exporter normalises `sample_weight` to
    mean 1 within each kind (MEASURED on data/rl/sft.jsonl: 0.6784679608549855 of rows,
    0.6784679608549308 of the loss). A stop weight separates them on purpose, so the number the
    paper quotes is measured here rather than inferred from 68% and 0.5 by a reader.

    `mean_row_weight` is beside it because `accum_scale` normalises the accumulation window by
    `per_device_batch * grad_accum` on the strength of that mean being 1 -- see `SFTConfig`.
    """
    ws = [row_weight(r, use_sample_weight=use_sample_weight, stop_weight=stop_weight) for r in rows]
    total = sum(ws)
    stop = sum(w for w, r in zip(ws, rows) if is_stop_row(r))
    return {
        "stop_weight": stop_weight,
        "stop_share_of_loss": (stop / total) if total else 0.0,
        "mean_row_weight": (total / len(ws)) if ws else 0.0,
        # BESIDE `dataset_report`'s `n_stop`, which counts the `is_stop` FLAG, so the two
        # predicates are visible in one report. They agree on every row of the corpus today
        # (43,837 of 43,837) and a file where they ever stop agreeing is a file whose
        # `stop_share` describes different rows from its `stop_share_of_loss`.
        "n_stop_by_action_json": sum(1 for r in rows if is_stop_row(r)),
    }


def build_examples(
    tok: Tokenizer | ChatTokenizer,
    rows: Iterable[dict[str, Any]],
    *,
    max_seq_len: int,
    chat_template: bool = False,
    enable_thinking: bool = False,
    reasoning_effort: str | None = None,
    use_sample_weight: bool = True,
    stop_weight: float | None = None,
    refuse_over_length: bool = True,
) -> BuiltExamples:
    """Exported rows -> masked examples. One row is one decision point.

    TWO PATHS, AND THEY HANDLE OVER-LENGTH DIFFERENTLY ON PURPOSE. The raw path left-truncates:
    the evidence block's head is the least load-bearing part of a concatenated state and the
    action lives at the right-hand end. The TEMPLATED path refuses, because its front is the chat
    header and a row missing `<|im_start|>user` is a row the server can never produce -- cutting
    it would reintroduce the train/serve gap the template exists to close.

    `refuse_over_length=True` (the default) raises on the first offender, which is what a human
    running a preflight should see. `train()` passes False so it can count them instead and put
    the number in the manifest.
    """
    out: list[MaskedExample] = []
    kept: list[int] = []
    n_over = 0
    for row_index, r in enumerate(rows):
        if chat_template:
            ex = build_chat_masked_example(
                tok,  # type: ignore[arg-type]
                str(r["state_text"]),
                str(r["action_json"]),
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )
            if len(ex.input_ids) > max_seq_len:
                if refuse_over_length:
                    raise OverLength(
                        f"a templated row is {len(ex.input_ids)} tokens, over the "
                        f"{max_seq_len}-token window. Left truncation would cut the chat header, "
                        "which is the one thing that makes the trainer's bytes the server's "
                        "bytes. Raise max_seq_len (see `pi train tokstats`) or drop the row."
                    )
                n_over += 1
                continue
        else:
            ex = build_masked_example(tok, str(r["state_text"]), str(r["action_json"]))  # type: ignore[arg-type]
            if len(ex.input_ids) > max_seq_len:
                ex = truncate_left(ex, max_seq_len)
        ex = dataclasses.replace(
            ex, weight=row_weight(r, use_sample_weight=use_sample_weight, stop_weight=stop_weight)
        )
        out.append(ex)
        kept.append(row_index)
    return BuiltExamples(examples=tuple(out), n_over_length=n_over, kept_row_indices=tuple(kept))


def dataset_report(
    rows: Sequence[dict[str, Any]], *, floor: float = DISTINCT3_FLOOR
) -> dict[str, Any]:
    """What the dataset is, before a gradient step: counts, stop share, and the diversity gate.

    The stop share matters as much as the count. Rejection-sampling SFT turns every below-floor
    state into a STOP target, so a dataset that is 90% STOP will train a policy that stops
    immediately -- and it will do so while every per-example loss looks healthy.
    """
    n = len(rows)
    stops = sum(1 for r in rows if r.get("is_stop"))
    # ONE PASS for both lists. Building them separately and zipping would let the two drift
    # under any future change to the predicate, and a misaligned zip scores one task's
    # questions under another's id -- the gate would then read a number describing nothing.
    questions: list[str] = []
    question_task_ids: list[str] = []
    for r in rows:
        if r.get("is_stop") or not _is_json(r.get("action_json")):
            continue
        q = str(json.loads(r["action_json"]).get("question", ""))
        if not q:
            continue
        questions.append(q)
        question_task_ids.append(str(r.get("task_id", "")))

    # NO `if questions else None`. `assert_diverse` refuses an empty sample on purpose -- "an
    # empty sample passes every diversity test ever written" -- and guarding the call with the
    # very condition it exists to reject meant a FULLY collapsed dataset (100% STOP, zero
    # questions) skipped the gate entirely and reported `distinct_3: None`. The gate was
    # unreachable precisely in the state it guards against.
    # TASK IDS PASSED, so the gate measures WITHIN-task variation. Pooled distinct-3 scores
    # the gold answer key at 0.2250 -- a perfectly state-dependent policy -- so gating on it
    # rejected exactly the datasets it exists to protect. It is still reported below.
    # `floor` IS REPORTED, NOT ENFORCED, and saying so is the point of this comment. With
    # `task_ids` supplied the gate inside `assert_diverse` is WITHIN_TASK_FLOOR (0.65); `floor`
    # only sets `DiversityReport.floor`/`.ok`, and that branch never reads `.ok`. Wiring
    # `cfg.distinct3_floor` here makes the POOLED diagnostic comparable against the configured
    # number instead of a hard-coded one; it does not create a second gate, and a reader who
    # believed it did would trust a threshold that cannot fire.
    div = assert_diverse(questions, floor=floor, task_ids=question_task_ids)

    stop_share = (stops / n) if n else 0.0
    if stop_share > STOP_SHARE_CEILING:
        # Reported AND gated. This docstring already says a 90%-STOP dataset trains a policy
        # that stops immediately "while every per-example loss looks healthy" -- a number that
        # is computed, printed and never acted on is a number that documents the failure
        # instead of preventing it.
        raise ModeCollapse(
            f"{stop_share:.0%} of examples are STOP, above the {STOP_SHARE_CEILING:.0%} "
            f"ceiling ({stops} of {n}). Rejection-sampling SFT turns every below-floor state "
            "into a STOP target, so this trains a policy that stops immediately -- and every "
            "per-example loss will look healthy while it does."
        )

    return {
        "n_examples": n,
        "n_stop": stops,
        "stop_share": stop_share,
        "n_questions": len(questions),
        # The two counts that say how much of the dataset the loss and the wall can actually
        # vouch for. An unweighted row is trained at weight 1.0, which is a DEFAULT and not a
        # measurement; a row with no `split` is a legacy row the held-out check below waved
        # through. Both are silent unless counted, so both are counted.
        "n_unweighted_rows": sum(1 for r in rows if r.get("sample_weight") is None),
        "n_rows_without_split": sum(1 for r in rows if r.get("split") is None),
        "distinct3_floor_reported": floor,
        # BOTH reported. `distinct_3` is kept as a diagnostic, and because every earlier
        # report carries it -- dropping it would make old and new reports silently
        # incomparable. The other key is named `..._of_dataset` ON PURPOSE: this number is a
        # property of the TRAINING FILE, identical for every arm trained on the same export
        # regardless of base model or seed. `gate.py`'s `criteria["distinct3"]` answers to the
        # same "within-task distinct-3" prose but is measured on the CHECKPOINT's own generated
        # questions at inference -- a different number per checkpoint on this same file. Do
        # not rename this back to the bare `within_task_distinct_3`: that name is what let a
        # dataset-level and a policy-level figure sit under one label in the first place. See
        # `tests/test_train_gate.py::test_a_per_model_report_cannot_present_the_datasets_distinct_3_as_the_policys`.
        "distinct_3": div.distinct_3 if div else None,
        "within_task_distinct_3_of_dataset": within_task_distinct_n(
            list(zip(question_task_ids, questions))
        ),
        "suites": sorted({str(r.get("suite_id", "")) for r in rows}),
    }


def _is_json(text: object) -> bool:
    try:
        json.loads(str(text))
    except (json.JSONDecodeError, TypeError):
        return False
    return True


# ----------------------------------------------------------------------------- the weighted loss
#
# Plan I.5 defines the batch loss as `sum_i w_i L_i / sum_i w_i`, with `L_i` the per-example
# token-mean cross-entropy over the supervised positions and `w_i` the exporter's `sample_weight`.
# Both functions below are PURE and module-level, which is the whole design: the only part of this
# that needs a GPU is one call into torch, and the arithmetic that decides what the checkpoint
# actually optimises is tested on a laptop.


def weighted_loss(per_example_token_means: Sequence[float], weights: Sequence[float]) -> float:
    """`sum(w*m) / sum(w)`. Equal to the plain mean when every weight is 1."""
    if len(per_example_token_means) != len(weights):
        raise ValueError(
            f"{len(per_example_token_means)} losses but {len(weights)} weights: a misaligned zip "
            "would score one example under another's weight, and nothing downstream could see it."
        )
    if not weights:
        raise ValueError("no examples: an empty batch has no mean, weighted or otherwise")
    total = sum(float(w) for w in weights)
    if total <= 0:
        raise ValueError(
            f"the weights sum to {total}. A non-positive total either divides by zero or flips "
            "the sign of the gradient, and a flipped gradient trains the policy to do the "
            "opposite of what the data says while the loss curve still goes down."
        )
    return sum(float(w) * float(m) for w, m in zip(weights, per_example_token_means)) / total


def accum_scale(
    weights: Sequence[float],
    *,
    per_device_batch: int,
    grad_accum: int,
    num_items_in_batch: int | None,
) -> float:
    """What to multiply this micro-batch's `weighted_loss` by so the WINDOW is weighted correctly.

    THE BUG THIS EXISTS TO PREVENT. `sum(w*m)/sum(w)` does not decompose across micro-batches. At
    the shipped hyperparameters -- `per_device_batch=1`, `grad_accum=16` -- each micro-batch holds
    exactly one example, and `w*m/w = m`: the weights cancel EXACTLY and the weighting becomes a
    silent no-op. No loss curve, no metric and no existing test in this repository could show it.

    So the normaliser is the WINDOW's weight sum rather than the micro-batch's. We do not know the
    window's weights from inside one micro-batch, but we know the dataset's mean weight is 1: the
    exporter normalises `sample_weight` to mean 1 within kind, and
    `tests/test_sft_weighting.py::test_the_exported_dataset_has_mean_weight_one` measures it on
    the real file (44,624 rows, mean 0.9999999999999636). So `sum_window(w) ~= N_window =
    per_device_batch * grad_accum`, and each micro-batch contributes

        weighted_loss(m, w) * sum(w) / N_window  ==  sum_micro(w*m) / N_window

    which sums over the window to `sum_window(w*m) / N_window`, the weighted mean.

    HOW `num_items_in_batch` IS HONOURED. Its VALUE is a token count and our normalisation is per
    EXAMPLE, so the value is not usable here. Its PRESENCE is: `Trainer.training_step` divides the
    returned loss by `gradient_accumulation_steps` only when it is absent. Present -> we divide by
    the full `N_window`; absent -> we divide by `per_device_batch` and let the Trainer supply the
    other factor. Both branches produce the identical window loss, which is what
    `test_sixteen_accumulated_micro_batches_equal_one_batch_of_sixteen` pins.

    TWO HONEST CAVEATS. (1) The last, partial accumulation window of an epoch is normalised by the
    full `N_window` and is therefore slightly down-weighted -- one window in 2,789 at 44,624 rows
    and `grad_accum=16`. (2) The whole construction rests on mean(w)=1 over the dataset, which is
    why `dataset_report` counts `n_unweighted_rows`: rows defaulted to 1.0 pull the true mean away
    from 1 and are the one way this quietly stops being exact.
    """
    if per_device_batch < 1 or grad_accum < 1:
        raise ValueError(
            f"per_device_batch={per_device_batch}, grad_accum={grad_accum}: both must be >= 1"
        )
    total = sum(float(w) for w in weights)
    denom = (per_device_batch * grad_accum) if num_items_in_batch is not None else per_device_batch
    return total / float(denom)


def estimate_gpu_hours(cfg: SFTConfig, n_tokens: int) -> float:
    """Wall hours on one card, from the packed token count. Arithmetic, not a benchmark."""
    if cfg.tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be positive")
    return (n_tokens * cfg.epochs) / cfg.tokens_per_second / 3600.0


def training_argument_kwargs(cfg: SFTConfig, field_names: Container[str]) -> dict[str, Any]:
    """The `TrainingArguments` kwargs, spelled for the transformers that is INSTALLED.

    PURE, and module-level, for the same reason `accum_scale` is: `_build_trainer` needs
    `transformers` and is therefore the one line of this rung no test on a laptop can reach, so
    what it passes was pinned by nothing. `tests/test_trainer_arguments_across_transformers.py`
    drives this against both field sets with no library present at all.

    TWO NAMES THIS TRAINER PASSED WERE REMOVED IN transformers 5.0, and `[train]` allows
    `transformers>=4.42` -- both sides of that break:

      * `warmup_ratio` is gone; `warmup_steps` absorbed it and is now a FLOAT, where an integer
        is an exact step count and a value in [0, 1) is a ratio of total steps. So 0.03 is
        `warmup_ratio=0.03` on 4.x and `warmup_steps=0.03` on 5.x. This branches rather than
        renaming because `warmup_steps=0.03` handed to 4.x is an integer step count that
        truncates to zero -- no warmup, no error, no trace.
      * `group_by_length` is gone outright. It was only ever passed as False ("OFF, always":
        a length-ordered epoch is a curriculum nobody chose), and a library with no length
        grouping needs no flag to disable it, so under 5.x the key is simply not emitted. It
        stays False under `cfg.curriculum` too: ordering by evidence size is a SECOND ordering
        intervention, and an arm that ran both could not be read.

    `num_train_epochs` IS 1 UNDER A CURRICULUM, AND THAT IS NOT A SHORTER RUN. `Trainer` draws a
    fresh `RandomSampler` permutation at every epoch boundary, so a row order handed to it
    survives exactly one epoch; the ordering is therefore materialised as ONE dataset of
    `epochs x N` rows (`train()`) that the trainer makes one pass over, with a sequential
    sampler. The optimiser sees the same rows the same number of times in the same total number
    of steps -- `floor(epochs*N / (batch*accum))` here against `epochs * floor(N /
    (batch*accum))` in the null arm, equal when that division is exact and within `epochs - 1`
    steps of it otherwise -- and `warmup_ratio`, `learning_rate` and the cadence keys are
    untouched, so the LR schedule is the same schedule. It is spelled HERE, in the pure
    function, rather than inside `_build_trainer`, because this is the half of the mapping a
    laptop can test.

    MEASURED, transformers 5.17.0, first `pi train rung1 --train` of the Mac smoke:
    `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'`
    -- raised after the weights had loaded and after preflight had printed a clean report.

    THE CHECKPOINT CADENCE IS PROBED THE SAME WAY. `save_strategy="epoch"` wrote one checkpoint
    per epoch, and rung 1 is 2 epochs over ~44k rows: a job preempted at 95% of epoch 1 left
    nothing on disk to resume from. A step cadence bounds that loss to `cfg.save_steps`. Where
    the installed class declares no `save_steps`, neither it nor `save_strategy="steps"` is
    emitted -- a strategy naming a cadence the library cannot express is the `warmup_ratio`
    failure in a new place -- and the epoch strategy stands.
    """
    kw: dict[str, Any] = {
        "output_dir": cfg.out_dir,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": 1 if cfg.curriculum != SHUFFLED else cfg.epochs,
        "per_device_train_batch_size": cfg.per_device_batch,
        "gradient_accumulation_steps": cfg.grad_accum,
        "seed": cfg.seed,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": [],
        "bf16": cfg.bf16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        # THE COLUMN FILTER, OFF. `_Rows` is a plain torch Dataset, so `Trainer._get_dataloader`
        # wraps the collator in a `RemoveColumnsCollator` keyed on `inspect.signature(
        # model.forward)` -- and `Qwen3ForCausalLM.forward` has no `sample_weight` parameter, so
        # the one column the weighted loss exists to read was dropped between the dataset and
        # `compute_loss`. MEASURED: `KeyError: 'sample_weight'` at step 0 of 19.
        #
        # Unconditional rather than gated on `use_sample_weight`: a hand-built dataset yields
        # exactly the keys this trainer put there, so a filter over it can only subtract
        # something deliberate. The default is True, and that default is old -- this is not a
        # transformers 5 regression, it is the reason the weighted path had never run anywhere.
        "remove_unused_columns": False,
    }
    if "warmup_ratio" in field_names:
        kw["warmup_ratio"] = cfg.warmup_ratio
    else:
        kw["warmup_steps"] = cfg.warmup_ratio
    if "group_by_length" in field_names:
        kw["group_by_length"] = False
    # --- resume ---
    if "save_steps" in field_names:
        kw["save_strategy"] = "steps"
        kw["save_steps"] = cfg.save_steps
    if "save_total_limit" in field_names:
        # TWO, not one. `Trainer` writes the new checkpoint before deleting the old, so a limit
        # of 1 still needs room for two -- and a preemption during that window would otherwise
        # leave the only surviving checkpoint the half-written one.
        kw["save_total_limit"] = 2
    # --- end resume ---
    return kw


# --------------------------------------------------------------------------- the one GPU line


def base_model_load_kwargs(n_visible_gpus: int, *, dtype: Any) -> dict[str, Any]:
    """The `from_pretrained` kwargs that put the base model where it fits. PURE.

    ONE CARD IS THE UNCHANGED PATH and returns `{}` -- the bare
    `AutoModelForCausalLM.from_pretrained(cfg.base_model)` this rung has always used, which is
    what the 8B arms of record ran. A placement kwarg appearing there would change how a
    finished arm's weights were loaded without changing anything that names it.

    MORE THAN ONE CARD gets accelerate's naive model parallelism: `device_map="auto"` splits
    the layers across the visible devices and leaves `model.hf_device_map` behind, which is the
    flag the HF Trainer reads. Qwen3-32B is ~64 GB of bf16 weights and does not fit on one
    A100-80GB beside its activations, so the 32B is the first arm that needs it.

    WHY THE TRAINER MUST SEE `hf_device_map`, transcribed from the installed transformers
    (5.17.0, the HPC training venv):

      * `trainer.py:449-455` -- `is_model_parallel` is True only when `hf_device_map` spans
        more than one non-cpu/disk device;
      * `trainer.py:489-491` -- "Force n_gpu to 1 to avoid DataParallel as MP will manage the
        GPUs"; without it `training_args.py:1917` leaves `n_gpu = torch.cuda.device_count()`
        and `trainer.py:2543-2548` wraps the model in `nn.DataParallel`, which REPLICATES a
        model that did not fit once;
      * `trainer.py:465-477` -- the same flag turns off `place_model_on_device`, so the
        Trainer does not then call `.to(cuda:0)` on layers that are deliberately elsewhere.

    THE DTYPE IS STATED RATHER THAN DEFAULTED because `device_map="auto"` computes the split
    FROM it: transformers 5.x takes the checkpoint's own (bf16 for every Qwen3) but 4.x
    upcast to float32, and 128 GB over two 80 GB cards is answered by offloading layers to
    CPU, not by raising -- a run that is 10x slower and says nothing. `dtype=` is the
    transformers 5 spelling; 4.x calls it `torch_dtype` (see `pinq_train.merge._hf_loaders`,
    which carries the retry for the one caller that must work on both).

    NOT A CONFIG FIELD, on purpose. How many cards the job was given is a property of the
    machine, like `wall_ms`: it is recorded in `rung1.manifest.json` beside the checkpoint and
    stays out of `cfg.sha`, so the same recipe on one card and on two is one experiment.
    """
    if n_visible_gpus > 1:
        return {"device_map": "auto", "dtype": dtype}
    return {}


# --------------------------------------------------- the loss, and where it is supervised


def logits_to_keep_for(labels: Any, ignore_index: int = IGNORE_INDEX) -> int | None:
    """How many TRAILING positions of the logits the masked loss can reach, or None for all.

    `_weighted_ce` materialises `[T, V]` three times and the model's own `ForCausalLMLoss` adds
    a fourth by upcasting the whole block to fp32; with V = 151,669 one of them is 3.1 GB at
    T = 5,120. The mask leaves only the ACTION supervised and the action is the tail, so every
    logit before it is multiplied by zero. `logits_to_keep` makes the model slice the hidden
    states BEFORE the LM head, so the big tensor is never built.

    PLUS ONE, because the loss is shifted: the logit that predicts the first supervised label
    sits one position earlier. Keeping `n_action` exactly drops that token from the loss and
    raises nothing.

    READ OFF THE LABELS, not off `n_action` or any config field. The earliest supervised
    position is the only thing that is true by construction, and it stays true for a template
    that supervises more than one span. `None` means nothing in the batch is supervised, which
    is not "keep nothing" -- it is a batch this has no opinion about, so the old path runs.

    MEASURED, Qwen3-0.6B on CPU in fp32, one forward + backward at T = 5,120 with a 128-token
    action, peak RSS from `ru_maxrss` in a fresh process per arm:

        full logits (1, 5120, 151936):  32.667 GB      loss 10.118711
        kept span   (1,  129, 151936):  15.527 GB      loss 10.118711

    17.14 GB, 52.5%, and the same number to every printed digit. The equality is not incidental
    -- it is what `test_the_tail_loss_equals_the_full_logits_loss` pins at 1e-6.
    """
    rows = labels.tolist() if hasattr(labels, "tolist") else [list(r) for r in labels]
    if rows and not isinstance(rows[0], list):
        rows = [rows]
    width = len(rows[0]) if rows else 0
    first = min(
        (i for row in rows for i, v in enumerate(row) if v != ignore_index),
        default=None,
    )
    return None if first is None else min(width, width - first + 1)


def accepts_logits_to_keep(model: Any) -> bool:
    """Whether `model.forward` can be handed `logits_to_keep`.

    `[train]` allows `transformers>=4.42` and the kwarg arrives in 4.45, so this is probed, not
    assumed -- an unexpected keyword argument would be a TypeError at step 0 on the rented box.

    `**kwargs` COUNTS. `PeftModelForCausalLM.forward` names `input_ids`, `labels` and `**kwargs`
    and passes the rest down, so a probe for the name alone would disable this on every LoRA
    run -- which is every run this rung makes. A wrapper that accepts the kwarg and ignores it
    costs memory and nothing else, because `weighted_ce` aligns to the logits it was RETURNED.
    """
    import inspect

    try:
        params = inspect.signature(model.forward).parameters
    except (AttributeError, TypeError, ValueError):
        return False
    if "logits_to_keep" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def forward_for_loss(model: Any, inputs: dict[str, Any]) -> Any:
    """Run the model, materialising only the logits `weighted_ce` can reach.

    THE LABELS ARE NOT PASSED DOWN with a tail. `ForCausalLMLoss` shifts the FULL label sequence
    against whatever logits it is handed, so a tail plus full labels is a shape error; and the
    loss it computes was thrown away here anyway, which means dropping it also removes the
    `logits.float()` copy of the whole `[T, V]` block -- the largest allocation of the step.
    """
    keep = logits_to_keep_for(inputs["labels"])
    if keep is None or not accepts_logits_to_keep(model):
        return model(**inputs)
    return model(**{k: v for k, v in inputs.items() if k != "labels"}, logits_to_keep=keep)


def weighted_ce(
    logits: Any,
    labels: Any,
    weights: Any,
    *,
    per_device_batch: int,
    grad_accum: int,
    num_items_in_batch: int | None,
) -> Any:
    """Per-example token-mean CE over `labels != -100`, combined by `weighted_loss`.

    All of the arithmetic that decides what the checkpoint optimises lives in the pure functions
    above; this is the torch adapter and nothing else. It is module-level rather than a closure
    so that the alignment below can be tested against the full-logits number it has to equal.

    `logits` MAY BE A TAIL. The labels are aligned to what the model actually returned -- not to
    what `forward_for_loss` asked for -- so the same code path serves the full call and the
    kept-span call, and a model that swallowed the kwarg still gets the right number. A tail
    that does not line up raises: silently aligning it would put the loss on the wrong tokens.
    """
    import torch

    kept = logits.shape[-2]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., labels.shape[-1] - kept + 1 :].contiguous()
    if tuple(shift_logits.shape[:-1]) != tuple(shift_labels.shape):
        raise ValueError(
            f"logits of {tuple(logits.shape)} and labels of {tuple(labels.shape)} do not line "
            "up: the kept span must be a TAIL of the label sequence. Aligning them anyway "
            "would compute the loss against the wrong tokens and report nothing."
        )
    per_token = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view(shift_labels.shape)
    supervised = (shift_labels != IGNORE_INDEX).float()
    means = (per_token * supervised).sum(-1) / supervised.sum(-1).clamp(min=1.0)
    w = weights.to(means.dtype).view(-1)
    scale = accum_scale(
        [float(x) for x in w.detach().cpu().tolist()],
        per_device_batch=per_device_batch,
        grad_accum=grad_accum,
        num_items_in_batch=num_items_in_batch,
    )
    # `weighted_loss` in torch, term for term: sum(w*m)/sum(w), then the window scale.
    return (w * means).sum() / w.sum().clamp(min=1e-12) * scale


def assert_train_split(rows: Sequence[dict[str, Any]]) -> None:
    """Refuse a dataset carrying any row stamped with a split other than `train`.

    PRESENT-AND-WRONG is refused; ABSENT is waved through and counted. A row written before the
    exporter stamped splits carries no `split` key at all, and refusing those would reject every
    historical dataset over a field they could not have had -- so they are reported as
    `n_rows_without_split` instead, where a reader can see how much of the check was vacuous. An
    EMPTY split is present-and-wrong: it means the exporter looked and could not tell, which is
    not the same as nobody having asked.
    """
    seen: dict[str, int] = {}
    for r in rows:
        v = r.get("split")
        if v is None or str(v) == "train":
            continue
        seen[str(v)] = seen.get(str(v), 0) + 1
    if seen:
        detail = ", ".join(f"{k or '<empty>'}={v}" for k, v in sorted(seen.items()))
        raise HeldOutDataset(
            f"{sum(seen.values())} of {len(rows)} rows are not on the train side of the wall "
            f"({detail}). A checkpoint that saw dev is a checkpoint whose dev number is a "
            "training number, and nothing downstream can unlearn it. Export with "
            "`pi train export --split train`, or point --dataset at the train file."
        )


def preflight(cfg: SFTConfig, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Everything that can be checked before a device exists. Runs in milliseconds."""
    cfg.validate()
    assert_train_split(rows)
    rep = dataset_report(rows, floor=cfg.distinct3_floor)
    rep["margin_threshold"] = cfg.margin
    rep["config_sha"] = cfg.sha
    # WHICH ARM THIS IS, printed where an operator will actually see it. `pi train rung1`
    # without `--train` prints this report and nothing else, so a run whose only difference
    # from the null is the row ORDER would otherwise be indistinguishable from it on screen --
    # and `--curriculum` is the one flag whose effect leaves no other trace in the numbers
    # below. Reported unconditionally, like `stop_weight`: the null arm's value is what the
    # ablation's arms are read against.
    rep["curriculum"] = cfg.curriculum
    # What the STOP rows are worth in the GRADIENT, not in the file. Reported unconditionally:
    # the null arm's number is what the two ablation arms are read against, and a number that
    # only appears when the knob is set is a number with nothing to compare to.
    rep.update(
        stop_loss_share(rows, use_sample_weight=cfg.use_sample_weight, stop_weight=cfg.stop_weight)
    )
    return rep


def train(
    cfg: SFTConfig,
    *,
    acknowledge_untested: bool = False,
    tokenizer: Tokenizer | None = None,
) -> dict[str, Any]:
    """LoRA SFT with the masked loss. THE ONLY function here that needs a GPU.

    `acknowledge_untested` is required because no run of this function has ever completed on
    this project's hardware -- there is none. Passing it is a statement that the caller knows
    the first execution is also the first test.
    """
    if not acknowledge_untested:
        raise NotValidatedOnHardware(
            "rung 1 has never been executed: this project has no local CUDA device (Apple "
            "M1 Max) and no rented GPU has been booked. Everything except this call is "
            "covered by tests; this call is not. Pass acknowledge_untested=True to run it, "
            "and treat the first run as a debugging session rather than as an experiment."
        )
    cfg.validate()
    # Lazy on purpose: `make gate` and the default test run must never import these.
    import torch  # `torch.cuda.device_count()` below; also fails early and clearly if absent
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = tokenizer or AutoTokenizer.from_pretrained(cfg.base_model)
    rows = read_jsonl(cfg.dataset)
    report = preflight(cfg, rows)
    built = build_examples(
        tok,
        rows,
        max_seq_len=cfg.max_seq_len,
        chat_template=cfg.chat_template,
        enable_thinking=cfg.enable_thinking,
        reasoning_effort=cfg.reasoning_effort,
        use_sample_weight=cfg.use_sample_weight,
        stop_weight=cfg.stop_weight,
        # COUNT rather than raise: a preflight run by a human should stop on the first
        # over-length row, but a training run that has already reached a rented GPU should
        # report how many it dropped, in the manifest, beside the checkpoint.
        refuse_over_length=False,
    )
    report["n_over_length_refused"] = built.n_over_length
    # PACK ONLY WHEN ASKED. `validate()` has already refused packing together with the template,
    # so under the default config this is off and every row is one sequence -- which is what makes
    # the trained bytes the served bytes. A packed window is a concatenation of examples and
    # carries no single row's weight, so it trains at 1.0 by construction.
    if cfg.pack_sequences:
        items = [(ids, labels, 1.0) for ids, labels in pack(list(built.examples), cfg.max_seq_len)]
    else:
        items = [(list(e.input_ids), list(e.labels), e.weight) for e in built.examples]

    # IN WHAT ORDER, AND HOW THE EPOCHS MAP ONTO IT. Under the default `shuffled` this is the
    # untouched path: N items, `num_train_epochs = cfg.epochs`, the library's own per-epoch
    # shuffle. Under a curriculum the whole `epochs`-pass sequence is materialised here, ONCE,
    # and `_build_trainer` walks it sequentially -- because `Trainer` reshuffles at every epoch
    # boundary, so an ordering handed to it survives exactly one epoch.
    #
    # ORDERED OVER THE ROWS THAT SURVIVED, not over the file: `build_examples` drops rows that
    # do not fit the window, and indexing the kept examples by an original row's position would
    # order one row's depth against another row's tokens with nothing to see it.
    sequence = items
    if cfg.curriculum != SHUFFLED:
        kept_rows = [rows[i] for i in built.kept_row_indices]
        if len(kept_rows) != len(items):
            raise ValueError(
                f"{len(kept_rows)} rows survived the window but there are {len(items)} "
                f"training items: a curriculum orders ROWS, and it cannot be applied to a "
                "sequence whose units are not rows."
            )
        order = order_examples(kept_rows, cfg.curriculum, cfg.seed, cfg.epochs)
        sequence = [items[i] for i in order]
    # WHAT THE TRAINER'S ONE PASS ACTUALLY HOLDS, recorded because `num_train_epochs` alone no
    # longer says it: under a curriculum it is 1 over `epochs x N` rows, and a reader of the
    # manifest would otherwise have to infer the multiplication.
    report["n_train_sequences"] = len(sequence)

    # WHERE THE WEIGHTS GO. `device_count()` counts VISIBLE devices -- LSF sets
    # CUDA_VISIBLE_DEVICES to the ordinals it allocated -- so this is the job's allocation and
    # not the node's inventory, and a 2-GPU job on an 8-GPU node splits across 2. On one card
    # (and on a machine with none) `base_model_load_kwargs` returns `{}` and this line is the
    # bare load it has always been.
    n_gpus = torch.cuda.device_count()
    load_kwargs = base_model_load_kwargs(n_gpus, dtype="bfloat16" if cfg.bf16 else None)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs)
    # BELT AND SUSPENDERS on tied embeddings. `from_pretrained` is expected to tie
    # `lm_head.weight` to the input embedding automatically for any `tie_word_embeddings=True`
    # architecture (Granite-3.3-8B is one; Qwen3-8B is not) as part of its normal load path.
    # Verified this is NOT automatic on the meta-device construction path used to cross-check
    # `lora_capacity` below (`AutoModelForCausalLM.from_config()` under
    # `accelerate.init_empty_weights()` leaves `lm_head.weight is embed_tokens.weight` False
    # until `tie_weights()` is called, which silently double-counts one embedding matrix in
    # `total_params`). `from_pretrained` is documented to call this itself already, so this is
    # a no-op here in the ordinary case -- but a no-op is cheaper than a doubted number, and
    # `total_params` below is the only place this repo reports a base parameter count at all.
    model.tie_weights()
    # NAMED BEFORE THE PEFT WRAP, not re-derived from the wrapped model: `type(model).__name__`
    # after `get_peft_model` reads back as `PeftModel`/`PeftModelForCausalLM`, which is the
    # wrapper's class, not the base architecture the manifest needs for a cross-family
    # comparison.
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
    # `get_nb_trainable_parameters` IS `peft`'s own implementation of the trainable/total
    # count -- not a second, independently-written counting method that could silently drift
    # from what the trainer itself sees as trainable.
    trainable_params, total_params = model.get_nb_trainable_parameters()
    trainer = _build_trainer(cfg, model, tok, sequence)
    # WHERE THE PREEMPTED JOB LEFT OFF. Read BEFORE the output directory is created, so a first
    # submission sees no directory and resumes nothing. `None` is passed explicitly under
    # `--no-resume`: "told to start fresh" and "never told anything" are different facts, and
    # `trainer.train()` with no argument -- what this line used to be -- is the second one.
    resumed_from = latest_checkpoint(cfg.out_dir) if cfg.resume else None
    trainer.train(resume_from_checkpoint=resumed_from)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    # HOW THE WEIGHTS WERE PLACED, read back off the kwargs that were actually passed rather
    # than re-derived, so the record cannot drift from the load. Beside the config, not in it:
    # the card count is a machine artefact like `wall_ms`, and a run that had to be split over
    # two GPUs is the same experiment as one that fit on one. Null `device_map` is the
    # single-card load, which is what every 8B arm of record ran.
    placement = {
        "n_gpus": n_gpus,
        "device_map": load_kwargs.get("device_map"),
        "dtype": load_kwargs.get("dtype"),
    }
    # THE FIGURE A CROSS-FAMILY CAPACITY CLAIM NEEDS, so it can be READ rather than retyped
    # from a paper draft. Beside `placement`, not in `config`: `cfg.lora.target_modules` (what
    # the run was ASKED for, which may name modules absent from this architecture -- see
    # `resolve_lora_targets`) already lives in `config` and is part of `cfg.sha`; `architecture`,
    # the RESOLVED module list and the two parameter counts are facts about what this run's
    # PEFT wrap actually produced for THIS base model, discovered only after the load, the same
    # reason `placement` itself is out here and not in `config`. Adding this dict cannot move
    # `cfg.sha`: the property reads only `asdict(self)` of the config dataclass (see
    # `tests/test_model_placement.py::test_lora_capacity_does_not_move_the_config_sha`).
    lora_capacity = {
        "architecture": architecture,
        "target_modules_resolved": list(target_modules_resolved),
        "trainable_params": trainable_params,
        # NAMED `total_params`, NOT `base_params`: this is `peft`'s `all_param` -- base model
        # PLUS the LoRA adapter's own parameters -- not the pure base-only count. A reader who
        # wants base-only subtracts `trainable_params` and, for a rank/alpha this repo trains
        # at, is off by a few thousand LoRA-only buffers that are not `requires_grad`; naming
        # this field for what it actually counts is cheaper than a second, easily-confused key.
        "total_params": total_params,
    }
    (out / "rung1.manifest.json").write_text(
        json.dumps(
            # `resumed_from` is beside the config, not in it: a resumed run and the run it
            # continues share one `cfg.sha`, so this is the only place that can say which of
            # the two this directory holds.
            {
                "config": asdict(cfg),
                "dataset": report,
                "resumed_from": resumed_from,
                "placement": placement,
                "lora_capacity": lora_capacity,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "out_dir": str(out),
        # THE FILE'S UNITS, not the trainer's pass. `n_packed` has always been "how many
        # sequences one epoch is"; under a curriculum the trainer's dataset is `epochs` of
        # those, and that number is `n_train_sequences` in the report above.
        "n_packed": len(items),
        "resumed_from": resumed_from,
        "placement": placement,
        "lora_capacity": lora_capacity,
        **report,
    }


def _build_trainer(cfg: SFTConfig, model: Any, tok: Any, items: Sequence[Any]) -> Any:
    """Assemble the HF Trainer over PRE-MASKED sequences, with the exporter's weights in the loss.

    The labels are built here, not by a collator's `mlm=False` path: the default causal collator
    copies `input_ids` into `labels` wholesale, which is precisely the whole-sequence loss this
    rung exists to avoid. Handing the trainer labels it did not derive is the only way to be
    certain the mask survives.

    `items` are `(input_ids, labels, weight)` triples -- one row each under the chat template, one
    packed window each under `pack_sequences`, and under a curriculum the `epochs x N` SEQUENCE
    `train()` already ordered. THE DATASET RETURNS NO PADDING, so
    `per_device_train_batch_size > 1` over unpacked rows will raise in the default collator rather
    than silently mis-batch. That is the intended failure: the shipped config is batch 1 x
    accumulation 16, and a padding collator would need its own attention-mask test before it could
    be trusted here.
    """
    import torch
    from torch.utils.data import Dataset, SequentialSampler
    from transformers import Trainer, TrainingArguments

    class _Rows(Dataset):  # pragma: no cover - requires torch
        def __len__(self) -> int:
            return len(items)

        def __getitem__(self, i: int) -> dict[str, Any]:
            ids, labels, weight = items[i]
            item = {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.ones(len(ids), dtype=torch.long),
            }
            # ONLY under the weighted trainer. The stock `Trainer.compute_loss` forwards the whole
            # item into `model(**inputs)`, where an extra key is an unexpected keyword argument.
            if cfg.use_sample_weight:
                item["sample_weight"] = torch.tensor(float(weight), dtype=torch.float)
            return item

    class _Ordered:  # pragma: no cover - requires torch
        """Walk the dataset in the order it was built, instead of resampling it.

        THE ONE OVERRIDE A CURRICULUM NEEDS, and the reason it cannot be avoided: `Trainer`'s
        default `_get_train_sampler` returns a `RandomSampler`, so a dataset handed to it in a
        chosen order is trained in a different one -- and nothing in the loss curve, the
        manifest or the checkpoint would say so. It is a MIXIN rather than a branch inside
        `_WeightedTrainer` because the unweighted path (`use_sample_weight=False`) uses the
        stock `Trainer` and would otherwise silently drop the ordering.

        `dataset` is accepted because transformers 5.x passes one and 4.x does not, the same
        break `training_argument_kwargs` is written around.
        """

        def _get_train_sampler(self, dataset=None, *args, **kwargs):
            return SequentialSampler(self.train_dataset if dataset is None else dataset)

    class _WeightedTrainer(Trainer):  # pragma: no cover - requires torch
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            w = inputs.pop("sample_weight")
            out = forward_for_loss(model, inputs)
            loss = weighted_ce(
                out.logits,
                inputs["labels"],
                w,
                per_device_batch=cfg.per_device_batch,
                grad_accum=cfg.grad_accum,
                num_items_in_batch=num_items_in_batch,
            )
            return (loss, out) if return_outputs else loss

    # SPELLED FOR THE INSTALLED LIBRARY. transformers 5.0 removed `warmup_ratio` and
    # `group_by_length`, and `[train]` allows both sides of that break; see
    # `training_argument_kwargs`, which is pure so the mapping is tested without transformers.
    # `group_by_length` is still passed as False wherever it exists: length-ordered batches make
    # the epoch's example order depend on evidence size, which correlates with task difficulty --
    # a cheaper epoch bought with a curriculum nobody chose. Same reason `pack()` is first-fit.
    args = TrainingArguments(
        **training_argument_kwargs(cfg, {f.name for f in dataclasses.fields(TrainingArguments)})
    )
    cls = _WeightedTrainer if cfg.use_sample_weight else Trainer
    if cfg.curriculum != SHUFFLED:
        # The mixin FIRST, so its `_get_train_sampler` wins over `Trainer`'s. Composed here
        # rather than declared as two more classes: the ordering and the weighting are
        # independent knobs, and the four combinations would otherwise be four class bodies.
        cls = type(f"_Ordered{cls.__name__.lstrip('_')}", (_Ordered, cls), {})
    return cls(model=model, args=args, train_dataset=_Rows())
