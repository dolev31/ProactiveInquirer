"""Offline dev metrics: what a checkpoint learned, before any rollout is bought.

THE DIVISION OF LABOUR (I.8/I.9). These three functions are Tier A. They are free and
deterministic and they answer one question: did the checkpoint learn the LABEL. They cannot
answer whether the label was worth learning -- a model can reach 90% pair accuracy by learning
"prefer the longer question" if length leaks -- which is what the online gate (`pinq_train.gate`)
and the length guard are for. (Half of that one exception is now answerable here and for free:
`acc_by_len_sign` below says whether the ORDERING read length, though not whether the questions
the policy goes on to emit get longer, which is still the gate's.) Running Tier A first is what
keeps a failed checkpoint from costing a rollout budget before the paper knows it failed.

WHY THE MODEL IS A CALLABLE. Every function here takes

    render:  (state_text, action_json) -> Rendered(prompt_ids, completion_ids)
    logprob: (prompt_ids, completion_ids) -> float      # SUM of completion token log-probs

and nothing else. The tokenizer and the weights enter only through those two, so the decision
rules are testable against a scripted model whose right answer is known by construction, and
`torch` stays out of this module's import graph. `torch_logprob_fn` builds the real pair.

WHY `stop_confusion` NORMALISES BY LENGTH AND `pair_accuracy` DOES NOT.
`STOP_ACTION_JSON` is ~7 tokens and an ASK is tens. Argmaxing the SUM of completion log-probs
over those two strings therefore prefers STOP on nearly every state, for a reason that has
nothing to do with stopping; the 2x2 would read "the policy stops well" on a policy that had
learned nothing. So the STOP-vs-ASK decision is made on the PER-TOKEN mean.

`pair_accuracy` is the opposite case and is deliberately NOT normalised: DPO's implicit reward
is a difference of SEQUENCE log-probs, so the ordering the loss actually induces is the ordering
of sums. Length-normalising the accuracy would report a quantity the objective does not
optimise -- a number that could move while the trained policy's behaviour did not. The length
confound in the pairs is held by `PreferencePair.len_delta` and the export-side length guard,
which is where a confound belongs: in the data, not hidden inside the metric that reads it.

AND THE METRIC STILL HAS TO SAY WHETHER THE POLICY READ IT. The guard bounds the length
asymmetry in the pairs; it cannot say whether the checkpoint ordered them on what is left of it.
A policy that has learned "prefer the longer question" scores 1.0 on the pairs whose chosen side
is longer and 0.0 on the pairs whose chosen side is shorter, and the two halves cancel to a
pooled 0.5 -- which is also exactly what "learned nothing" reads. So `pair_accuracy` reports
`acc_by_len_sign`, the same accuracy SPLIT by the sign of the chosen-minus-rejected question
length, and `len_sign_gap`, its difference. That is a breakdown of the existing number, not a
normalisation of it: `acc` is untouched and still the quantity DPO's loss orders.

This module must not import `pi_eval` (contract 3) and does not: every input is a plain row
dict, as written by the dev exporter.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pinq.actions import STOP_ACTION_JSON, action_kind_of

# THE EXPORTER'S OWN PARSE, not a second one. `question_len_delta` is `abs(len(_action_text(a)) -
# len(_action_text(b)))`; the length sign below is the same subtraction with the `abs` removed, so
# it has to measure the same text -- the parsed, whitespace-collapsed, casefolded question. A
# private import rather than a re-implementation for the reason `question_len_delta` gives for
# being one function: two implementations of one measure drift, and the drift is silent.
from pinq_train.export.dataset import _action_text

__all__ = [
    "Rendered",
    "RenderFn",
    "LogprobFn",
    "sft_nll",
    "stop_confusion",
    "pair_accuracy",
    "torch_logprob_fn",
    "AdapterIdentityMismatch",
    "PROVENANCE_SCHEMA",
    "WEIGHTS_FILE",
    "sha256_file",
    "weights_sha256",
    "dataset_provenance",
    "checkpoint_provenance",
    "provenance",
]


@dataclass(frozen=True, slots=True)
class Rendered:
    """One example, split at the point where supervision starts."""

    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]


RenderFn = Callable[[str, str], Rendered]
LogprobFn = Callable[[Sequence[int], Sequence[int]], float]


def _score(render: RenderFn, logprob: LogprobFn, state: str, action: str) -> tuple[float, int]:
    r = render(state, action)
    n = len(r.completion_ids)
    if n == 0:
        raise ValueError(
            f"action {action!r} rendered to zero completion tokens: a per-token mean over it "
            "is undefined, and a zero-length completion is a rendering bug, not a datum."
        )
    return float(logprob(r.prompt_ids, r.completion_ids)), n


# --------------------------------------------------------------------------- masked NLL


def sft_nll(
    rows: Sequence[Mapping[str, Any]], *, render: RenderFn, logprob: LogprobFn
) -> dict[str, float]:
    """Masked negative log-likelihood on the dev SFT rows.

    BOTH NORMALISATIONS, because they fail differently. `nll_per_token` is the overfitting
    readout (train falls while dev rises). `nll_per_example` is not length-normalised, so a
    checkpoint whose actions got LONGER moves it without getting worse per token -- and length
    creep is a named failure mode of this training run, not a nuisance.
    """
    total_nll = 0.0
    total_tok = 0
    per_example = 0.0
    n = 0
    for row in rows:
        lp, n_tok = _score(render, logprob, str(row["state_text"]), str(row["action_json"]))
        total_nll += -lp
        total_tok += n_tok
        per_example += -lp
        n += 1
    return {
        "nll_per_token": (total_nll / total_tok) if total_tok else float("nan"),
        "nll_per_example": (per_example / n) if n else float("nan"),
        "n": float(n),
    }


# --------------------------------------------------------------------------- the STOP 2x2


def stop_confusion(
    rows: Sequence[Mapping[str, Any]],
    *,
    render: RenderFn,
    logprob: LogprobFn,
    reference_ask: str | None = None,
    condition_fields: Sequence[str] = ("done_before",),
) -> dict[str, Any]:
    """P(STOP | done) and P(ASK | not done), by length-normalised argmax over two strings.

    The two candidates at a state are `STOP_ACTION_JSON` and an ASK. The ASK is the row's own
    `action_json` when the row IS an ask; a STOP row has no ask of its own, so it uses
    `reference_ask` when one is supplied and is otherwise SKIPPED AND COUNTED. Skipping
    silently would shrink a denominator with nothing in the output saying by how much.

    A condition field that is None means unlabelled, never False: such a row is skipped and
    counted too, because reading None as not-done would quietly move every legacy row into the
    second cell.

    BOTH CELLS ARE RETURNED AND NEITHER SUMMARISES THE OTHER. A checkpoint can raise
    P(STOP|done) from 0.6 to 0.9 by stopping more everywhere, which also raises "stopped while
    needs remained"; a real improvement moves the first without moving the second.

    `condition_fields` NAMES THE GOLD FACT EACH TABLE IS KEYED ON, and this used to be the
    hardcoded string `"done_before"`. That made every such number this programme has produced a
    statement about POOLED required-evidence completeness, whatever the arm was called. Lane
    L6.1's question is about the ANSWER-BEARING node, and re-exporting the dev rows under an
    answer-node STOP label does NOT move what this table conditions on -- it moves which rows are
    STOP targets. The field has to be selectable or the two questions cannot be told apart.

    SCORED ONCE, BUCKETED PER FIELD. The expensive half is two forward passes per row; a
    condition is only a bucketing of rows already scored, so a second table costs nothing and
    reporting both is the default rather than a choice. `n_skipped_no_reference` and
    `n_skipped_unparseable` are properties of the ROW, not of the condition, so they are counted
    identically in every table -- counting them once would leave the denominators incomparable.

    THE FIRST FIELD ALSO KEEPS THE FLAT KEYS, with exactly their old meaning. Sixty Tier-A
    verdicts on the cluster carry `p_stop_given_done` at the top level and
    `scripts/figures_programme.py` reads them there; a verdict written before and after this
    change has to mean the same thing.
    """
    fields = tuple(condition_fields)
    if not fields:
        raise ValueError(
            "condition_fields is empty: a stop 2x2 needs at least one gold fact to be keyed on. "
            "An empty result would read as 'no rows were labelled', which is a claim about the "
            "data rather than about the call."
        )

    cells = {
        f: {"n_done": 0, "n_not_done": 0, "stop_when_done": 0, "ask_when_not_done": 0, "skip": 0}
        for f in fields
    }
    skipped_no_reference = skipped_unparseable = 0

    for row in rows:
        labels = {f: row.get(f) for f in fields}
        if all(v is None for v in labels.values()):
            # Unlabelled under EVERY condition: nothing to score it into, and scoring it would
            # cost two forward passes to reach no cell.
            for f in fields:
                cells[f]["skip"] += 1
            continue
        action = str(row["action_json"])
        kind = action_kind_of(action)
        if kind == "ask":
            ask_json = action
        elif kind == "stop":
            if reference_ask is None:
                skipped_no_reference += 1
                continue
            ask_json = reference_ask
        else:
            # Neither action. The parser calls this `unparseable` and the policy would have
            # been charged a malformed; it is not a STOP row and must not be scored as one.
            skipped_unparseable += 1
            continue

        state = str(row["state_text"])
        stop_lp, stop_n = _score(render, logprob, state, STOP_ACTION_JSON)
        ask_lp, ask_n = _score(render, logprob, state, ask_json)
        predicts_stop = (stop_lp / stop_n) > (ask_lp / ask_n)

        for f, done in labels.items():
            c = cells[f]
            if done is None:
                c["skip"] += 1
            elif bool(done):
                c["n_done"] += 1
                c["stop_when_done"] += int(predicts_stop)
            else:
                c["n_not_done"] += 1
                c["ask_when_not_done"] += int(not predicts_stop)

    def table(f: str) -> dict[str, float]:
        c = cells[f]
        nd, nnd = c["n_done"], c["n_not_done"]
        return {
            "p_stop_given_done": (c["stop_when_done"] / nd) if nd else float("nan"),
            "p_ask_given_not_done": (c["ask_when_not_done"] / nnd) if nnd else float("nan"),
            "n_done": float(nd),
            "n_not_done": float(nnd),
            "n_skipped_no_reference": float(skipped_no_reference),
            "n_skipped_no_label": float(c["skip"]),
            "n_skipped_unparseable": float(skipped_unparseable),
        }

    by_condition = {f: table(f) for f in fields}
    out: dict[str, Any] = dict(by_condition[fields[0]])
    out["condition_field"] = fields[0]
    out["by_condition"] = by_condition
    return out


# --------------------------------------------------------------------------- pair accuracy

LEN_SIGNS = ("chosen_longer", "chosen_shorter", "equal")


def _len_sign(chosen_json: str, rejected_json: str) -> str:
    """Which side asked the longer QUESTION, by the export-side length guard's own measure.

    Recomputed from the payloads and never read off `PreferencePair.len_delta`: that field is an
    `abs()`, so it carries a magnitude and no sign at all. Reading its magnitude would file every
    pair under `chosen_longer` and the diagnostic would be a constant that always passes.
    """
    delta = len(_action_text(chosen_json)) - len(_action_text(rejected_json))
    return "chosen_longer" if delta > 0 else ("chosen_shorter" if delta < 0 else "equal")


def pair_accuracy(
    pairs: Sequence[Mapping[str, Any]], *, render: RenderFn, logprob: LogprobFn
) -> dict[str, Any]:
    """Share of dev pairs the model orders the way the label does, on SEQUENCE log-prob.

    Not length-normalised: see the module docstring. 0.5 is "learned nothing" and 0.0 is
    "learned the opposite", and I.9's Tier A reading needs the two to be distinguishable.

    Broken down by `pair_kind` and by `decided_by` because an ask_stop pair carries a ~60
    character length asymmetry that no guard can remove -- it is what the two actions ARE -- so
    a pooled number cannot say which population moved.

    AND BY THE SIGN OF THE QUESTION-LENGTH DELTA (`acc_by_len_sign`, `len_sign_gap`), which is
    the length-exploitation diagnostic every rung-2 checkpoint is checked on. See the module
    docstring for why the pooled `acc` cannot answer it. The kill rule -- chosen-shorter accuracy
    below 0.50 while chosen-longer is above 0.60 -- is applied by the reviewer (docs/GPU_RUNBOOK
    section 5), not here: this function reports, and a metric that also decided would have to be
    re-run to be re-read under a revised threshold.
    """
    n = 0
    n_correct = 0
    margins: list[float] = []
    by_kind: dict[str, list[int]] = {}
    by_decided: dict[str, list[int]] = {}
    by_len_sign: dict[str, list[int]] = {k: [] for k in LEN_SIGNS}

    for p in pairs:
        state = str(p["state_text"])
        chosen_json = str(p["chosen_json"])
        rejected_json = str(p["rejected_json"])
        chosen_lp, _ = _score(render, logprob, state, chosen_json)
        rejected_lp, _ = _score(render, logprob, state, rejected_json)
        ok = int(chosen_lp > rejected_lp)
        n += 1
        n_correct += ok
        margins.append(chosen_lp - rejected_lp)
        by_kind.setdefault(str(p.get("pair_kind") or ""), []).append(ok)
        by_decided.setdefault(str(p.get("decided_by") or ""), []).append(ok)
        # ASK-VS-ASK PAIRS ONLY. `STOP_ACTION_JSON` is 18 bytes and every ASK is longer, so an
        # ask_stop pair falls on one side of the sign BY CONSTRUCTION -- chosen_longer whenever
        # the ASK is the chosen side, chosen_shorter whenever it is not. Pooling them in would
        # fill the cells with a fact about what the two actions ARE and bury the fact about what
        # the policy learned -- and they would dominate, not merely dilute: 970 of the 4,962 rows
        # in `data/rl/dev/pairs.dev.jsonl` are ask_ask (docs/GPU_RUNBOOK.md section 1, measured
        # 2026-09-14), so four pairs in five carry a STOP on one side.
        #
        # Derived from the payloads rather than read off `pair_kind`, for `pair_kind_of`'s
        # reason: that field is a string on a jsonl file that can be filtered, concatenated or
        # hand-edited between the export and this command, and a mislabelled row would put a
        # STOP back into the very split that exists to detect length. `action_kind_of` is the
        # policy parser's own normalisation, so this and the exporter cannot disagree about a row.
        if "stop" not in (action_kind_of(chosen_json), action_kind_of(rejected_json)):
            by_len_sign[_len_sign(chosen_json, rejected_json)].append(ok)

    # `acc: None` on an empty cell, not 0.0 and not NaN: `_finite`'s convention, because "no
    # chosen-shorter pairs were scored" and "every chosen-shorter pair was ordered wrong" are
    # opposite readings of the kill rule and must not print the same.
    acc_by_len_sign = {
        k: {"acc": (sum(v) / len(v)) if v else None, "n": len(v)} for k, v in by_len_sign.items()
    }
    longer = acc_by_len_sign["chosen_longer"]["acc"]
    shorter = acc_by_len_sign["chosen_shorter"]["acc"]

    return {
        "acc": (n_correct / n) if n else float("nan"),
        "acc_by_kind": {k: sum(v) / len(v) for k, v in sorted(by_kind.items())},
        "acc_by_decided_by": {k: sum(v) / len(v) for k, v in sorted(by_decided.items())},
        "acc_by_len_sign": acc_by_len_sign,
        "len_sign_gap": (longer - shorter)
        if (longer is not None and shorter is not None)
        else None,
        "mean_margin": (sum(margins) / len(margins)) if margins else float("nan"),
        "n": n,
    }


# --------------------------------------------------------------------------- the real model


def torch_logprob_fn(
    model: Any, tok: Any, *, chat_template: bool = True, enable_thinking: bool = False
) -> tuple[RenderFn, LogprobFn]:
    """The (render, logprob) pair for a real checkpoint. `torch` is imported lazily, here.

    RENDERING MUST MATCH RUNG 1 EXACTLY. A dev NLL is comparable to a training NLL only if the
    two were computed over the same token sequence, so this calls the same builder the trainer
    does. `build_chat_masked_example` is track A's (plan I.13.1) and may not exist yet; when it
    does not, a chat-template request RAISES rather than falling back to the raw concatenation.
    The fallback would be silent and wrong: a checkpoint fitted under the chat template scored
    without its header yields an NLL that is off by the header's contribution, with nothing
    downstream able to notice.
    """
    from pinq_train.rung1_sft import mask as mask_mod

    if chat_template:
        builder = getattr(mask_mod, "build_chat_masked_example", None)
        if builder is None:
            raise NotImplementedError(
                "chat_template=True needs pinq_train.rung1_sft.mask.build_chat_masked_example, "
                "which plan I.13.1 (track A) adds and this checkout does not have. Refusing "
                "rather than falling back to the raw concatenation: the raw path omits the chat "
                "header the checkpoint was trained under, so every NLL here would be wrong by "
                "its contribution and no assertion downstream could see it. Pass "
                "chat_template=False only for a checkpoint trained WITHOUT the template."
            )

        def _render(state_text: str, action_json: str) -> Rendered:
            ex = builder(tok, state_text, action_json, enable_thinking=enable_thinking)
            return Rendered(
                prompt_ids=tuple(ex.input_ids[: ex.n_prompt]),
                completion_ids=tuple(ex.input_ids[ex.n_prompt :]),
            )
    else:

        def _render(state_text: str, action_json: str) -> Rendered:
            ex = mask_mod.build_masked_example(tok, state_text, action_json)
            return Rendered(
                prompt_ids=tuple(ex.input_ids[: ex.n_prompt]),
                completion_ids=tuple(ex.input_ids[ex.n_prompt :]),
            )

    def _logprob(prompt_ids: Sequence[int], completion_ids: Sequence[int]) -> float:
        import torch

        ids = list(prompt_ids) + list(completion_ids)
        with torch.no_grad():
            batch = torch.tensor([ids], device=getattr(model, "device", "cpu"))
            logits = model(batch).logits.float()
            # position i predicts token i+1, so the completion's first token is predicted at
            # index len(prompt_ids) - 1. An off-by-one here is the classic silent masked-SFT
            # bug and it shows up as a plausible-looking NLL, never as an error.
            logp = torch.log_softmax(logits[0, len(prompt_ids) - 1 : -1, :], dim=-1)
            want = torch.tensor(list(completion_ids), device=logp.device)
            return float(logp.gather(-1, want.unsqueeze(-1)).sum().item())

    return _render, _logprob


def _finite(x: float) -> float | None:
    """NaN is not JSON. `null` says 'not measured'; 0.0 would say 'measured, and zero'."""
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else x


# --------------------------------------------------------------------------- provenance
#
# WHY THIS BLOCK EXISTS AT ALL. Until 2026-09-18 this module computed three numbers and wrote
# them beside a checkpoint path and a tokenizer name, and nothing else. CONTRIBUTING.md rule 1 is that
# a number without provenance is not a result: every reported value traces to a run identity, a
# `scorer_hash` and a `graph_version`. Seven \NOTREADY markers in the paper said, in seven
# different wordings, that the offline half of the stopping comparison could not be tabulated
# because the instrument wrote none of the three. This is that fix. It adds a `provenance` block
# and changes no metric: `sft_nll`, `stop_confusion` and `pair_accuracy` are untouched, and
# `tests/test_eval_offline_provenance.py` pins their outputs on a fixture against the values the
# pre-change code produced, so a future edit here that moved a metric would fail rather than
# silently republish.
#
# WHAT A TIER-A RUN'S "RUN IDENTITY" IS. There is no `run_id`: the unit here is a (checkpoint,
# dataset file) pair and not a rollout, so inventing one would be a name for nothing. What takes
# its place is the pair of identities a reader would have to re-derive to reproduce the number --
# the adapter's own weights digest and the dataset file's sha256 -- plus the code version that
# read them. `scorer_hash` and `graph_version` are NOT properties of this evaluation: they are
# properties of the rows it scores, carried on every row by the exporter, and they are reported
# as the SET of distinct values present rather than as one value, because a file that mixes two
# scorers is a fact a reader needs and an average of two hashes is not a hash.

PROVENANCE_SCHEMA = "tierA-provenance-1"

# The file `conf/checkpoints.json` rows hash, and the ONLY file they hash. See
# `checkpoint_provenance` for the two-identity trap this names its way out of.
WEIGHTS_FILE = "adapter_model.safetensors"

# The row fields that are provenance rather than data. `scorer_hash` and `graph_version` are
# CONTRIBUTING.md rule 1's second and third; `code_version` is the exporter's own, which is a different
# commit from the one running this command and must not be confused with it; `matcher_id` and
# `pins_sha` are carried because the live gate verdicts this table is read beside quote them.
_ROW_PROVENANCE_FIELDS = (
    "scorer_hash",
    "graph_version",
    "code_version",
    "matcher_id",
    "pins_sha",
    "split",
)


class AdapterIdentityMismatch(RuntimeError):
    """The registry row named does not hash to the weights actually being scored.

    A REFUSAL AND NOT A WARNING. The whole point of writing a registry name into the output is
    that a later reader may look the row up and read `base_model`, `rung`, `init_from` and
    `dataset_sha` off it. If the name is wrong every one of those is wrong too, and all of them
    are plausible, so nothing downstream could catch it. Raising is the only outcome that cannot
    be mistaken for a measurement.
    """


def sha256_file(path: str | os.PathLike[str]) -> str:
    """The same bytes `sha256sum` reads, so a remote `sha256sum` and this agree.

    Deliberately the same definition, and the same docstring sentence, as
    `scripts/hpc/register_checkpoint.py:sha256_file`, which is the registry's own writer. That
    script cannot be imported (it is a script and not a package), so the test asserts the two
    agree on a fixture rather than trusting that they do.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_sha256(checkpoint: str | os.PathLike[str]) -> str | None:
    """`sha256(adapter_model.safetensors)` at the root of an adapter directory, or None.

    TWO ADAPTER IDENTITIES EXIST AND THEY DISAGREE BY DESIGN. This one -- the weights file alone
    -- is what `conf/checkpoints.json` rows carry and therefore what enters `ModelPin.key ->
    model_pin_hash -> semantic_hash -> run_id`. `pinq_train.merge.adapter_sha` hashes the config
    AND the weights and is the merge identity; `artifacts/<run>/adapter_sha.txt` holds THAT one.
    Recording the merge identity under the name the registry uses would name weights that never
    answered a request, so this reads the one file the registry reads and nothing else.

    None, not a raise, when the path is a full model rather than an adapter: a base-model
    reading (the step-0 point of a training curve) is a legitimate Tier-A row and has no adapter
    to hash. The field then says "absent" by being null, which is not the same as "zero".
    """
    w = Path(checkpoint) / WEIGHTS_FILE
    return sha256_file(w) if w.is_file() else None


def _distinct(rows: Sequence[Mapping[str, Any]], field: str) -> list[str]:
    """Every distinct value of `field` across the rows, sorted, as strings.

    A LIST AND NOT A VALUE. `data/rl/dev/sft.dev.jsonl` happens to carry one `scorer_hash`
    today; nothing enforces that, and a concatenated or re-exported file could carry two. A
    single value would have to pick one and would pick it silently. The list makes a mixed file
    visible in the output of every checkpoint scored on it, which is where a reader is looking.
    """
    seen = {str(r[field]) for r in rows if r.get(field) is not None}
    return sorted(seen)


def dataset_provenance(
    path: str | os.PathLike[str], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """What file was read, how many rows it held, and what the rows say they were scored under.

    THE SHA IS OF THE FILE'S BYTES, not of the rows. A reader reproducing this number needs to
    know they hold the same file, and `sha256sum` on either machine answers that in one command
    -- which is exactly how the 2026-09-17 cluster evaluator logged the two dev files, so the
    values here are comparable to that log rather than to nothing.

    THE MANIFEST IS READ FOR ITS OWN IDENTITY AND IS NOT WHERE THE SCORER COMES FROM.
    `sft.dev.manifest.json` and `pairs.dev.manifest.json` carry `split` and `train_id_set_hash`
    and carry NO `scorer_hash` and NO `graph_version` (measured 2026-09-18). The rows do carry
    both, stamped through the export from the scored runs the rows were built from, which is
    where `pi_eval` puts them in the first place. So both fields are derived from the rows and
    labelled with where they came from; a reader who wants the manifest's word for it finds the
    manifest's sha256 here and can go and look.
    """
    p = Path(path)
    # `<stem>.jsonl` -> `<stem>.manifest.json`, which is `${DATASET%.jsonl}.manifest.json`, the
    # expression `scripts/hpc/rung1.sh` uses for the same pairing. One trailing suffix is
    # replaced, so `sft.dev.jsonl` gives `sft.dev.manifest.json` and not `sft.manifest.json`.
    manifest = p.with_suffix(".manifest.json")
    if not manifest.is_file():
        # `sft.dev.jsonl` -> `sft.dev.manifest.json` is the export's own naming, but a sampled
        # file (`sft.dev.sample.jsonl`) has no manifest of its own. Absent is recorded as absent.
        manifest_block: dict[str, Any] = {"path": str(manifest), "present": False}
    else:
        m = json.loads(manifest.read_text())
        manifest_block = {
            "path": str(manifest),
            "present": True,
            "sha256": sha256_file(manifest),
            "split": m.get("split"),
            "n_examples": m.get("n_examples"),
            "n_pairs": m.get("n_pairs"),
            "train_id_set_hash": m.get("train_id_set_hash"),
            "rank_rule": m.get("rank_rule"),
            "stop_action_json": m.get("stop_action_json"),
            # Stated, not implied: the two fields rule 1 asks for are the ones this file does
            # not have, and a reader must not conclude from their absence here that they were
            # not recorded anywhere.
            "carries_scorer_hash": "scorer_hash" in m,
            "carries_graph_version": "graph_version" in m,
        }
    out: dict[str, Any] = {
        "path": str(p),
        "sha256": sha256_file(p),
        "n_rows": len(rows),
        "manifest": manifest_block,
        "row_field_source": (
            "the values below are read off the rows, which carry them from the scored runs the "
            "exporter built them from; the manifest carries neither scorer_hash nor graph_version"
        ),
    }
    for field in _ROW_PROVENANCE_FIELDS:
        out[field] = _distinct(rows, field)
    return out


def checkpoint_provenance(
    checkpoint: str | os.PathLike[str],
    *,
    registry_path: str | os.PathLike[str] | None = None,
    registry_name: str | None = None,
) -> dict[str, Any]:
    """The weights' digest, the registry row that claims them, and the base they adapt.

    REFUSES on a name that does not hash to these weights (`AdapterIdentityMismatch`). It also
    reports every registry name whose `adapter_sha` equals the measured digest, so a run whose
    caller passed no name still records whether the weights are registered at all -- which is
    the question "does this checkpoint have an identity a run_id could carry" and is worth
    answering even when the answer is no.
    """
    ckpt = Path(checkpoint)
    measured = weights_sha256(ckpt)

    cfg = ckpt / "adapter_config.json"
    adapter_cfg = json.loads(cfg.read_text()) if cfg.is_file() else {}

    reg_path = Path(registry_path) if registry_path else None
    registry: dict[str, Any] = {}
    if reg_path is not None and reg_path.is_file():
        registry = {
            k: v
            for k, v in json.loads(reg_path.read_text()).items()
            if not k.startswith("_") and isinstance(v, dict)
        }

    matches = sorted(n for n, e in registry.items() if e.get("adapter_sha") == measured)
    row = registry.get(registry_name or "") if registry_name else None

    if registry_name is not None:
        if row is None:
            raise AdapterIdentityMismatch(
                f"--registry-name {registry_name!r} is not a row in "
                f"{reg_path}. Recording an unregistered name would put a base model, a rung and "
                "an init_from lineage in the output that no row backs."
            )
        if measured is None:
            raise AdapterIdentityMismatch(
                f"--registry-name {registry_name!r} was passed but {ckpt} holds no "
                f"{WEIGHTS_FILE}, so the row's adapter_sha cannot be checked against anything. "
                "A registry name on unverifiable weights is a claim, not a measurement."
            )
        if row.get("adapter_sha") != measured:
            raise AdapterIdentityMismatch(
                f"--registry-name {registry_name!r} carries adapter_sha "
                f"{row.get('adapter_sha')} but {ckpt / WEIGHTS_FILE} hashes to {measured}. "
                "Every other field this output would copy off that row -- base_model, rung, "
                "init_from, dataset_sha -- would then be wrong and would look right."
                + (f" These weights are registered as {matches}." if matches else "")
            )

    base_from_cfg = adapter_cfg.get("base_model_name_or_path")
    if row is not None:
        base_model, base_source = row.get("base_model"), "registry"
    elif base_from_cfg:
        base_model, base_source = base_from_cfg, "adapter_config.json"
    else:
        base_model, base_source = None, "neither a registry row nor an adapter_config.json"

    return {
        "path": str(ckpt),
        "weights_file": WEIGHTS_FILE if measured is not None else None,
        "adapter_sha": measured,
        "adapter_sha_definition": (
            f"sha256({WEIGHTS_FILE}); the identity conf/checkpoints.json rows carry and "
            "run_id is built from, NOT pinq_train.merge.adapter_sha (config + weights, written "
            "to adapter_sha.txt), which is a different digest of the same directory"
        ),
        "registry_path": str(reg_path) if reg_path else None,
        "registry_name": registry_name,
        "registry_adapter_sha": (row or {}).get("adapter_sha"),
        "adapter_sha_verified": (
            None if registry_name is None else (row or {}).get("adapter_sha") == measured
        ),
        "registry_names_matching_weights": matches,
        "base_model": base_model,
        "base_model_source": base_source,
        "rung": (row or {}).get("rung"),
        "init_from": (row or {}).get("init_from"),
        "training_manifest_sha": (row or {}).get("training_manifest_sha"),
        "dataset_sha": (row or {}).get("dataset_sha"),
        "train_id_set_hash": (row or {}).get("train_id_set_hash"),
    }


def _git_code_version(root: str | os.PathLike[str]) -> dict[str, Any]:
    """The commit that computed the number, and whether its tree was dirty.

    A SEPARATE, SMALLER IMPLEMENTATION THAN `pi_run.manifest.git_info` ON PURPOSE. Contract 4
    forbids anything importing `pinq_train`, and `pinq_train` importing `pi_run` would reach
    `pi_eval` in two hops and break contract 3. So this module cannot call that one, and the
    only honest response is to keep this to the two facts it needs and to name the difference
    here rather than let a reader assume one function serves both.

    An absent or broken git is `"unknown"` and dirty, matching `git_info`'s own convention: an
    unnameable code version is exactly the state the `dev-` run prefix exists for.
    """
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout
        return {"sha": sha, "dirty": bool(porcelain.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"sha": "unknown", "dirty": True}


def provenance(
    *,
    checkpoint: str | os.PathLike[str],
    dev_sft: str | os.PathLike[str],
    sft_rows: Sequence[Mapping[str, Any]],
    dev_pairs: str | os.PathLike[str],
    pair_rows: Sequence[Mapping[str, Any]],
    argv: Sequence[str],
    repo_root: str | os.PathLike[str],
    registry_path: str | os.PathLike[str] | None = None,
    registry_name: str | None = None,
    reference_ask: str | None = None,
    tokenizer: str | None = None,
) -> dict[str, Any]:
    """The whole block, assembled. Raises `AdapterIdentityMismatch` before any number is kept.

    `argv` IS RECORDED VERBATIM AND IS NOT RECONSTRUCTED. `--reference-ask` is the flag whose
    omission reads as a model finding rather than as an error -- without it every STOP row is
    skipped, `n_done` is 0 and `P(STOP|done)` is NaN, which prints as a missing cell and not as
    a missing flag. Three of the Tier-A files on the cluster were written that way. The exact
    command line in the output is what makes that case diagnosable from the file alone.

    The STOP bytes are recorded for the same reason: `stop_confusion` argmaxes against
    `STOP_ACTION_JSON`, so a change to those bytes would move every P(STOP|done) ever computed,
    and a file that does not carry them cannot be compared with one written after the change.
    """
    return {
        "schema": PROVENANCE_SCHEMA,
        "cli": list(argv),
        "code_version": _git_code_version(repo_root),
        "stop_action_json": STOP_ACTION_JSON,
        "reference_ask": reference_ask,
        "tokenizer": tokenizer,
        "checkpoint": checkpoint_provenance(
            checkpoint, registry_path=registry_path, registry_name=registry_name
        ),
        "dev_sft": dataset_provenance(dev_sft, sft_rows),
        "dev_pairs": dataset_provenance(dev_pairs, pair_rows),
    }
