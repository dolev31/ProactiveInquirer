"""Which weights answered: a deployment model id -> the adapter that served it.

WHY THIS EXISTS. `ModelPin.adapter_sha` has been a field since pins were introduced and has
been populated by nothing. Under I.14 one vLLM process serves several LoRA adapters, each under
its own OpenAI-visible model id, so the model id names a DEPLOYMENT and not a set of weights: a
re-trained adapter pushed under the same name is invisible to the manifest, produces the same
`run_id`, and is skipped by `--resume` as work already done.

WHY IT IS OPT-IN, AND WHY THAT IS NOT TIMIDITY. `adapter_sha` sits inside `ModelPin.key` ->
`model_pin_hash` -> `semantic_hash` -> `run_id`. Filling it in for a model id that has already
run would rename every finished run that used it -- retroactively, in a repo whose tables cite
run ids. So the registry answers `None` for anything it does not list, every existing id is
unlisted, and `tests/test_checkpoint_registry.py` pins that as a guarantee rather than an
accident.

STDLIB ONLY, ON PURPOSE. `pinq_adapters` may not import `pinq_train` (contract 4) and may not
import `pi_eval` (contract 1), and reaching `pi_run` for its `repo_root` would drag
`pi_run -> pi_eval` into this package's transitive graph. The four-line walk to `pyproject.toml`
below is the same rule `pi_run.manifest.repo_root` and `pinq.tripwire` already use.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "registry",
    "adapter_sha_of",
    "train_id_set_hash_of",
    "reset_cache",
    "REQUIRED_FIELDS",
    "INIT_FROM",
    "NO_ANCESTOR",
    "DEFAULT_PATH",
]

# What a row must carry to be worth having. A PARTIAL row is worse than no row: it makes a
# manifest look provenanced while the field that would let anyone re-derive the weights is
# missing, so a row is refused rather than half-read.
REQUIRED_FIELDS: tuple[str, ...] = (
    "adapter_sha",
    "training_manifest_sha",
    "base_model",
    "rung",
    "dataset_sha",
    "train_id_set_hash",
)

# `dataset_sha` and `train_id_set_hash` are ONE quantity under two names AT RUNG 1: the export
# manifest's `train_id_set_hash`, the set-hash of the `suite_id/task_id` pairs the adapter was
# fitted on.
#
# WHY BOTH NAMES EXIST RATHER THAN ONE. `dataset_sha` is the older name and is what the file's
# `_schema`, `register_checkpoint.py` and six shipped rows already say. `train_id_set_hash` is
# the name the quantity has EVERYWHERE ELSE it is written down -- the export manifest, the
# sidecar, `RunManifest.train_id_set_hash`, the `runs.parquet` column and the contamination
# check that compares them -- and a stamp is only readable if the thing stamping it and the
# thing reading it use one word. Renaming `dataset_sha` would rewrite rows that are cited;
# adding the second name costs one field and the invariant below.
#
# AND WHY THEY ARE CHECKED. Two fields that must agree eventually will not, and a silent
# disagreement here stamps a run with an id set the checkpoint was NOT fitted on -- which is a
# contamination check answering confidently about the wrong dataset.
_ONE_QUANTITY_TWO_NAMES = ("dataset_sha", "train_id_set_hash")

# ---------------------------------------------------------------------------------- LINEAGE
#
# AT RUNG >= 2 THEY ARE TWO QUANTITIES, and reading them as one is how a contamination check
# came to answer about a strict subset of the training set. A DPO adapter is INITIALISED from an
# SFT adapter -- `pinq_train.rung2_dpo.train.DPOConfig.validate` refuses an empty `adapter`
# under either reference mode, because DPO on the raw base optimises a preference the base has
# no policy to express -- so its weights were fitted on the SFT export's tasks AND the pairs
# export's. Its row names only the pairs export.
#
# MEASURED 2026-09-15 on the shipped headline export: the SFT id set holds 2,525 ids, the pairs
# set 1,027, SFT - pairs is 1,499 and the union is 2,526. So `qwen3-8b-dpo-headline-control`
# stamped its runs with a hash resolving to 1,027 of the 2,526 tasks its weights had seen. The
# split predicate in `pi_eval.report.train_split_violations` still covered those rows; the
# id-set canary beside it did not. Those 1,499 are what a leak through a served NAME could
# reach with nothing to catch it -- which is the whole reason layer 4 exists, since layers 1-3
# stop leakage through code and only this one catches leakage through a string. (The pooled
# export is the same shape: 2,664 SFT ids, 1,933 pairs, 732 outside the pairs set.)
#
# THE CANARY ALSO COVERS WHAT THE SPLIT PREDICATE STRUCTURALLY CANNOT SEE -- a task exported
# into a training file and since re-bucketed, where `split_of` answers `test` today and is
# RIGHT while the weights still saw it. That argument is structural, not a count: MEASURED
# 2026-09-15, 0 of the 2,525 SFT ids and 0 of the 1,027 pairs ids are test or dev today. An
# earlier draft of this comment said 94 were, from calling `split_of(suite, task_id)` without
# the template id; musique's bucket hashes `template_id or task_id`, so for its 453 ids that
# call is simply the wrong one and every one of the 148 it mislabels is musique. The reader
# itself was never affected: it keys on the parquet's `template_id`.
#
# SO: `dataset_sha` keeps its meaning (the export THIS adapter was fitted on) and
# `train_id_set_hash` becomes the UNION over the lineage, with `init_from` naming the row it
# was unioned with. Because an init row's own `train_id_set_hash` is ALREADY a union when that
# row is itself a lineage row, a stacked arm -- a second DPO pass from a DPO adapter, which is
# rung 2 initialised from rung 2 -- recurses for free and no row ever needs more than one union. `scripts/hpc/register_checkpoint.py --init-from` writes the union's preimage as a
# `train_ids.<hash16>.txt` sidecar beside the export manifest, which is what makes the stamp
# resolvable by the same filename rule the reader already uses.
INIT_FROM = "init_from"

# THE ONE ROW SHAPE THAT IS RUNG >= 2 AND HAS NO ANCESTOR. `pi train rung2 --reference base`
# (the dpo_from_base ablation) trains a fresh LoRA on the UNTRAINED base: no rung-1 adapter is
# loaded, merged or recorded. Its weights saw exactly one export, so its two id-set names are
# one quantity again -- and the field says `base` rather than being absent, because "there was
# no ancestor" and "nobody filled this in" are different facts and only one of them is checked.
NO_ANCESTOR = "base"

DEFAULT_PATH = "conf/checkpoints.json"

_CACHE: dict[str, Mapping[str, Mapping[str, Any]]] = {}


def _root() -> Path:
    for cand in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (cand / "pyproject.toml").is_file():
            return cand
    return Path.cwd()


def _path() -> Path:
    env = os.environ.get("PI_CHECKPOINTS")
    return Path(env) if env else _root() / DEFAULT_PATH


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """JSON's own rule is that a later key wins, SILENTLY. Here that would mean a checkpoint
    quietly re-pointed at other weights by a careless paste, with the manifest still naming the
    first sha. Two rows for one deployment is never intentional."""
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(
                f"{_path()}: duplicate checkpoint key {k!r}. JSON would silently keep the last "
                "one, so the registry would name weights the manifest did not use."
            )
        out[k] = v
    return out


def reset_cache() -> None:
    """Drop the memo. For tests and for a process that rewrites the file mid-run."""
    _CACHE.clear()


def registry(path: str | os.PathLike[str] | None = None) -> Mapping[str, Mapping[str, Any]]:
    """Deployment model id -> its provenance. `{}` when the file is absent.

    Absent is not an error: a checkout with no rented box and no adapters must run exactly as
    it did before this module existed.
    """
    p = Path(path) if path is not None else _path()
    key = str(p)
    if key in _CACHE:
        return _CACHE[key]
    if not p.is_file():
        _CACHE[key] = {}
        return _CACHE[key]

    raw = json.loads(p.read_text(), object_pairs_hook=_no_duplicates)
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: expected an object mapping model id -> provenance")

    out: dict[str, Mapping[str, Any]] = {}
    for name, entry in raw.items():
        if name.startswith("_"):
            continue  # documentation, not a deployment
        if name == NO_ANCESTOR:
            raise ValueError(
                f"{p}: {NO_ANCESTOR!r} is not available as a deployment name. It is the literal "
                f"{INIT_FROM} takes for a rung >= 2 row that was trained on the untrained base, "
                "so a row wearing it would make that field name a checkpoint and mean 'no "
                "ancestor' at the same time, with nothing downstream able to tell which."
            )
        if not isinstance(entry, dict):
            raise ValueError(f"{p}: entry {name!r} is not an object")
        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        if missing:
            hint = (
                " `train_id_set_hash` is the same value as `dataset_sha`; "
                "`scripts/hpc/register_checkpoint.py --backfill` copies it across."
                if "train_id_set_hash" in missing
                else ""
            )
            raise ValueError(
                f"{p}: checkpoint {name!r} is missing {missing}. A partial entry names a "
                "deployment without naming the weights, which is provenance in appearance "
                f"only.{hint}"
            )
        try:
            rung = int(entry["rung"])
        except (TypeError, ValueError):
            raise ValueError(
                f"{p}: checkpoint {name!r} has rung {entry['rung']!r}, which is not a number. "
                "The rung decides whether this row's two id-set names are one quantity or two."
            ) from None
        a, b = (str(entry[f]) for f in _ONE_QUANTITY_TWO_NAMES)
        init = entry.get(INIT_FROM)
        if rung <= 1:
            if init is not None:
                raise ValueError(
                    f"{p}: checkpoint {name!r} is rung {rung} and carries {INIT_FROM}={init!r}. "
                    "Rung 1 starts from the raw base, so there is no ancestor whose id set "
                    "could be unioned in -- and the field is what licenses this row's two "
                    "id-set names to differ, which at rung 1 they may not."
                )
            if a != b:
                raise ValueError(
                    f"{p}: checkpoint {name!r} has dataset_sha {a} and train_id_set_hash {b}. "
                    "They are one quantity under two names -- the export manifest's "
                    "train_id_set_hash -- so a disagreement means one of them names an id set "
                    "this adapter was not fitted on, and a run stamped from it would be "
                    "checked against the wrong tasks."
                )
        elif str(init) == NO_ANCESTOR:
            if a != b:
                raise ValueError(
                    f"{p}: checkpoint {name!r} has {INIT_FROM}={NO_ANCESTOR!r} but dataset_sha "
                    f"{a} and train_id_set_hash {b}. {NO_ANCESTOR!r} means the run started "
                    "from the untrained base, so there is no ancestor id set to union in and "
                    "the two names are one quantity -- a difference means one of them names an "
                    "id set these weights were not fitted on."
                )
        elif not init:
            raise ValueError(
                f"{p}: checkpoint {name!r} is rung {rung} and names no {INIT_FROM}. A rung-2 "
                "adapter is initialised from a rung-1 one, so it was fitted on that export's "
                "ids as well as its own; a row that names only its own export stamps its runs "
                "with a strict subset (MEASURED on the headline export: 1,027 of 2,526) and "
                "the id-set check answers about the wrong tasks. Register it with "
                "`scripts/hpc/register_checkpoint.py --init-from <model id>`, which writes the "
                "union and its sidecar."
            )
        out[name] = dict(entry)

    # THE CROSS-REFERENCES, in a second pass because `init_from` names another ROW and the
    # first pass has not seen them all yet. All three checks are about the union being
    # re-derivable by a reader holding this file.
    #
    # THE RUNG DOES NOT GUARANTEE TERMINATION, AND MUST NOT BE ASKED TO. This demanded a
    # STRICTLY LOWER parent rung, on the reasoning that a strictly decreasing chain must end.
    # That reasoning is sound and the rule is wrong: DPO-ON-DPO IS STILL RUNG 2 -- a second DPO
    # pass from the `qwen3-8b-dpo-headline-control` adapter is a planned arm -- so the rule made
    # a legitimate checkpoint unregisterable. All the rung has to say is that lineage does not
    # run BACKWARDS (a DPO cannot have started from a GRPO); termination is established
    # directly, by walking the chain to a base case with a visited set.
    for name, entry in out.items():
        init = entry.get(INIT_FROM)
        if init is None or str(init) == NO_ANCESTOR:
            continue
        parent = out.get(str(init))
        if parent is None:
            raise ValueError(
                f"{p}: checkpoint {name!r} has {INIT_FROM}={init!r}, which is not a row in this "
                "registry. That name is how a reader re-derives the union in "
                "train_id_set_hash; a dangling one makes the union unverifiable while the row "
                "still looks provenanced."
            )
        if int(parent["rung"]) > int(entry["rung"]):
            raise ValueError(
                f"{p}: checkpoint {name!r} is rung {entry['rung']} and inits from {init!r}, "
                f"which is rung {parent['rung']}. Lineage does not run backwards: a rung-"
                f"{entry['rung']} pass cannot have started from a rung-{parent['rung']} one."
            )
        # THE BASE CASES ARE A ROW WITH NO `init_from` (rung 1, fitted on one file) AND THE
        # LITERAL `base`. Anything else that ends the walk is a cycle, including a row naming
        # itself -- which a non-strict rung comparison waves straight through.
        seen = {name}
        cur = str(init)
        while cur != NO_ANCESTOR:
            if cur in seen:
                raise ValueError(
                    f"{p}: the {INIT_FROM} chain from checkpoint {name!r} is a cycle -- it "
                    f"reaches {cur!r} twice. A cycle has no base case, so the union it claims "
                    "to describe is not a finite set and no reader can re-derive the stamp."
                )
            seen.add(cur)
            nxt = out.get(cur)
            if nxt is None:
                raise ValueError(
                    f"{p}: the {INIT_FROM} chain from checkpoint {name!r} reaches {cur!r}, "
                    "which is not a row in this registry."
                )
            nxt_init = nxt.get(INIT_FROM)
            if nxt_init is None:
                break
            cur = str(nxt_init)
    _CACHE[key] = out
    return out


def adapter_sha_of(model_id: str) -> str | None:
    """The adapter sha for a REGISTERED deployment, else None.

    None is the load-bearing half. See the module docstring: a value here for an id that has
    already run rewrites that run's identity.
    """
    entry = registry().get(str(model_id))
    return None if entry is None else str(entry["adapter_sha"])


def train_id_set_hash_of(model_id: str) -> str | None:
    """The id set a REGISTERED deployment was fitted on, else None.

    NOT IN RUN IDENTITY, unlike `adapter_sha_of` above, and the difference is the whole reason
    this is a separate function. `adapter_sha` is inside `ModelPin.key` -> `run_id`, so it may
    only ever be filled in for a name that has not yet run. This is stamped on
    `RunManifest.train_id_set_hash`, which `semantic_hash` does not cover -- it is a property
    of weights the pin already names, so it adds nothing to identity and renames nothing. That
    is what lets it be read for a deployment that is already serving.

    THE ROW'S `train_id_set_hash` UNCHANGED, which at rung >= 2 is the UNION over the lineage
    rather than the row's own export -- see LINEAGE above. Nothing here walks `init_from`: the
    union is computed once at registration and its preimage written as a sidecar, so the stamp
    stays one hash that resolves to one file by name, with no index in between. Walking the
    chain here would be a second place for the mapping to be wrong.

    None for an unregistered id, which is every id in use today; the contamination check reads
    a None as "no id set declared" and falls back to the split predicate alone.
    """
    entry = registry().get(str(model_id))
    return None if entry is None else str(entry["train_id_set_hash"])
