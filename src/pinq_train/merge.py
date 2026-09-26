"""Fold rung 1's LoRA into the base weights: a fused checkpoint, and rung 2's older reference.

THIS IS NO LONGER RUNG 2'S DEFAULT, AND THE REASON IS A MEASUREMENT
===================================================================
`DPOConfig.reference` defaults to `"adapter"`, which loads rung 1's LoRA TWICE onto one base --
a trainable copy and a frozen one -- and merges nothing. Merging is exact only in a dtype whose
ulp is below the update, and rung 1's is not: MEASURED on the Mac smoke run (Qwen3-0.6B, LoRA
r=32, six optimiser steps), mean |LoRA delta| was 2.819e-05 against a typical |W| of 2.363e-02
whose bf16 ulp is 9.229e-05, and **44.7% of `sum|delta|` did not survive the save** (6862.69 of
12413.99; 307,036,664 weights had a non-zero fp32 delta rounded exactly to zero). `MERGE_DTYPE`
fixes that by storing fp32 -- and a merged 8B base is then ~32 GB instead of ~16 GB, which rung
2 has to load. The two-adapter recipe costs one base in memory, loses nothing, and gives
"which adapter is the reference" a NAME rather than a library default.

WHAT THIS MODULE IS STILL FOR
=============================
  * SERVING. A fused checkpoint is one directory of weights with no adapter at inference time:
    no LoRA merge per request, no `--lora-modules` wiring, no adapter sha to pin separately.
  * A rung 2 whose LoRA shape must DIFFER from rung 1's. Under `reference="adapter"` the policy
    IS rung 1's adapter, so r and alpha are rung 1's; a run that needs another shape has to
    start from merged weights and fit a fresh LoRA (`pi train rung2 --reference merged`).
  * Any downstream step that wants rung 1 as plain weights.

WHY IT WAS WRITTEN IN THE FIRST PLACE
======================================
Rung 2's claim is "DPO moved the policy away from the SFT checkpoint". That sentence has
content only if the reference distribution `pi_ref` in

    log pi(y_w|x) - log pi_ref(y_w|x)  -  [ log pi(y_l|x) - log pi_ref(y_l|x) ]

is the rung-1 policy. TRL computes `pi_ref` by DISABLING the adapter it is training. The old
trainer both loaded rung 1's LoRA onto the base AND handed `DPOTrainer` a fresh `peft_config`,
so "the adapter" named two different things and which one got disabled was a property of the
installed trl/peft versions rather than of anything this repository wrote down. Either reading
trains, reports a falling loss and a rising reward margin, and produces a checkpoint; only one
of them makes the paper's sentence true, and no artifact recorded which happened.

Merging removes the ambiguity instead of documenting it. After `merge_adapter`, rung 1 is IN
the weights, the only adapter in the process is the one DPO trains, and "adapter disabled"
has exactly one meaning. The two-adapter default removes the same ambiguity the other way --
two adapters, two NAMES, and the reference named explicitly -- at no cost in precision; see
`rung2_dpo.train.POLICY_ADAPTER` and `REFERENCE_ADAPTER`, whose values are dictated by peft's
save layout and by which name the installed TRL selects.

WHAT THE RETURNED SHA IS, AND WHAT IT DELIBERATELY IS NOT
=========================================================
It is a SHA-256 over the weight files (`*.safetensors`) in sorted relative-path order, with
each path hashed alongside its bytes so that renaming a shard is a different merge. It is
`DPOConfig.merged_base_sha`: the run's checkable claim about which weights its reference was.

It deliberately does NOT cover `config.json` or the tokenizer files. `save_pretrained` stamps
`transformers_version` into the config, so hashing it would give two different shas for one
set of weights merged on two boxes -- a provenance field that changes when nothing about the
model did is a field people learn to ignore. The weights are what the log-probs are a function
of; the rest is recorded in the manifest beside them.

NOTHING HERE HAS RUN ON A GPU. As with the rungs, `merge_adapter`'s plumbing is tested with
doubles (`Loaders`) and the numerical property -- that a reloaded merged model gives the same
log-probs as the in-memory one -- is an integration test gated on `PI_SMOKE_MODEL`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

# The weight files the output sha is computed over. See the module docstring for why config
# and tokenizer files are recorded but not hashed.
WEIGHT_SUFFIXES = (".safetensors",)
# WHAT AN ADAPTER'S IDENTITY IS: the tensors, plus the config that says where they are applied.
# `adapter_config.json` carries r, alpha and the target modules, and the same weights under a
# different config are a different adapter.
#
# AND WHAT IT IS NOT, BY NAME RATHER THAN BY SUFFIX. This was `(".safetensors", ".bin", ".json")`
# over the whole tree, which swept in everything the trainer leaves beside the adapter:
# `checkpoint-*/` (save_strategy="steps" writes one every `cfg.save_steps`, each with its own
# adapter_model.safetensors and optimizer.pt), `training_args.bin` (a pickle of the output_dir
# and the seed), `rung1.manifest.json`, the model card. MEASURED on the smoke run: DELETING an
# intermediate checkpoint changed `adapter_sha` while the adapter did not change a byte. That
# sha sits inside `ModelPin.key` -> `model_pin_hash` -> `semantic_hash` -> `run_id`, so a field
# that moves when nothing moved renames finished runs in a repository whose tables cite run ids.
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
ADAPTER_FILES = (*ADAPTER_WEIGHTS, "adapter_config.json")
MANIFEST = "merge.manifest.json"
_CHUNK = 1 << 20  # merged shards are gigabytes; never read one into memory whole

# THE DTYPE THE MERGED BASE IS STORED IN, AND WHY IT IS NOT THE CHECKPOINT'S OWN.
#
# `Qwen/Qwen3-0.6B` (and every Qwen3) ships `torch_dtype: "bfloat16"`, and transformers 5.x
# makes `from_pretrained` default to the CHECKPOINT's dtype where 4.x upcast to float32. Under
# that default this module loaded bf16, folded in bf16, and saved bf16 -- and a bf16 weight has
# eight mantissa bits, so an update smaller than one ulp of the weight it is added to lands on
# the same representable number and is gone.
#
# MEASURED on the Mac smoke run (Qwen3-0.6B, LoRA r=32, 6 optimiser steps): mean |LoRA delta|
# 2.819e-05 against a typical |W| of 2.363e-02 whose bf16 ulp is 9.229e-05 -- the update is a
# third of one representable step. 44.7% of sum|delta| did not survive the save (6862.69 of
# 12413.99); 307,036,664 weights had a non-zero fp32 delta that became exactly zero; and
# `test_reference_logprobs_equal_a_freshly_loaded_merged_model` failed at 2.74e-02 against its
# 1e-04 tolerance -- the first time that test had ever been run.
#
# IT IS STORAGE, NOT ARITHMETIC. Folding in fp32 and storing bf16 keeps the same 55.3%, to the
# digit, so a higher-precision matmul fixes nothing. Only the stored dtype can.
#
# The cost is real: a merged 8B base is ~32 GB at fp32 rather than ~16 GB, and rung 2 then
# loads that. A run whose update is large relative to a bf16 ulp may legitimately choose
# bfloat16 -- `merge_adapter(..., dtype=...)` is how, and the manifest records which. What is
# refused is choosing it by accident, which is what a library default amounts to.
MERGE_DTYPE = "float32"


class NothingMerged(RuntimeError):
    """A merge that wrote no weight files.

    Refused rather than hashed: the SHA-256 of an empty file set is a perfectly good hex
    string, and it would land in a manifest as provenance for a merge that never happened.
    """


class MergeMismatch(ValueError):
    """`merged_base_sha` disagrees with the merge manifest in the directory it names."""


@dataclass(frozen=True, slots=True)
class Loaders:
    """The three transformers/peft entry points a merge needs, as data.

    Injected rather than imported so the plumbing -- what is loaded from where, in what order,
    and what ends up on disk -- is testable on a laptop with no torch. The default is built
    lazily inside `_hf_loaders`, which keeps the heavy imports inside the `[train]` extra.
    """

    model: Callable[[str], Any]
    tokenizer: Callable[[str], Any]
    adapter: Callable[[Any, str], Any]


def _hf_loaders(dtype: str = MERGE_DTYPE) -> Loaders:
    """`from_pretrained` bound to an EXPLICIT dtype. See `MERGE_DTYPE` for why that matters.

    `dtype=` is the transformers 5.x spelling; 4.x calls it `torch_dtype` and does not accept
    the new name, and `[train]` allows `transformers>=4.42`. The fallback is a retry rather
    than a version check because `from_pretrained` takes `**kwargs` and no signature
    inspection can tell the two apart.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _model(name: str) -> Any:
        try:
            return AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)

    return Loaders(
        model=_model,
        tokenizer=AutoTokenizer.from_pretrained,
        adapter=PeftModel.from_pretrained,
    )


def _file_sha(p: Path) -> bytes:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            d.update(chunk)
    return d.digest()


def _sha_over(root: Path, files: Sequence[Path]) -> str:
    """SHA-256 over `files`, each path hashed alongside its bytes, in the order given."""
    dig = hashlib.sha256()
    for p in files:
        dig.update(p.relative_to(root).as_posix().encode())
        dig.update(b"\x1f")
        dig.update(_file_sha(p))
    return dig.hexdigest()


def dir_sha(d: str | Path, suffixes: tuple[str, ...] = WEIGHT_SUFFIXES) -> str:
    """SHA-256 over the named files of a MERGED model directory, sorted by relative path.

    Recursive, because a merged checkpoint is sharded and the shards may sit in subdirectories.
    The path is hashed with the bytes, so renaming or re-sharding a checkpoint is a different
    sha even when the tensors are identical. `merge.manifest.json` is never part of it -- it
    records the sha, so including it would be a hash of its own value.

    NOT FOR ADAPTER DIRECTORIES. See `adapter_sha`: a rung-1 output holds the trainer's
    bookkeeping and one subdirectory per epoch, and none of it is the adapter.
    """
    root = Path(d)
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes and p.name != MANIFEST
    )
    if not files:
        raise NothingMerged(
            f"{root} holds no {' or no '.join(suffixes)} file, so there is nothing whose "
            "identity could be recorded. A merge that saved nothing must not return a sha: "
            "that sha would name a checkpoint that does not exist."
        )
    return _sha_over(root, files)


def adapter_sha(d: str | Path) -> str:
    """SHA-256 over an adapter's OWN files at the ROOT of `d`, and nothing else.

    `ADAPTER_FILES` by NAME, not by suffix, and not recursively -- see that constant for the
    measurement. Everything a trainer leaves beside the adapter (`checkpoint-*/`,
    `training_args.bin`, manifests, the model card) is excluded, so deleting an intermediate
    checkpoint to reclaim disk does not rename every run that pinned the adapter.

    A directory with no adapter weights is REFUSED rather than hashed: the sha of an empty file
    set is a perfectly good hex string, and it would name an adapter that does not exist. A
    `checkpoint-*/adapter_model.safetensors` does not rescue it -- an intermediate checkpoint is
    not the adapter, and pointing peft at this directory would fail.
    """
    root = Path(d)
    files = [root / name for name in ADAPTER_FILES]
    if not any((root / w).is_file() for w in ADAPTER_WEIGHTS):
        raise NothingMerged(
            f"{root} holds no {' or no '.join(ADAPTER_WEIGHTS)} at its root, so there is no "
            "adapter here whose identity could be recorded. An intermediate checkpoint under "
            "checkpoint-*/ is not the adapter: peft cannot load this directory either."
        )
    return _sha_over(root, [p for p in files if p.is_file()])


def merge_adapter(
    base_model: str,
    adapter_dir: str,
    out_dir: str,
    *,
    dtype: str = MERGE_DTYPE,
    _loaders: Loaders | None = None,
) -> str:
    """Fold `adapter_dir` into `base_model`, save to `out_dir`, return the output sha.

    The tokenizer is saved from the BASE: a LoRA does not change it, and a merged directory
    without one cannot be served or tokenised, which would be discovered on the rented box.

    `dtype` is what the merged weights are STORED in, and it decides whether the fold preserves
    rung 1 or rounds it away -- see `MERGE_DTYPE`, which carries the measurement. It is written
    into the manifest, because two merges at two dtypes are two reference policies.
    """
    ld = _loaders or _hf_loaders(dtype)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = ld.model(base_model)
    ld.adapter(base, str(adapter_dir)).merge_and_unload().save_pretrained(str(out))
    ld.tokenizer(base_model).save_pretrained(str(out))

    sha = dir_sha(out)
    (out / MANIFEST).write_text(
        json.dumps(
            {
                "base_model": base_model,
                "adapter": str(adapter_dir),
                "adapter_sha": adapter_sha(adapter_dir),
                # WHAT THE WEIGHTS ARE STORED IN. Not cosmetic: at bfloat16 a LoRA update
                # smaller than one ulp of the weight it is added to is gone, and the merged
                # base is then a different distribution from the checkpoint it claims to be.
                # See MERGE_DTYPE for the measurement.
                "dtype": dtype,
                "output_sha": sha,
                "out_dir": str(out),
                "weight_suffixes": list(WEIGHT_SUFFIXES),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return sha


def verify_merged_base(base_model: str, merged_base_sha: str) -> dict[str, Any]:
    """Reconcile a run's `merged_base_sha` against the manifest in the directory it names.

    Cheap on purpose: it compares against what `merge_adapter` recorded rather than re-hashing
    tens of gigabytes on every run. `dir_sha(base_model)` is the expensive check, and it is
    what an auditor with the directory runs.

    A base with no manifest -- a hub id, or a directory merged by some other tool -- cannot be
    checked here. That is RECORDED as unverified rather than raised: refusing would make a
    legitimate configuration unusable, and claiming it was verified would be a lie in a
    provenance field.
    """
    man = Path(base_model) / MANIFEST
    if not man.is_file():
        return {"verified": False, "output_sha": "", "why": "no merge.manifest.json"}
    recorded = str(json.loads(man.read_text()).get("output_sha") or "")
    if recorded != merged_base_sha:
        raise MergeMismatch(
            f"merged_base_sha={merged_base_sha!r} but {man} records output_sha={recorded!r}. "
            "The config names weights that are not the ones in this directory, so the run "
            "cannot say what its reference policy was. Re-run `pi train merge`, or point "
            "--merged-base at the directory the sha came from."
        )
    return {"verified": True, "output_sha": recorded, "why": ""}
