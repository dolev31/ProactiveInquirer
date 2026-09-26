#!/usr/bin/env python3
"""Register a served checkpoint: one row in the registry, one $0 row in the price table.

    register_checkpoint.py --name qwen3-8b-sft-headline \\
        --adapter artifacts/rung1-8b-headline \\
        --base Qwen/Qwen3-8B --rung 1 \\
        --run-dir artifacts/rung1-8b-headline \\
        --dataset-manifest data/rl/headline/sft.manifest.json

    # the adapter is 350 MB on the cluster and this runs on the Mac:
    #   ssh box sha256sum ~/ProactiveInquirer/artifacts/rung1-8b-headline/adapter_model.safetensors
    register_checkpoint.py --name ... --adapter-sha <hex> --run-dir <pulled dir> ...

    # rung >= 2 was INITIALISED from another checkpoint, so it saw that one's ids too:
    register_checkpoint.py --name qwen3-8b-dpo-headline-control --rung 2 \\
        --init-from qwen3-8b-sft-headline \\
        --dataset-manifest data/rl/headline/pairs.manifest.json ...

WHY A SCRIPT. `conf/checkpoints.json`'s first row was typed by hand, and so was the $0 row in
`scripts/price_tables/2026-09.json` that must accompany it (`PriceTable.rates` is strict: a
served name with no price row raises `LLMConfigError` before a sweep starts). Two files and
five shas, one of them the sha256 of a 350 MB file on another machine. Seventeen more
checkpoints are coming; by hand, a wrong sha is a schedule rather than a risk.

WHAT IT REFUSES. `adapter_sha` is inside `ModelPin.key` -> `model_pin_hash` ->
`semantic_hash` -> `run_id` (`src/pinq_adapters/llm/checkpoints.py`). Re-pointing an existing
name at different weights renames every finished run that used it, retroactively, in a repo
whose tables cite run ids -- so a changed sha under an existing name stops the script.
`--force` exists for the case where nothing has run yet, and it is loud.

THE SAME REFUSAL, ARRIVING THE OTHER WAY. The check above compares against the REGISTRY FILE's
own prior row, so it is silent the first time a name is registered -- `old` is `None` whether
nothing has ever run under that name, or whether a great deal has and every run recorded
`adapter_sha: None` because the name was unregistered at the time. `granite33-8b-sft-headline-
s0/s1/s2` were registered exactly that way (commit `60bd644`, 2026-09-18), months into their own
run history, and `tests/test_checkpoint_registry.py::test_no_registry_row_RENAMES_A_RUN_THAT_
HAS_ALREADY_FINISHED` measures the result at 5,611 finished runs whose recomputed `run_id` no
longer matches what is on disk -- 3,600 of them `split=test` and already the population of a
committed artifact (`artifacts/granite_family_20260918/run_ids.test_split_isolated_store.txt`).
So this script also checks `calls.parquet` and `runs/<run_id>/manifest.json` for a FIRST-time
name, not only a repointed one: see `_history_offenders`. `--force` is the same escape, for the
same reason -- a human said these specific runs may be renamed.

WHAT IS PROVISIONAL, AND WHY THAT IS A FIELD AND NOT A COMMENT. `training_manifest_sha`
should be the sha of `rung{1,2}.manifest.json`, which the trainer writes when it FINISHES. A
checkpoint served out of a still-running job has none, so the trainer's own
`trainer_state.json` stands in and the entry carries `"provisional": true`. Re-running this
command once the job has ended replaces the sha and drops the flag; the adapter is unchanged,
so that is an update and not the refusal above.

LINEAGE, AND WHY `--init-from` IS REQUIRED AT RUNG >= 2. A rung-1 row's `dataset_sha` and
`train_id_set_hash` are one quantity because the adapter was fitted on exactly one file. A DPO
adapter was not: it is INITIALISED from an SFT adapter (`DPOConfig.validate` refuses an empty
`adapter` under either reference mode), so its weights saw the SFT export's tasks AND the pairs
export's, while its row names only the pairs export. MEASURED 2026-09-15 on the shipped headline
export: SFT 2,525 ids, pairs 1,027, SFT - pairs 1,499, union 2,526 -- a row registered without
this flag stamps 1,027 of the 2,526 tasks its weights saw, and the id-set canary in
`pi_eval.report.train_split_violations` is blind to the other 1,499. That is what a leak
through a served NAME could reach unseen, which is the case layer 4 of the firewall exists for.
So `--init-from` resolves the
named row's id set, unions it with this export's, writes the union's preimage as a
`train_ids.<hash16>.txt` sidecar beside the dataset manifest, and puts the union hash in
`train_id_set_hash` while `dataset_sha` keeps naming THIS export. An init row's own
`train_id_set_hash` is already a union when it is itself rung >= 2, so a stacked arm recurses
for free.

STDLIB ONLY AT IMPORT TIME: it runs on the Mac against files pulled from the cluster, and it
must not need the project's install to be importable. The union is the one operation that does
need it -- it hashes with `pinq_train.split.id_set_hash`, the SAME function the exporter hashed
every sidecar with, because a second implementation of a set hash is a second answer -- so that
import is made lazily, inside the union, and refuses with a message rather than failing at
startup for the commands that do not need it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO / "conf" / "checkpoints.json"
DEFAULT_PRICE_TABLE = REPO / "scripts" / "price_tables" / "2026-09.json"
# Where the FINISHED-RUN-HISTORY refusal reads. Same two paths
# `tests/test_checkpoint_registry.py::_runs_using` reads, for the same reason: `calls.parquet`
# finds the small candidate set of run ids for one model id fast, and `runs/<id>/manifest.json`
# is where each one's actual pin lives. Absent in a stranger's checkout, which is not an error
# (see `_history_offenders`).
DEFAULT_CALLS_PARQUET = REPO / "scores" / "parquet" / "calls.parquet"
DEFAULT_RUNS_DIR = REPO / "runs"

WEIGHTS = "adapter_model.safetensors"
TRAINER_STATE = "trainer_state.json"
REQUIRED_FIELDS = (
    "adapter_sha",
    "training_manifest_sha",
    "base_model",
    "rung",
    "dataset_sha",
    "train_id_set_hash",
)
# Written only on a rung >= 2 row, and ordered immediately after the six above so a lineage row
# reads as one block. `pinq_adapters.llm.checkpoints` REQUIRES it there and REFUSES it at rung 1.
INIT_FROM = "init_from"
# THE ONE ROW SHAPE THAT IS RUNG >= 2 AND HAS NO ANCESTOR. `pi train rung2 --reference base`
# (the dpo_from_base ablation) trains a fresh LoRA on the UNTRAINED base -- no rung-1 adapter
# loaded, merged or recorded -- so its weights saw exactly one export and there is nothing to
# union. `--init-from base` records THAT, rather than leaving the field off: "there was no
# ancestor" and "nobody filled this in" are different facts and only one of them is checked.
NO_ANCESTOR = "base"
# What `rung2.manifest.json` calls the same choice. The cross-check reads the first that is
# there; `reference_policy` is where the trainer states it, `config.reference` is the field it
# was set from, and they are one quantity -- reading either makes the check live on manifests
# written before the former existed.
REFERENCE_KEYS = (("reference_policy", "reference"), ("config", "reference"))

# Where the exporter leaves its id sets when the dataset manifest's own directory does not hold
# them. The same `data/rl` `pi_eval.report._train_ids_roots` looks under, for the same reason.
IDS_SUBDIR = ("data", "rl")

# Locally served weights: nobody bills per token for them. See the price table's own
# `_why_local_deployments` for why $0 here is not the silent zero that table exists to prevent.
ZERO_PRICE = {"cached_input": 0.0, "input": 0.0, "output": 0.0, "reasoning": None}

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class Refused(Exception):
    """An input that would produce a wrong row. Nothing is written."""


def sha256_file(path: Path) -> str:
    """The same bytes `sha256sum` reads, so a remote `sha256sum` and this agree."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def adapter_sha(adapter: Path | None, supplied: str | None) -> str:
    """The sha of the weights, from the file or from a remote `sha256sum`.

    TWO IDENTITIES EXIST AND MUST NOT BE CONFUSED. This one is sha256(adapter_model.safetensors)
    alone: the served-weights identity that 1,395 finished runs already carry in their pins.
    `artifacts/<run>/adapter_sha.txt`, written by scripts/hpc/rung1.sh, is
    `pinq_train.merge.adapter_sha`: a hash over the adapter's config AND weights, the merge
    identity `verify_merged_base` checks. MEASURED 2026-09-15 on rung1-8b-headline: file sha
    a255a40b..., sidecar 06ee9c1d... -- different quantities, not a corrupted file. Never
    register from the sidecar.

    Given BOTH, they are a cross-check and not a preference: a disagreement means the file
    hashed on the cluster is not the file that is here, and one of the two is the wrong
    checkpoint.
    """
    local = None
    if adapter is not None:
        w = adapter / WEIGHTS
        if not w.is_file():
            raise Refused(f"{w} is not a file -- that directory is not an adapter")
        local = sha256_file(w)
    if supplied is not None:
        supplied = supplied.strip().lower()
        if not _HEX64.match(supplied):
            raise Refused(f"--adapter-sha {supplied!r} is not 64 lowercase hex characters")
        if local is not None and local != supplied:
            raise Refused(
                f"--adapter-sha {supplied} does not match {local}, the sha256 of "
                f"{adapter / WEIGHTS}. One of the two is a different checkpoint; nothing is "
                "written until they agree."
            )
        return supplied
    if local is None:
        raise Refused("one of --adapter <dir> or --adapter-sha <hex> is required")
    return local


def training_manifest(run_dir: Path, rung: int) -> tuple[str, str | None]:
    """`(sha, provisional_source)` -- `provisional_source` is None when the run finished.

    The manifest for the run's own rung is preferred, then the other rung's (a directory
    holding both is a mistake worth surfacing elsewhere, not here), then the trainer's state.
    """
    for cand in (f"rung{rung}.manifest.json", "rung1.manifest.json", "rung2.manifest.json"):
        p = run_dir / cand
        if p.is_file():
            return sha256_file(p), None
    state = run_dir / TRAINER_STATE
    if state.is_file():
        return sha256_file(state), TRAINER_STATE
    raise Refused(
        f"{run_dir} holds neither rung{rung}.manifest.json nor {TRAINER_STATE}: there is "
        "nothing to name the run that produced these weights, and a row without that is a "
        "deployment with no provenance."
    )


def dataset_sha(manifest: Path) -> str:
    """`train_id_set_hash` -- the id set the adapter was fitted on, not the sha of the file.

    Two exports of the same ids differ byte-for-byte (timestamps, counters) and are the same
    training set; the id-set hash is what makes two runs comparable.
    """
    raw = json.loads(manifest.read_text())
    got = str(raw.get("train_id_set_hash") or "")
    if not got:
        raise Refused(
            f"{manifest} carries no train_id_set_hash. That field IS dataset_sha; without it "
            "the row cannot say what the adapter was fitted on."
        )
    return got


# --------------------------------------------------------------------------------- lineage


def id_set_hash(ids) -> str:
    """The EXPORTER's set hash, imported rather than written again.

    A second implementation of "the set-hash of these ids" is a second answer, and the whole
    point of the union is that `pi_eval.report` can find its preimage by the filename the hash
    produces. `sys.path` is nudged rather than required so this still works from a bare checkout
    on the Mac (see the module docstring's stdlib note); if the import fails the operation is
    REFUSED, because falling back to a local sha256 would be exactly the second answer.
    """
    try:
        from pinq_train.split import id_set_hash as _h
    except ImportError:
        sys.path.insert(0, str(REPO / "src"))
        try:
            from pinq_train.split import id_set_hash as _h
        except ImportError as e:
            raise Refused(
                "the union needs pinq_train.split.id_set_hash -- the same function the "
                f"exporter hashed every sidecar with -- and it is not importable from {REPO}. "
                "Run this from the repository, or with PYTHONPATH pointing at its src/. It is "
                "not reimplemented here: a second set hash is a second answer, and the sidecar "
                "would then be named after a hash no reader computes."
            ) from e
    return _h(ids)


def train_ids_name(set_hash: str) -> str:
    """The sidecar's filename, derived from the hash it is the preimage of.

    Spelled here rather than imported from `pinq_train.export.dataset`, which
    `pi_eval.report._train_id_set` also does and for the same stated reason: the mapping from
    hash to file must be the FILENAME RULE and nothing else, with no index in between. Three
    copies of one f-string is the cost of there being no fourth place to look it up.
    """
    return f"train_ids.{set_hash[:16]}.txt"


def ids_roots(dataset_manifest: Path, ids_dir: Path | None) -> list[Path]:
    """Where to look for a sidecar, best first.

    AN EXPLICIT `--ids-dir` IS EXCLUSIVE, not first-in-line -- the same rule, and the same
    wording, as `pi_eval.report._train_ids_roots`: an override that silently fell through to
    the repo would let a caller who pointed at the wrong directory get an answer from a
    different one.
    """
    if ids_dir is not None:
        return [Path(ids_dir)]
    return list(dict.fromkeys([manifest_dir(dataset_manifest), REPO.joinpath(*IDS_SUBDIR)]))


def manifest_dir(dataset_manifest: Path) -> Path:
    """The directory the manifest was NAMED in -- `.parent.resolve()`, never `.resolve().parent`.

    THE DIFFERENCE IS A SYMLINKED MANIFEST, which is how a worktree gets read-only access to
    one export without copying it. Resolving the FILE dereferences the link and lands the
    sidecar in the directory the link points into -- another checkout -- while resolving the
    DIRECTORY keeps it beside the manifest as the caller spelled it, which is where that
    caller's reader will look.
    """
    return dataset_manifest.parent.resolve()


def read_id_set(set_hash: str, roots: list[Path]) -> tuple[set[str], Path]:
    """The `suite_id/task_id` set a hash names, FOUND BY THE HASH with no index in between.

    AND VERIFIED. The sidecar is a proof of membership only if its contents reproduce the hash
    its name claims; one that does not is a proof of something else, under a name a reader
    trusts. `export.dataset.write_jsonl` refuses to WRITE such a file, and this refuses to read
    one, so a hand-edit between the two is caught rather than unioned in.
    """
    name = train_ids_name(set_hash)
    for root in roots:
        if not root.is_dir():
            continue
        for hit in sorted(root.rglob(name)):
            ids = {ln.strip() for ln in hit.read_text().splitlines() if ln.strip()}
            got = id_set_hash(ids)
            if got != set_hash:
                raise Refused(
                    f"{hit} holds {len(ids)} ids that do not reproduce the train_id_set_hash "
                    f"its name claims: it hashes to {got}, not {set_hash}. A sidecar is a "
                    "preimage or it is nothing, and unioning this one would put ids into a "
                    "set the adapter may never have seen."
                )
            return ids, hit
    raise Refused(
        f"{name} is in none of {[str(r) for r in roots]}. That file is the preimage of "
        f"train_id_set_hash {set_hash}, and the union cannot be taken over a set nobody holds. "
        "Point --ids-dir at the directory holding the export's sidecars, or re-run "
        "`pi train export` to write one."
    )


def declared_reference(run_dir: Path, rung: int) -> str | None:
    """What the trainer's own manifest says `pi_ref` was: "adapter", "merged", "base" -- or
    None when there is no manifest to ask.

    NONE IS NOT A FAILURE. A checkpoint served out of a still-running job has no manifest (that
    is what `provisional` records), and refusing it would make `--init-from` unusable exactly
    when a deployment is being registered in a hurry. The check fires when there is something
    to check against.
    """
    path = run_dir / f"rung{rung}.manifest.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
    except ValueError:
        return None
    for outer, inner in REFERENCE_KEYS:
        block = raw.get(outer)
        if isinstance(block, dict) and block.get(inner):
            return str(block[inner])
    return None


def check_reference_agrees(run_dir: Path, rung: int, init_from: str) -> None:
    """`init_from` against the trainer's own record of what it initialised from.

    THE ONE FIELD ON THE ROW A READER CANNOT RE-DERIVE. `adapter_sha` is checkable against the
    weights and `dataset_sha` against the export manifest, but nothing in the served artifact
    says which checkpoint it started from -- so a row claiming `base` for a run that loaded
    rung 1's adapter would hide 1,499 ids behind a field that looks answered.
    """
    got = declared_reference(run_dir, rung)
    if got is None:
        return
    on_base = got == NO_ANCESTOR
    if init_from == NO_ANCESTOR and not on_base:
        raise Refused(
            f"--init-from {NO_ANCESTOR} but rung{rung}.manifest.json in {run_dir} says "
            f"reference={got!r}: this run's pi_ref WAS a rung-1 checkpoint, so the weights saw "
            "that export's ids too and the row would hide them behind a field that looks "
            "answered. Name the row it was initialised from."
        )
    if init_from != NO_ANCESTOR and on_base:
        raise Refused(
            f"--init-from {init_from!r} but rung{rung}.manifest.json in {run_dir} says "
            f"reference={NO_ANCESTOR!r}: this run trained a fresh LoRA on the untrained base "
            "and was initialised from no checkpoint, so unioning that row's id set in would "
            "claim a lineage the training run says it did not have. Use "
            f"--init-from {NO_ANCESTOR}."
        )


def init_id_set_hash(name: str, entry: dict) -> str:
    """The id set an init row names: its `train_id_set_hash`, which is already a union when
    that row is itself rung >= 2.

    THE FALLBACK IS EXACT, NOT A GUESS. A row written before `train_id_set_hash` existed has
    only `dataset_sha`, and at rung 1 those are one quantity by the rule the loader enforces --
    so reading it there is the migration, not a preference. At rung >= 2 they are two, and
    `dataset_sha` is the row's own export alone; falling back to it would silently rebuild the
    very subset this flag exists to stop.
    """
    have = entry.get("train_id_set_hash")
    if have:
        return str(have)
    rung = int(entry.get("rung") or 0)
    ds = entry.get("dataset_sha")
    if rung <= 1 and ds:
        return str(ds)
    raise Refused(
        f"{name!r} is rung {rung} and carries no train_id_set_hash, so the id set it saw is "
        "not stated anywhere. Its own dataset_sha names only its own export, which for a "
        "lineage row is a strict subset. Run --backfill-init-from on it first."
    )


def check_init_row(
    registry: dict, init_from: str, *, name: str, base_model: str, rung: int
) -> dict:
    """The init row, or a refusal. Every check is about the union meaning anything."""
    # SELF-INIT IS A CYCLE, and saying so beats "not a registered row" -- which is what the
    # lookup below would say for a name being registered for the first time, sending the reader
    # off to check a registry that is fine.
    if init_from == name:
        raise Refused(
            f"--init-from {init_from!r} on --name {name!r}: a row cannot be initialised from "
            "itself. That is a cycle with no base case, so there is no finite union to compute."
        )
    entry = registry.get(init_from)
    if not isinstance(entry, dict) or init_from.startswith("_"):
        raise Refused(
            f"--init-from {init_from!r} is not a row in this registry. It must be a REGISTERED "
            "model id: the union is only re-derivable by a reader who can look that row up."
        )
    got = str(entry.get("base_model") or "")
    if got != base_model:
        raise Refused(
            f"--init-from {init_from!r} has base_model {got!r} and this row has {base_model!r}. "
            "A LoRA is applied to a base, so an adapter cannot have been initialised from one "
            "fitted on another; the union would be arithmetic over unrelated training."
        )
    parent_rung = int(entry.get("rung") or 0)
    if parent_rung > rung:
        raise Refused(
            f"--init-from {init_from!r} is rung {parent_rung} and this row is rung {rung}. "
            f"Lineage does not run backwards: a rung-{rung} pass cannot have started from a "
            f"rung-{parent_rung} one."
        )
    walk_chain(registry, name, init_from)
    return entry


def walk_chain(registry: dict, name: str, init_from: str) -> None:
    """Follow `init_from` to a base case, refusing a cycle or a dangling name.

    WHY THE RUNG NO LONGER CARRIES THIS. The rule used to be a STRICTLY LOWER parent rung, on
    the reasoning that a strictly decreasing chain must end. Sound reasoning, wrong rule:
    DPO-ON-DPO IS STILL RUNG 2 -- a second DPO pass from the `qwen3-8b-dpo-headline-control`
    adapter is a planned arm -- so it made a legitimate checkpoint unregisterable. Termination
    is now established directly. The base cases are a row with no `init_from` (rung 1, fitted
    on one file) and the literal `base`; anything else that ends the walk is a cycle, including
    a row naming itself, which a non-strict rung comparison waves straight through.
    """
    seen = {name}
    cur = init_from
    while cur != NO_ANCESTOR:
        if cur in seen:
            raise Refused(
                f"--init-from {init_from!r} would close a cycle: the chain from {name!r} "
                f"reaches {cur!r} twice. A cycle has no base case, so the union it claims to "
                "describe is not a finite set and no reader can re-derive the stamp."
            )
        seen.add(cur)
        nxt = registry.get(cur)
        if not isinstance(nxt, dict):
            raise Refused(
                f"the {INIT_FROM} chain from {name!r} reaches {cur!r}, which is not a row in "
                "this registry, so the union cannot be re-derived from it."
            )
        nxt_init = nxt.get(INIT_FROM)
        if nxt_init is None:
            return
        cur = str(nxt_init)


def union_sidecar(
    *,
    init_from: str,
    init_entry: dict,
    own_hash: str,
    dataset_manifest: Path,
    ids_dir: Path | None,
) -> tuple[str, Path, int, bool]:
    """`(union hash, sidecar path, n ids, was already there)`. Writes the preimage beside the
    dataset manifest, and does NOT rewrite one whose bytes already match.

    BESIDE THE MANIFEST, because that is where every other sidecar is and where
    `pi_eval.report._train_ids_roots` already looks. NAMED BY ITS OWN HASH, so a file of that
    name either IS this union or is not there -- which is also why an existing file with other
    bytes is REFUSED rather than overwritten: under that name it is already a corrupt proof,
    and silently replacing it would hide which of the two was wrong.
    """
    roots = ids_roots(dataset_manifest, ids_dir)
    init_ids, init_path = read_id_set(init_id_set_hash(init_from, init_entry), roots)
    own_ids, own_path = read_id_set(own_hash, roots)
    union = init_ids | own_ids
    uh = id_set_hash(union)
    dest = manifest_dir(dataset_manifest) / train_ids_name(uh)
    body = "".join(f"{i}\n" for i in sorted(union))
    if dest.is_file() and dest.read_text() != body:
        raise Refused(
            f"{dest} already exists and holds different ids. Its name says it is the preimage "
            f"of {uh}, so the file that is there is already a proof of something else; "
            "overwriting it would hide which of the two is wrong. Delete it deliberately, or "
            "find out what wrote it."
        )
    # IDENTICAL BYTES MEAN NO WRITE AT ALL, not a write that happens to be harmless. The
    # refusal above has already excluded every other content, so reaching here with the file
    # present PROVES it holds exactly `body`. Rewriting it would change nothing but the mtime
    # -- and these sidecars live under the shared `data/rl/`, where a bumped mtime is a false
    # freshness signal to every `find -newermt` that looks there.
    unchanged = dest.is_file()
    if not unchanged:
        dest.write_text(body)
    print(
        f"  union: {len(init_ids)} ({init_path.name}) | {len(own_ids)} ({own_path.name}) "
        f"-> {len(union)} -> {dest} ({'unchanged' if unchanged else 'written'})"
    )
    return uh, dest, len(union), unchanged


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise Refused(f"{path} does not exist")
    return json.loads(path.read_text())


def order_fields(entry: dict) -> dict:
    """Stable key order so a re-registration is a no-op diff."""
    entry = dict(entry)
    ordered = {f: entry.pop(f) for f in REQUIRED_FIELDS}
    if INIT_FROM in entry:
        ordered[INIT_FROM] = entry.pop(INIT_FROM)
    ordered.update(entry)
    return ordered


def build_entry(
    *,
    existing: dict | None,
    sha: str,
    manifest_sha: str,
    provisional_source: str | None,
    base_model: str,
    rung: int,
    ds_sha: str,
    id_set: str | None = None,
    init_from: str | None = None,
) -> dict:
    """The row, preserving any extra keys a previous hand-edit put there."""
    entry = dict(existing or {})
    entry["adapter_sha"] = sha
    entry["training_manifest_sha"] = manifest_sha
    entry["base_model"] = base_model
    entry["rung"] = rung
    entry["dataset_sha"] = ds_sha
    # AT RUNG 1, THE SAME VALUE UNDER THE NAME EVERYTHING ELSE USES. `dataset_sha` is what this
    # file has always called it; `train_id_set_hash` is what the export manifest, the id-set
    # sidecar, `RunManifest`, the `runs.parquet` column and
    # `pi_eval.report.train_split_violations` all call it, and a stamp is only readable when the
    # writer and the reader use one word. `checkpoints.registry()` refuses a row whose two names
    # disagree -- unless the row is rung >= 2, where they are two quantities and `id_set` is the
    # UNION over the lineage; see the module docstring for the 1,499 that made that necessary.
    entry["train_id_set_hash"] = ds_sha if id_set is None else id_set
    if init_from is None:
        # A name re-registered at rung 1 after a rung-2 row wore it must not keep the claim.
        entry.pop(INIT_FROM, None)
    else:
        entry[INIT_FROM] = init_from
    if provisional_source is None:
        entry.pop("provisional", None)
        entry.pop("provisional_source", None)
    else:
        entry["provisional"] = True
        entry["provisional_source"] = provisional_source
    return order_fields(entry)


def _history_offenders(
    name: str, sha: str, calls_parquet: Path, runs_dir: Path
) -> list[tuple[str, str, "str | None"]]:
    """Finished runs already on disk that a registration of `name` at `sha` would rename.

    Mirrors `tests/test_checkpoint_registry.py::_runs_using` /
    `_assert_no_finished_run_is_renamed` exactly, because they are the same question asked at
    two different times: that test asks it of the registry that already shipped, this asks it
    of the registry about to be written. `adapter_sha` is inside `ModelPin.key` ->
    `model_pin_hash` -> `semantic_hash` -> `run_id`, so ANY row that disagrees with what a
    finished run already recorded renames that run -- whether `name` is being re-pointed (a
    prior row existed and named a different sha) or registered for the FIRST time (no prior row
    at all, which is silent to the `old = registry.get(name)` check above and is exactly how
    `granite33-8b-sft-headline-s0/s1/s2` reached 5,611 offenders: nothing compared the row about
    to be written against calls.parquet, only against the registry file's own prior state).

    BEST-EFFORT, LIKE THE TEST IT MIRRORS. A checkout with no `calls.parquet` has no rented box
    and no history to disagree with, and must register exactly as it always could; a checkout
    with no `duckdb` installed is the same case by a different absence. Returns `[]` in both.
    """
    if not calls_parquet.is_file():
        return []
    try:
        import duckdb
    except ImportError:
        return []
    con = duckdb.connect()
    rows = con.execute(
        "SELECT DISTINCT run_id FROM read_parquet(?) WHERE model = ?",
        [str(calls_parquet), name],
    ).fetchall()
    offenders: list[tuple[str, str, str | None]] = []
    for (run_id,) in rows:
        m = runs_dir / str(run_id) / "manifest.json"
        if not m.is_file():
            continue  # a pruned run dir is not evidence either way -- same as the guard test
        pins = json.loads(m.read_text()).get("pins") or {}
        for role, pin in pins.items():
            if pin.get("model_id") != name:
                continue
            recorded = pin.get("adapter_sha")
            if recorded != sha:
                offenders.append((str(run_id), str(role), recorded))
    return offenders


def register(
    *,
    name: str,
    adapter: Path | None,
    supplied_sha: str | None,
    base_model: str,
    rung: int,
    run_dir: Path,
    dataset_manifest: Path,
    registry_path: Path,
    price_path: Path,
    force: bool,
    init_from: str | None = None,
    ids_dir: Path | None = None,
    calls_parquet: Path = DEFAULT_CALLS_PARQUET,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> dict:
    """Validate everything, then write both files. Returns the entry."""
    if not _NAME.match(name):
        raise Refused(f"--name {name!r} is not a served model id (it is a vLLM --lora-modules key)")
    if name.startswith("_"):
        raise Refused(f"--name {name!r} would be read as a documentation key, not a deployment")
    if name == NO_ANCESTOR:
        raise Refused(
            f"--name {NO_ANCESTOR!r} is reserved: it is the literal {INIT_FROM} takes for a "
            "rung >= 2 row trained on the untrained base, so a deployment wearing it would "
            "make that field name a checkpoint and mean 'no ancestor' at once."
        )
    if rung not in (1, 2, 3):
        raise Refused(f"--rung {rung} is not 1 (SFT), 2 (DPO) or 3 (GRPO)")
    # THE FLAG AND THE RUNG DECIDE EACH OTHER, checked before anything is hashed so a wrong
    # pairing costs no work and writes no file.
    if rung == 1 and init_from:
        raise Refused(
            f"--init-from {init_from!r} at --rung 1. Rung 1 is fitted on one file and starts "
            "from the raw base, so there is no ancestor to union in -- and this field is what "
            "licenses a row's two id-set names to differ, which at rung 1 they may not."
        )
    if rung >= 2 and not init_from:
        raise Refused(
            f"--rung {rung} requires --init-from <registered model id>, or --init-from "
            f"{NO_ANCESTOR} if this run trained a fresh LoRA on the untrained base "
            f"(`pi train rung2 --reference base`). A rung-{rung} adapter is otherwise "
            "initialised from a lower rung, so it was fitted on that export's ids as well as "
            "its own; a row that names only its own export stamps its runs with a strict "
            "subset (MEASURED on the headline export: 1,027 of 2,526) and the id-set check in "
            "pi_eval.report.train_split_violations then answers about the wrong tasks."
        )

    sha = adapter_sha(adapter, supplied_sha)
    manifest_sha, provisional_source = training_manifest(run_dir, rung)
    ds_sha = dataset_sha(dataset_manifest)

    registry = _load_json(registry_path)
    price = _load_json(price_path)
    models = price.get("models")
    if not isinstance(models, dict):
        raise Refused(f"{price_path} has no 'models' object")

    old = registry.get(name)
    if isinstance(old, dict) and old.get("adapter_sha") not in (None, sha) and not force:
        raise Refused(
            f"{name!r} is already registered with adapter_sha {old['adapter_sha']}, and these "
            f"weights hash to {sha}. adapter_sha is inside run_id, so re-pointing this name "
            "renames every finished run that used it. Serve the new weights under a NEW name, "
            "or pass --force if nothing has run under this one."
        )

    # THE FIRST-TIME CASE THE CHECK ABOVE CANNOT SEE: no row in the registry FILE, but finished
    # runs already used this bare name and recorded a pin that disagrees with `sha` -- every one
    # of them `adapter_sha: None`, if `name` has never been registered before at all. See
    # `_history_offenders` and the module docstring's "THE SAME REFUSAL, ARRIVING THE OTHER WAY".
    if not force:
        offenders = _history_offenders(name, sha, calls_parquet, runs_dir)
        if offenders:
            run_id, role, recorded = offenders[0]
            raise Refused(
                f"{name!r} would be registered at adapter_sha {sha}, but {len(offenders)} "
                f"finished run(s) already used this name and recorded a different pin -- e.g. "
                f"run {run_id} recorded {recorded!r} for role {role}. adapter_sha is inside "
                "run_id, so writing this row renames every one of them. Serve the new weights "
                "under a NEW deployment name, or pass --force if these runs may be renamed."
            )

    priced = models.get(name)
    if priced is not None and any(float(priced.get(k) or 0.0) != 0.0 for k in ("input", "output")):
        raise Refused(
            f"the price table already prices {name!r} at input={priced.get('input')} "
            f"output={priced.get('output')}. A locally served checkpoint is $0; this name "
            "belongs to something that is billed."
        )

    # LAST, so that every refusal above fires before a file is written. The sidecar is content
    # addressed -- its name IS the hash of its contents -- so writing it is idempotent, but a
    # refusal that had already written one would still leave a file nobody asked for.
    id_set = None
    if init_from:
        check_reference_agrees(run_dir, rung, init_from)
    if init_from and init_from != NO_ANCESTOR:
        init_entry = check_init_row(
            registry, init_from, name=name, base_model=base_model, rung=rung
        )
        id_set, _, _, _ = union_sidecar(
            init_from=init_from,
            init_entry=init_entry,
            own_hash=ds_sha,
            dataset_manifest=dataset_manifest,
            ids_dir=ids_dir,
        )

    entry = build_entry(
        existing=old if isinstance(old, dict) else None,
        sha=sha,
        manifest_sha=manifest_sha,
        provisional_source=provisional_source,
        base_model=base_model,
        rung=rung,
        ds_sha=ds_sha,
        id_set=id_set,
        init_from=init_from,
    )

    registry[name] = entry
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    if priced is None:
        models[name] = dict(ZERO_PRICE)
        price_path.write_text(json.dumps(price, indent=1, sort_keys=True) + "\n")
    return entry


def backfill(registry_path: Path) -> list[str]:
    """Copy `dataset_sha` into `train_id_set_hash` on every row that lacks it. Returns the
    names changed; writing nothing and returning `[]` when they all already carry it.

    THE MIGRATION FOR SIX ROWS THAT PREDATE THE FIELD. `checkpoints.registry()` now requires
    `train_id_set_hash`, so without this every row written before it existed makes the loader
    raise -- and the loader runs on every rollout that builds a pin.

    IT REFUSES RATHER THAN CHOOSING. A row where the two names already disagree is not a row
    this can migrate: at rung 1 they are one quantity, so a difference means one of them is
    wrong, and "which one" is not a question a migration may answer by picking. Nothing is
    written.

    AND IT PASSES OVER RUNG >= 2 ENTIRELY, refusing any such row that does not already carry
    `init_from`. There the two names are two quantities and the copy this function performs
    would write the subset stamp `--init-from` exists to prevent; `--backfill-init-from` is the
    per-row migration that computes the union instead.
    """
    reg = _load_json(registry_path)
    changed: list[str] = []
    for name, entry in reg.items():
        if name.startswith("_") or not isinstance(entry, dict):
            continue
        ds = entry.get("dataset_sha")
        if ds is None:
            raise Refused(
                f"{name!r} has no dataset_sha, so there is nothing to backfill from. A row "
                "without it cannot say what the adapter was fitted on under either name."
            )
        # A LINEAGE ROW IS NOT A COPY, AND THIS IS THE ONE PLACE IT WOULD SILENTLY BECOME ONE.
        # At rung >= 2 `dataset_sha` names this row's own export and the id set the weights saw
        # is the UNION with the init row's; copying one onto the other would write exactly the
        # subset stamp this flag exists to stop, under a migration nobody would re-read.
        if int(entry.get("rung") or 0) >= 2:
            if not entry.get(INIT_FROM):
                raise Refused(
                    f"{name!r} is rung {entry.get('rung')} and names no {INIT_FROM}, so its "
                    "train_id_set_hash cannot be copied from dataset_sha: a rung-2 adapter is "
                    "initialised from a rung-1 one and saw that export's ids too (MEASURED on "
                    "the headline export: 1,027 of 2,526). A row registered from now on gets "
                    "this from `--init-from <model id>`; an existing row is migrated with\n"
                    f"  register_checkpoint.py --backfill-init-from <init model id> --name "
                    f"{name} --dataset-manifest <its export manifest>\n"
                    "first; --backfill then passes over it."
                )
            if not entry.get("train_id_set_hash"):
                raise Refused(
                    f"{name!r} names {INIT_FROM}={entry[INIT_FROM]!r} but carries no "
                    "train_id_set_hash. The union has to be computed from the two id sets, "
                    "which is --backfill-init-from; a copy here would name the wrong set."
                )
            continue
        have = entry.get("train_id_set_hash")
        if have is None:
            entry["train_id_set_hash"] = ds
            changed.append(name)
        elif str(have) != str(ds):
            raise Refused(
                f"{name!r} has dataset_sha {ds} and train_id_set_hash {have}. They are one "
                "quantity under two names -- the export manifest's train_id_set_hash -- so a "
                "disagreement means one of them names an id set this adapter was not fitted "
                "on. Fix the row by hand; a migration must not pick a winner."
            )
    if changed:
        registry_path.write_text(json.dumps(reg, indent=2) + "\n")
    return changed


def backfill_init_from(
    *,
    registry_path: Path,
    name: str,
    init_from: str,
    dataset_manifest: Path,
    ids_dir: Path | None,
) -> dict:
    """The per-row migration for a rung >= 2 row registered before `--init-from` existed.

    THE SAME UNION BY THE SAME FUNCTION; only the row already exists, so nothing here touches
    `adapter_sha` (which is inside `run_id`) or `training_manifest_sha`. `train_id_set_hash` is
    outside `RunManifest.semantic_hash` by construction, so changing it renames no finished run
    -- which is the only reason a row that has already served may be edited at all.
    """
    reg = _load_json(registry_path)
    entry = reg.get(name)
    if not isinstance(entry, dict) or name.startswith("_"):
        raise Refused(f"--name {name!r} is not a row in {registry_path}")
    rung = int(entry.get("rung") or 0)
    if rung < 2:
        raise Refused(
            f"{name!r} is rung {rung}. Only a rung >= 2 row has an ancestor to union in; at "
            "rung 1 the two id-set names are one quantity and --backfill is the migration."
        )
    ds = str(entry.get("dataset_sha") or "")
    if not ds:
        raise Refused(f"{name!r} has no dataset_sha, so its own export is not named anywhere")
    # THE MANIFEST MUST BE THIS ROW'S EXPORT, not merely an export. Without this the union
    # could be taken over someone else's dataset and the row would claim a lineage it has not
    # got -- and the sidecar would land next to the wrong manifest, where no reader looks.
    got = dataset_sha(dataset_manifest)
    if got != ds:
        raise Refused(
            f"{dataset_manifest} has train_id_set_hash {got} but {name!r} has dataset_sha "
            f"{ds}. That manifest is not this row's export, so the union would be over another "
            "dataset and the sidecar would be written beside a manifest that does not name it."
        )
    if init_from == NO_ANCESTOR:
        # No ancestor, so no union: the weights saw exactly this export and the two names are
        # one quantity, as at rung 1. The field still records WHY, which is the whole point.
        uh, dest, n, unchanged = ds, None, None, False
    else:
        init_entry = check_init_row(
            reg,
            init_from,
            name=name,
            base_model=str(entry.get("base_model") or ""),
            rung=rung,
        )
        uh, dest, n, unchanged = union_sidecar(
            init_from=init_from,
            init_entry=init_entry,
            own_hash=ds,
            dataset_manifest=dataset_manifest,
            ids_dir=ids_dir,
        )
    entry["train_id_set_hash"] = uh
    entry[INIT_FROM] = init_from
    reg[name] = order_fields(entry)
    registry_path.write_text(json.dumps(reg, indent=2) + "\n")
    if dest is not None:
        print(f"  sidecar: {dest} ({n} ids, {'unchanged' if unchanged else 'written'})")
    return reg[name]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", help="the served model id (PI_MODEL_INQUIRER)")
    ap.add_argument("--adapter", type=Path, help="adapter directory, if it is on this machine")
    ap.add_argument("--adapter-sha", help="sha256 of adapter_model.safetensors, if it is not")
    ap.add_argument("--base", dest="base_model")
    ap.add_argument("--rung", type=int)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--dataset-manifest", type=Path)
    ap.add_argument(
        "--init-from",
        help="the REGISTERED model id this adapter was initialised from. Required at --rung "
        ">= 2, refused at --rung 1: train_id_set_hash then names the UNION of that row's id "
        "set and this export's, and the union's preimage is written beside the manifest.",
    )
    ap.add_argument(
        "--ids-dir",
        type=Path,
        help="where the train_ids.<hash16>.txt sidecars are. EXCLUSIVE when given (the same "
        "rule as pi agg --ids-dir); otherwise the dataset manifest's own directory and "
        f"{'/'.join(IDS_SUBDIR)}/ are searched.",
    )
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--price-table", type=Path, default=DEFAULT_PRICE_TABLE)
    ap.add_argument(
        "--calls-parquet",
        type=Path,
        default=DEFAULT_CALLS_PARQUET,
        help="where the finished-run-history refusal reads calls from. Absent is not an "
        "error -- a stranger's checkout has no rented box and no history to disagree with.",
    )
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="where the finished-run-history refusal reads each candidate run's manifest.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite a different adapter_sha, or register over disagreeing finished-run history",
    )
    ap.add_argument(
        "--backfill",
        action="store_true",
        help="one-time migration: copy dataset_sha into train_id_set_hash where it is missing",
    )
    ap.add_argument(
        "--backfill-init-from",
        metavar="MODEL_ID",
        help="per-row migration for a rung >= 2 row registered before --init-from existed: "
        "with --name and --dataset-manifest, computes the union and rewrites that one row. "
        "adapter_sha is untouched, so no finished run is renamed.",
    )
    a = ap.parse_args(argv)

    if a.backfill_init_from:
        missing = [
            f
            for f, v in (("--name", a.name), ("--dataset-manifest", a.dataset_manifest))
            if v is None
        ]
        if missing:
            ap.error(
                f"--backfill-init-from also requires: {', '.join(missing)} -- the row to "
                "migrate, and the export manifest whose sidecar it is unioned with."
            )
        try:
            entry = backfill_init_from(
                registry_path=a.registry,
                name=a.name,
                init_from=a.backfill_init_from,
                dataset_manifest=a.dataset_manifest,
                ids_dir=a.ids_dir,
            )
        except (Refused, OSError, ValueError) as e:
            print(f"register_checkpoint: REFUSED: {e}", file=sys.stderr)
            return 1
        print(f"{a.name} -> {a.registry}")
        for k, v in entry.items():
            print(f"  {k}: {v}")
        return 0

    if a.backfill:
        try:
            changed = backfill(a.registry)
        except (Refused, OSError, ValueError) as e:
            print(f"register_checkpoint: REFUSED: {e}", file=sys.stderr)
            return 1
        print(f"backfilled train_id_set_hash on {len(changed)} row(s) in {a.registry}")
        for name in changed:
            print(f"  {name}")
        return 0

    # NOT `required=True` on the parser, because --backfill takes none of them. The message is
    # the same one argparse gave, and a missing flag is still an exit code rather than a
    # half-written row.
    missing = [
        f
        for f, v in (
            ("--name", a.name),
            ("--base", a.base_model),
            ("--rung", a.rung),
            ("--run-dir", a.run_dir),
            ("--dataset-manifest", a.dataset_manifest),
        )
        if v is None
    ]
    if missing:
        ap.error(f"the following arguments are required: {', '.join(missing)}")

    try:
        entry = register(
            name=a.name,
            adapter=a.adapter,
            supplied_sha=a.adapter_sha,
            base_model=a.base_model,
            rung=a.rung,
            run_dir=a.run_dir,
            dataset_manifest=a.dataset_manifest,
            registry_path=a.registry,
            price_path=a.price_table,
            force=a.force,
            init_from=a.init_from,
            ids_dir=a.ids_dir,
            calls_parquet=a.calls_parquet,
            runs_dir=a.runs_dir,
        )
    except (Refused, OSError, ValueError) as e:
        print(f"register_checkpoint: REFUSED: {e}", file=sys.stderr)
        return 1

    print(f"{a.name} -> {a.registry}")
    for k, v in entry.items():
        print(f"  {k}: {v}")
    if entry.get("provisional"):
        print(
            f"  NOTE: provisional -- training_manifest_sha is {TRAINER_STATE}'s, not "
            f"rung{a.rung}.manifest.json's. Re-run this once the trainer finishes."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
