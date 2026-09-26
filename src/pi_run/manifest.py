"""Run identity: turn a configuration plus the state of the working tree into a RunManifest.

The one behaviour worth arguing about is `dirty`. A run produced from an uncommitted tree
cannot be reproduced by anyone, including its author next week, so it must not be silently
comparable with a clean run. `RunManifest.run_id` gives such runs a `dev-` prefix, which
makes exclusion a `str.startswith` at aggregate time rather than a judgement call about
which afternoon's edits were "only cosmetic".
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping

from pinq.splitting import split_of as _split_of
from pinq.types import ModelPin, RunManifest

Split = Literal["train", "dev", "test"]

# 0-59 train, 60-74 dev, 75-99 test. Hashing on template_id where a suite has one keeps
# near-duplicate paraphrases of the same task on the same side of the wall.
_TRAIN_MAX, _DEV_MAX = 59, 74


@dataclass(frozen=True, slots=True)
class GitInfo:
    sha: str
    dirty: bool
    # WHICH FILES MADE IT DIRTY. `dirty` alone says a run is unreproducible and never says why,
    # so nothing could ever clear it -- and the comparison that would settle it empirically
    # cannot exist, because a dev- run and a clean run never share a code_version (committing
    # is exactly what flips one into the other; verified, 0 such pairs in 10,719 runs).
    #
    # It matters because the flag is repo-wide. Measured here: 6,754 of 8,811 ok musique runs
    # came out dev- because a CONCURRENT session had five files open, three of them in
    # `pi_eval`, which a rollout worker provably cannot import. With the list recorded, such a
    # run can be adjudicated from its own manifest instead of being excluded forever.
    dirty_files: tuple[str, ...] = ()


def _dirty_paths(porcelain_z: str) -> tuple[str, ...]:
    """The paths named by `git status --porcelain -z`, parsed BY FIELD, never by offset.

    A fixed offset was wrong here twice over, and both were silent:

      * A porcelain record is `XY<space><path>`, and for an UNSTAGED modification X is a
        space -- ` M paper/sections/mechanism.tex`. The old code applied `.strip()` to the
        whole output and then sliced every line at `line[3:]`, so the first line, and only the
        first, arrived one character short of where the slice looked. Measured before the fix:
        `(' M paper/sections/mechanism.tex', ' M src/pi_run/cli.py')` was recorded as
        `('aper/sections/mechanism.tex', 'src/pi_run/cli.py')` -- a path that does not exist,
        printed to the operator whose sweep `dirty_sweep_refusal` had just refused. It reached
        no manifest, which is the only reason no run record holds a corrupted name: until
        `RunManifest.dirty_files` was added, `manifest_to_dict` wrote the key off an attribute
        no field could fill, so it was `[]` on all 109,503 manifests under runs/ (9,094 of them
        dirty=True). Both halves are fixed now, and neither is backfilled.
      * Line-oriented porcelain also QUOTES and octal-escapes paths, so a dirty file with a
        space or a non-ASCII character was recorded as `"unicode_caf\\303\\251.txt"` -- quotes
        and all. `-z` emits raw bytes and no quoting, which is why the argv changed rather
        than the slice being nudged by one.

    A rename record is `R<space><space><new>NUL<old>NUL`, and BOTH paths go in. The field
    exists so a run can be adjudicated from its own manifest ("this diff touched only
    tests/"); a file renamed OUT of `src/pi_eval/` touched `src/pi_eval/`, and recording only
    the destination would make that diff read as though the gold side was never involved.
    """
    fields = [f for f in porcelain_z.split("\0") if f]
    out: list[str] = []
    i = 0
    while i < len(fields):
        entry = fields[i]
        i += 1
        if len(entry) < 4 or entry[2] != " ":
            # A record shape this parse does not know. Recorded VERBATIM rather than dropped:
            # a strange string in the list is a visible defect, a missing one reads as "that
            # file was clean", which is the failure mode this whole field exists to close.
            out.append(entry)
            continue
        status, path = entry[:2], entry[3:]
        out.append(path)
        # The source path follows ONLY for a rename/copy, which porcelain v1 reports in the
        # INDEX column. Keyed on `status[0]` rather than "R anywhere in status" on purpose: if
        # git ever put an R in the worktree column WITHOUT a second field, a lenient test
        # would eat the next record's field and silently mis-name everything after it.
        if status[0] in "RC" and i < len(fields):
            out.append(fields[i])
            i += 1
    # Sorted, not deduplicated: two runs of one tree give the same list and the manifest stays
    # diffable, while a path git named twice stays visible instead of being quietly merged.
    return tuple(sorted(out))


@lru_cache(maxsize=8)
def git_info(root: str) -> GitInfo:
    """Cached per process: a 30k-unit sweep must not fork git 30k times.

    An absent or broken git is not fatal but IS dirty — an unknown code version is exactly
    the situation the dev- prefix exists for.
    """
    try:
        sha = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "-C", root, "status", "--porcelain", "-z", "--untracked-files=no"],
            capture_output=True,
            text=True,
            # `-z` hands over the path's real bytes, so a filename this locale cannot decode
            # now reaches Python instead of arriving pre-escaped by git. A replacement
            # character is a mangled NAME; an uncaught UnicodeDecodeError here would be a
            # crash in the run-identity path of every unit in the sweep. Precautionary, not
            # measured: APFS on the mac refuses to create such a filename at all
            # ("[Errno 92] Illegal byte sequence"), so the case cannot be exercised here.
            errors="replace",
            check=True,
            timeout=20,
        ).stdout
        # `dirty` is computed exactly as before -- truthiness of the stripped output -- because
        # the dev- prefix rule rests on it and runs already stamped must not be reinterpreted.
        # Only the accuracy of the NAMES changes.
        return GitInfo(sha=sha, dirty=bool(porcelain.strip()), dirty_files=_dirty_paths(porcelain))
    except (OSError, subprocess.SubprocessError):
        # An absent or broken git can name no files, which is the one case where a dirty flag
        # beside an empty list is correct rather than contradictory.
        return GitInfo(sha="unknown", dirty=True, dirty_files=())


def split_of(suite_id: str, key: str) -> Split:
    """Delegates to pinq.splitting, the ONE definition.

    There used to be a second, differently-hashed copy in pinq_train, so a run stamped `train`
    here could be `test` for the exporter. Nothing leaked -- the exporter intersects -- but it
    silently discarded most rows.
    """
    return _split_of(suite_id, key)


def template_id_of(suite: Any, task_id: str) -> str | None:
    """A suite's template id for one task, or None when it has no notion of one.

    WHY THIS MATTERS AND IS NOT COSMETIC. `split_of` hashes on the template id where one
    exists, so near-duplicate instantiations of the same scenario land on the same side of
    the train/dev/test wall; and the tau2 primary endpoint is CLUSTERED at template_id,
    because 97 tau2 tasks are instantiations of ~18 banking scenarios and resampling tasks
    rather than templates understates the SE. A None here is not neutral: it silently makes
    every task its own cluster, which is the same arithmetic as no clustering at all.

    `getattr` rather than a protocol method: most suites genuinely have no templates, and
    forcing every adapter to implement `template_id` would mean five adapters returning None
    to satisfy a signature.
    """
    fn = getattr(suite, "template_id", None)
    if not callable(fn):
        return None
    value = fn(task_id)
    return str(value) if value else None


# The role whose weights decide what this run may be scored on. The Drafter and the Answerer
# are FROZEN across arms by construction, so a checkpoint in either of those seats is not a
# thing that exists; the Inquirer is the only seat a trained adapter is ever served into.
INQUIRER_ROLE = "inquirer"


def train_id_set_hash_of_pins(pins: Mapping[str, ModelPin] | None) -> str | None:
    """The id set the Inquirer's weights were fitted on, or None when nothing says.

    DERIVED, NOT A PARAMETER. A caller that could pass this could pass a hash the pin does not
    support, and the whole value of the stamp is that it is a fact about the weights that
    answered rather than a claim the grid made about them. The registry is the one place that
    maps a served model id to the dataset it came from, and `--backfill` is what put the field
    on the rows that predate it.

    None FOR AN UNREGISTERED ID, which is every id in use today: the Inquirer of every finished
    run is `openai/aws/gpt-oss-120b`, which trained on nothing here and has no row. A None is
    what `train_split_violations` reads as "no id set declared", and it falls back to the split
    predicate alone rather than refusing.

    NOT IN `semantic_hash`, so this stamp renames NOTHING -- unlike `adapter_sha`, which is
    inside `ModelPin.key` and therefore inside `run_id`. That is why this can be filled in for a
    deployment that is already serving, and `adapter_sha` cannot.
    """
    pin = (pins or {}).get(INQUIRER_ROLE)
    if pin is None:
        return None
    # Function-local: `pinq_adapters.llm` pulls litellm in at package import, and `pi_run`
    # imports this module to build a run id in places that never open a socket.
    from pinq_adapters.llm.checkpoints import train_id_set_hash_of

    return train_id_set_hash_of(pin.model_id)


def _canary_nonces_loaded() -> int | None:
    """What firewall layer 4's arming check saw in THIS process, or None if it never ran."""
    from pinq import tripwire

    st = tripwire.last_status()
    return None if st is None else st.n_canaries


def build_manifest(
    *,
    suite_id: str,
    task_id: str,
    arm_id: str,
    policy_id: str,
    seed: int,
    corpus_hash: str,
    budget_cap: int,
    max_turns: int,
    word_cap: int,
    code_version: str,
    dirty: bool,
    # Defaults to (), which is what a caller that cannot name the files honestly reports:
    # `git_info` returns an empty tuple beside dirty=True when git itself failed. So an
    # empty list here never means "clean" -- `dirty` is the only field that says that.
    dirty_files: tuple[str, ...] = (),
    pins: Mapping[str, ModelPin] | None = None,
    prompt_hashes: Mapping[str, str] | None = None,
    upstream_pins: Mapping[str, str] | None = None,
    template_id: str | None = None,
    pilot_flag: bool = False,
    exploratory: bool = False,
    gold_exposed: bool = False,
    prompt_variant_id: str = "v1",
    concurrency: int = 1,
    corpus_dir: str = "",
    k: int = 0,
    questions_hash: str = "",
    grid_sha256: str = "",
    grid_name: str = "",
    branch_of_run_id: str | None = None,
    branch_turn_idx: int | None = None,
    branch_seed: int | None = None,
    foreign_trace_sha: str | None = None,
    foreign_prefix_k: int | None = None,
) -> RunManifest:
    return RunManifest(
        suite_id=suite_id,
        task_id=task_id,
        arm_id=arm_id,
        policy_id=policy_id,
        seed=seed,
        split=split_of(suite_id, template_id or task_id),
        corpus_hash=corpus_hash,
        budget_cap=budget_cap,
        max_turns=max_turns,
        word_cap=word_cap,
        pins=dict(pins or {}),
        # WHICH TASKS THE INQUIRER'S WEIGHTS SAW. Recorded on every run so a trained arm's row
        # can be checked against the id set its checkpoint was fitted on -- the reader
        # `train_id_set_hash` never had. None for every unregistered model id, which is every
        # id in use today, and None changes nothing about the manifest those runs already have.
        train_id_set_hash=train_id_set_hash_of_pins(pins),
        prompt_hashes=dict(prompt_hashes or {}),
        upstream_pins=dict(upstream_pins or {}),
        template_id=template_id,
        pilot_flag=pilot_flag,
        exploratory=exploratory,
        # OR, never AND. `pi run` scrubs PI_GOLD_ROOT from the process tree, so the
        # environment test alone was structurally False on every rollout ever written and
        # `--assert-no-gold-exposed` passed vacuously. The caller passes the arm's own
        # `requires_gold`, which is the part that is true regardless of environment; the env
        # check stays because a run launched some other way must still be marked.
        # The aggregator additionally re-derives exposure from canary scanning: this flag is
        # a declaration, and a declaration is not evidence.
        gold_exposed=bool(gold_exposed) or bool(os.environ.get("PI_GOLD_ROOT")),
        # READ FROM THE ARMING CHECK, NOT RECOMPUTED. `assert_firewall` runs
        # `tripwire.assert_armed()` at unit start in this same process, and this records what
        # THAT call saw. Recomputing here could report a different root -- the failure being
        # closed is precisely a launch root whose canary registry is not where it was assumed to
        # be. None when no arming check ran in this process: unknown, never defaulted.
        canary_nonces_loaded=_canary_nonces_loaded(),
        code_version=code_version,
        dirty=dirty,
        dirty_files=tuple(dirty_files),
        prompt_variant_id=prompt_variant_id,
        concurrency=concurrency,
        corpus_dir=corpus_dir,
        k=k,
        questions_hash=questions_hash,
        grid_sha256=grid_sha256,
        grid_name=grid_name,
        branch_of_run_id=branch_of_run_id,
        branch_turn_idx=branch_turn_idx,
        branch_seed=branch_seed,
        foreign_trace_sha=foreign_trace_sha,
        foreign_prefix_k=foreign_prefix_k,
    )


def manifest_to_dict(m: RunManifest) -> dict:
    """Flat JSON for runs/<run_id>/manifest.json.

    run_id and semantic_hash are derived properties, but they are written out anyway: the
    file has to be readable by a human debugging a directory name, and by `pi compact`
    without re-instantiating the dataclass.
    """
    return {
        "run_id": m.run_id,
        "semantic_hash": m.semantic_hash,
        "model_pin_hash": m.model_pin_hash,
        "suite_id": m.suite_id,
        "task_id": m.task_id,
        "arm_id": m.arm_id,
        "policy_id": m.policy_id,
        "seed": m.seed,
        "split": m.split,
        "corpus_hash": m.corpus_hash,
        "corpus_dir": m.corpus_dir,
        # PROVENANCE: which grid file produced this run, and its exact bytes. Not identity --
        # see RunManifest.grid_sha256 for why it stays out of semantic_hash.
        "grid_sha256": m.grid_sha256,
        "grid_name": m.grid_name,
        "k": m.k,
        "questions_hash": m.questions_hash,
        "budget_cap": m.budget_cap,
        "max_turns": m.max_turns,
        "word_cap": m.word_cap,
        "pins": {
            r: {
                "role": p.role,
                "model_id": p.model_id,
                "provider": p.provider,
                "base_url_sha": p.base_url_sha,
                "adapter_sha": p.adapter_sha,
                "system_prompt_sha": p.system_prompt_sha,
                "sampling_sha": p.sampling_sha,
                "price_table_version": p.price_table_version,
            }
            for r, p in sorted(m.pins.items())
        },
        "prompt_hashes": dict(sorted(m.prompt_hashes.items())),
        "upstream_pins": dict(sorted(m.upstream_pins.items())),
        "template_id": m.template_id,
        "pilot_flag": m.pilot_flag,
        "exploratory": m.exploratory,
        "gold_exposed": m.gold_exposed,
        "counterfactual_kind": m.counterfactual_kind,
        "prefix_of_run_id": m.prefix_of_run_id,
        "prefix_k": m.prefix_k,
        "branch_of_run_id": m.branch_of_run_id,
        "branch_turn_idx": m.branch_turn_idx,
        "branch_seed": m.branch_seed,
        # NONE PASSES THROUGH AS None, never as "". `ids.semantic_hash` pops a None-valued
        # foreign field so a non-fork keeps the id it has always had; `""` is not popped and
        # would rename every non-fork run on disk. See tests/test_run_id_stability.py.
        "foreign_trace_sha": m.foreign_trace_sha,
        "foreign_prefix_k": m.foreign_prefix_k,
        "permutation_id": m.permutation_id,
        "prompt_variant_id": m.prompt_variant_id,
        "train_id_set_hash": m.train_id_set_hash,
        "code_version": m.code_version,
        "dirty": m.dirty,
        # Carried onto the run so the diff can be adjudicated later from the artifact
        # alone, rather than needing a git history that no longer holds it. Empty for every
        # manifest written before RunManifest carried the field, so `[]` beside `"dirty": true`
        # is an OLD RECORD, not a clean tree; `[]` beside `"dirty": false` is a clean tree.
        "dirty_files": list(m.dirty_files),
        "canary_hit": m.canary_hit,
        "firewall_ok": m.firewall_ok,
        # Firewall layer 4's own evidence. See RunManifest.canary_nonces_loaded: without it a
        # run with the tripwire armed and a run whose tripwire never loaded a single needle
        # produce byte-identical manifests (measured 2026-09-17: 0 differing keys).
        "canary_nonces_loaded": m.canary_nonces_loaded,
        "concurrency": m.concurrency,
    }


def repo_root(start: str | os.PathLike[str] | None = None) -> Path:
    p = Path(start or Path.cwd()).resolve()
    for cand in (p, *p.parents):
        if (cand / "pyproject.toml").exists():
            return cand
    return p
