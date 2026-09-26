"""Deterministic identity for every object in the system.

Never call an LLM from this module. Every id here must be reproducible from bytes alone,
because these hashes are cache keys, memo keys and join keys simultaneously.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePath
from typing import Any, Iterable

_SEP = b"\x1f"  # ASCII unit separator: cannot appear in the JSON we hash, so it is unambiguous


class NonCanonical(TypeError):
    """A value reached `canon` that cannot be serialized reproducibly."""


def _default(o: Any) -> Any:
    """The ONLY non-JSON values allowed into an id, and how they are canonicalized.

    This used to be `default=str`, which silently turned anything unknown into its repr. Two
    ways that produced an id that is not a function of its inputs:

      * AN OBJECT WITHOUT __repr__ CONTRIBUTES ITS MEMORY ADDRESS.
            canon({"x": Thing()}) -> {"x":"<__main__.Thing object at 0x102efcf80>"}
        Two structurally identical values hash differently, and the same value hashes
        differently on every call.
      * A SET'S repr ORDER DEPENDS ON PYTHONHASHSEED. Measured, same set, three processes:
            seed=1 -> {"x":"{'beta', 'delta', 'gamma', 'alpha'}"}
            seed=2 -> {"x":"{'delta', 'gamma', 'alpha', 'beta'}"}
            seed=3 -> {"x":"{'gamma', 'beta', 'delta', 'alpha'}"}

    `canon` feeds `request_sha` (the per-call LLM cache key) and `semantic_hash` (run identity),
    so either one means a worker misses a cache entry it paid for, or two identical experiments
    get different run_ids and `--resume` cannot match them -- both silent, both expensive, and
    the second one corrupts the resume sentinel this repository relies on.

    So sets are canonicalized (sorted by a total, seed-independent key) and paths are stringified,
    because both have one obviously right answer. Everything else RAISES: an id that cannot be
    computed reproducibly must not be computed at all.
    """
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=lambda x: (type(x).__name__, canon(x)))
    if isinstance(o, PurePath):
        return str(o)
    raise NonCanonical(
        f"{type(o).__name__} cannot be canonicalized reproducibly, and this value feeds a "
        "cache key and a run_id. Convert it at the call site to something JSON already "
        "describes -- a str, a number, a list or a dict -- so the id is a function of the "
        "value rather than of the process that computed it."
    )


def canon(obj: Any) -> str:
    """Canonical JSON. Sorting keys is what makes ids stable across processes and runs."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_default
    )


def h(*parts: str) -> str:
    """Domain-separated SHA-256 over an ordered list of strings.

    Parts are separated by 0x1f so that h("ab", "c") != h("a", "bc"), which a naive
    concatenation would collide.
    """
    d = hashlib.sha256()
    for p in parts:
        d.update(p.encode("utf-8"))
        d.update(_SEP)
    return d.hexdigest()


def short(digest: str, n: int = 12) -> str:
    return digest[:n]


def evidence_uid(corpus_id: str, doc_id: str, span: str) -> str:
    """Identity of an evidence atom is CONTENT LOCATION and nothing else.

    Deliberately NOT a function of any model-produced canonicalization: if a proposition
    string entered this hash, every leave-one-out memo key would become nondeterministic
    and the whole attribution layer would silently degrade into cache misses.
    """
    return h("ev", corpus_id, doc_id, span)


def corpus_hash(rows: Iterable[tuple[str, str, str]]) -> str:
    """Hash a corpus from (doc_id, title, sha256(text)) triples. Order-independent."""
    return h("corpus", canon(sorted(rows)))


def subset_hash(uids: Iterable[str]) -> str:
    """Identity of a SET of evidence. Order-independent and duplicate-insensitive, so that
    two trajectories that retrieved the same units in different orders memoize together."""
    return h("subset", canon(sorted(set(uids))))


def question_id(text: str) -> str:
    return h("q", text)


def request_sha(payload: Any) -> str:
    """The per-call LLM cache key: the EXACT serialized request.

    Deliberately distinct from RunManifest.semantic_hash (run identity). Keeping them
    separate is what stops a prompt typo from evicting a 30k-rollout sweep.
    """
    return h("req", canon(payload))


# EXACTLY the fields that enter run identity. Held equal to the dict RunManifest.semantic_hash
# builds, by a test, because these are two lists of the same thing and two lists of one thing
# drift.
SEMANTIC_FIELDS: tuple[str, ...] = (
    "suite_id",
    "arm_id",
    "seed",
    "task_id",
    "corpus_hash",
    "model_pin_hash",
    "prompt_hashes",
    "budget_cap",
    "max_turns",
    # Treatment parameters. k is pinned per suite by the grids; questions_hash identifies the
    # recorded list a replay or ceiling arm was seeded with.
    "k",
    "questions_hash",
    # A pilot run and a confirmatory run of one cell are not the same run: the pilot's results
    # were looked at, which is what disqualifies them from the confirmatory analysis.
    "pilot_flag",
    # Same argument, same failure mode. A canary, a rehearsal or an oracle probe exists to be
    # LOOKED AT before the analysis is fixed. Without this the two share a run_id, a run
    # directory and a --resume sentinel, so the exploratory one is published as confirmatory.
    "exploratory",
    # WHICH PROMPT SET PRODUCED THE RUN. `prompt_hashes` covers the templates an arm actually
    # RENDERS, so an overlay that rewrites a template this arm never touches leaves the hash
    # unchanged -- and an LLM-free arm renders none at all. Without this, two runs under
    # DIFFERENT overlays shared a run_id and the second silently resumed the first, inheriting
    # its label. Verified: `PI_PROMPT_OVERLAY=... pi run --arm fake_chain` produced ONE run
    # directory, not two, and it was stamped with the non-overlay variant.
    "prompt_variant_id",
    # A candidate-0 branch reuses the parent's seed by construction, so without these it
    # shared the parent's run_id and run directory and `--resume` skipped it.
    "branch_of_run_id",
    "branch_turn_idx",
    "branch_seed",
    # A FOREIGN-TRACE FORK has the same shape as a branch and a different cause: suite, arm,
    # seed, task, corpus, prompts and code are all identical to a fresh rollout of that task,
    # and what differs is the PREFIX it was handed -- which public trace, cut where. Without
    # these, forking one task at k=3 and at k=7 produces a single run directory and the second
    # silently resumes the first.
    "foreign_trace_sha",
    "foreign_prefix_k",
    "code_version",
    "upstream_pins",
)


def semantic_hash(manifest_fields: dict[str, Any]) -> str:
    """Run identity: only what would change the science.

    Excludes wall-clock, hostname, cache paths and concurrency, so the same experiment run
    on a different machine at a different time collapses to the same identity.

    A SECOND ALLOWLIST LIVES HERE ON PURPOSE, and it is a guard rather than a convenience: a
    caller that passed wall_ms or a hostname would otherwise make two identical experiments
    hash differently. But an allowlist is also a place a new field can be silently dropped --
    `k` and `questions_hash` were added to `RunManifest.semantic_hash`'s dict and had NO effect,
    because they were not named here. `tests/test_runtime.py` holds the two lists equal, so the
    next addition fails loudly instead of quietly not counting.
    """
    fields = {k: manifest_fields.get(k) for k in SEMANTIC_FIELDS}
    # OMITTED WHEN ABSENT, NOT HASHED AS None. These were added long after 736 runs were on
    # disk, and a key present with value None still changes `canon`, so simply listing them
    # above moved the semantic_hash of EVERY existing run: measured, 736 of 736 changed, which
    # would rename every run directory, make `--resume` re-roll a completed corpus and leave
    # every recorded run_id unreproducible. `None` here means "this run is not a branch",
    # which is precisely what every historical run is -- so the correct hash input for such a
    # run is the one that has no branch keys at all.
    #
    # THE FOREIGN PAIR JOINS THE LIST FOR THE SAME REASON AND ON THE SAME EVIDENCE. Measured
    # over all 77,109 manifests under `runs/` before they were added: 49,408 non-forks hash
    # identically with and without them, 5,096 forks reproduce their recorded id only WITH
    # them, and 0 runs that reproduced before stop reproducing. The remaining 22,605 are the
    # `rehydrated_from` stubs, which carry no pins and reproduce under no rule at all.
    # `tests/test_run_id_stability.py` pins seven of those ids, in both manifest shapes.
    #
    # `""` IS NOT `None` AND IS NOT POPPED. A writer that coerced an absent trace to the empty
    # string would give every non-fork a third identity and rename the whole corpus, so the
    # writers pass None through unchanged; that edge has its own test.
    for k in (
        "branch_of_run_id",
        "branch_turn_idx",
        "branch_seed",
        "foreign_trace_sha",
        "foreign_prefix_k",
    ):
        if fields.get(k) is None:
            fields.pop(k, None)
    return h("run", canon(fields))
