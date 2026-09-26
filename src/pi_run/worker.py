"""One (task, arm, seed) unit, in one OS process, with no shared state of any kind.

The design is deliberately boring, and each boring choice buys something:

  * ONE LEDGER PER PROCESS. Budget accounting never crosses a process boundary, so there is
    no lock, no shared counter and no reduction step that could be wrong. `pinq.budget`
    already says this; the worker is where it becomes true.

  * status.json IS THE SENTINEL, WRITTEN LAST. Every other artifact is flushed first, so
    `--resume` is exactly `os.path.exists(status.json)`. A unit killed mid-write leaves no
    status file and is simply redone; there is no partial-state repair path to get wrong.

  * NO pi_eval IMPORT, ANYWHERE ON THIS PATH. A rollout worker runs with PI_GOLD_ROOT unset
    and asserts it at entry. Importing gold-reading code into this address space would make
    that assertion the only thing standing between a bug and a contaminated result, and one
    assertion is not enough of a wall.
"""

from __future__ import annotations

import json
import os
import signal
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from pinq import promptlib, tripwire
from pinq.budget import BudgetLedger
from pinq.ids import canon, h
from pinq.loop import run_loop
from pinq.types import Ask, Trajectory, Usage
from pinq_expt import arms as arm_table
from pinq_expt.components import prompt_hashes_of

RUN_FILES = ("manifest.json", "turns.jsonl", "calls.jsonl", "ledger.jsonl", "evidence.jsonl")
STATUS = "status.json"


class FirewallError(RuntimeError):
    """PI_GOLD_ROOT was set in a rollout worker. The process split is not advisory."""


class ReconcileError(RuntimeError):
    """The ledger and the trajectory disagree about what was spent.

    Never a warning. A run whose accounting is wrong is worse than a missing run, because a
    missing run is visible in a count and a mis-metered one silently shifts a parity table.
    """


class UnknownSuite(KeyError):
    pass


@dataclass(frozen=True, slots=True)
class UnitSpec:
    """Everything a worker needs, as plain data. No open handles, no callables, no Path
    objects: it crosses a process boundary by pickle, and every field here survives that.

    code_version/dirty are computed ONCE in the parent and carried down. Forking git 30,000
    times to learn the same answer is the kind of thing that turns a 4-hour sweep into a
    6-hour one.
    """

    suite_id: str
    corpus_dir: str
    task_id: str
    arm_id: str
    seed: int
    runs_root: str
    cache_root: str
    max_turns: int = 16
    k: int = 5
    budget_cap: int | None = None
    pilot: bool = False
    # See RunManifest.exploratory: a run that exists to be looked at is inadmissible.
    exploratory: bool = False
    resume: bool = True
    code_version: str = ""
    dirty: bool = True
    # The names behind that flag, measured ONCE in the parent with the same `git_info`
    # call. Not derived in the child: a worker runs with a different cwd and must not
    # disagree with the parent about the tree it was launched from. Not in `key` --
    # provenance, never identity.
    dirty_files: tuple[str, ...] = ()
    # Replay arms (parallel_replay, oracle_vreq) are seeded from a prior sweep. Plain dicts,
    # because a UnitSpec crosses the process boundary by pickle and a generator would not.
    questions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    ask_counts: Mapping[str, int] = field(default_factory=dict)
    concurrency: int = 1
    # None defers to PI_UNIT_TIMEOUT_S, then to DEFAULT_UNIT_TIMEOUT_S. Carried on the spec
    # rather than read in the child so that one sweep cannot have two different watchdogs.
    timeout_s: int | None = None
    # PROVENANCE, carried so a run records which grid file produced it. `grids.load` has
    # always computed the hash and `cli.py` printed it, but it reached no record at all.
    # Not in semantic_hash: two grids may legitimately produce the same run.
    grid_sha256: str = ""
    grid_name: str = ""
    # RUNG 2. The parent run this unit forks, and the turn it forks at. Both are in
    # semantic_hash (see RunManifest.semantic_hash): candidate 0 reuses the parent's seed by
    # construction, so without them a candidate-0 branch would land in the parent's own run
    # directory and --resume would skip it.
    branch_of_run_id: str | None = None
    branch_turn_idx: int | None = None
    branch_seed: int | None = None
    # The prefix a foreign-trace fork was handed: which public trace, and where it was cut.
    # In `key` for the same reason the branch triple is -- the sweep dedupes and resumes on
    # `key`, so identity in the manifest alone would let it skip the second fork of one task
    # before a manifest ever existed.
    foreign_trace_sha: str | None = None
    foreign_prefix_k: int | None = None
    # SAMPLING, passed through to the client verbatim. `RolloutRequest.sampling` has existed
    # on the wire type since the serve path was written and `build_spec` DROPPED IT: a caller
    # asking for temperature 1.0 silently got 0.0 and no error. Rung 2 cannot sample two
    # actions at one state without it.
    #
    # Empty means GREEDY, and greedy stays the default: every confirmatory arm depends on
    # temperature 0.0, and a sampled confirmatory run is a different measurement.
    sampling: Mapping[str, float | int | str] = field(default_factory=dict)
    # A NAMED PROMPT VARIANT, read by `tau2_runner.apply_prompt_variant` and by nothing else.
    # Empty is the shipped prompt set, which is what every run on disk rendered. The name
    # reaches the manifest as `prompt_variant_id`, which is in `pinq.ids.SEMANTIC_FIELDS`, so
    # a variant run can never land in a shipped-prompt run's directory.
    prompt_variant: str = ""
    # WHICH HARNESS PROTOCOL A DIALOGUE RAN UNDER, read by `tau2_runner` and by nothing else.
    # Empty is this repository's own (max_steps 100, max_errors 30, DB-state reward with no
    # termination gate), which is what every tau2 run on disk carries; `tau2_upstream` is
    # tau2-bench's. The name reaches `upstream_pins`, which is in `pinq.ids.SEMANTIC_FIELDS`,
    # so two protocols can never share a run id, a run directory or a resume sentinel.
    protocol: str = ""

    @property
    def key(self) -> tuple[str, str, str, int, str, int, int, str, int, str, str]:
        """The reassembly key. Results are collected by this, never by completion order.

        THE BRANCH TRIPLE IS PART OF IT. `candidate_specs` gives every candidate the PARENT's
        suite, task, arm AND seed -- branch identity lives entirely in branch_of_run_id /
        branch_turn_idx / branch_seed -- so without them all n candidates shared one slot.
        `run_sweep` returns `[by_key[spec.key] for spec in specs]`, so the result list was
        n ALIASES of one dict: `ok`/`failed` counted copies of a single status, an errored
        candidate was invisible, and `pi train sample-candidates` reported ONE run_id where
        eight branch directories had been written. `drain_uncollected` skips
        `if key in by_key`, so a spend cap under-reported the siblings still in flight --
        which sweep.py's own docstring calls "worse than no cap".

        The run DIRECTORIES were always distinct (each branch has its own semantic_hash),
        which is why this never corrupted data and never surfaced as a failure.
        """
        return (
            self.suite_id,
            self.task_id,
            self.arm_id,
            self.seed,
            self.branch_of_run_id or "",
            -1 if self.branch_turn_idx is None else int(self.branch_turn_idx),
            -1 if self.branch_seed is None else int(self.branch_seed),
            # THE FORK PREFIX, for the reason the branch triple is here. Two forks of one task
            # differ only in which trace they were cut from and where; sharing a key would make
            # `run_sweep`'s `[by_key[spec.key] for spec in specs]` return aliases of one status.
            self.foreign_trace_sha or "",
            -1 if self.foreign_prefix_k is None else int(self.foreign_prefix_k),
            # THE PROMPT VARIANT, for the reason the two above are here. Two units of one fork
            # point under the shipped prompt and under `tau2_stop` differ in nothing else, so
            # sharing a key would make `run_sweep`'s `[by_key[spec.key] for spec in specs]`
            # hand back two aliases of one status and hide whichever errored.
            self.prompt_variant or "",
            # THE HARNESS PROTOCOL, for the reason the prompt variant is here. One task under
            # this repository's caps and under tau2-bench's differ in nothing else in this
            # tuple, so sharing a key would hand back two aliases of one status.
            self.protocol or "",
        )


# --------------------------------------------------------------------------- suites


# Suites that read a corpus this repo BUILT (data/corpora/<suite>/<hash>/) versus suites that
# source their own data from an upstream checkout or a hosted API. The distinction is not
# cosmetic: only the first kind has a corpus_hash we control, and only the second kind can fail
# for a reason the researcher must fix in the environment rather than by running a builder.
# DRGYM IS ON THE CORPUS-BACKED SIDE, and it was on the other one.
#
# The distinction is "did this repository BUILD the task list?", and for drgym the answer is
# yes: `pi data build --suite drgym` writes data/corpora/drgym/<hash>/tasks.jsonl and
# `pi data status` reports `n=976` for it. What drgym does NOT own is its DOCUMENTS -- those
# live behind a hosted search API -- but that is a separate fact the adapter already handles
# with `offline=True` and a key check, and it is not what this tuple decides.
#
# Being on the self-sourced side made `_resolve_corpus` hand back the bare
# `data/corpora/drgym` directory instead of resolving the content-addressed one under it, so
# `pi run --suite drgym` died on
# `FileNotFoundError: .../data/corpora/drgym/tasks.jsonl` -- a path that never exists, with a
# traceback rather than a remedy.
CORPUS_BACKED = ("synth", "musique", "strategyqa", "wiki2", "drgym", "frames", "musique_x2")
# `tau2_retail` joins them: `retail_build` deliberately writes NO corpus tree, because a second
# copy of upstream's db.json would carry a second corpus_hash and the adapter's uids would then
# disagree with gold's -- invisible downstream, since an unmatched uid reads as "the policy
# retrieved nothing relevant". Demanding data/corpora/tau2_retail/ would block every run on a
# directory that is never supposed to exist.
# `userbench` joins them for the same reason as tau2: its 3,122 scenarios (~200 MB of JSON)
# ship INSIDE the upstream repository, so there is nothing for `pi data build` to build and
# nothing this repo may redistribute. USERBENCH_DIR points at the checkout; the adapter
# raises with that remedy when it is unset.
SELF_SOURCED = ("tau2", "tau2_retail", "tau2_airline", "pare", "userbench")


def sampling_temperature(sampling: Mapping[str, Any]) -> float:
    """The client's temperature for this unit. 0.0 -- greedy -- unless asked otherwise.

    Extracted so the default is testable without running a unit. It was a literal 0.0 at the
    construction site, which is why `RolloutRequest.sampling` could be dropped in
    `build_spec` and nothing anywhere noticed.
    """
    try:
        return float(sampling.get("temperature", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


class DrGymOnlineWithoutKey(RuntimeError):
    """PI_DRGYM_ONLINE was set but no DRGYM_API_KEY is present."""


def drgym_offline(env: Mapping[str, str]) -> bool:
    """Whether the drgym suite replays its cache (True) or queries the hosted index (False).

    OFFLINE IS THE DEFAULT AND HOLDING A KEY DOES NOT CHANGE IT. A confirmatory sweep must
    replay a recorded search cache rather than re-query an index whose contents can move
    under it; that is a reproducibility property, not a convenience, and it must not flip
    just because a credential appeared in the environment.

    Going online is an explicit `PI_DRGYM_ONLINE=1`. Without a key that is REFUSED rather
    than attempted: every searching unit would otherwise walk into a 401 one query at a
    time, which reads as a broken suite rather than a missing credential.
    """
    want_online = str(env.get("PI_DRGYM_ONLINE", "")).strip().lower() in {"1", "true", "yes", "on"}
    if not want_online:
        return True
    if not str(env.get("DRGYM_API_KEY", "")).strip():
        raise DrGymOnlineWithoutKey(
            "PI_DRGYM_ONLINE is set but DRGYM_API_KEY is not. Request a free key from "
            "deepresearchgym@cmu.edu, or unset PI_DRGYM_ONLINE to replay the cache."
        )
    return False


def load_suite(suite_id: str, corpus_dir: str):
    """Construct the adapter for a suite.

    Every import is local on purpose: a worker for the synthetic suite must not pay for, or
    fail on, an adapter whose extras are not installed. An unregistered suite raises rather
    than silently falling back, because a sweep that quietly ran the wrong corpus is worse
    than one that did not run.
    """
    if suite_id == "synth":
        from pinq_adapters.synth.suite import SynthSuite

        return SynthSuite(Path(corpus_dir))

    if suite_id == "musique":
        from pinq_adapters.musique.suite import MusiqueSuite

        return MusiqueSuite(Path(corpus_dir))

    if suite_id == "strategyqa":
        # The CONCRETE subclass, not the shared ParagraphSuite base: the base declares
        # suite_id = "" and make_view rightly refuses an empty suite_id, so wiring the base
        # here made every strategyqa/wiki2 view raise LeakageError.
        from pinq_adapters.strategyqa.suite import StrategyQASuite

        return StrategyQASuite(Path(corpus_dir))

    if suite_id == "wiki2":
        from pinq_adapters.wiki2.suite import Wiki2Suite

        return Wiki2Suite(Path(corpus_dir))

    if suite_id == "frames":
        # Corpus-backed like the three above, and the only one whose corpus was assembled
        # from a live third party rather than a frozen release -- so its constructor also
        # re-checks the per-page digests the builder recorded. That refusal is the point of
        # the suite being here rather than in SELF_SOURCED.
        from pinq_adapters.frames.suite import FramesSuite

        return FramesSuite(Path(corpus_dir))

    if suite_id == "musique_x2":
        # Composed pairs of held-out MuSiQue TEST tasks, built by pi_eval.build.compose_build
        # into its own content-addressed corpus with its own corpus_id. A separate suite id
        # rather than a flag on musique: its uids, its gold and its split (eval-only) all differ.
        from pinq_adapters.musique_x2.suite import MusiqueX2Suite

        return MusiqueX2Suite(Path(corpus_dir))

    if suite_id == "tau2_airline":
        # Self-sourced like the other tau2 domains: the wheel ships no data/, so this needs
        # TAU2_DATA_DIR pointed at a checkout and `corpus_dir` is ignored rather than faked.
        # A SEPARATE SUITE ID for the reason retail is one -- same upstream checkout, a
        # different domain, so a fallback between them would load the wrong corpus and every
        # gold uid would silently miss. Its split is upstream's, not ours: see
        # `pinq.splitting.FIXED_SPLITS`.
        from pinq_adapters.tau2.airline_suite import Tau2AirlineSuite

        return Tau2AirlineSuite()

    if suite_id == "tau2_retail":
        # A SEPARATE SUITE ID, not a flag on tau2. `tau2` stays in EVAL_ONLY_SUITES as the
        # zero-shot transfer target; retail exists to be trainable. They read the same upstream
        # checkout and different domains, so a fallback between them would load the wrong
        # corpus and every gold uid would silently miss.
        from pinq_adapters.tau2.retail_suite import Tau2RetailSuite

        return Tau2RetailSuite()

    if suite_id == "tau2":
        # Self-sourced: the tau2 wheel ships NO data/, so this needs TAU2_DATA_DIR pointed at a
        # repo checkout. The adapter raises with that remedy if it is unset, which is why the
        # corpus_dir argument is ignored here rather than being faked.
        from pinq_adapters.tau2.suite import Tau2Suite

        return Tau2Suite()

    if suite_id == "pare":
        from pinq_adapters.pare.suite import PareSuite

        return PareSuite()

    if suite_id == "userbench":
        # Self-sourced from a UserBench checkout via USERBENCH_DIR, so `corpus_dir` is
        # ignored here rather than faked -- exactly as for tau2. EVAL-ONLY: the adapter reads
        # upstream's `test` split and has no parameter that could point it at `train`.
        #
        # Constructing the suite does NOT construct a TravelEnv and spends nothing: the split
        # index is a few kilobytes of parquet. A live episode goes through
        # `UserBenchSuite.session(task_id)`, which arms the user-simulator failure counter
        # before the env exists.
        from pinq_adapters.userbench.suite import UserBenchSuite

        return UserBenchSuite()

    if suite_id == "drgym":
        # offline by default: a confirmatory sweep must replay the recorded search cache
        # rather than re-querying a hosted index whose contents can move under it. Holding a
        # key does NOT flip that; `PI_DRGYM_ONLINE=1` does. See `drgym_offline`.
        from pinq_adapters.drgym.suite import DrGymSuite

        return DrGymSuite(Path(corpus_dir), offline=drgym_offline(os.environ))

    raise UnknownSuite(
        f"no adapter registered for suite {suite_id!r}. "
        f"corpus-backed: {CORPUS_BACKED}; self-sourced: {SELF_SOURCED}"
    )


# --------------------------------------------------------------------------- serialization


def _action_row(action: Any) -> dict[str, Any]:
    if isinstance(action, Ask):
        return {
            "action_kind": "ask",
            "question": action.text,
            "question_id": action.qid,
            "rationale": action.rationale,
            "target": action.target,
            # SELF-REPORTED. Audited, plotted, never scored: see pinq.types.Ask.
            "parent_uids": list(action.parent_uids),
        }
    return {
        "action_kind": "stop",
        "question": "",
        "question_id": "",
        "rationale": getattr(action, "reason", ""),
        "target": "",
        "parent_uids": [],
    }


def turn_rows(traj: Trajectory) -> list[dict[str, Any]]:
    rows = []
    for t in traj.turns:
        row: dict[str, Any] = {
            "turn_idx": t.turn_idx,
            "response_text": t.response_text,
            "retrieved_uids": list(t.retrieved_uids),
            "new_uids": list(t.new_uids),
            "n_retrieved": len(t.retrieved_uids),
            "n_new": len(t.new_uids),
            "draft_sha": t.draft_sha,
            "draft_text": t.draft_text,
            "subset_hash_before": t.subset_hash_before,
            "subset_hash_after": t.subset_hash_after,
            "branch_of_run_id": t.branch_of_run_id,
            "branch_turn_idx": t.branch_turn_idx,
            "candidate_id": t.candidate_id,
            "candidate_rank": t.candidate_rank,
            "forced": t.forced,
            "policy_conf": t.policy_conf,
            "depth_pred": t.depth_pred,
            "bm25_hits": t.bm25_hits,
            "spec_bits": t.spec_bits,
        }
        row.update(_action_row(t.action))
        row.update({f"usage_{k}": v for k, v in t.usage.as_dict().items()})
        rows.append(row)
    return rows


def call_rows(traj: Trajectory) -> list[dict[str, Any]]:
    return [asdict(c) for c in traj.calls]


def ledger_rows(ledger: BudgetLedger) -> list[dict[str, Any]]:
    return [asdict(r) for r in ledger.rows]


def evidence_rows(traj: Trajectory) -> list[dict[str, Any]]:
    """One row per retrieved unit, with the turn that first surfaced it.

    Text is NOT stored: the corpus is content-addressed and frozen, so a uid plus corpus_hash
    already identifies the bytes, and copying them into every run multiplies the artifact
    size by the number of arms for no recoverable information.
    """
    first: dict[str, int] = {}
    for t in traj.turns:
        for uid in t.retrieved_uids:
            first.setdefault(uid, t.turn_idx)
    return [
        {
            "uid": u.uid,
            "corpus_id": u.corpus_id,
            "doc_id": u.doc_id,
            "span": u.span,
            "title": u.title,
            "score": float(u.score),
            "n_chars": len(u.text),
            "first_turn_idx": first.get(u.uid),
        }
        for u in traj.evidence.units
    ]


def outcome_dict(traj: Trajectory) -> dict[str, Any]:
    a = traj.outcome.answer
    return {
        "answer": None
        if a is None
        else {
            "text": a.text,
            "evidence_hash": a.evidence_hash,
            "cited_unit_ids": list(a.cited_unit_ids),
            "n_words": a.n_words,
            "stop_reason": a.stop_reason,
        },
        "env_calls": [asdict(e) for e in traj.outcome.env_calls],
        "env_final_hashes": dict(traj.outcome.env_final_hashes),
        "native": dict(traj.outcome.native),
        "artifacts": dict(traj.outcome.artifacts),
        "transcript_digest": traj.outcome.transcript_digest,
        "stop_reason": traj.stop_reason,
        "subset_hash": traj.evidence.subset_hash,
    }


def write_atomic(path: Path, text: str) -> None:
    """Write via a same-directory temp file plus os.replace, which is atomic on POSIX.

    status.json is the RESUME SENTINEL, so a torn one is not a lost unit but a lost SWEEP: the
    resume path json.loads() it, the JSONDecodeError propagated out of run_unit, through
    fut.result(), and killed every unit still queued. One kill -9 during the final write of
    unit 4,000 therefore discarded the 11,600 units that had not started yet.

    os.replace also means a reader can never observe a half-file: it sees the old bytes or the
    new bytes. Applied to every artifact rather than only the sentinel, because `pi compact`
    walks run directories that may have been killed mid-write and a truncated manifest.json
    fails compaction for the whole campaign, not for one run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    write_atomic(path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def read_status(path: Path) -> dict[str, Any] | None:
    """The prior status, or None if there isn't a usable one.

    A file that exists but does not parse is treated as ABSENT, i.e. the unit is re-run. That
    is the safe direction: re-running costs one unit's tokens, whereas propagating the decode
    error costs the remainder of the sweep, and trusting a partial parse would silently resume
    from a status whose `usage` block was truncated.
    """
    try:
        obj = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


# --------------------------------------------------------------------------- watchdog


class UnitTimeout(RuntimeError):
    """One unit exceeded its wall-clock allowance and was abandoned."""


# SET FROM MEASUREMENT, not from a round number. Measured over 10 live units on 2026-08-24:
#
#     suite    arm                  n   max wall   median
#     musique  drafter_only         2       17 s     14 s
#     musique  inquirer_prompted    2       86 s     57 s
#     tau2     drafter_only         3      300 s    165 s
#     tau2     inquirer_prompted    3      726 s    426 s
#
# THE SPREAD IS THE POINT. The treatment is 5x the baseline on musique and 2.4x on tau2, so at
# 1800 s the treatment arm was the ONLY arm that could realistically time out -- and one
# rate-limit ladder is ~148 s, so a handful of stalls on a 726 s unit reaches it. A timed-out
# unit is excluded, and an exclusion that only one arm can suffer is a selection effect.
#
# `report.estimate` pairs on the intersection of the two arms' task sets, so a missing
# treatment unit drops that task from BOTH arms -- which removes the arm asymmetry but not the
# selection: the tasks lost are the ones the treatment worked hardest on. So the cap is set to
# 5x the slowest unit ever observed rather than to a comfortable-sounding number, and
# `summarize()` warns when the margin erodes.
DEFAULT_UNIT_TIMEOUT_S = 3600


def unit_timeout_s(explicit: int | None = None) -> int:
    """Seconds one unit may run before SIGALRM fires. 0 or negative disables the watchdog.

    A wedged unit is invisible without this. The client retries 8 times from a 4s base and the
    ladder spans about 148s per call; a stalled tunnel that never returns and never resets can
    hold one worker for the ~4.6h an operator eventually notices by watching `runs/` stop
    growing -- during which that worker's share of the pool is simply gone. `summarize()` has
    always counted a "timeout" terminal status; nothing produced one, so the counter read as a
    guarantee that no unit had ever hung.

    SIGALRM and not a thread: the unit is already alone in its process, the blocking call is in
    C, and a daemon timer thread cannot interrupt it. Platforms without SIGALRM (Windows) get
    no watchdog rather than a silently-different one.
    """
    if explicit is not None:
        return int(explicit)
    raw = os.environ.get("PI_UNIT_TIMEOUT_S")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return DEFAULT_UNIT_TIMEOUT_S


class _Watchdog:
    """Arm SIGALRM for `seconds`, and restore whatever handler was there before.

    Restoring matters because the single-worker path runs run_unit in the PARENT process, and
    a sweep that left a stale SIGALRM handler installed would fire it during summarize().
    """

    def __init__(self, seconds: int) -> None:
        self._on = seconds > 0 and hasattr(signal, "SIGALRM")
        self._seconds = seconds
        self._prev: Any = None

    def __enter__(self) -> "_Watchdog":
        if self._on:
            self._prev = signal.signal(signal.SIGALRM, self._fire)
            signal.alarm(self._seconds)
        return self

    def _fire(self, signum: int, frame: Any) -> None:
        raise UnitTimeout(
            f"unit exceeded PI_UNIT_TIMEOUT_S={self._seconds}s and was abandoned. "
            "Re-issuing the sweep retries it; raise the limit if the unit is legitimately slow."
        )

    def __exit__(self, *exc: Any) -> None:
        if self._on:
            signal.alarm(0)
            if self._prev is not None:
                signal.signal(signal.SIGALRM, self._prev)


# --------------------------------------------------------------------------- reconciliation


def _fold(usages) -> Usage:
    total = Usage()
    for u in usages:
        total = total + u
    return total


def reconcile(traj: Trajectory, ledger: BudgetLedger) -> dict[str, Any]:
    """Two independent accounts of the same rollout, compared.

    `tokens_ok` is pinq.budget.BudgetLedger.reconcile: per-turn usage against the ledger.

    `docs_ok` is the detector the ledger cannot provide on its own, and it is the one that
    catches an UNMETERED NESTED RETRIEVAL. run_loop meters every unit it merges via
    note_docs, so the set of documents in the final Evidence must equal the set the ledger
    saw. A Drafter that quietly retrieves inside resolve() and folds the result into its
    returned Evidence grows the first set and not the second — which is invisible in tokens,
    invisible in retrieval_calls, and fatal to a budget-parity claim.
    """
    turn_usage = _fold(t.usage for t in traj.turns)
    # The terminal half is what makes this bite. Compared against turns alone the equality was
    # FALSE on every LLM arm -- a real rollout charges the STOP act(), the final draft and the
    # Answerer to no turn -- and TRUE only on llm_free arms whose ledgers are empty, so a
    # field named `tokens_ok` said False everywhere it mattered and was therefore never read.
    tokens_ok = ledger.reconcile(turn_usage, traj.terminal_usage)
    doc_ids = traj.evidence.doc_ids
    docs_ok = ledger.unique_docs == len(doc_ids)
    return {
        "tokens_ok": bool(tokens_ok),
        "docs_ok": bool(docs_ok),
        "ledger_unique_docs": ledger.unique_docs,
        "evidence_unique_docs": len(doc_ids),
        "turn_usage": turn_usage.as_dict(),
        "terminal_usage": traj.terminal_usage.as_dict(),
        "ledger_usage": ledger.usage.as_dict(),
    }


# --------------------------------------------------------------------------- the unit


def assert_firewall(env: dict[str, str] | None = None) -> None:
    """Both firewall layers this process can check about ITSELF, at unit start.

    Layer 3 (the process split) and layer 4 (the canaries). The other two are static -- the
    import-linter contract and the type wall hold at build time -- but these two are properties
    of THIS process, and both fail silently in the direction of looking fine: an unset
    PI_GOLD_ROOT is indistinguishable from a correctly-scrubbed one until something reads gold,
    and an unloaded canary registry is indistinguishable from a clean scan on every request.
    """
    env = env if env is not None else dict(os.environ)
    if env.get("PI_GOLD_ROOT"):
        raise FirewallError(
            "PI_GOLD_ROOT is set in a rollout worker. Rollouts must not be able to read gold; "
            "unset it in the parent before launching the sweep."
        )
    # Layer 4 is the only one that catches a leak through a STRING rather than through code, and
    # `tripwire.scan` cannot distinguish "no canaries registered" from "no canaries present".
    # Checked once, loudly, here -- rather than absorbed silently on every request.
    tripwire.assert_armed()


def firewall_probe() -> dict[str, Any]:
    """Runs INSIDE a pool worker so `pi verify firewall` reports the real child environment
    rather than the parent's, which is the only one that matters."""
    return {
        "pid": os.getpid(),
        "gold_root_set": bool(os.environ.get("PI_GOLD_ROOT")),
        "pi_eval_imported": "pi_eval" in _loaded_modules(),
    }


def _loaded_modules() -> set[str]:
    import sys

    return {m.split(".")[0] for m in sys.modules}


def questions_hash(spec: UnitSpec, inquirer: Any) -> str:
    """Identity of the recorded question list THIS unit replays. Empty when it replays none.

    Two shapes, because two kinds of arm depend on two different things:

      * A ScriptedInquirer (parallel_replay, gold_evidence, oracle_vreq) issues its OWN task's
        list, so that list is what identifies the run. Adding an unrelated task to the sweep
        must not change its run_id.
      * `random_q` samples from the questions of OTHER tasks -- that is what makes it a null
        rather than a weaker copy of the treatment -- so the whole pool is its input, and
        adding a task genuinely changes what it could have drawn.

    Without this, seeding a ceiling arm from `--questions-from-file A` and then from B produced
    IDENTICAL run ids: the second invocation resumed the first's results and the new file was
    silently ignored. That is the same failure that let both ceiling arms share one input.
    """
    if not spec.questions:
        return ""
    if getattr(inquirer, "uses_question_pool", False):
        return h("questions_pool", canon({k: list(v) for k, v in sorted(spec.questions.items())}))
    own = spec.questions.get(spec.task_id)
    return h("questions", canon(list(own))) if own else ""


def _billed(ledger: BudgetLedger) -> float:
    """Dollars the provider will actually charge: the ledger's CALLS minus the cache hits.

    `ledger.calls` and not `ledger.rows`: rows are the per-currency accounting entries and
    carry no provenance, while a CallTelemetry knows whether it was served from disk.
    """
    return round(sum(float(c.usd) for c in ledger.calls if not c.cache_hit), 6)


def collect_pins(llm: Any, components: Sequence[Any]) -> dict[str, Any]:
    """RunManifest.pins, read off the components that were actually built.

    WHY THIS IS NOT COSMETIC. `model_pin_hash` is inside `semantic_hash`, which is inside
    `run_id`. With `pins` empty -- as it was in all 50 manifests written before this existed --
    `inquirer_trained` and `inquirer_prompted` hash IDENTICALLY: swapping in a fine-tuned
    checkpoint produced the same run_id, so `--resume` skipped the new arm as already done and
    the trained-vs-prompted comparison would have been one arm's numbers printed twice. The
    manifest also could not answer the first question anyone asks of a result table, which is
    which model produced which column.

    Roles come from the components rather than from ROLE_ENV, so an arm whose Inquirer is
    scripted (parallel_replay, gold_evidence) records a drafter and an answerer pin and no
    inquirer pin -- which is exactly true of what it ran, and keeps its run_id insensitive to
    an inquirer model swap that could not have affected it.

    A client with no `pin()` (the fakes and ReplayClient in tests) yields a placeholder naming
    the class, never an empty dict: "no pins" and "pins from a fake" must not look alike.
    """
    from pinq.types import ModelPin

    pin_of = getattr(llm, "pin", None)
    out: dict[str, Any] = {}
    for comp in components:
        role = getattr(comp, "llm_role", None)
        if not role or role in out:
            continue
        if pin_of is not None:
            try:
                out[role] = pin_of(role)
            except Exception as exc:  # noqa: BLE001 - an unresolved pin is a RECORD, not a raise
                # A role with no model configured must still fail this unit -- and it does, at
                # the first complete() inside run_loop, where the failure becomes status=error
                # with the manifest already on disk. Raising HERE instead would abort before
                # build_manifest, so the one artifact that says which prompts the unit would
                # have rendered would never be written, and the unit would take the sweep down
                # with it. So the gap is recorded rather than thrown: never a silent {}, and
                # never a pin that claims a model nobody named.
                out[role] = ModelPin(
                    role=role,  # type: ignore[arg-type]
                    model_id=f"<unresolved:{type(exc).__name__}>",
                    provider="unconfigured",
                )
        else:
            out[role] = ModelPin(
                role=role,  # type: ignore[arg-type]
                model_id=f"<{type(llm).__name__}>",
                provider="fake",
            )
    return out


def run_unit(spec: UnitSpec) -> dict[str, Any]:
    """Execute one unit. Returns a plain dict so the result pickles back cheaply.

    Never raises for a task-level failure: a crashed unit becomes status='error' with the
    message recorded, because one bad task must not take down a 30k sweep. A FIREWALL or
    RECONCILE failure is different and does propagate — those mean the results are wrong,
    not that one task is missing.
    """
    assert_firewall()
    from pi_run.stages.tau2_runner import DIALOGUE_SUITES

    if spec.suite_id in DIALOGUE_SUITES:
        # tau2 is a DIALOGUE benchmark and cannot be run flat: the only task text upstream
        # ships is the customer's roleplay script, and its reward is a hash over a transcript
        # the Orchestrator has to produce. The driver mirrors this function's contract exactly
        # -- same status dict, same resume sentinel, same never-raise rule -- so the sweep
        # reassembles both kinds of unit by the same key. Imported here, not at module scope,
        # because the driver imports this module back.
        from pi_run.stages.tau2_runner import run_tau2_unit

        return run_tau2_unit(spec)

    started = time.time()
    t0 = time.perf_counter()

    suite = load_suite(spec.suite_id, spec.corpus_dir)
    arm = arm_table.get(spec.arm_id)
    cap = spec.budget_cap if spec.budget_cap is not None else arm.budget_cap

    # The ledger is created BEFORE the components because an LLM-backed arm needs it twice
    # over: the client debits it, and the policy reports a malformed generation through it as
    # a write-only Recorder. It cannot wait for the manifest, because the manifest has to
    # hash the prompts these very objects will render -- a prompt hash computed from anything
    # other than the object that renders it is a hash of a hope.
    ledger = BudgetLedger(cap=cap)

    # THE CLIENT IS BUILT BEFORE THE RETRIEVER. It used to be built after, so a tool-backed
    # suite's retriever was handed None and could not search at all: every Ask resolved to
    # nothing while the run reported success with a real reward. The fork path had the identical
    # defect. See tests/test_retriever_is_wired.py.
    llm = None
    if not arm.llm_free:
        # Constructed here, per process, holding this unit's ledger. Sharing one client
        # across units would cross-charge their budgets and silently break parity.
        from pi_run.cache import CachingClient, DiskCache
        from pinq_adapters.llm.litellm_client import MeteredClient

        # Read-through cache, or the reproducibility artifact is never written. Without the
        # wrapper every rerun re-bills the provider and `pi cache stats` reports zero, which
        # is exactly what "cache-replayable, not bitwise deterministic" depends on.
        llm = CachingClient(
            MeteredClient(ledger, temperature=sampling_temperature(spec.sampling)),
            DiskCache(spec.cache_root),
            ledger=ledger,
        )

    from pi_run.stages.tau2_runner import _wire_retriever

    retriever = _wire_retriever(
        suite, spec.task_id, llm=llm, llm_free=arm.llm_free, seed=spec.seed, recorder=ledger
    )

    try:
        components = arm_table.build(
            arm,
            llm=llm,
            retriever=retriever,
            recorder=ledger,
            questions=spec.questions or None,
            ask_counts=spec.ask_counts or None,
            # The action channel. Without it a tau2 sweep produces zero env_calls and
            # tau_reward is structurally 0 for every arm -- P1 then reads as a flat null
            # rather than as an unwired path. Suites with no tools return () and are
            # unaffected.
            tools=(suite.tool_schemas(spec.task_id) if hasattr(suite, "tool_schemas") else ()),
        )
    except Exception as exc:
        # A construction failure is a fact about THIS unit's configuration -- a replay arm
        # with no recorded questions, a missing model pin -- not about the sweep. It used to
        # propagate out of run_unit, through fut.result(), and kill every remaining unit:
        # plan() is task-major, so tier1_pilot died around unit 5 of 840 and four of its
        # seven arms (two of them kill switches) never ran at all.
        #
        # Firewall and reconcile failures are deliberately NOT caught here; they still
        # propagate, because a leak invalidates the whole sweep rather than one unit.
        return {
            "run_id": "",
            "key": list(spec.key),
            "suite_id": spec.suite_id,
            "task_id": spec.task_id,
            "arm_id": spec.arm_id,
            "seed": spec.seed,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "started_at": started,
            "wall_ms": int((time.perf_counter() - t0) * 1000),
            "code_version": spec.code_version,
            "dirty": spec.dirty,
            "pilot": spec.pilot,
        }
    inquirer, drafter, answerer = components.inquirer, components.drafter, components.answerer

    from pi_run.manifest import build_manifest, manifest_to_dict, template_id_of

    view = suite.view(spec.task_id)
    manifest = build_manifest(
        suite_id=spec.suite_id,
        task_id=spec.task_id,
        arm_id=spec.arm_id,
        policy_id=str(getattr(inquirer, "policy_id", spec.arm_id)),
        seed=spec.seed,
        corpus_hash=view.corpus_hash,
        # The DIRECTORY, so the scorer can find the public corpus this run read.
        corpus_dir=Path(spec.corpus_dir).name,
        # Treatment parameters, both inside semantic_hash. See RunManifest.
        k=spec.k,
        questions_hash=questions_hash(spec, inquirer),
        grid_sha256=spec.grid_sha256,
        grid_name=spec.grid_name,
        budget_cap=cap,
        max_turns=spec.max_turns,
        word_cap=view.word_cap,
        code_version=spec.code_version,
        dirty=spec.dirty,
        dirty_files=spec.dirty_files,
        # P1's preregistered test clusters at template_id: tau2 tasks derive from a
        # small number of templates, so resampling tasks would understate the SE.
        template_id=template_id_of(suite, spec.task_id),
        # Every template these components will render, by name -> sha256. semantic_hash
        # covers prompt_hashes, so editing a prompt changes run_id and two rows of one table
        # can never have come from two different prompts. The bare "answerer" key predates
        # the named templates and is kept so LLM-free run ids do not move.
        prompt_hashes={
            "answerer": str(getattr(answerer, "prompt_hash", "")),
            **prompt_hashes_of(inquirer, drafter, answerer),
        },
        upstream_pins={"suite_version": str(getattr(suite, "suite_version", ""))},
        # WHICH PROMPT SET PRODUCED THIS RUN. The field was declared on RunManifest, defaulted
        # to "v1", written into every manifest and compacted into a parquet column -- and
        # nothing ever set it, so the column was constant across every run ever made and rows
        # from different prompts pooled into one cell. `PI_PROMPT_OVERLAY` already changes the
        # templates; the label to GROUP BY was the missing half. Identity was never at risk --
        # prompt_hashes is inside semantic_hash -- but a 15-template hash is not a label.
        prompt_variant_id=promptlib.variant_id(),
        # Empty for an llm_free arm, and per-role for every other. Inside semantic_hash.
        pins=collect_pins(llm, (inquirer, drafter, answerer)) if llm is not None else {},
        pilot_flag=spec.pilot,
        exploratory=spec.exploratory,
        # A CEILING ARM IS GOLD-EXPOSED BY CONSTRUCTION, not by environment. `pi run` scrubs
        # PI_GOLD_ROOT from the process tree, so deriving this flag from the environment alone
        # made it structurally False for every run ever written -- and
        # `pi agg --assert-no-gold-exposed`, whose entire job is to keep gold_evidence and
        # oracle_vreq out of a confirmatory table, therefore passed by finding nothing to find.
        # The arm declares what it is; the environment can only add to that, never subtract.
        gold_exposed=arm.requires_gold,
        concurrency=spec.concurrency,
        # IN RUN IDENTITY, not decoration: candidate 0 reuses its parent's seed, so a branch
        # that omitted these would be written into the parent's own run directory.
        branch_of_run_id=spec.branch_of_run_id,
        branch_turn_idx=spec.branch_turn_idx,
        branch_seed=spec.branch_seed,
    )
    run_dir = Path(spec.runs_root) / manifest.run_id
    status_path = run_dir / STATUS

    if spec.resume and status_path.exists():
        # Idempotence, in one line. No row is rewritten, so compact cannot double-count.
        # read_status returns None for a torn file, so a kill -9 during the final write costs
        # this unit and not the rest of the sweep.
        prior = read_status(status_path)
        # ...but resume must NOT bake in a failure. A unit that died to a 429 or a dropped
        # tunnel is exactly the unit re-issuing the sweep is supposed to recover, and the
        # sweep docstring calls that "the correct recovery procedure". Skipping only
        # successes makes it true; the alternative was --no-resume, which re-runs all 15,600.
        if prior is not None and str(prior.get("status")) != "ok":
            prior = None
    else:
        prior = None

    if prior is not None:
        # Carry the PRIOR outcome forward. Without this a resumed error reports only
        # "resumed", and a sweep in which every unit had failed reads as a success -- the
        # next stage then produces an empty table instead of an error.
        return {
            **prior,
            "key": list(spec.key),
            "status": "resumed",
            "prior_status": prior.get("status"),
            "run_id": manifest.run_id,
        }

    status: dict[str, Any] = {
        "run_id": manifest.run_id,
        "key": list(spec.key),
        "suite_id": spec.suite_id,
        "task_id": spec.task_id,
        "arm_id": spec.arm_id,
        "seed": spec.seed,
        "started_at": started,
        "code_version": spec.code_version,
        "dirty": spec.dirty,
        "pilot": spec.pilot,
    }

    try:
        with _Watchdog(unit_timeout_s(spec.timeout_s)):
            traj = run_loop(
                view=view,
                inquirer=inquirer,
                retriever=retriever,
                drafter=drafter,
                answerer=answerer,
                ledger=ledger,
                actuator=suite.actuator(spec.task_id),
                max_turns=spec.max_turns,
                k=spec.k,
                seed=spec.seed,
                branch_at=spec.branch_turn_idx,
                branch_seed=spec.branch_seed,
            )
    except Exception as exc:  # noqa: BLE001 - one task's failure is data, not a sweep abort
        status.update(
            {
                # A watchdog kill is its own terminal status, not a generic error: "one unit
                # hung" and "one unit raised" call for different responses, and summarize()
                # has always had the counter for it.
                "status": "timeout" if isinstance(exc, UnitTimeout) else "error",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": time.time(),
                "wall_ms": int((time.perf_counter() - t0) * 1000),
                "usd_billed": _billed(ledger),
            }
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(run_dir / "manifest.json", json.dumps(manifest_to_dict(manifest), indent=2))
        write_atomic(status_path, json.dumps(status, indent=2, sort_keys=True))
        return status

    rec = reconcile(traj, ledger)
    if not rec["docs_ok"]:
        raise ReconcileError(
            f"{manifest.run_id}: ledger saw {rec['ledger_unique_docs']} unique docs but the "
            f"trajectory carries {rec['evidence_unique_docs']}. Unmetered nested retrieval."
        )
    if arm.llm_free and not rec["tokens_ok"]:
        raise ReconcileError(
            f"{manifest.run_id}: arm {arm.arm_id!r} declares llm_free but the ledger recorded "
            f"{rec['ledger_usage']}. Either the arm is mis-declared or a call went unmetered."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(run_dir / "manifest.json", json.dumps(manifest_to_dict(manifest), indent=2))
    _write_jsonl(run_dir / "turns.jsonl", turn_rows(traj))
    _write_jsonl(run_dir / "calls.jsonl", call_rows(traj))
    _write_jsonl(run_dir / "ledger.jsonl", ledger_rows(ledger))
    _write_jsonl(run_dir / "evidence.jsonl", evidence_rows(traj))
    write_atomic(run_dir / "outcome.json", json.dumps(outcome_dict(traj), indent=2, sort_keys=True))

    status.update(
        {
            "status": "ok",
            "finished_at": time.time(),
            "wall_ms": int((time.perf_counter() - t0) * 1000),
            "n_turns": len(traj.turns),
            "n_asks": traj.n_asks,
            "n_calls": len(traj.calls),
            # Lost cache write races: calls we paid for whose answer was discarded for the
            # canonical one (see pi_run.cache). Nonzero is not an error -- it is the count of
            # places where this run and the cache would have disagreed and now do not.
            "cache_races": int(getattr(llm, "races", 0) or 0),
            "n_evidence": len(traj.evidence),
            "stop_reason": traj.stop_reason,
            "usage": traj.usage.as_dict(),
            # The INVOICE, as opposed to usage["usd"] which charges cache hits as-if so
            # that two arms stay comparable however warm the cache was. The campaign spend
            # cap reads this one; billing a replayed sweep for tokens nobody sold would
            # cap a $0 rerun at grid A's $60.
            "usd_billed": _billed(ledger),
            "spent": dict(sorted(ledger.spent.items())),
            "unique_docs": ledger.unique_docs,
            "reconciled": rec["tokens_ok"] and rec["docs_ok"],
            "reconcile": rec,
        }
    )
    # LAST, and atomically. Everything above is on disk before this file exists, which is
    # what makes --resume a single exists() check with no repair path.
    write_atomic(status_path, json.dumps(status, indent=2, sort_keys=True))
    return status


def with_seed(spec: UnitSpec, seed: int) -> UnitSpec:
    return replace(spec, seed=seed)
