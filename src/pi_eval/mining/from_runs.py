"""S0, driven from THIS repository's own recorded rollouts.

`pool.py` says what a trace pool must BE — a factorial over (model family x policy form x
retrieval variant), successes and failures kept apart — and says nothing about where the
traces come from. This module is the only place that maps `runs/<run_id>/` onto it, so the
question "which generators actually stand behind this mined node" is answered by reading one
file rather than by trusting a pipeline.

THE THREE MAPPINGS, AND WHY EACH IS EXPLICIT RATHER THAN INFERRED

  * MODEL FAMILY comes from `calls.jsonl`, not from the manifest. `RunManifest.pins` is
    populated by whoever builds the manifest, and `pi_run.worker` does not currently pass
    one, so `model_pin_hash` is a constant and two different models produce the SAME run_id.
    `calls.jsonl` records the model and provider of every dispatched call, so it is the only
    place in a recorded run where the model that generated the questions is actually
    written down. A run with no calls at all is `llm_free` — a real, distinct label, and
    NOT a second family: a pool of nothing but LLM-free arms has one generator family and
    `pool_is_admissible` is supposed to reject it.

  * POLICY FORM is a table keyed on `policy_id`, not a guess. An unmapped policy is
    REPORTED, never bucketed into a default: quietly filing an unknown policy under "chain"
    would let one generator's habits masquerade as two forms and defeat the exact
    contamination control the factorial exists to provide.

  * RETRIEVAL VARIANT is the suite's retriever, which is a property of the suite and not of
    the run, so it is passed in by the caller (`bm25` for tau2, `token` for synth) and
    recorded on the cell rather than being invented here.

WHAT A CANDIDATE IS. Every ASK a policy issued is one candidate need proposition, and the
units retrieved for that ask are its `ev_uids`. That is the weakest possible reading of a
trace and it is the right one: it claims only "this generator asked this, and got these
documents back", which is exactly the correlational statement S3 is allowed to make before
the ablation in S4 turns anything into a causal claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .canon import Candidate
from .pool import Cell, PolicyForm, Trace, TracePool, is_success

# policy_id -> policy form. The `policy_id` a run records is the class attribute on the
# Inquirer that produced it (pinq_expt.policies / pinq_expt.fakes), so this table is keyed
# on the thing the run actually wrote rather than on the arm id, which several arms share.
POLICY_FORMS: dict[str, PolicyForm] = {
    # one question at a time, each conditioned on what the last one returned
    "chain": "chain",
    "verbatim": "chain",
    "inquirer_prompted": "chain",
    "self_ask": "chain",
    # questions drawn only from what x names, with no conditioning on evidence
    "depth1": "breadth",
    "inquirer_noevidence": "breadth",
    # the whole question list is produced up front, then walked
    "inquirer_depth1": "plan_then_act",
    "checklist": "plan_then_act",
    "parallel_replay": "plan_then_act",
    "random_q": "plan_then_act",
    "gold_evidence": "plan_then_act",
    "oracle_vreq": "plan_then_act",
    # reasoning and retrieval interleaved, with no explicit separate question
    "ircot": "react",
    "self_inquire": "react",
}

# A run that never asks contributes no candidate and no ordering, so it is not a generator.
NON_GENERATOR_POLICIES = frozenset({"never_ask"})

# substring -> family label. A FAMILY, not a model: two pins inside one family are one
# generator for contamination purposes, which is the whole point of the >= 2 families rule.
FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("gpt-oss", "gpt_oss"),
    ("gemma", "gemma"),
    ("gemini", "gemini"),
    ("claude", "claude"),
    ("llama", "llama"),
    ("qwen", "qwen"),
    ("mistral", "mistral"),
    ("deepseek", "deepseek"),
    ("phi-", "phi"),
    ("gpt-", "openai_gpt"),
    ("o1", "openai_gpt"),
    ("o3", "openai_gpt"),
    ("o4", "openai_gpt"),
)

LLM_FREE_FAMILY = "llm_free"
UNKNOWN_FAMILY = "unknown"


class UnmappedPolicy(KeyError):
    """A recorded policy_id with no declared policy form.

    Loud rather than defaulted: a silent default would collapse two generators into one and
    the factorial would report a diversity it does not have.
    """


def family_of(model_ids: Iterable[str]) -> str:
    """Family label for the models a single run dispatched to.

    A run that used two families is labelled `mixed:<a>+<b>` rather than being assigned to
    either one. That is rare (it means the role pins disagreed) and it must not silently
    count as membership of either family.
    """
    fams: set[str] = set()
    for mid in model_ids:
        low = str(mid).lower()
        for needle, label in FAMILY_PATTERNS:
            if needle in low:
                fams.add(label)
                break
        else:
            fams.add(UNKNOWN_FAMILY)
    if not fams:
        return LLM_FREE_FAMILY
    if len(fams) == 1:
        return next(iter(fams))
    return "mixed:" + "+".join(sorted(fams))


@dataclass(frozen=True, slots=True)
class RecordedRun:
    """One `runs/<run_id>/` directory, read into the fields S0 needs and nothing else."""

    run_id: str
    suite_id: str
    task_id: str
    arm_id: str
    policy_id: str
    seed: int
    status: str
    model_family: str
    n_calls: int
    turn_uids: tuple[tuple[int, tuple[str, ...]], ...]
    asks: tuple[tuple[int, str, tuple[str, ...]], ...]  # (turn_idx, question, uids)

    @property
    def is_generator(self) -> bool:
        """A run that asked nothing cannot support a candidate or an ordering."""
        return bool(self.asks) and self.policy_id not in NON_GENERATOR_POLICIES

    def policy_form(self) -> PolicyForm:
        try:
            return POLICY_FORMS[self.policy_id]
        except KeyError:
            raise UnmappedPolicy(
                f"policy_id {self.policy_id!r} (run {self.run_id}, arm {self.arm_id!r}) has no "
                f"declared policy form. Add it to pi_eval.mining.from_runs.POLICY_FORMS; do not "
                f"let it default, or one generator will be counted as two."
            ) from None

    def cell(self, retrieval_variant: str) -> Cell:
        return Cell(self.model_family, self.policy_form(), retrieval_variant)

    def trace(self, *, retrieval_variant: str, score: float, success: bool) -> Trace:
        return Trace(
            trace_id=self.run_id,
            suite_id=self.suite_id,
            task_id=self.task_id,
            cell=self.cell(retrieval_variant),
            seed=self.seed,
            success=success,
            native_score=score,
            retrieved_uids=tuple(u for _, uids in self.turn_uids for u in uids),
            turn_uids=self.turn_uids,
        )

    def candidates(self, *, retrieval_variant: str) -> list[Candidate]:
        cell = self.cell(retrieval_variant)
        # A deterministic LLM-free policy extracts its questions from the environment by
        # rule, so its candidates are `mechanical`; anything a model wrote is weak
        # supervision and says so.
        provenance = "mechanical" if self.model_family == LLM_FREE_FAMILY else "llm_elicited"
        return [
            Candidate(
                text=question,
                trace_id=self.run_id,
                cell_id=cell.cell_id,
                model_family=cell.model_family,
                policy_form=cell.policy_form,
                ev_uids=uids,
                turn_idx=turn_idx,
                provenance=provenance,
            )
            for turn_idx, question, uids in self.asks
            if question.strip()
        ]


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def read_run(run_dir: Path) -> RecordedRun | None:
    """Read one run directory, or None if it is not a completed rollout.

    `status.json` is written LAST by the worker, so its absence means the unit was killed
    mid-write and has no trustworthy artifacts. An errored unit is skipped for the same
    reason: it has a manifest but no turns.
    """
    manifest_path = run_dir / "manifest.json"
    status_path = run_dir / "status.json"
    if not manifest_path.exists() or not status_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    state = str(status.get("prior_status") or status.get("status") or "")
    if state not in ("ok", "resumed"):
        return None

    turns = sorted(_jsonl(run_dir / "turns.jsonl"), key=lambda r: int(r.get("turn_idx", 0)))
    turn_uids: list[tuple[int, tuple[str, ...]]] = []
    asks: list[tuple[int, str, tuple[str, ...]]] = []
    for t in turns:
        idx = int(t.get("turn_idx", 0))
        uids = tuple(str(u) for u in (t.get("retrieved_uids") or ()))
        turn_uids.append((idx, uids))
        if t.get("action_kind") == "ask":
            asks.append((idx, str(t.get("question") or ""), uids))

    calls = _jsonl(run_dir / "calls.jsonl")
    return RecordedRun(
        run_id=str(manifest["run_id"]),
        suite_id=str(manifest["suite_id"]),
        task_id=str(manifest["task_id"]),
        arm_id=str(manifest["arm_id"]),
        policy_id=str(manifest.get("policy_id") or manifest["arm_id"]),
        seed=int(manifest.get("seed", 0)),
        status=state,
        # THE INQUIRER'S MODEL, NOT THE RUN'S. The admissibility gate demands >= 2 model
        # families before a candidate need may be promoted, because "one generator's habits
        # would otherwise manufacture a phantom need". The generator of a need is whatever
        # produced the ASK -- the Inquirer. Deriving the label from EVERY call folded in the
        # Drafter's and the frozen Answerer's models too, so a single Inquirer paired with a
        # differently-pinned Answerer already looked like two families. `tier1_trained` runs
        # exactly that shape on purpose: a trained Inquirer against the same frozen Drafter and
        # Answerer as the prompted arm.
        #
        # A run whose Inquirer made no calls generated no question, so it contributes no
        # candidate; `family_of` already labels an empty iterable LLM-free, which is the right
        # answer rather than borrowing another role's family.
        model_family=family_of(
            str(c.get("model", ""))
            for c in calls
            if c.get("model") and str(c.get("actor", "")) == "inquirer"
        ),
        n_calls=len(calls),
        turn_uids=tuple(turn_uids),
        asks=tuple(asks),
    )


def read_runs(
    runs_root: Path, *, suite_id: str, task_ids: Sequence[str] | None = None
) -> list[RecordedRun]:
    """Every completed run of one suite, oldest run_id first (a stable, content-derived order)."""
    keep = set(task_ids) if task_ids else None
    out: list[RecordedRun] = []
    for d in sorted(Path(runs_root).iterdir()):
        if not d.is_dir():
            continue
        run = read_run(d)
        if run is None or run.suite_id != suite_id:
            continue
        if keep is not None and run.task_id not in keep:
            continue
        out.append(run)
    return out


@dataclass(frozen=True, slots=True)
class PoolBuild:
    """A pool plus the accounting for everything that did NOT reach it.

    Skips are DATA. A task whose pool is thin because half its runs errored is a different
    situation from one that is thin because only one generator was ever run, and a driver
    that reported only the pool would make those two indistinguishable.
    """

    task_id: str
    pool: TracePool
    candidates: tuple[Candidate, ...]
    n_runs: int
    n_unscored: int
    n_non_generator: int


def build_pools(
    runs: Sequence[RecordedRun],
    scores: Mapping[str, float],
    *,
    suite_id: str,
    retrieval_variant: str,
    success_threshold: float | None = None,
) -> list[PoolBuild]:
    """Group completed runs into one pool per task.

    A run with no score is EXCLUDED, not defaulted to failure. `pool.failures` is the
    negative control that decides whether a need is diagnostic at all, so filling it with
    runs that were never scored would bias every lift statistic in the direction of
    "everything is diagnostic".
    """
    by_task: dict[str, list[RecordedRun]] = {}
    for r in runs:
        by_task.setdefault(r.task_id, []).append(r)

    out: list[PoolBuild] = []
    for task_id, group in sorted(by_task.items()):
        pool = TracePool(suite_id=suite_id, task_id=task_id)
        cands: list[Candidate] = []
        unscored = non_gen = 0
        for run in group:
            if not run.is_generator:
                non_gen += 1
                continue
            if run.run_id not in scores:
                unscored += 1
                continue
            score = float(scores[run.run_id])
            ok = is_success(suite_id, score, override=success_threshold)
            trace = run.trace(retrieval_variant=retrieval_variant, score=score, success=ok)
            (pool.successes if ok else pool.failures).append(trace)
            if ok:
                cands.extend(run.candidates(retrieval_variant=retrieval_variant))
        out.append(
            PoolBuild(
                task_id=task_id,
                pool=pool,
                candidates=tuple(cands),
                n_runs=len(group),
                n_unscored=unscored,
                n_non_generator=non_gen,
            )
        )
    return out


def corpus_documents(corpus_dir: Path, task_id: str) -> list[dict[str, str]]:
    """The task's public documents, as `partition.discoverability_of` wants them.

    Reads `tasks.jsonl` from the PUBLIC corpus. The discoverability cut asks "could an
    autonomous inquirer ever retrieve this?", and the honest denominator for that question
    is exactly what the adapter can see — never the gold side.
    """
    path = Path(corpus_dir) / "tasks.jsonl"
    if not path.exists():
        return []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if str(rec.get("id")) != str(task_id):
            continue
        # Two corpus shapes are in use: `docs` (synth) and `paragraphs` (musique, strategyqa,
        # wiki2). Both carry the text under a different key, so both are accepted here rather
        # than in three copies of a caller.
        rows = rec.get("docs") or rec.get("paragraphs") or []
        out = []
        for i, d in enumerate(rows):
            out.append(
                {
                    "id": str(d.get("doc_id") or d.get("idx") or i),
                    "content": str(d.get("text") or d.get("content") or ""),
                }
            )
        return out
    return []
