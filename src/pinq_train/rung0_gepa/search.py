"""Rung 0: reflective prompt evolution over the Inquirer template, scored through the seam.

RUNG 0 IS NOT OPTIONAL AND IT IS NOT AN OPTIMISATION FOR ITS OWN SAKE. It produces the
Inquirer prompt that the UNTRAINED and FRONTIER families both run. Without it,
"trained beats prompted" reduces to "our fine-tune beat a prompt we did not try very hard on",
which is a prompt contest with a fine-tune attached. With it, all three families share one
optimised prompt and the only thing that differs between them is
`ModelPin.base_url_sha`/the model pin -- which is the comparison the paper claims to make.

NO GPU. Every rollout happens on the far side of `POST /rollout`; every score comes back from
`POST /score`. This module holds no model, allocates no tensor and imports no training
library. It is the one rung that must ship, and it must be runnable from a laptop.

WHY A PARETO FRONT RATHER THAN AN ARGMAX ON THE MEAN. Averaging reward over tasks throws away
the fact that different prompts win on different tasks: a mutation that fixes ten hard tasks
and costs a hair on forty easy ones loses on the mean and is exactly the mutation worth
keeping. GEPA's answer, adopted here, is to keep every candidate that is best on at least ONE
task and to sample parents in proportion to how many tasks they own. `mean` is used only to
break ties inside the front, and only over a HELD-OUT slice, so the winner is not the
candidate that overfitted the tasks the search looked at.

WHY THE CANDIDATE REACHES THE POLICY AS A FILE. `RolloutRequest.prompt_overlay` points
`pinq.promptlib` at a directory of candidate templates, so a candidate is rendered by the same
policy class, through the same loop, as the shipped template -- and `promptlib.sha()` hashes
the bytes that were actually loaded, so a candidate rollout carries different
`prompt_hashes` and therefore a different `run_id`. A candidate can never be mistaken for a
baseline rollout, which is the only property that makes running the search against the same
`runs/` directory safe.

WHAT THE SEARCH IS ALLOWED TO SEE. The seam refuses any task outside `train`
(`ScoreRequest.split_assert`), so the search is structurally incapable of touching dev or
test. Selecting the winner on dev would be legitimate practice and is DELIBERATELY NOT DONE
HERE: it would require a scoring endpoint that answers for dev, and such an endpoint, once it
exists, is reachable by every later rung too. The winner is validated on dev afterwards
through the ordinary gold-side eval path (`pi run` + `pi score`), which no trainer can call.
"""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from pinq.ids import h, short
from pinq.wire import RolloutRequest
from pinq_train.client import SeamClient, ServerRefused
from pinq_train.reward import DEFAULT_WEIGHTS, RewardWeights, reward_of
from pinq_train.rung0_gepa.config import Rung0Config

# gpt-oss and its relatives bill REASONING tokens inside max_tokens: a request asking for 700
# content tokens can come back with 700 reasoning tokens and an EMPTY content field. Every
# completion this module asks for is therefore multiplied by this headroom, exactly as
# pinq_adapters.llm.litellm_client does for the roles.
REASONING_HEADROOM = 3

# What the reflector is asked to produce. Kept small: a mutation that rewrites the output
# contract produces a prompt whose actions no longer parse, and every one of its rollouts is
# scored as malformed rather than as a worse question.
REFLECTION_MAX_TOKENS = 1200


class SearchRefused(RuntimeError):
    """The search will not start. Always a configuration or a budget decision, never a bug."""


# --------------------------------------------------------------------------- candidates


@dataclass(frozen=True, slots=True)
class Candidate:
    """One prompt, with its ancestry.

    `parent` and `generation` are stored because the interesting artifact of a prompt search
    is not the winner alone but the LINEAGE: which mutation bought which tasks. A winner with
    no ancestry is a string somebody found.
    """

    text: str
    generation: int = 0
    parent: str = ""
    note: str = ""

    @property
    def cid(self) -> str:
        return h("rung0-candidate", self.text)

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "cid": self.cid}


@dataclass(frozen=True, slots=True)
class TaskKey:
    suite_id: str
    task_id: str
    seed: int = 0

    def __str__(self) -> str:  # used as the per-task key in the Pareto tables
        return f"{self.suite_id}/{self.task_id}#{self.seed}"


@dataclass(frozen=True, slots=True)
class TaskResult:
    """One (candidate, task) evaluation: the number, and enough text to reflect on."""

    key: str
    reward: float
    run_id: str = ""
    n_asks: int = 0
    stop_reason: str = ""
    trace: str = ""
    error: str = ""


# --------------------------------------------------------------------------- Pareto


def pareto_front(scores: Mapping[str, Mapping[str, float]]) -> tuple[str, ...]:
    """Candidate ids that are best on at least one task.

    Ties keep EVERY tied candidate rather than the first one seen: dropping ties would make
    the front depend on dict ordering, and a search whose survivors depend on iteration order
    is not reproducible from its seed.
    """
    owners: set[str] = set()
    tasks: set[str] = set()
    for per_task in scores.values():
        tasks |= set(per_task)
    for task in sorted(tasks):
        best: float | None = None
        for cid in sorted(scores):
            v = scores[cid].get(task)
            if v is None:
                continue
            if best is None or v > best:
                best = v
        if best is None:
            continue
        owners |= {cid for cid in scores if scores[cid].get(task) == best}
    return tuple(sorted(owners))


def ownership(scores: Mapping[str, Mapping[str, float]]) -> dict[str, int]:
    """How many tasks each candidate is (jointly) best on. The parent-sampling weight."""
    counts = {cid: 0 for cid in scores}
    tasks: set[str] = set()
    for per_task in scores.values():
        tasks |= set(per_task)
    for task in sorted(tasks):
        vals = [(cid, scores[cid][task]) for cid in sorted(scores) if task in scores[cid]]
        if not vals:
            continue
        best = max(v for _, v in vals)
        for cid, v in vals:
            if v == best:
                counts[cid] += 1
    return counts


def sample_parent(
    front: Sequence[str], scores: Mapping[str, Mapping[str, float]], rng: random.Random
) -> str:
    """Sample from the front in proportion to how many tasks a candidate owns."""
    if not front:
        raise SearchRefused("empty Pareto front: nothing was evaluated")
    counts = ownership(scores)
    weights = [max(1, counts.get(cid, 0)) for cid in front]
    return rng.choices(list(front), weights=weights, k=1)[0]


def mean_reward(per_task: Mapping[str, float]) -> float:
    return sum(per_task.values()) / len(per_task) if per_task else float("-inf")


# --------------------------------------------------------------------------- evaluation


class Evaluator(Protocol):
    """Anything that can price one candidate on one task. A protocol so the search is
    testable with a deterministic fake and never needs a server to be unit-tested."""

    def __call__(self, candidate: Candidate, key: TaskKey) -> TaskResult: ...


@dataclass
class SeamEvaluator:
    """The real evaluator: overlay the candidate, roll out, score, apply the reward weights.

    The overlay directory is written ONCE PER CANDIDATE and reused across its tasks, because
    the file's bytes are what `promptlib.sha()` hashes into the run id -- rewriting it between
    tasks would be a no-op at best and a race at worst when two evaluations overlap.
    """

    client: SeamClient
    cfg: Rung0Config
    overlay_root: Path
    weights: RewardWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)

    def overlay_for(self, candidate: Candidate) -> Path:
        d = Path(self.overlay_root) / short(candidate.cid)
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{self.cfg.base_prompt}.txt"
        if not f.exists() or f.read_text(encoding="utf-8") != candidate.text:
            f.write_text(candidate.text, encoding="utf-8")
        return d

    def __call__(self, candidate: Candidate, key: TaskKey) -> TaskResult:
        overlay = self.overlay_for(candidate)
        req = RolloutRequest(
            task_id=key.task_id,
            suite_id=key.suite_id,
            seed=key.seed,
            max_turns=self.cfg.max_turns,
            arm_id=self.cfg.arm_id,
            policy_model=self.cfg.policy_model,
            policy_base_url=self.cfg.policy_base_url,
            k=self.cfg.k_for(key.suite_id),
            budget_cap=self.cfg.budget_cap,
            prompt_overlay=str(overlay),
        )
        try:
            roll = self.client.rollout(req)
        except ServerRefused as exc:
            # A refusal is a policy decision (split, frozen roles). Recording it as a failed
            # task rather than crashing the search keeps one bad task from ending a six-hour
            # run, and the reason survives into the result row so it is not silent.
            return TaskResult(key=str(key), reward=float("-inf"), error=str(exc))
        if not rollout_succeeded(roll):
            return TaskResult(key=str(key), reward=float("-inf"), error=roll.error or roll.status)
        resp = self.client.score(roll.run_id, mode="prefix_ladder")
        breakdown = reward_of(resp, self.weights)
        return TaskResult(
            key=str(key),
            reward=breakdown.total,
            run_id=roll.run_id,
            n_asks=roll.n_asks,
            stop_reason=roll.stop_reason,
            trace=summarise(roll.run_id, resp, breakdown.total),
        )


def summarise(run_id: str, resp: object, reward: float) -> str:
    """A compact, textual account of one episode for the reflector to read.

    Deliberately built from the MEASURED record (turn count, retrieved/new counts, the
    potential ladder) rather than from the policy's own narration. A reflector shown the
    policy's rationale would be optimising the story the policy tells about itself.
    """
    turns = getattr(resp, "turns", ()) or ()
    lines = [f"episode {short(run_id)}  reward={reward:.4f}"]
    for t in turns:
        lines.append(
            f"  t{t.turn_idx}: retrieved={t.n_retrieved} new={t.n_new} "
            f"phi_loo={'' if t.phi_loo is None else format(t.phi_loo, '.3f')}"
        )
    pot = getattr(resp, "potential", ()) or ()
    if pot:
        lines.append("  coverage ladder: " + " -> ".join(f"{p:.3f}" for p in pot))
    lines.append(f"  stop_reason={getattr(resp, 'stop_reason', '')!r}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- reflection


class Reflector(Protocol):
    """Proposes a mutated prompt from a parent and a minibatch of its traces."""

    def __call__(self, parent: Candidate, traces: Sequence[TaskResult]) -> str: ...


REFLECTION_INSTRUCTION = """\
You are improving the SYSTEM PROMPT of an "Inquirer": a policy that reads a task, the evidence
retrieved so far and its own question history, then emits EITHER one internal question to a
Drafter OR a decision to stop. It never talks to a user.

Below is the current prompt, then measured traces of episodes it produced. Each trace reports
what the RETRIEVER returned per turn (retrieved / new documents) and the evidence-coverage
ladder. Higher coverage earlier is better; every question costs a retrieval call, so questions
that return nothing new are worse than not asking.

Rewrite the prompt so it produces better questions. HARD CONSTRAINTS -- a rewrite that breaks
any of these is discarded:
  * keep EVERY {{placeholder}} that appears in the current prompt, spelled identically;
  * keep the output contract (the exact JSON shape the policy must emit) byte-identical;
  * never mention a budget, a cap, a number of turns remaining, or the arm/condition name;
  * do not add examples that name specific datasets or tasks.

Reply with the complete rewritten prompt and nothing else. No preamble, no code fence.
"""


# A SEARCH MUST SURVIVE BEING STOPPED. Rollouts already do -- `status.json` is the sentinel and
# is written LAST, so a unit killed mid-write leaves no status file and is simply redone. The
# SEARCH state did not: the Pareto front, the per-candidate scores and the candidate pool lived
# only in memory.
#
# That makes a restart worse than it looks. The seed re-scores for free (its rollouts are
# "resumed"), but the reflector proposes FRESH mutations on the way back up, and a different
# mutation is a different prompt, a different `prompt_hashes` and a different run_id -- so every
# generation past the first is paid for a second time.
CHECKPOINT_NAME = "rung0.checkpoint.json"


def save_checkpoint(
    out_dir: str | Path,
    *,
    pool: Mapping[str, "Candidate"],
    scores: Mapping[str, Mapping[str, float]],
    generation: int,
) -> Path:
    """Write the search state. ATOMICALLY, because being killed mid-write is the case this
    exists for -- the write must not itself be what corrupts it.

    Stores candidate TEXTS, not just their ids: a `cid` names a text nothing else keeps, so an
    id alone would name something nothing can reconstruct, evaluate or ship as the winner.
    """
    import json as _json
    import os as _os
    import tempfile

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "generation": int(generation),
        "scores": {k: dict(v) for k, v in scores.items()},
        "pool": [
            {"text": c.text, "generation": c.generation, "parent": c.parent, "note": c.note}
            for c in pool.values()
        ],
    }
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".ckpt.", suffix=".tmp")
    try:
        with _os.fdopen(fd, "w", encoding="utf-8") as fh:
            _json.dump(payload, fh)
        _os.replace(tmp, d / CHECKPOINT_NAME)
    except BaseException:
        try:
            _os.unlink(tmp)
        except OSError:
            pass
        raise
    return d / CHECKPOINT_NAME


def load_checkpoint(out_dir: str | Path) -> dict | None:
    """The saved state, or None. A missing checkpoint is a first run, not an error; a CORRUPT
    one starts afresh rather than making the search unstartable -- the same direction
    `pi_run.cache.get` takes for a torn entry."""
    import json as _json

    p = Path(out_dir) / CHECKPOINT_NAME
    try:
        raw = _json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        pool = {}
        for c in raw.get("pool") or ():
            cand = Candidate(
                text=str(c["text"]),
                generation=int(c.get("generation", 0)),
                parent=c.get("parent"),
                note=str(c.get("note", "")),
            )
            pool[cand.cid] = cand
        return {
            "generation": int(raw.get("generation", 0)),
            "scores": {k: dict(v) for k, v in (raw.get("scores") or {}).items()},
            "pool": pool,
        }
    except (KeyError, TypeError, ValueError):
        return None


# THE REFLECTOR WAS THE ONLY LLM CALLER WITH NO RETRY, and one 429 ended a multi-hour search.
#
# `litellm_client` runs an 8-attempt ladder spanning ~148s (three 60s rate-limit windows); the
# tau2 user simulator was given num_retries=8 after a 429 killed a dialogue unit. The reflector
# posted once and raised `SearchRefused`, which aborts `run_search` entirely -- observed after
# two completed generations of real work (gen 2 reached mean=0.8763), all of it discarded.
#
# Over a multi-hour search sharing a proxy, a 429 is expected rather than exceptional. Matched
# to `litellm_client`'s attempt count so both behave the same under one outage.
REFLECT_MAX_ATTEMPTS = 8
REFLECT_INITIAL_WAIT_S = 4.0
REFLECT_MAX_WAIT_S = 30.0

# Retried: rate limits and server-side faults, which pass. NOT retried: 4xx other than 429 --
# a bad key, a wrong URL or a missing model will not fix itself, and spending two minutes on it
# only delays a message the operator needs immediately. The 403 from a base_url missing /v1 was
# exactly that case.
_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})


def is_transient_http(exc: BaseException) -> bool:
    """Is this worth trying again?"""
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):
        return int(getattr(exc, "code", 0)) in _TRANSIENT_STATUS
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, OSError))


def call_with_retry(fn, *, sleep=None):
    """Run `fn`, retrying transient failures with exponential backoff.

    Re-raises the LAST exception when the ladder is exhausted rather than wrapping it, so the
    caller's own error message still names the status that actually happened.
    """
    import time as _time

    nap = sleep or _time.sleep
    wait = REFLECT_INITIAL_WAIT_S
    last: BaseException | None = None
    for attempt in range(REFLECT_MAX_ATTEMPTS):
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            last = exc
            if not is_transient_http(exc) or attempt == REFLECT_MAX_ATTEMPTS - 1:
                raise
            nap(wait)
            wait = min(wait * 2, REFLECT_MAX_WAIT_S)
    raise last  # pragma: no cover - loop always returns or raises


# A ROLLOUT THAT ALREADY RAN IS A ROLLOUT THAT SUCCEEDED. `run_unit` returns "resumed" only
# when a run whose status was "ok" already exists for that run_id -- a failed prior is discarded
# and re-run, so that "resume must NOT bake in a failure". So "resumed" means the artifacts are
# on disk and scoreable.
#
# Treating it as a failure was fatal and scaled with how much work had already been done: the
# seed sits in the Pareto front and is re-scored every generation, so its first evaluation was
# "ok" and every later one "resumed" -> -inf; and on a repository that already holds runs for
# those tasks, even the FIRST evaluation was "resumed" -> -inf. Observed:
# `gen 0  seed 85ae43a5b538 mean=-inf` with every underlying run `status: ok` on disk.
SUCCESSFUL_ROLLOUT_STATUSES = frozenset({"ok", "resumed"})


def rollout_succeeded(roll: object) -> bool:
    """Did this rollout leave scoreable artifacts behind?"""
    return str(getattr(roll, "status", "")) in SUCCESSFUL_ROLLOUT_STATUSES


def openai_base(base_url: str) -> str:
    """A litellm proxy root -> its OpenAI-compatible route prefix.

    `OpenAIReflector` posts to `{base_url}/chat/completions`, and `LITELLM_BASE_URL` is the
    proxy ROOT -- the OpenAI routes live under `/v1`. Defaulting to the root produced
    `https://host/chat/completions` and HTTP 403, silently, until the first reflection: the
    seed generation had already been rolled out and paid for when the search died.

    Idempotent, and it leaves a non-/v1 path alone: a proxy mounted under a custom prefix is
    not ours to rewrite. Empty stays empty, so the caller's own "reflector is unreachable"
    message fires instead of a confusing request to "/v1".
    """
    u = (base_url or "").rstrip("/")
    if not u:
        return ""
    tail = u.rsplit("/", 1)[-1]
    return u if ("/" in u[8:] and tail != u) else f"{u}/v1"


def proxy_model_name(model: str) -> str:
    """Strip litellm's leading PROVIDER segment for a proxy addressed directly over HTTP.

    `openai/aws/gpt-oss-120b` tells the litellm CLIENT which provider to route through. A raw
    POST to the proxy wants the deployment name it knows -- `aws/gpt-oss-120b` -- and rejects
    the prefixed form. Only the leading `openai/` is removed, so a model whose own name
    contains "openai" survives.
    """
    m = (model or "").strip()
    return m[len("openai/") :] if m.startswith("openai/") else m


@dataclass
class OpenAIReflector:
    """Reflection through an OpenAI-compatible endpoint, over stdlib urllib.

    Same rationale as `pinq_train.client`: this package has to import cleanly in a container
    whose dependency set is dictated by torch and vllm pins, so it does not take an SDK to
    post one JSON body. It is also the same shape of endpoint the policy itself is addressed
    by, which keeps "one code path for a checkpoint, an open model and a frontier API" true
    for the reflector too.
    """

    base_url: str
    model: str
    api_key: str = "local"
    timeout: float = 180.0
    max_tokens: int = REFLECTION_MAX_TOKENS
    temperature: float = 1.0

    def __call__(self, parent: Candidate, traces: Sequence[TaskResult]) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": REFLECTION_INSTRUCTION},
                {
                    "role": "user",
                    "content": (
                        "CURRENT PROMPT\n-----\n"
                        + parent.text
                        + "\n-----\n\nMEASURED TRACES\n-----\n"
                        + "\n\n".join(t.trace or f"{t.key}: {t.error}" for t in traces)
                    ),
                },
            ],
            # Reasoning tokens are billed INSIDE max_tokens on gpt-oss: without the headroom a
            # request for 1,200 content tokens returns 1,200 reasoning tokens and EMPTY
            # content, and the mutation silently becomes "no change".
            "max_tokens": self.max_tokens * REASONING_HEADROOM,
            "temperature": self.temperature,
        }
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        def _post() -> dict:
            with urllib.request.urlopen(req, timeout=self.timeout) as fh:  # noqa: S310
                return json.loads(fh.read().decode("utf-8"))

        try:
            # RETRIED, because one 429 used to end the whole search. A multi-hour run sharing a
            # proxy meets rate limits as a matter of course; only transient statuses are
            # retried, so a bad key or URL still fails immediately with its own message.
            payload = call_with_retry(_post)
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError) as exc:
            raise SearchRefused(f"reflector at {self.base_url} is unreachable: {exc}") from None
        choices = payload.get("choices") or []
        return str((choices[0].get("message") or {}).get("content", "") if choices else "")


def placeholders_of(text: str) -> frozenset[str]:
    import re

    return frozenset(re.findall(r"\{\{(\w+)\}\}", text))


def _budget_from(cfg: "Rung0Config") -> int | None:
    """`cfg.max_prompt_tokens`: "family" derives it, an int is used as-is, None disables it."""
    want = getattr(cfg, "max_prompt_tokens", "family")
    if want is None:
        return None
    if isinstance(want, int):
        return want
    if want != "family":
        raise SearchRefused(f"max_prompt_tokens={want!r}: expected 'family', an int, or None")
    return _parity_budget_for(cfg.base_prompt)


def _parity_budget_for(prompt_name: str) -> int | None:
    """The prompt's parity budget, or None when it belongs to no family."""
    from pinq import promptlib

    for family, members in promptlib.PARITY_FAMILIES.items():
        if prompt_name in members:
            return promptlib.parity_budget(prompt_name, family)
    return None


def accept_mutation(
    parent: Candidate, proposed: str, *, max_tokens: int | None = None
) -> tuple[bool, str]:
    """Structural admissibility of a proposed prompt. Reasons, never a bare bool.

    A prompt that drops a placeholder renders with an EMPTY evidence or history section, which
    is a silent ablation nobody declared -- `promptlib.render` would not even raise, because
    the hole is gone rather than unresolved. A prompt that gains one raises
    `UnresolvedPlaceholder` on the first rollout and burns a whole generation. Both are
    refused here, cheaply, before any tokens are spent.

    `max_tokens` is the PARITY BUDGET, and it is the constraint the first rung-0 search did not
    have. Arms are compared slot by slot within 10% (`promptlib.PARITY_FAMILIES`), so a prompt
    that outgrows its family stops being comparable to the ablations it is measured against.
    Unconstrained, the 2026-09-01 search spent its search budget on length: of 25 scored
    candidates exactly one fit the cap -- the seed -- and its 522-token winner turned out to
    beat the 425-token seed by no more than padding the seed with inert text did
    (winner - lenctl = +0.003, p=0.96, n=36; see artifacts/rung0/lenctl/RESULT.md).

    Left None, length is not a reason to refuse: callers that are not feeding a parity family
    keep the old behaviour.
    """
    from pinq.promptlib import FORBIDDEN_PLACEHOLDERS, count_tokens

    text = (proposed or "").strip()
    if not text:
        return False, "empty proposal (a reasoning-token overrun returns empty content)"
    want, got = placeholders_of(parent.text), placeholders_of(text)
    if want != got:
        return (
            False,
            f"placeholders changed: missing {sorted(want - got)}, added {sorted(got - want)}",
        )
    banned = sorted(got & FORBIDDEN_PLACEHOLDERS)
    if banned:
        return False, f"forbidden placeholders {banned}: a policy must not be told its cap"
    if text == parent.text:
        return False, "identical to parent"
    if max_tokens is not None:
        n = count_tokens(text)
        if n > max_tokens:
            return False, f"breaks parity: {n} tokens over the family budget of {max_tokens}"
    return True, "ok"


# --------------------------------------------------------------------------- the search


@dataclass(frozen=True, slots=True)
class SearchResult:
    winner: Candidate
    winner_mean: float
    baseline_mean: float
    front: tuple[str, ...]
    generations: int
    n_evaluations: int
    rejected: dict[str, int]
    config_sha: str
    weights_sha: str

    def as_dict(self) -> dict[str, object]:
        return {
            "winner": self.winner.as_dict(),
            "winner_mean": self.winner_mean,
            "baseline_mean": self.baseline_mean,
            "front": list(self.front),
            "generations": self.generations,
            "n_evaluations": self.n_evaluations,
            "rejected": dict(self.rejected),
            "config_sha": self.config_sha,
            "weights_sha": self.weights_sha,
        }


def run_search(
    cfg: Rung0Config,
    *,
    seed_prompt: str,
    tasks: Sequence[TaskKey],
    evaluate: Evaluator,
    reflect: Reflector,
    val_tasks: Sequence[TaskKey] = (),
    weights: RewardWeights | None = None,
    on_event: Callable[[str], None] = print,
    checkpoint_dir: str | Path | None = None,
) -> SearchResult:
    """Evolve `seed_prompt` for `cfg.n_generations`, and return the tie-broken winner.

    Deterministic in `cfg.seed` given a deterministic evaluator and reflector: parent sampling
    is the only stochastic step this module owns, and it draws from one seeded `random.Random`.
    """
    if not tasks:
        raise SearchRefused("no tasks: a prompt cannot be scored against an empty set")
    w = weights or DEFAULT_WEIGHTS
    rng = random.Random(cfg.seed)
    root = Candidate(text=seed_prompt, generation=0, note="seed: the shipped template")
    scores: dict[str, dict[str, float]] = {}
    pool: dict[str, Candidate] = {root.cid: root}
    rejected: dict[str, int] = {}
    n_eval = 0
    # THE PARITY BUDGET. Derived from the family the base prompt belongs to, so it tracks the
    # shipped prompts instead of restating them; None when the prompt is in no family, which
    # keeps this a no-op for callers that are not feeding an arm comparison.
    budget = _budget_from(cfg)
    on_event(
        f"parity budget: {budget} tokens for {cfg.base_prompt}"
        if budget is not None
        else "parity budget: NONE -- an unconstrained search optimises length; the winner "
        "will not be comparable to its ablations without padding them to match"
    )

    def score_candidate(c: Candidate, keys: Sequence[TaskKey]) -> dict[str, TaskResult]:
        nonlocal n_eval
        out: dict[str, TaskResult] = {}
        for key in keys:
            out[str(key)] = evaluate(c, key)
            n_eval += 1
        return out

    # RESUME. A restart without this re-pays for every generation past the first: the seed's
    # rollouts are "resumed" and free, but the reflector proposes FRESH mutations on the way
    # back up, and a different mutation is a different run_id.
    start_gen = 1
    ck = load_checkpoint(checkpoint_dir) if checkpoint_dir else None
    if ck and ck["pool"]:
        pool.update(ck["pool"])
        scores.update(ck["scores"])
        start_gen = max(1, int(ck["generation"]) + 1)
        on_event(
            f"resumed from checkpoint: {len(pool)} candidates, "
            f"{len(scores)} scored, continuing at generation {start_gen}"
        )

    traces: dict[str, dict[str, TaskResult]] = {}
    if root.cid not in scores:
        base_results = score_candidate(root, tasks)
        scores[root.cid] = {k: r.reward for k, r in base_results.items()}
        traces[root.cid] = base_results
        on_event(f"gen 0  seed {short(root.cid)} mean={mean_reward(scores[root.cid]):.4f}")

    for gen in range(start_gen, cfg.n_generations + 1):
        front = pareto_front(scores)
        for _ in range(cfg.n_candidates):
            parent_cid = sample_parent(front, scores, rng)
            parent = pool[parent_cid]
            batch = sorted(traces[parent_cid].values(), key=lambda r: r.reward)[: cfg.minibatch]
            proposed = reflect(parent, batch)
            ok, why = accept_mutation(parent, proposed, max_tokens=budget)
            if not ok:
                rejected[why.split(":")[0]] = rejected.get(why.split(":")[0], 0) + 1
                continue
            child = Candidate(
                text=proposed.strip(),
                generation=gen,
                parent=parent_cid,
                note=f"mutant of {short(parent_cid)}",
            )
            if child.cid in pool:
                rejected["duplicate"] = rejected.get("duplicate", 0) + 1
                continue
            res = score_candidate(child, tasks)
            pool[child.cid] = child
            traces[child.cid] = res
            scores[child.cid] = {k: r.reward for k, r in res.items()}
            on_event(
                f"gen {gen}  {short(child.cid)} <- {short(parent_cid)} "
                f"mean={mean_reward(scores[child.cid]):.4f}"
            )
            # AFTER EVERY CANDIDATE. Per-generation was the first design and its reasoning was
            # wrong: a restart does not discard a partial generation, because `pool` and
            # `scores` are restored wholesale and every scored candidate goes on feeding the
            # Pareto front whichever generation produced it. Measured the hard way -- the
            # search died mid-generation-1 twice to the same VPN drop, after 3 candidates and
            # then 5, and wrote nothing either time. A per-candidate save can at worst
            # duplicate one candidate; a per-generation save was losing six.
            if checkpoint_dir:
                save_checkpoint(checkpoint_dir, pool=pool, scores=scores, generation=gen)

        if checkpoint_dir:
            save_checkpoint(checkpoint_dir, pool=pool, scores=scores, generation=gen)

    front = pareto_front(scores)
    # The tie-break runs on HELD-OUT tasks. Picking the front member with the best mean over
    # the tasks the search itself optimised would select the candidate that overfitted them,
    # and the search has no other defence against that.
    if val_tasks:
        val = {cid: score_candidate(pool[cid], val_tasks) for cid in front}
        ranked = sorted(
            front,
            key=lambda cid: mean_reward({k: r.reward for k, r in val[cid].items()}),
            reverse=True,
        )
        winner_mean = mean_reward({k: r.reward for k, r in val[ranked[0]].items()})

        # THE SEED, ON THE SAME TASKS AS THE WINNER.
        #
        # `baseline_mean` used to be `mean_reward(scores[root.cid])` unconditionally -- the
        # seed's mean over the TRAIN tasks -- while `winner_mean` came from the disjoint VAL
        # slice. `cmd_train` prints them as "mean X (baseline Y)" and `write_winner` serialises
        # both into rung0.manifest.json, the provenance record for the prompt that the
        # untrained, trained and frontier families all share. So the headline "the optimised
        # prompt beats the seed" was a TASK-SET contrast wearing a prompt contrast's label.
        #
        # Demonstrated with an evaluator whose reward depends only on the task, so the true
        # prompt gain is exactly zero: the search returned the byte-identical seed as winner
        # and still reported mean 9.0000 (baseline 1.0000), an apparent +8.0 from a search that
        # changed nothing. With a real +0.10-per-mutation effect plus a difficulty offset it
        # reported +4.2 against a true held-out gain of +0.2, a 21x overstatement.
        #
        # `val` only covers the Pareto front, and the seed is often dominated off it -- in the
        # second reproduction it was -- so its held-out mean has to be scored explicitly rather
        # than looked up. That costs one more held-out pass, which `Rung0Config.n_rollouts`
        # now prices.
        seed_val = val.get(root.cid)
        if seed_val is None:
            seed_val = score_candidate(root, val_tasks)
        baseline_mean = mean_reward({k: r.reward for k, r in seed_val.items()})
    else:
        ranked = sorted(front, key=lambda cid: mean_reward(scores[cid]), reverse=True)
        winner_mean = mean_reward(scores[ranked[0]])
        baseline_mean = mean_reward(scores[root.cid])
    return SearchResult(
        winner=pool[ranked[0]],
        winner_mean=winner_mean,
        baseline_mean=baseline_mean,
        front=front,
        generations=cfg.n_generations,
        n_evaluations=n_eval,
        rejected=rejected,
        config_sha=cfg.sha,
        weights_sha=w.sha,
    )


def write_winner(out_dir: str | Path, cfg: Rung0Config, result: SearchResult) -> Path:
    """Write the winning template as an OVERLAY DIRECTORY, plus its provenance.

    An overlay directory rather than a patch to `src/pinq/prompts/`: the untrained, trained and
    frontier families all point `PI_PROMPT_OVERLAY` at this one directory, so the three share
    one prompt by construction. Editing the shipped template instead would change the run
    identity of every previously recorded rollout's ancestor and make the pilot incomparable.
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{cfg.base_prompt}.txt").write_text(result.winner.text, encoding="utf-8")
    (d / "rung0.manifest.json").write_text(
        json.dumps({"config": asdict(cfg), **result.as_dict()}, indent=2, sort_keys=True) + "\n"
    )
    return d


def task_keys(
    suite_ids: Iterable[str], task_ids_by_suite: Mapping[str, Sequence[str]], seeds: Sequence[int]
) -> tuple[TaskKey, ...]:
    return tuple(
        TaskKey(suite_id=s, task_id=t, seed=seed)
        for s in suite_ids
        for t in task_ids_by_suite.get(s, ())
        for seed in seeds
    )
