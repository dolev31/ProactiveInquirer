"""The real Drafters and the frozen Answerer: the components an arm varies around.

The one invariant that governs this file is `Drafter.draft()` PURITY. phi_LOO, the prefix
ladder and the stop test are all statements about re-drafting over a subset of the evidence
long after the rollout ended, and none of them is defined if `draft()` can return two
different strings for the same `(view, evidence.subset_hash, seed)`.

Purity here is BY CONSTRUCTION, not by memoization:

  * the evidence is serialized canonically — `Evidence.units` is already deduplicated and
    sorted by uid, so the rendered block is a function of `subset_hash` and nothing else;
  * `draft()` makes ONE non-agentic call whose request bytes are that block plus the view
    plus the seed, and reads no field of `self` that a previous call could have written;
  * the retrieval a Drafter is allowed to do lives in `resolve()`, which is agentic and
    metered, and never in `draft()`.

A memo would have hidden an impure implementation behind a dict and made the purity test
tautological. Determinism of the model itself is supplied one layer down, by the
content-addressed cache: identical request bytes replay identical response bytes.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar, Mapping, Sequence

from pinq import promptlib
from pinq.budget import BudgetExceeded, BudgetLedger
from pinq.ids import canon
from pinq.protocols import LLM, Retriever
from pinq.types import Answer, Ask, Draft, Evidence, EvidenceUnit, TaskView

# How much of one unit's text reaches a prompt. A cap is needed (a 20-paragraph pool does not
# fit twice), and it must be a CONSTANT rather than a budget-derived number, or the prompt
# would start carrying information about the cap.
#
# SIZED FOR QA PARAGRAPHS, AND THE DEFAULT EVERYWHERE. A suite whose units are records rather
# than paragraphs passes its own window to the Drafter and the Answerer as a CONSTRUCTOR
# constant (`unit_chars=`), never per call -- see `pi_run.stages.tau2_runner`, where 1200 cut
# every airline flight record and hid the dates on which a flight is available. The
# Inquirers never take one: their view, which the trained questioner learned on, stays here.
UNIT_CHARS = 1200

DRAFT_MAX_TOKENS = 700
RESOLVE_MAX_TOKENS = 300
ANSWER_MAX_TOKENS = 700


class RetrieverRequired(RuntimeError):
    """An arm whose whole claim is "equal retrieval calls" was built without a retriever.

    Raised at first use rather than absorbed. Without it this arm degrades into a single
    verbatim ask and still writes a row, and that row would sit in a table under a name
    that says it expanded queries.
    """


class LLMRequired(RuntimeError):
    """A component that needs a model was built without one.

    Raised at reset/first use rather than swallowed: an arm declared non-`llm_free` that
    silently degrades to a no-op contributes a row to a table that says something false.
    """


# --------------------------------------------------------------------------- serialization


def render_evidence(ev: Evidence, *, unit_chars: int = UNIT_CHARS) -> str:
    """Canonical, order-independent rendering of an evidence SET.

    `Evidence.__post_init__` has already deduplicated by uid and sorted by uid, so two
    trajectories that retrieved the same units in different orders produce the same bytes
    here — which is precisely the statement `subset_hash` makes.
    """
    if not ev.units:
        return "(nothing retrieved yet)"
    return "\n\n".join(f"[{u.uid[:12]}] {u.title}\n{u.text[:unit_chars]}" for u in ev.units)


def render_history(history: Sequence[Any], *, answers: bool = True) -> str:
    """Past asks, and what came back for each.

    run_loop stores the Drafter's reply to an ask on `Turn.response_text`; reading it here is
    what makes the Inquirer's next question conditional on the previous answer rather than
    on the question list alone.
    """
    lines: list[str] = []
    n = 0
    for turn in history:
        action = getattr(turn, "action", None)
        if not isinstance(action, Ask):
            continue
        n += 1
        lines.append(f"Q{n}: {action.text}")
        if answers:
            lines.append(f"A{n}: {getattr(turn, 'response_text', '') or '(no answer recorded)'}")
    return "\n".join(lines) if lines else "(nothing asked yet)"


def render_tools(tools: Sequence[Mapping[str, Any]]) -> str:
    """The advertised tool list, as canonical JSON, one tool per line.

    Sorted by name and serialized with `canon` so the rendered block is a pure function of
    the tool set. A dict repr would reorder across processes and break `draft()` purity in a
    way no test on a single machine could see.
    """
    return "\n".join(
        canon(
            {
                "name": str(t.get("name", "")),
                "description": str(t.get("description", "")),
                "parameters": t.get("parameters") or {},
            }
        )
        for t in sorted(tools, key=lambda t: str(t.get("name", "")))
    )


def _balanced_objects(text: str) -> list[str]:
    """Every top-level balanced {...} substring, in order. String-aware, so a brace inside a
    quoted value does not terminate an object."""
    out: list[str] = []
    depth = start = 0
    in_str = escape = False
    for i, c in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                out.append(text[start : i + 1])
            elif depth < 0:
                depth = 0
    return out


def normalize_tool_plan(raw: Any) -> tuple[Mapping[str, Any], ...]:
    """A model's `tool_plan` list -> the steps `Actuator.execute` accepts.

    Malformed steps are DROPPED, never repaired: a step with no name has nowhere to go, and
    inventing one would put a call in the environment log that no policy chose.

    `requestor` is stripped. `Tau2Actuator.execute` honours it — the gold-action replay needs
    to be able to say "the customer did this" — but a DRAFTER may never emit it. tau2's user
    tools belong to the customer, and a policy that could act as the customer would be
    scored for a mutation the agent under test never had the authority to make.
    """
    if not isinstance(raw, list):
        return ()
    out: list[Mapping[str, Any]] = []
    for step in raw:
        if not isinstance(step, Mapping):
            continue
        name = str(step.get("name") or step.get("tool_name") or "").strip()
        if not name:
            continue
        args = step.get("args")
        if not isinstance(args, Mapping):
            args = step.get("arguments") if isinstance(step.get("arguments"), Mapping) else {}
        out.append({"name": name, "args": dict(args)})
    return tuple(out)


def split_tool_plan(text: str) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    """(prose, plan). The LAST balanced object carrying a `tool_plan` key wins.

    The last rather than the first: the fragment asks for the object at the END of the reply,
    and a draft that quotes a JSON example from a knowledge-base document would otherwise
    have that example executed against a live bank.

    The object is removed from the prose so the frozen Answerer — which is shared by every
    arm and knows nothing about tools — is never handed a block of tool JSON to paraphrase.
    """
    for blob in reversed(_balanced_objects(text)):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "tool_plan" not in obj:
            continue
        prose = text.replace(blob, "")
        # the fence the block came wrapped in, now empty
        prose = re.sub(r"```(?:json)?\s*```", "", prose)
        return prose.strip(), normalize_tool_plan(obj.get("tool_plan"))
    return text, ()


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """The first balanced JSON object in a generation, or None.

    None is a VALUE, not an exception: a malformed generation must degrade a single turn to
    STOP and be counted, never abort a 30k-rollout sweep. A fenced block is unwrapped because
    models emit them constantly and refusing one would inflate the malformed rate with a
    formatting artifact rather than a policy failure.
    """
    if not text:
        return None
    body = _FENCE.sub("", text).strip()
    start = body.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(body)):
        c = body[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(body[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


# --------------------------------------------------------------------------- drafters


class LLMDrafter:
    """The real Drafter. `resolve()` is agentic and metered; `draft()` is pure.

    `resolve()` may fire its own retrieval, and every unit it brings back is charged to the
    SHARED ledger (`charge_retrieval` + `note_docs`) and returned inside the Evidence, so
    `pi_run.worker.reconcile` can prove that the documents in the trajectory and the
    documents the ledger saw are the same set. A Drafter that retrieved privately would be
    invisible in `retrieval_calls`, invisible in tokens, and fatal to a budget-parity claim.
    """

    # The provider role this component bills against. Collected by pi_run.worker to build
    # RunManifest.pins; see the note there for why an undeclared role is not a cosmetic gap.
    llm_role: ClassVar[str] = "drafter"

    draft_prompt = "drafter_draft"
    resolve_prompt = "drafter_resolve"
    tool_plan_prompt = "fragment_tool_plan"

    def __init__(
        self,
        llm: LLM | None = None,
        *,
        retriever: Retriever | None = None,
        agentic_resolve: bool = True,
        k: int = 3,
        draft_max_tokens: int = DRAFT_MAX_TOKENS,
        tools: Sequence[Mapping[str, Any]] = (),
        unit_chars: int = UNIT_CHARS,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._agentic_resolve = agentic_resolve
        self._k = k
        self._draft_max_tokens = draft_max_tokens
        # The evidence window, a CONSTRUCTOR constant for the reason `tools` is one below:
        # `draft()` stays a pure function of (view, evidence.subset_hash, seed) only if
        # nothing that shapes its prompt can change between two calls.
        self._unit_chars = int(unit_chars)
        # The tools this suite EXPOSES, frozen at construction. Construction-time and not
        # per-call on purpose: `draft()` has to stay a pure function of
        # (view, evidence.subset_hash, seed), and a tool list that could change between two
        # calls would be a fourth argument that nothing hashes.
        self._tools = tuple(dict(t) for t in tools or ())

    # ------------------------------------------------------------------ prompt identity

    @property
    def prompt_hashes(self) -> dict[str, str]:
        """Only the templates this instance will actually render.

        A Drafter configured not to resolve never sends `drafter_resolve`, so reporting its
        hash would put a prompt in the manifest that had no effect on the run — and a
        manifest that names a prompt the run did not use is a provenance claim that is
        false in the direction that matters.
        """
        names = [self.draft_prompt]
        if self._agentic_resolve:
            names.append(self.resolve_prompt)
        if self._tools:
            names.append(self.tool_plan_prompt)
        return promptlib.hashes(*names)

    def _call(self, prompt: str, *, seed: int, max_tokens: int) -> str:
        if self._llm is None:
            raise LLMRequired(f"{type(self).__name__} needs an LLM; build the arm with one")
        text, _ = self._llm.complete(
            role="drafter",
            messages=[{"role": "user", "content": prompt}],
            seed=seed,
            max_tokens=max_tokens,
            actor="drafter",
        )
        return text

    # ------------------------------------------------------------------ agentic half

    def resolve(
        self, view: TaskView, ask: Ask, ev: Evidence, *, seed: int, ledger: BudgetLedger
    ) -> tuple[str, Evidence]:
        ev = self._sub_retrieve(self._queries(view, ask, ev, seed=seed, ledger=ledger), ev, ledger)
        if not self._agentic_resolve:
            return ("", ev)
        prompt = promptlib.render(
            self.resolve_prompt,
            question=view.question,
            ask=ask.text,
            evidence=render_evidence(ev, unit_chars=self._unit_chars),
        )
        return (self._call(prompt, seed=seed, max_tokens=RESOLVE_MAX_TOKENS), ev)

    def _queries(
        self, view: TaskView, ask: Ask, ev: Evidence, *, seed: int, ledger: BudgetLedger
    ) -> tuple[str, ...]:
        """Extra retrieval queries this Drafter wants to fire for `ask`.

        Empty for the plain Drafter: run_loop already retrieved for the ask itself, and
        re-issuing it would spend a second call on the same units. QueryExpansionDrafter
        overrides this and is the reason the seam exists.
        """
        return ()

    def _sub_retrieve(self, queries: Sequence[str], ev: Evidence, ledger: BudgetLedger) -> Evidence:
        if not queries or self._retriever is None:
            return ev
        units: list[EvidenceUnit] = []
        for q in queries:
            try:
                ledger.charge_retrieval(1)
            except BudgetExceeded:
                # The hard cap binds here exactly as it does in run_loop. Stopping the
                # expansion is the honest response: raising would turn a budget boundary
                # into a task-level error and drop the row from the table entirely.
                ledger.record("expansion_truncated", 1)
                break
            found = list(self._retriever.search(q, self._k))
            ledger.note_docs(u.doc_id for u in found)
            units.extend(found)
        return ev.with_units(units) if units else ev

    # ------------------------------------------------------------------ the pure half

    def draft(self, view: TaskView, ev: Evidence, *, seed: int, ledger: BudgetLedger) -> Draft:
        """One call, then a PURE parse. The plan is DECIDED here and EXECUTED by the Actuator.

        Executing here would put an environment mutation inside the one function the whole
        measurement layer re-runs over arbitrary evidence subsets: every phi_LOO re-draft and
        every rung of the prefix ladder would fire a real tool call against a live bank, and
        the same subset re-scored twice would produce two different worlds. `run_loop` already
        hands `Draft.tool_plan` to `Actuator.execute` exactly once, at the end of the rollout,
        which is the only place a side effect can happen without breaking that.
        """
        return self._to_draft(
            self._call(self._draft_prompt(view, ev), seed=seed, max_tokens=self._draft_max_tokens)
        )

    def _to_draft(self, text: str) -> Draft:
        """Generation -> Draft, in ONE place.

        Shared by every subclass so a drafter that overrides `draft()` cannot become the one
        arm on a tool-bearing suite that silently emits no plan — which is the same class of
        confound as an arm that was never handed the tool schemas at all.
        """
        if not self._tools:
            return Draft(text=text)
        prose, plan = split_tool_plan(text)
        return Draft(text=prose, tool_plan=plan)

    def _draft_prompt(self, view: TaskView, ev: Evidence) -> str:
        prompt = promptlib.render(
            self.draft_prompt,
            question=view.question,
            instructions=view.instructions,
            evidence=render_evidence(ev, unit_chars=self._unit_chars),
        )
        if not self._tools:
            return prompt
        # APPENDED, never substituted. Keeping `drafter_draft` (and `drafter_verbosity`, and
        # `drafter_compute_matched`) byte-identical is what keeps promptlib's `drafter` parity
        # family intact: a tool-bearing suite adds the same block to every arm in the slot, so
        # the arms still differ in what their prompt says and not in how much of it there is.
        return (
            prompt
            + "\n\n"
            + promptlib.render(self.tool_plan_prompt, tools=render_tools(self._tools))
        )


class VerbosityDrafter(LLMDrafter):
    """THE KILL SWITCH. The Drafter alone, prompted for exhaustive multi-facet coverage and
    given the treatment's completion budget.

    If this matches `inquirer_prompted`, the effect the paper reports is answer length and
    there is no paper — which is why it is preregistered as a stop rule rather than as an
    appendix row.

    Its retrieval count is zero, exactly like `drafter_only`: `resolve()` is never called
    because the arm pairs it with a policy that never asks, and a Drafter cannot retrieve
    from inside `draft()` without breaking purity and the doc-reconciliation check at once.
    Parity for this arm is therefore TOKEN parity, and it is the generous side of it: the
    kill switch is supposed to be given every advantage that is not evidence.
    """

    draft_prompt = "drafter_verbosity"

    def __init__(
        self,
        llm: LLM | None = None,
        *,
        tools: Sequence[Mapping[str, Any]] = (),
        unit_chars: int = UNIT_CHARS,
        **kw: Any,
    ) -> None:
        # `tools` is named EXPLICITLY, exactly as `retriever` is on QueryExpansionDrafter and
        # for exactly the same reason: `pinq_expt.arms.build` passes only the arguments a
        # constructor DECLARES, so a parameter absorbed into **kw is never passed at all. It
        # is not a style choice. Absorbed here, the kill switch would be the one arm on a
        # tool-bearing suite that cannot act, and "verbosity did not match the Inquirer"
        # would mean "verbosity was not given the action channel".
        # `unit_chars` is named for the same reason as `tools`.
        kw.setdefault("agentic_resolve", False)
        super().__init__(llm, tools=tools, unit_chars=unit_chars, **kw)


class ComputeMatchedDrafter(LLMDrafter):
    """Self-consistency@n at token parity with the Inquirer arm.

    Purity survives sampling n candidates because the derived seeds are a function of the
    seed argument and the winner is chosen by a deterministic rule: most frequent normalized
    text, ties broken lexicographically. Nothing here reads the clock, a global RNG, or
    `self`.
    """

    draft_prompt = "drafter_compute_matched"

    def __init__(
        self,
        llm: LLM | None = None,
        *,
        n: int = 5,
        tools: Sequence[Mapping[str, Any]] = (),
        unit_chars: int = UNIT_CHARS,
        **kw: Any,
    ) -> None:
        kw.setdefault("agentic_resolve", False)
        super().__init__(llm, tools=tools, unit_chars=unit_chars, **kw)
        self.n = n

    def draft(self, view: TaskView, ev: Evidence, *, seed: int, ledger: BudgetLedger) -> Draft:
        prompt = self._draft_prompt(view, ev)
        cands = [
            self._call(prompt, seed=seed + i, max_tokens=self._draft_max_tokens)
            for i in range(self.n)
        ]
        # The winner is chosen over the WHOLE generation and the plan is split out of the
        # winner, so the prose and the plan can never come from two different samples.
        return self._to_draft(_majority(cands))


def _majority(candidates: Sequence[str]) -> str:
    """Most frequent by normalized text; ties broken lexicographically on the raw string.

    Deterministic on purpose: `max(counts, key=counts.get)` would depend on insertion order
    and make the same evidence set produce two different drafts across processes.
    """
    if not candidates:
        return ""
    counts: dict[str, int] = {}
    for c in candidates:
        key = " ".join(c.split())
        counts[key] = counts.get(key, 0) + 1
    best = sorted(counts, key=lambda k: (-counts[k], k))[0]
    for c in candidates:  # return the raw candidate, not the normalized key
        if " ".join(c.split()) == best:
            return c
    return candidates[0]


class QueryExpansionDrafter(LLMDrafter):
    """Multi-query / HyDE at equal retrieval calls.

    The expansion happens in `resolve()`, which is where a Drafter is allowed to retrieve and
    where the ledger can see it. The arm is budget-matched by the single hard-capped
    currency: every expanded query is a `charge_retrieval(1)` against the same cap the
    Inquirer arm spends its questions from, so "more queries" costs exactly what "more
    questions" costs.
    """

    expand_prompt = "drafter_query_expansion"

    def __init__(
        self,
        llm: LLM | None = None,
        *,
        retriever: Retriever | None = None,
        n_queries: int = 3,
        tools: Sequence[Mapping[str, Any]] = (),
        unit_chars: int = UNIT_CHARS,
        **kw: Any,
    ) -> None:
        # `retriever` and `tools` are named explicitly rather than swallowed by **kw because
        # pinq_expt.arms.build passes only the arguments a constructor DECLARES: absorbed
        # into **kw they would never be passed, and the arm would silently stop expanding /
        # silently stop acting.
        super().__init__(llm, retriever=retriever, tools=tools, unit_chars=unit_chars, **kw)
        self.n_queries = n_queries

    @property
    def prompt_hashes(self) -> dict[str, str]:
        return promptlib.hashes(self.draft_prompt, self.resolve_prompt, self.expand_prompt)

    def _queries(
        self, view: TaskView, ask: Ask, ev: Evidence, *, seed: int, ledger: BudgetLedger
    ) -> tuple[str, ...]:
        if self._retriever is None:
            raise RetrieverRequired(
                "QueryExpansionDrafter has no retriever, so its expanded queries would go "
                "nowhere and the arm would silently become a single verbatim ask. Build it "
                "with pinq_expt.arms.build(arm, llm=..., retriever=suite.retriever(tid))."
            )
        prompt = promptlib.render(
            self.expand_prompt, ask=ask.text, question=view.question, n=self.n_queries
        )
        obj = parse_json_object(self._call(prompt, seed=seed, max_tokens=RESOLVE_MAX_TOKENS))
        raw = (obj or {}).get("queries")
        if not isinstance(raw, list):
            # A malformed expansion degrades to no expansion. The arm then spends fewer
            # retrieval calls than its cap allows, which is visible in the ledger rather
            # than papered over with a fabricated query.
            ledger.record("malformed", 1)
            return ()
        return tuple(str(q) for q in raw if str(q).strip())[: self.n_queries]


# --------------------------------------------------------------------------- answerer


class FrozenLLMAnswerer:
    """FROZEN across every arm: one prompt, one word cap, one model pin, and blind to both
    the arm id and the budget.

    This is what removes answer-length bias BY CONSTRUCTION rather than by a post-hoc length
    regression. Note what the signature cannot express and this class therefore refuses to
    accept: there is no `arm_id` parameter and no per-arm configuration of any kind, so two
    arms cannot be given two answerers without that being visible in the arm table.

    The word cap is applied twice: asked for in the prompt, and enforced on the returned
    string. The model's compliance is not part of the experiment's design.
    """

    # The provider role this component bills against. Collected by pi_run.worker to build
    # RunManifest.pins; see the note there for why an undeclared role is not a cosmetic gap.
    llm_role: ClassVar[str] = "answerer"

    prompt_name = "answerer_frozen"

    def __init__(self, llm: LLM | None = None, *, unit_chars: int = UNIT_CHARS) -> None:
        self._llm = llm
        # THE EVIDENCE WINDOW IS A PROPERTY OF THE SUITE, NOT OF THE ARM, and so does not
        # break the freeze above: every arm on one suite is built with the same value by
        # the suite's driver, and `pinq_expt.arms.build` has no per-arm way to set it.
        self._unit_chars = int(unit_chars)

    @property
    def prompt_hash(self) -> str:
        return promptlib.sha(self.prompt_name)

    def answer(
        self,
        view: TaskView,
        ev: Evidence,
        draft: Draft | None,
        *,
        seed: int,
        ledger: BudgetLedger,
    ) -> Answer:
        if self._llm is None:
            raise LLMRequired("FrozenLLMAnswerer needs an LLM; build the arm with one")
        prompt = promptlib.render(
            self.prompt_name,
            question=view.question,
            instructions=view.instructions,
            evidence=render_evidence(ev, unit_chars=self._unit_chars),
            draft=(draft.text if draft else "(no draft)"),
            word_cap=view.word_cap,
        )
        text, _ = self._llm.complete(
            role="answerer",
            messages=[{"role": "user", "content": prompt}],
            seed=seed,
            max_tokens=ANSWER_MAX_TOKENS,
            actor="answerer",
        )
        capped = " ".join(text.split()[: view.word_cap])
        return Answer(
            text=capped,
            evidence_hash=ev.subset_hash,
            cited_unit_ids=tuple(u.uid for u in ev.units),
            n_words=len(capped.split()),
            stop_reason="policy_stop",
        )


def prompt_hashes_of(*components: object) -> Mapping[str, str]:
    """Every prompt an arm's components will render, for the manifest.

    A prompt edit therefore changes `semantic_hash` and so changes `run_id`, which is what
    stops two rows of one table from having come from two different prompts.
    """
    out: dict[str, str] = {}
    for c in components:
        multi = getattr(c, "prompt_hashes", None)
        if isinstance(multi, Mapping):
            out.update(multi)
        name = getattr(c, "prompt_name", None)
        if isinstance(name, str) and name:
            out[name] = promptlib.sha(name)
    return dict(sorted(out.items()))
