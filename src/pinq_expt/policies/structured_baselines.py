"""A structured, training-free retrieval baseline, ported faithfully or not at all.

TRACK 3 (2026-09-22): two reviewers named a family of structured/learned retrieval
alternatives -- PRISM, PAR2-RAG, PyRAG, ConRAG, A2RAG, TREX, REAR -- as the missing
comparator class. Both PAR2-RAG (arXiv:2603.29085) and PRISM (arXiv:2510.14278) were read
in full before anything below was written.

PAR2-RAG is ported, as `Par2RagInquirer`. PRISM is NOT, and the reason is a mechanism-level
one rather than a naming coincidence:

  * PAR2-RAG's own two stages are (1) Coverage Anchoring -- a Planner decomposes the task
    into m sub-queries, retrieved BREADTH-FIRST and unconditioned on any evidence -- and
    (2) Iterative Chain Refinement -- an Evidence Sufficiency Controller ("ESC") reads the
    accumulated context and either stops or hands a Query Formulator the next focused
    query, DEPTH-FIRST. Both stages are sequences of ask-or-stop decisions, each consuming
    exactly one retrieval call, which is exactly the shape `Inquirer.act(s: State)` is built
    for: stage 1 is `Depth1Inquirer`'s enumerate-once-then-walk pattern (see
    pinq_expt.policies.prompted), and stage 2 is a sufficiency-based stopper reading only
    `s.evidence` / `s.draft` / `s.history` -- never a budget -- which is precisely the
    pattern this codebase's own comment on `Inquirer.act` sanctions.

  * PRISM's mechanism is a Selector/Adder cycle that operates on ONE ALREADY-RETRIEVED
    candidate pool: a Question Analyzer decomposes the query, a precision-focused Selector
    filters candidate passages per sub-question, and a recall-focused Adder re-examines the
    UNSELECTED remainder for missed bridging facts, cycling with the Selector for a fixed
    N=3 iterations before the results are merged. Nothing in that cycle is a sequence of
    Ask actions each spending a `retrieval_calls` unit -- it is evidence CURATION over a
    pool the retriever already returned, i.e. a resolve/selection mechanism. Porting it
    faithfully means moving Selector/Adder into the Drafter's resolve path, not the
    Inquirer's act() loop, and every arm in this table is built on the premise that arms
    "differ only in what occupies the Inquirer's stage" (see pinq_expt.arms, Grid E's own
    comment on this). Moving PRISM's mechanism into the Drafter would make the drafter
    differ between the PRISM arm and every other arm, which is the one thing guard (2) of
    this track's task forbids. A caricature that flattens Selector/Adder into per-turn
    Ask/Stop decisions would not be PRISM -- it would be a strawman wearing its name, which
    the task explicitly calls WORSE than no baseline. So PRISM is documented here, and
    ported nowhere, rather than ported wrong.

WHAT WAS LEFT OUT OF PAR2-RAG, AND WHY:

  * The paper's Stage-2 loop is three separate LLM calls per step -- a Writer produces a
    step response r_t, the ESC judges (q, r_t, C_t) -> {CONTINUE, STOP} and, on CONTINUE, a
    Query Formulator proposes q*_{t+1}. `LLMPolicy.act` (pinq_expt.policies.base) makes
    exactly ONE `_complete` call per turn, by a design this track's task forbids changing
    (an extra call per turn is an extra `retrieval_calls`-free LLM call that every other arm
    in the table does not make, which would confound "this baseline asks more" with "this
    baseline thinks more per ask"). Three deviations follow, each folding a paper role into
    an existing harness role rather than adding a call:
      1. The Writer's step response r_t is the harness's own Drafter output, `s.draft` --
         which the Inquirer already reads in the treatment arm (`PromptedInquirer._prompt`)
         -- rather than a second inquirer-side generation of the same thing.
      2. The ESC's sufficiency judgment and the Query Formulator's next query are elicited
         by ONE prompt (`par2_rag_refine.txt`) and ONE strict-JSON action, reusing
         `LLMPolicy._action_from` verbatim: a "stop" action IS the ESC's STOP, an "ask"
         action IS "CONTINUE, with q*_{t+1} attached". Nothing about the semantic content of
         the two-function split is lost -- the model is still asked to judge sufficiency
         BEFORE it is asked to name the next query, in that order, inside the same prompt --
         only the call boundary moves.
      3. The paper's default Stage-2 step budget (5, tested at {3,5,7,10}) is a policy
         hyperparameter, not a read of this harness's `budget_cap`: `max_asks` is already
         how `Depth1Inquirer` and every other capped ablation in this file's siblings
         express "this policy's own count of its own actions", and it is set the same way
         here. The ESC's own STOP is still what actually ends refinement in the common case;
         the cap only bounds the tail, exactly as budget_cap bounds every other arm's tail.
"""

from __future__ import annotations

from pinq import promptlib
from pinq.types import Action, Ask, State, Stop, TaskView
from pinq_expt.components import render_evidence, render_history
from pinq_expt.policies.base import LLMPolicy, parse_or_none

# The paper's own default (tested at {3, 5, 7, 10}); a policy hyperparameter, read nowhere
# near the harness's budget_cap. See the module docstring, deviation 3.
DEFAULT_STAGE2_STEPS = 5


class Par2RagInquirer(LLMPolicy):
    """PAR2-RAG (arXiv:2603.29085): Coverage Anchoring, then Iterative Chain Refinement.

    Stage 1 (breadth, unconditioned on evidence): one model call enumerates the sub-queries
    nameable from the task alone, exactly as `Depth1Inquirer._enumerate` does, and they are
    walked in order without looking at what came back -- Coverage Anchoring is explicitly
    breadth-first in the paper, not adaptive.

    Stage 2 (depth, ESC + Query Formulator): once the stage-1 queue is empty, every
    subsequent `act()` renders `par2_rag_refine.txt` over `(s.view, s.evidence, s.draft,
    s.history)` and reuses `_action_from` on the result -- a "stop" is the ESC's STOP; an
    "ask" is CONTINUE with the Query Formulator's next focused query attached. This stage is
    capped at `self.stage2_steps` steps of its OWN counting, never at a read of the run's
    budget_cap (see the module docstring).
    """

    policy_id = "par2_rag"
    prompt_name = "par2_rag_plan"
    stage2_prompt_name = "par2_rag_refine"
    stage2_steps: int = DEFAULT_STAGE2_STEPS

    def __init__(
        self,
        llm=None,
        *,
        recorder=None,
        max_asks: int | None = None,
        may_ask_user: bool = False,
        stage2_steps: int | None = None,
    ) -> None:
        # Named explicitly, not `*args, **kwargs`: `pinq_expt.arms._make` inspects this
        # signature by PARAMETER NAME to decide what to hand the constructor (llm, recorder,
        # ...), and a bare `**kwargs` is invisible to that check -- it would build this
        # policy with every dependency silently dropped.
        super().__init__(llm, recorder=recorder, max_asks=max_asks, may_ask_user=may_ask_user)
        if stage2_steps is not None:
            self.stage2_steps = stage2_steps

    @property
    def prompt_hashes(self) -> dict[str, str]:
        # TWO templates, not one -- the whole point of the CRITICAL LESSON this track was
        # briefed with: a manifest that recorded only `self.prompt_name` would show the same
        # single key as `inquirer_depth1` on any rollout that never left stage 1, which is
        # indistinguishable from the base-prompt leak this note exists to catch. Both keys
        # are always present, on every rollout, regardless of how many stage-1 questions the
        # task happened to name.
        return promptlib.hashes(
            self.prompt_name, self.stage2_prompt_name, self._user_channel_fragment()
        )

    def _on_reset(self, view: TaskView, seed: int) -> None:
        self._queue: list[str] | None = None
        self._stage2_asks = 0

    # ---- stage 1: coverage anchoring (breadth, x alone) -------------------------------

    def _prompt(self, s: State) -> str:
        """Stage-1 prompt. Deliberately `s.view` ONLY -- see `Depth1Inquirer._prompt` for
        why this is a property of what the prompt can contain, not a claim about a model."""
        return promptlib.render(
            self.prompt_name,
            question=s.view.question,
            instructions=s.view.instructions,
            user_channel=promptlib.load(self._user_channel_fragment()).strip(),
        )

    def _enumerate(self, s: State) -> list[str]:
        obj = parse_or_none(self._complete(self._prompt(s)))
        raw = (obj or {}).get("questions")
        if not isinstance(raw, list):
            self._malformed("not a sub-query list")
            return []
        seen: set[str] = set()
        out: list[str] = []
        for q in raw:
            q = str(q).strip()
            if q and q not in seen:
                seen.add(q)
                out.append(q)
        return out

    # ---- stage 2: iterative chain refinement (depth, ESC + query formulator) ----------

    def _stage2_prompt(self, s: State) -> str:
        return promptlib.render(
            self.stage2_prompt_name,
            question=s.view.question,
            instructions=s.view.instructions,
            evidence=render_evidence(s.evidence),
            draft=(s.draft.text if s.draft else "(no draft yet)"),
            history=render_history(s.history),
            user_channel=promptlib.load(self._user_channel_fragment()).strip(),
        )

    def act(self, s: State) -> Action:
        if self.max_asks is not None and self._n_asks >= self.max_asks:
            return Stop(reason="policy_stop")
        if self._queue is None:
            self._queue = self._enumerate(s)
        if self._queue:
            question = self._queue.pop(0)
            self._n_asks += 1
            return Ask(text=question, rationale="par2_rag stage 1: coverage anchoring")
        # Stage 1 exhausted (possibly empty on the first call): stage 2 takes over.
        if self._stage2_asks >= self.stage2_steps:
            return Stop(reason="policy_stop")
        action = self._decide(self._complete(self._stage2_prompt(s)), s)
        if isinstance(action, Ask):
            self._n_asks += 1
            self._stage2_asks += 1
        return action

    def _decide(self, text: str, s: State) -> Action:
        return self._action_from(parse_or_none(text), s) or self._malformed(
            "not an action (stage 2: ESC + query formulator)"
        )
