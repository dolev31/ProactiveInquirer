"""The arm table. One dict, no registry, no plugin discovery, no decorators.

An arm is a triple of components plus a budget cap plus a few flags, and that is the entire
abstraction. The reason to keep it a literal dict rather than a registry is auditability: the
reviewer's question is always "what exactly differed between these two columns of Table 1",
and the answer has to be readable in one screen without following an import-time side effect.

THREE THINGS IN A ROW ARE ASSERTIONS, NOT DECORATION.

`llm_free` is what the worker checks: an arm declared LLM-free that records a single token
has an unmetered call somewhere, and the run fails rather than quietly contributing an
inflated row to a token-parity table.

`budget_cap` is the SAME number for every confirmatory arm. Budget parity is the precondition
for the whole comparison, so an arm that needed a bigger cap to be interesting would be
reporting the cap, not the policy. The two ceiling arms are the exception and are marked.

`prompts` names the template that occupies each SLOT. Arms are compared slot by slot, and
tests/test_arms.py holds the templates in one slot to within 10% of each other in tokens, so
"this arm won" can never reduce to "this arm's scaffolding was longer".

WHY THE FACTORIES TAKE ARGUMENTS. An LLM-backed component needs a client, and some need a
retriever or a recorded question list. `build()` supplies exactly what each constructor
declares and nothing else, so no component can reach a dependency it did not ask for — in
particular, none of them can be handed the ledger, which is how `Inquirer.act` stays
budget-blind by construction rather than by review.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from pinq_expt.components import (
    UNIT_CHARS,
    ComputeMatchedDrafter,
    FrozenLLMAnswerer,
    LLMDrafter,
    QueryExpansionDrafter,
    VerbosityDrafter,
)
from pinq_expt.fakes import (
    BreadthOnlyInquirer,
    ChainInquirer,
    EchoDrafter,
    FrozenAnswerer,
    NeverAsk,
    VerbatimInquirer,
)
from pinq_expt.policies import (
    CapAwarePromptedInquirer,
    ChecklistInquirer,
    Depth1Inquirer,
    GoldEvidenceInquirer,
    IRCoTInquirer,
    NoEvidenceInquirer,
    OracleVreqInquirer,
    Par2RagInquirer,
    ParallelReplayInquirer,
    PromptedInquirer,
    RandomQInquirer,
    SelfAskInquirer,
    SelfInquireInquirer,
)

# The single hard-capped currency is retrieval_calls (pinq.budget.HARD_CURRENCY). 16 is the
# main-grid cap: high enough that the chain policy on the synthetic suite is never truncated,
# so a STOP is always the policy's decision and never the cap's.
DEFAULT_CAP = 16

# Slots. Two arms differing in one slot differ in one prompt, which is what makes the token
# parity check meaningful and the manifest diff readable.
#
# "inquirer_stage2" exists for exactly one arm, par2_rag: PAR2-RAG's own two stages (Coverage
# Anchoring, then Iterative Chain Refinement) are two genuinely different templates, not one
# template rendered twice, and declaring only "inquirer" here would leave the second one
# unenumerated -- the exact MissingPrompt-shaped gap tests/test_prompt_templates_exist.py
# exists to catch (see its `_arm_table_templates`). There is nothing yet for it to be
# token-matched against, so its parity family is itself alone, honestly.
SLOTS = ("inquirer", "inquirer_stage2", "drafter", "resolve", "expand", "answerer")


@dataclass(frozen=True, slots=True)
class Arm:
    arm_id: str
    inquirer: Callable[..., Any]
    drafter: Callable[..., Any]
    answerer: Callable[..., Any]
    budget_cap: int = DEFAULT_CAP
    llm_free: bool = True
    kind: str = "control"  # control | treatment | ablation | killswitch | ceiling
    prompts: Mapping[str, str] = field(default_factory=dict)
    # DOES THIS ARM ASK ANYTHING? A `NeverAsk` arm with n_asks == 0 is working; any other arm
    # with n_asks == 0 is `drafter_only` in disguise wearing status=ok, which is the single
    # most dangerous silent failure a sweep can have -- the run costs money, produces a
    # well-formed row, and reports the baseline's behaviour under the treatment's name.
    # Measured on a real musique sweep: `random_q` and `parallel_replay` both did exactly that.
    # Declared rather than inferred, and held equal to the policy class by a test.
    expects_asks: bool = True
    # This arm's REQUESTS duplicate another arm's by construction, so 100% cache hits are
    # correct rather than a dead code path. Found by the tier0 canary, which reported
    # parallel_replay and pare_plus_inquirer_prompted as "0 of N calls reached the provider"
    # while all three arms emitted the byte-identical question on the same task at seed 7.
    # `pi verify arms` skips its live-call check for these; the property worth asserting
    # here is duplication, not novelty.
    input_identical_by_design: bool = False
    # The suites on which this arm's declared behaviour is MEANINGFUL. Empty = anywhere.
    #
    # `expects_asks` is a property of (arm, SUITE), not of the arm alone: ChainInquirer
    # follows the SYNTHETIC suite's dependency chain by construction, so fake_chain cannot
    # ask on musique. The tier0 canary measured exactly that -- 3 runs, n_asks 0.0,
    # n_turns 0.0, tok_total 0 -- and `pi verify arms` reported it as "expected to ask but
    # did not", describing a suite mismatch as a dead policy.
    home_suites: tuple[str, ...] = ()
    requires_questions: bool = False  # seeded from a prior run or from the gold side
    requires_gold: bool = False  # scorer-side only; never buildable on the rollout path
    note: str = ""


@dataclass(frozen=True, slots=True)
class Components:
    inquirer: Any
    drafter: Any
    answerer: Any


def _make(factory: Callable[..., Any], ctx: Mapping[str, Any]) -> Any:
    """Pass only what this constructor declares.

    Five lines instead of a dependency-injection container, and one property worth the
    explicitness: `ledger` is never in `ctx`, so no policy can receive one by accident.
    """
    params = inspect.signature(factory).parameters
    return factory(**{k: v for k, v in ctx.items() if k in params})


def build(
    arm: Arm,
    *,
    llm: Any = None,
    retriever: Any = None,
    recorder: Any = None,
    questions: Mapping[str, Any] | None = None,
    ask_counts: Mapping[str, int] | None = None,
    tools: Sequence[Mapping[str, Any]] = (),
    unit_chars: int | None = None,
) -> Components:
    """Instantiate an arm's three components.

    Deliberately NOT accepting a ledger. A Drafter gets one per call through the `Drafter`
    protocol, where retrieval is metered; an Inquirer never does, and the only thing it can
    report — a malformed generation — goes through the write-only `recorder`.

    `tools` is the ADVERTISED tool schema of a suite that has a stateful environment
    (`Tau2Suite.tool_schemas`), and it reaches only the constructors that declare it — the
    Drafters. An Inquirer must not receive it: its job is to decide what to ASK, and a policy
    that could see the action space would start planning actions inside `act(s)`, where
    nothing meters them and the budget-blindness argument no longer holds. Empty for every
    suite without an environment, which is what keeps a tau2 change from touching musique's
    prompt bytes.

    `unit_chars` is the evidence window of a suite whose units are RECORDS, not paragraphs
    (tau2; see `pi_run.stages.tau2_runner.TAU2_EVIDENCE_CHARS`). Like `tools` it reaches only
    the constructors that declare it -- the Drafters and the Answerer -- and NO Inquirer
    declares it, so the questioner's view is `UNIT_CHARS` on every suite: that is the view
    the trained questioner learned on. None, the default and every QA caller, leaves it out
    of the context entirely, so a QA arm is built exactly as before.
    """
    if arm.requires_questions and not questions:
        raise ValueError(
            f"arm {arm.arm_id!r} replays a recorded question list; pass questions=... "
            "(pinq_expt.policies.recorded_questions for a prior run)."
        )
    ctx: dict[str, Any] = {
        "llm": llm,
        "retriever": retriever,
        "recorder": recorder,
        "questions": questions,
        "ask_counts": ask_counts,
        "tools": tuple(tools or ()),
    }
    if unit_chars is not None:
        ctx["unit_chars"] = int(unit_chars)
    return Components(
        inquirer=_make(arm.inquirer, ctx),
        drafter=_make(arm.drafter, ctx),
        answerer=_make(arm.answerer, ctx),
    )


# --------------------------------------------------------------------------- the table

_LLM_SLOTS = {
    "drafter": "drafter_draft",
    "resolve": "drafter_resolve",
    "answerer": "answerer_frozen",
}

ARMS: dict[str, Arm] = {
    # ------------------------------------------------------------------ LLM-free reference
    # Built from pinq_expt.fakes: they cost nothing, run in CI with no key, and give every
    # metric a closed-form value on the synthetic suite. They are the arms the LOOP is
    # debugged on and they are named `fake_*` for one reason: the ids without that prefix
    # belong to the preregistered table (pi_eval.prereg.default_stage1), and an id that
    # means a scripted stub in the code and the system in the prereg is exactly how a wrong
    # number gets published.
    "fake_drafter_only": Arm(
        arm_id="fake_drafter_only",
        inquirer=NeverAsk,
        drafter=EchoDrafter,
        answerer=FrozenAnswerer,
        kind="control",
        note="Loop-debug stand-in for drafter_only: zero internal questions, zero tokens.",
        expects_asks=False,  # NeverAsk: zero asks is correct here
    ),
    "fake_chain": Arm(
        arm_id="fake_chain",
        home_suites=("synth",),  # its inquirer walks the synth dependency chain
        inquirer=ChainInquirer,
        drafter=EchoDrafter,
        answerer=FrozenAnswerer,
        kind="treatment",
        note="Loop-debug stand-in for inquirer_prompted: follows the synthetic suite's "
        "dependency chain by construction, so every metric has a hand-computable value.",
    ),
    "fake_depth1": Arm(
        arm_id="fake_depth1",
        home_suites=("synth",),  # its inquirer walks the synth dependency chain
        inquirer=BreadthOnlyInquirer,
        drafter=EchoDrafter,
        answerer=FrozenAnswerer,
        kind="ablation",
        note="Loop-debug stand-in for inquirer_depth1: asks only about needs stated in the "
        "question, and on a token-gated suite therefore cannot reach depth >= 1.",
    ),
    # ------------------------------------------------------------------ the comparator
    "drafter_only": Arm(
        arm_id="drafter_only",
        inquirer=NeverAsk,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="control",
        prompts=dict(_LLM_SLOTS),
        note="The comparator, not a strawman: the real Drafter alone, zero internal "
        "questions, the same frozen Answerer as every other arm.",
        expects_asks=False,  # NeverAsk: zero asks is correct here
    ),
    # ------------------------------------------------------- the benchmark's own comparator
    "tau2_stock": Arm(
        arm_id="tau2_stock",
        home_suites=("tau2", "tau2_retail", "tau2_airline"),
        # NONE OF THESE THREE IS EVER INVOKED. `pi_run.stages.tau2_runner._simulate` branches
        # on `arm_id` and hands the Orchestrator upstream's own `LLMAgent` instead of our
        # driver, so this arm runs no `run_loop`, renders none of our prompts at generation
        # time and emits no ask. They are the REAL components rather than the `fake_*` stubs
        # for two reasons that are both enforced by tests: `llm_free` is exactly the three
        # `fake_*` arms (tests/test_runtime.py), so relabelling a paper arm LLM-free is the
        # precise hazard that test exists to stop; and `collect_pins` reads `llm_role` off
        # these objects, which is how `PI_MODEL_DRAFTER` -- the model upstream's agent
        # actually runs on, see `tau2_runner._stock_model` -- reaches `model_pin_hash` and
        # therefore run identity. The manifest's `prompt_hashes` consequently name this
        # repository's drafter and answerer templates; they had NO effect on this arm's
        # number, and `upstream_pins.agent_model` is what distinguishes one stock run from
        # another.
        inquirer=NeverAsk,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        prompts=dict(_LLM_SLOTS),
        kind="control",
        note="THE BENCHMARK'S OWN AGENT. tau2's `LLMAgent`, unchanged, on the same "
        "environment and the same user simulator, pinned to the Drafter's role model so "
        "that stock-versus-augmented is an architecture comparison and not a model one. "
        "It is tau2-only: on any other suite it would be `drafter_only` with the loop "
        "removed, which is not a thing this repository measures.",
        expects_asks=False,  # upstream's agent has no Inquirer to ask with
    ),
    # ------------------------------------------------------------------ the treatment
    "inquirer_prompted": Arm(
        arm_id="inquirer_prompted",
        inquirer=PromptedInquirer,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="treatment",
        prompts={"inquirer": "inquirer_prompted", **_LLM_SLOTS},
        note="THE SYSTEM. Sees (x, D_t, E_t, H_t), emits ASK or STOP as strict JSON.",
    ),
    # A reviewer's objection, made testable. The equal-spend reading takes the comparator's first
    # k questions, which is exchangeable with a true k-question run only because the comparator is
    # budget-blind. If a comparator TOLD its allowance would ask different, better early questions,
    # the prefix would understate it. This arm is the prompted comparator with one block of its
    # prompt replaced: the RESERVED placebo, which exists to be displaced, becomes a statement of
    # the allowance in words. Same class, same slots, same drafter and answerer, same token count
    # to within five characters. It is NOT part of the sealed stage-1 set, so it is exploratory.
    "inquirer_prompted_capaware": Arm(
        arm_id="inquirer_prompted_capaware",
        inquirer=CapAwarePromptedInquirer,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="treatment",
        prompts={"inquirer": "inquirer_prompted_capaware", **_LLM_SLOTS},
        note="The comparator told its allowance, to test whether the prefix reading understates it.",
    ),
    "inquirer_trained": Arm(
        arm_id="inquirer_trained",
        inquirer=PromptedInquirer,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="treatment",
        prompts={"inquirer": "inquirer_prompted", **_LLM_SLOTS},
        note="Same policy class and the same prompt as inquirer_prompted: the ONLY "
        "difference is the model pin, which is what makes 'trained vs prompted' a fair "
        "fight rather than a prompt contest. The pin lives in the manifest, not here.",
    ),
    "inquirer_may_ask_user": Arm(
        arm_id="inquirer_may_ask_user",
        inquirer=lambda llm=None, recorder=None: PromptedInquirer(
            llm, recorder=recorder, may_ask_user=True
        ),
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="ablation",
        prompts={"inquirer": "inquirer_prompted", **_LLM_SLOTS},
        note="The user channel opened, to quantify the user-private ceiling. The extra "
        "paragraph is token-matched by a placebo in every other arm, so what changes is "
        "what the prompt says and not how much of it there is.",
    ),
    # ------------------------------------------------------------------ ablations
    "inquirer_noevidence": Arm(
        arm_id="inquirer_noevidence",
        inquirer=NoEvidenceInquirer,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="killswitch",
        prompts={"inquirer": "inquirer_noevidence", **_LLM_SLOTS},
        note="KILL SWITCH. Cannot see E_t. If it matches the treatment, the questions were "
        "never conditioned on what was read: vertical proactivity does not exist and this "
        "becomes a horizontal-facet paper.",
    ),
    "inquirer_depth1": Arm(
        arm_id="inquirer_depth1",
        inquirer=Depth1Inquirer,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        # PROMOTED from "ablation". It now carries the SEQUENCING kill switch, which used to
        # hang off `parallel_replay` -- an arm whose delta is identically zero by
        # construction, so that switch fired with certainty and was not evidence. This arm
        # enumerates its questions from x alone, so none of them saw an answer AND the
        # strings genuinely differ from the treatment's: the sequencing channel actually varies.
        kind="killswitch",
        prompts={"inquirer": "inquirer_depth1", **_LLM_SLOTS},
        note="Breadth only: restricted to facets nameable from x alone. On a token-gated "
        "suite it cannot reach depth >= 1, which is a property of its prompt rather than "
        "an observation about a model.",
    ),
    "checklist": Arm(
        arm_id="checklist",
        inquirer=ChecklistInquirer,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="ablation",
        prompts=dict(_LLM_SLOTS),
        note="A fixed facet checklist with no state conditioning at all. Tests whether the "
        "policy is conditioned on state, or is a coverage template with extra steps. Its "
        "inquirer slot declares no prompt because it sends none: the facet list is a "
        "template, not a model prompt, and it reaches run identity through the component's "
        "own prompt_hashes rather than through a token-parity family it cannot belong to.",
    ),
    "random_q": Arm(
        arm_id="random_q",
        inquirer=RandomQInquirer,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        # A KILL SWITCH, not a control, and the distinction is the written consequence rather
        # than the mechanism. `pi_eval.prereg.KILL_SWITCHES` says: if this matches, the gain is
        # retrieval VOLUME rather than question content, the Inquirer is not selecting, and the
        # thesis does not survive. An arm carrying a consequence of that shape is a kill switch
        # by definition; calling it a control was the third of three places this comparator's
        # status disagreed with itself.
        kind="killswitch",
        requires_questions=True,
        prompts=dict(_LLM_SLOTS),
        note="THE NULL FOR PHI. Real questions from another task, matched volume. If phi "
        "moves as much here, phi measures retrieval volume and not question content.",
    ),
    "parallel_replay": Arm(
        arm_id="parallel_replay",
        input_identical_by_design=True,  # replays a recorded question list, so its requests ARE the replayed arm's
        inquirer=ParallelReplayInquirer,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="killswitch",
        requires_questions=True,
        prompts=dict(_LLM_SLOTS),
        note="KILL SWITCH. The sequential run's own questions, issued without any of them "
        "seeing what the previous one returned. If it matches, drop the graph framing.",
    ),
    # ------------------------------------------------------------------ ancestors
    "self_ask": Arm(
        arm_id="self_ask",
        inquirer=SelfAskInquirer,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="control",
        prompts={"inquirer": "self_ask", **_LLM_SLOTS},
        note="Self-Ask (Press et al.), verbatim format and canonical exemplars. THE closest "
        "ancestor and the baseline that would kill the paper if it matched.",
    ),
    "self_inquire": Arm(
        arm_id="self_inquire",
        inquirer=SelfInquireInquirer,
        drafter=lambda llm=None, tools=(), unit_chars=UNIT_CHARS: LLMDrafter(
            llm, agentic_resolve=False, tools=tools, unit_chars=unit_chars
        ),
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="killswitch",
        prompts={
            "inquirer": "self_inquire",
            "drafter": "drafter_draft",
            "answerer": "answerer_frozen",
        },  # no resolve slot: with one agent there is nothing to resolve against
        note="KILL SWITCH. ONE agent given the Inquirer prompt verbatim as a self-directive, "
        "answering its own questions — hence a Drafter that does not resolve. If it "
        "matches, the contribution is the protocol and not the two-agent split.",
    ),
    "ircot": Arm(
        arm_id="ircot",
        inquirer=IRCoTInquirer,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="control",
        prompts={"inquirer": "ircot", **_LLM_SLOTS},
        note="IRCoT (Trivedi et al.): each generated reasoning sentence IS the next query. "
        "The contrast against an explicit self-question.",
    ),
    # TRACK 3 (2026-09-22): the structured/learned-retrieval comparator two reviewers named
    # as missing. Training-free, faithfully ported -- see the module docstring of
    # pinq_expt.policies.structured_baselines for exactly what was implemented, what was
    # left out and why, and for why PRISM (the other method named) is documented there and
    # not given an arm of its own: its Selector/Adder cycle curates an already-retrieved
    # pool rather than issuing a sequence of metered Ask actions, and porting it into this
    # slot would move its mechanism into the Drafter, which is the one thing every arm in
    # this table holds fixed.
    "par2_rag": Arm(
        arm_id="par2_rag",
        inquirer=Par2RagInquirer,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="control",
        prompts={
            "inquirer": "par2_rag_plan",
            "inquirer_stage2": "par2_rag_refine",
            **_LLM_SLOTS,
        },
        note="PAR2-RAG (arXiv:2603.29085), training-free: Coverage Anchoring (breadth-first "
        "sub-query enumeration from x alone, one model call) then Iterative Chain "
        "Refinement (an Evidence Sufficiency Controller reads (evidence, draft, history) "
        "and either stops or hands the Query Formulator the next focused query, one strict "
        "JSON action per step). The comparator this track exists to add: a structured, "
        "training-free retrieval baseline at the SAME drafter, answerer and budget_cap as "
        "inquirer_trained and inquirer_prompted, so only the inquirer differs.",
    ),
    # ------------------------------------------------------------------ drafter-side arms
    "verbosity": Arm(
        arm_id="verbosity",
        inquirer=NeverAsk,
        drafter=VerbosityDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="killswitch",
        prompts={"drafter": "drafter_verbosity", "answerer": "answerer_frozen"},
        note="THE KILL SWITCH. The Drafter alone, prompted for exhaustive multi-facet "
        "coverage at the treatment's completion budget. If it matches the Inquirer, the "
        "effect is length and there is no paper. Retrieval is zero, exactly as in "
        "drafter_only: a Drafter cannot retrieve inside draft() without breaking purity.",
        expects_asks=False,  # NeverAsk: zero asks is correct here
    ),
    "compute_matched": Arm(
        arm_id="compute_matched",
        inquirer=NeverAsk,
        drafter=ComputeMatchedDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        # A KILL SWITCH, not a plain control. It is the same shape as `verbosity` -- NeverAsk,
        # a Drafter handed the treatment's budget, a frozen Answerer -- and its match has the
        # same consequence: the gain was the compute, not the inquiry. It carried `control`
        # while `pi_eval.prereg.KILL_SWITCHES` named it nothing at all, so the one comparator
        # that separates the thesis from "spend more tokens" had a preregistered ENDPOINT and
        # no preregistered CONSEQUENCE.
        kind="killswitch",
        prompts={"drafter": "drafter_compute_matched", "answerer": "answerer_frozen"},
        note="Self-consistency@n at token parity with the Inquirer arm: same tokens spent, "
        "spent on sampling instead of on questions. If it matches, the gain is the budget.",
        expects_asks=False,  # NeverAsk: zero asks is correct here
    ),
    "query_expansion": Arm(
        arm_id="query_expansion",
        # One verbatim ask; the Drafter turns it into n queries inside resolve(), where
        # retrieval is metered. Parity is enforced by the hard cap both arms share.
        inquirer=lambda: VerbatimInquirer(max_asks=1),
        drafter=QueryExpansionDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="control",
        # `expand` is an EXTRA call, not a substitution: this arm still resolves with the
        # shared drafter_resolve. The two resolve-time templates are held to one size
        # (promptlib.PARITY_FAMILIES["resolve"]) so the extra call is not also a longer one.
        prompts={"expand": "drafter_query_expansion", **_LLM_SLOTS},
        note="Multi-query / HyDE at equal retrieval calls: the same cap, spent on query "
        "variants instead of on questions.",
    ),
    # ------------------------------------------------------------------ Grid E: PARE
    #
    # THE TRANSFER EXPERIMENT, AND THE ONE SLOT IT IS ALLOWED TO MOVE.
    # Grid E asks whether a policy prompted for proactive information ACQUISITION improves a
    # proactive ACTION agent. The only way that question has an answer is if exactly one
    # thing changes between the three arms: PARE's GOAL-INFERENCE stage, which is the
    # `inquirer` slot here. The executor -- drafter, frozen answerer, and the PareActuator
    # the worker hands the loop -- is IDENTICAL across all three, so a delta is attributable
    # to the inquiry stage rather than to a better actuator we happened to ship.
    #
    # SCORED ON PARE'S OWN ORACLE. `PareActuator.attach_validation` folds upstream's
    # PAREScenarioValidationResult into `native()`, so these arms need no new judge and no
    # new gold. That is what keeps Grid E at ~572 rollouts, and it is why the grid is
    # exploratory: carrying no confirmatory test, it spends none of the multiplicity budget.
    #
    # WHAT `pare_baseline` IS AND IS NOT. It is Observe-Execute with NO inserted inquiry
    # stage: the executor observes app state through the retriever and acts, exactly as the
    # other two arms' executors do. It is NOT a call into upstream's own runner -- that path
    # is `PareSuite.run_upstream`, it does not go through `pinq.loop.run_loop`, and its
    # number is reported as an UPSTREAM REFERENCE rather than as this grid's control.
    # Comparing against a differently-implemented executor would confound the Inquirer with
    # the actuator, which is the one confound this grid cannot afford.
    "pare_baseline": Arm(
        arm_id="pare_baseline",
        inquirer=NeverAsk,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="control",
        prompts=dict(_LLM_SLOTS),
        note="GRID E CONTROL. Observe-Execute with no inserted inquiry stage, on the same "
        "executor as the other two PARE arms. Scored on PARE's own oracle validation.",
        expects_asks=False,  # NeverAsk: zero asks is correct here
    ),
    "pare_plus_inquirer_prompted": Arm(
        arm_id="pare_plus_inquirer_prompted",
        input_identical_by_design=True,  # the SAME PromptedInquirer and template as inquirer_prompted
        inquirer=PromptedInquirer,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="treatment",
        prompts={"inquirer": "inquirer_prompted", **_LLM_SLOTS},
        note="GRID E TREATMENT. The SAME PromptedInquirer and the SAME template as "
        "inquirer_prompted, occupying PARE's goal-inference slot; the executor is held "
        "fixed. If the policy is a general acquisition policy rather than a QA trick, this "
        "is where that shows.",
    ),
    "pare_plus_verbosity": Arm(
        arm_id="pare_plus_verbosity",
        inquirer=NeverAsk,
        drafter=VerbosityDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="killswitch",
        prompts={"drafter": "drafter_verbosity", "answerer": "answerer_frozen"},
        note="GRID E KILL SWITCH, the same one every other grid carries. If proposing more, "
        "at greater length, matches the Inquirer on PARE's oracle, the transfer result is "
        "verbosity and there is nothing to generalize.",
        expects_asks=False,  # NeverAsk: zero asks is correct here
    ),
    # ------------------------------------------------------------------ ceilings
    "gold_evidence": Arm(
        arm_id="gold_evidence",
        inquirer=GoldEvidenceInquirer,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="ceiling",
        budget_cap=64,
        requires_questions=True,
        requires_gold=True,
        prompts=dict(_LLM_SLOTS),
        note="CEILING. Questions selected on the gold side and handed in as strings; "
        "nothing here reads gold, and the firewall forbids the import that would.",
    ),
    "oracle_vreq": Arm(
        arm_id="oracle_vreq",
        inquirer=OracleVreqInquirer,
        drafter=LLMDrafter,
        answerer=FrozenLLMAnswerer,
        llm_free=False,
        kind="ceiling",
        budget_cap=64,
        requires_questions=True,
        requires_gold=True,
        prompts=dict(_LLM_SLOTS),
        note="CEILING. Only the questions with positive measured value, in order. Selected "
        "where phi is computed and delivered here as a list of strings.",
    ),
}


class UnknownArm(KeyError):
    pass


def get(arm_id: str) -> Arm:
    try:
        return ARMS[arm_id]
    except KeyError:
        raise UnknownArm(f"{arm_id!r} is not an arm. Known: {sorted(ARMS)}") from None


def arm_ids() -> tuple[str, ...]:
    return tuple(sorted(ARMS))


def llm_free_arm_ids() -> tuple[str, ...]:
    """The arms a keyless CI run can execute end to end."""
    return tuple(sorted(a for a, arm in ARMS.items() if arm.llm_free))
