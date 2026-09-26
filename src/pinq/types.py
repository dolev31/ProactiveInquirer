"""The whole vocabulary, in one file. Frozen dataclasses, no behaviour beyond invariants.

Two rules govern everything here:
  1. Nothing in this module may share a field name with pi_eval.gold.GoldNode, so a gold
     field can never be smuggled into an agent-visible object and still typecheck.
  2. State carries no budget. A budget-aware policy makes prefix-k of a B=16 rollout
     non-exchangeable with a true B=k run, and confounds STOP with the cap you announced.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Sequence

from .ids import canon, evidence_uid, h, question_id, subset_hash

TaskId = str
RunId = str
Uid = str

Actor = Literal[
    "inquirer", "drafter", "answerer", "user_sim", "judge", "matcher", "extractor", "digest"
]

StopReason = Literal[
    "policy_stop", "budget", "max_turns", "truncated_prefix", "error", "no_user_channel"
]

CounterfactualKind = Literal[
    "none",
    "prefix",
    "loo",
    "forced_continue",
    "candidate_branch",
    "shapley_perm",
    "random_q_branch",
]


# --------------------------------------------------------------------------- task view


@dataclass(frozen=True, slots=True)
class TaskView:
    """THE LEAKAGE FIREWALL. The only task representation any agent-side code may touch.

    INVARIANTS
      * Constructed solely by pinq.view.make_view(), which raises on any unknown key.
      * Shares no base class and no field name with pi_eval.gold.GoldNode, so neither
        isinstance() nor dataclasses.asdict() can smuggle gold across the boundary.
        Asserted by tests/test_firewall.py.
      * Carries no catch-all mapping: there is nowhere for an unvetted field to hide.
      * corpus_id/corpus_hash are HANDLES to a frozen corpus, never its contents.
    """

    task_id: TaskId
    suite_id: str
    question: str
    instructions: str
    corpus_id: str
    corpus_hash: str
    word_cap: int

    @property
    def key(self) -> str:
        return f"{self.suite_id}/{self.task_id}"


# --------------------------------------------------------------------------- evidence


@dataclass(frozen=True, slots=True)
class EvidenceUnit:
    """The atom of retrieved information. Identity is (corpus, doc, span) — see ids.evidence_uid."""

    uid: Uid
    corpus_id: str
    doc_id: str
    span: str
    title: str
    text: str
    score: float = 0.0

    @classmethod
    def make(
        cls,
        *,
        corpus_id: str,
        doc_id: str,
        span: str,
        title: str,
        text: str,
        score: float = 0.0,
    ) -> "EvidenceUnit":
        return cls(
            evidence_uid(corpus_id, doc_id, span), corpus_id, doc_id, span, title, text, score
        )


@dataclass(frozen=True, slots=True)
class Evidence:
    """A canonically ordered, deduplicated, hashable SET of units.

    `without` IS the leave-one-out operator and `upto` IS the prefix ladder. Both must work
    without ever running the loop, because scoring re-drafts over arbitrary subsets long
    after the rollout finished.
    """

    units: tuple[EvidenceUnit, ...] = ()

    def __post_init__(self) -> None:
        # dedup by uid (first occurrence wins), then sort by uid so subset_hash is canonical
        seen: dict[str, EvidenceUnit] = {}
        for u in self.units:
            seen.setdefault(u.uid, u)
        object.__setattr__(self, "units", tuple(sorted(seen.values(), key=lambda u: u.uid)))

    @staticmethod
    def of(units: Sequence[EvidenceUnit]) -> "Evidence":
        return Evidence(tuple(units))

    def with_units(self, new: Sequence[EvidenceUnit]) -> "Evidence":
        return Evidence(self.units + tuple(new))

    def without(self, uids: frozenset[Uid]) -> "Evidence":
        return Evidence(tuple(u for u in self.units if u.uid not in uids))

    def only(self, uids: frozenset[Uid]) -> "Evidence":
        return Evidence(tuple(u for u in self.units if u.uid in uids))

    @property
    def uids(self) -> frozenset[Uid]:
        return frozenset(u.uid for u in self.units)

    @property
    def doc_ids(self) -> frozenset[str]:
        return frozenset(u.doc_id for u in self.units)

    @property
    def subset_hash(self) -> str:
        return subset_hash(u.uid for u in self.units)

    def __len__(self) -> int:
        return len(self.units)


# --------------------------------------------------------------------------- actions


@dataclass(frozen=True, slots=True)
class Ask:
    """An internal question from the Inquirer to the Drafter.

    `parent_uids` is SELF-REPORTED by the policy under test. It is audited and plotted
    descriptively but NEVER scored; every attribution metric uses Turn.retrieved_uids,
    which is an objective record of what the retriever actually returned.
    """

    text: str
    rationale: str = ""
    parent_uids: tuple[Uid, ...] = ()
    target: Literal["kb", "user"] = "kb"

    @property
    def qid(self) -> str:
        return question_id(self.text)


@dataclass(frozen=True, slots=True)
class Stop:
    reason: str = "policy_stop"


Action = Ask | Stop


# --------------------------------------------------------------------------- telemetry


@dataclass(frozen=True, slots=True)
class CallTelemetry:
    """One row per LLM/HTTP call.

    `usd` is computed as tokens x price_table[model] and is therefore deterministic and
    reproducible. `wall_ms`/`ttft_ms` are MACHINE ARTIFACTS and must never be compared
    across arms as evidence — enforced by excluding them from the parity assertion.
    """

    call_id: str
    actor: Actor
    model: str
    provider: str
    request_sha: str
    response_sha: str
    tok_prompt: int = 0
    tok_completion: int = 0
    tok_reasoning: int = 0
    tok_cached: int = 0
    usd: float = 0.0
    wall_ms: int = 0
    ttft_ms: int = 0
    cache_hit: bool = False
    retries: int = 0
    rate_limit_stall_ms: int = 0
    http_status: int = 200
    provider_error_code: str | None = None
    attempt: int = 0


@dataclass(frozen=True, slots=True)
class Usage:
    """A reducer over CallTelemetry. Addition must be associative and have Usage() as identity."""

    tok_prompt: int = 0
    tok_completion: int = 0
    tok_reasoning: int = 0
    tok_cached: int = 0
    usd: float = 0.0
    wall_ms: int = 0
    n_calls: int = 0

    @property
    def tok_total(self) -> int:
        return self.tok_prompt + self.tok_completion + self.tok_reasoning

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.tok_prompt + other.tok_prompt,
            self.tok_completion + other.tok_completion,
            self.tok_reasoning + other.tok_reasoning,
            self.tok_cached + other.tok_cached,
            self.usd + other.usd,
            self.wall_ms + other.wall_ms,
            self.n_calls + other.n_calls,
        )

    @classmethod
    def from_call(cls, c: CallTelemetry) -> "Usage":
        return cls(
            c.tok_prompt, c.tok_completion, c.tok_reasoning, c.tok_cached, c.usd, c.wall_ms, 1
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "tok_prompt": self.tok_prompt,
            "tok_completion": self.tok_completion,
            "tok_reasoning": self.tok_reasoning,
            "tok_cached": self.tok_cached,
            "tok_total": self.tok_total,
            "usd": self.usd,
            "wall_ms": self.wall_ms,
            "n_calls": self.n_calls,
        }


# --------------------------------------------------------------------------- drafts, turns


@dataclass(frozen=True, slots=True)
class Draft:
    text: str
    tool_plan: tuple[Mapping[str, Any], ...] = ()

    @property
    def sha(self) -> str:
        return h("draft", self.text, canon(self.tool_plan))


@dataclass(frozen=True, slots=True)
class Answer:
    """Produced by the FROZEN Answerer: same model pin, same prompt, same word cap for every
    arm, and blind to both the arm id and the budget. Length is therefore controlled by
    construction rather than by post-hoc regression alone."""

    text: str
    evidence_hash: str
    cited_unit_ids: tuple[Uid, ...] = ()
    n_words: int = 0
    stop_reason: StopReason = "policy_stop"


@dataclass(frozen=True, slots=True)
class Turn:
    turn_idx: int
    action: Action
    # THE DRAFTER'S REPLY TEXT, and named for it. This was `response_sha` -- the same name
    # the CALLS schema uses for an actual sha256 digest, one field apart in the same frozen
    # schema. Measured on the shipped turns.parquet: 53 non-empty values, 0 of them matching
    # [0-9a-f]{16,64}; every one was English prose ("The evidence needed is a source that
    # provides an estimate of...").
    #
    # The BEHAVIOUR was right and two components depend on it -- `render_history` and the
    # ancestors policy read it back as text to build prompts -- so the name was the defect, in
    # a schema whose whole purpose is that a column means one thing.
    response_text: str = ""
    retrieved_uids: tuple[Uid, ...] = ()  # OBJECTIVE ev(q_t). Scored metrics read only this.
    new_uids: tuple[Uid, ...] = ()
    draft_sha: str = ""
    # D_t AS TEXT, because the sha alone cannot reconstruct the prompt the policy saw.
    #
    # `cmd_train.render_state` rebuilds the Inquirer's prompt for SFT/DPO and must render the
    # draft slot exactly as the policy saw it. With only a sha it cannot, so it refuses -- and
    # `collect_rows` discards the WHOLE run on one such turn. Measured: 655 of 841 recorded
    # turns carry a draft_sha, leaving 21 exportable runs, all synth fakes. Persisting the text
    # is what makes the training track reachable at all.
    #
    # Same precedent as `response_text` one field up: the Drafter's reply is already stored in
    # full for exactly this reason. Drafts are short (the Answerer's word cap bounds them) and
    # this is the artifact a reader most wants when auditing why a question was asked.
    draft_text: str = ""
    subset_hash_before: str = ""
    subset_hash_after: str = ""
    usage: Usage = Usage()
    # counterfactual machinery
    branch_of_run_id: RunId | None = None
    branch_turn_idx: int | None = None
    candidate_id: str | None = None
    candidate_rank: int | None = None
    forced: bool = False
    # self-reported: AUDITED, NEVER SCORED
    policy_conf: float | None = None
    depth_pred: int | None = None
    # model-free corpus statistics
    bm25_hits: int | None = None
    spec_bits: float | None = None


# --------------------------------------------------------------------------- state


@dataclass(frozen=True, slots=True)
class State:
    """s_t = (x, D_t, E_t, H_t).

    DELIBERATELY ABSENT: remaining budget, cap, step limit, arm id, gold anything.
    Truncation is EXTERNAL, applied by the driver, never announced to the policy.
    Enforced by tests/test_firewall.py::test_state_has_no_budget_field.
    """

    view: TaskView
    evidence: Evidence = Evidence()
    draft: Draft | None = None
    history: tuple[Turn, ...] = ()

    @property
    def t(self) -> int:
        return len(self.history)

    @property
    def asks(self) -> tuple[Ask, ...]:
        return tuple(turn.action for turn in self.history if isinstance(turn.action, Ask))

    def with_turn(self, turn: Turn, units: Sequence[EvidenceUnit] = ()) -> "State":
        return replace(
            self,
            evidence=self.evidence.with_units(units),
            history=self.history + (turn,),
        )

    def with_draft(self, draft: Draft) -> "State":
        return replace(self, draft=draft)

    @property
    def key(self) -> str:
        return h(
            "state",
            self.view.key,
            self.evidence.subset_hash,
            canon([a.qid for a in self.asks]),
        )


# --------------------------------------------------------------------------- environment


@dataclass(frozen=True, slots=True)
class EnvCall:
    """One tool invocation against a stateful world. Reads AND writes: `mutating` disambiguates.

    This log is the substrate from which tau2's DB-hash reward and PARE's oracle validation
    are recomputed offline, which is why the Outcome is not a string.
    """

    seq: int
    turn_idx: int
    requestor: Literal["assistant", "user"]
    tool_name: str
    kwargs_json: str
    ok: bool
    result_digest: str
    mutating: bool
    unlock_required: bool = False
    unlock_satisfied: bool = False
    latency_ms: int = 0


@dataclass(frozen=True, slots=True)
class Outcome:
    """The unit of scoring. Not a string: a hash over an executed action sequence is not
    computable from an answer string, which is exactly what broke the earlier design."""

    answer: Answer | None = None
    env_calls: tuple[EnvCall, ...] = ()
    env_final_hashes: Mapping[str, str] = field(default_factory=dict)
    native: Mapping[str, float] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    transcript_digest: str = ""


@dataclass(frozen=True, slots=True)
class Trajectory:
    view: TaskView
    turns: tuple[Turn, ...]
    evidence: Evidence
    outcome: Outcome
    usage: Usage
    stop_reason: StopReason
    calls: tuple[CallTelemetry, ...] = ()
    # Everything charged OUTSIDE a turn window: the act() that returned STOP, the final draft,
    # and the frozen Answerer. It is not overhead to be hidden -- it is the fixed cost of
    # producing an answer at all, it is the same for every arm by construction, and
    # `sum(turn.usage) + terminal_usage == ledger.usage` is what makes the token-parity
    # reconciliation a real invariant rather than one that only llm_free arms can satisfy.
    terminal_usage: Usage = field(default_factory=Usage)
    # D_T: the Drafter's belief at the moment the Answerer was called. Carried on the
    # trajectory rather than dug out of the Outcome because a caller that must ACT on the
    # draft -- the tau2 driver, which emits `tool_plan` as real tool calls through the
    # Orchestrator -- needs the object, not its sha. `Outcome.artifacts` is a str->str map and
    # a tool plan is neither a string nor an artifact.
    final_draft: "Draft | None" = None

    @property
    def n_asks(self) -> int:
        return sum(1 for t in self.turns if isinstance(t.action, Ask))

    def prefix(self, k: int) -> "Trajectory":
        """EXTERNAL truncation to the first k turns.

        stop_reason becomes 'truncated_prefix', NEVER 'budget'. The gap between the
        truncated-prefix curve and the true budget-conditioned curve is the evidence of
        budget-adaptivity; labelling a truncation as a budget stop merges the two curves
        and destroys that evidence.
        """
        kept = self.turns[:k]
        keep_uids = frozenset(u for t in kept for u in t.retrieved_uids)
        return Trajectory(
            view=self.view,
            turns=kept,
            evidence=self.evidence.only(keep_uids),
            outcome=Outcome(),
            usage=self.usage,
            stop_reason="truncated_prefix",
            calls=self.calls,
            terminal_usage=self.terminal_usage,
            # NOT carried: D_T is the belief at turn T, and a k-prefix did not reach it.
            final_draft=None,
        )

    def evidence_of_turn(self, turn_idx: int) -> frozenset[Uid]:
        """ev(q_t) for the leave-one-out estimator. Objective: what the retriever returned."""
        for t in self.turns:
            if t.turn_idx == turn_idx:
                return frozenset(t.retrieved_uids)
        return frozenset()


# --------------------------------------------------------------------------- pins, manifest


@dataclass(frozen=True, slots=True)
class ModelPin:
    """Addresses a LoRA endpoint and a frontier API identically.

    base_url_sha is the ONLY field that differs across the trained / untrained / frontier
    experiment families, which is what makes those three comparable at all.
    """

    role: Actor
    model_id: str
    provider: str
    base_url_sha: str = ""
    adapter_sha: str | None = None
    system_prompt_sha: str = ""
    sampling_sha: str = ""
    price_table_version: str = ""

    @property
    def key(self) -> str:
        return h(
            "pin",
            canon(
                {
                    "role": self.role,
                    "model_id": self.model_id,
                    "provider": self.provider,
                    "base_url_sha": self.base_url_sha,
                    "adapter_sha": self.adapter_sha,
                    "system_prompt_sha": self.system_prompt_sha,
                    "sampling_sha": self.sampling_sha,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Identity and provenance of one (task, arm, seed) rollout — the join key for everything.

    semantic_hash governs RUN IDENTITY. It is NOT the per-call cache key (that is
    ids.request_sha over the exact serialized request), so a prompt typo invalidates one
    cache entry rather than an entire sweep.
    """

    suite_id: str
    task_id: TaskId
    arm_id: str
    policy_id: str
    seed: int
    split: Literal["train", "dev", "test"]
    corpus_hash: str
    budget_cap: int
    max_turns: int
    word_cap: int
    pins: Mapping[str, ModelPin] = field(default_factory=dict)
    prompt_hashes: Mapping[str, str] = field(default_factory=dict)
    upstream_pins: Mapping[str, str] = field(default_factory=dict)
    template_id: str | None = None
    # INSIDE semantic_hash. A pilot exists so its results can be LOOKED AT before the
    # confirmatory analysis is fixed, which is exactly what makes them inadmissible afterwards.
    # With this outside run identity, a pilot and a confirmatory run of the same cell produced
    # the same semantic_hash, the same run_id and therefore the same run directory and the same
    # resume sentinel -- so `--resume` skipped the confirmatory unit as already done and the
    # pilot's numbers were published as confirmatory. Verified before the fix: build_manifest
    # with pilot_flag=True and pilot_flag=False returned byte-identical run_ids.
    pilot_flag: bool = False
    # INSIDE semantic_hash, for exactly pilot_flag's reason. An EXPLORATORY run -- a canary,
    # a rehearsal, an oracle probe -- exists to be LOOKED AT before the analysis is fixed,
    # which is what makes it inadmissible afterwards. Measured after the tier0 canary: 60 of
    # its 66 runs passed `pi_eval.report.ELIGIBLE`, and its 3 musique/inquirer_prompted runs
    # at max_turns=1 sat in the same (suite, arm) cell as 14 runs at max_turns=16, one
    # groupby away from being averaged into a published mean. The grid said
    # `exploratory: true`; that reached the console banner and nothing else.
    exploratory: bool = False
    gold_exposed: bool = False
    counterfactual_kind: CounterfactualKind = "none"
    prefix_of_run_id: RunId | None = None
    prefix_k: int | None = None
    branch_of_run_id: RunId | None = None
    # The turn the branch forked at. IN RUN IDENTITY, with branch_of_run_id: branching one
    # parent at turn 2 and at turn 5 are different experiments.
    branch_turn_idx: int | None = None
    # The seed the Inquirer is re-reset with AT the branch turn -- i.e. WHICH candidate this
    # is. In identity because every candidate at one turn shares branch_of_run_id,
    # branch_turn_idx AND the parent's `seed` (the prefix must reproduce the parent), so
    # without this all n candidates hash identically and --resume keeps only the first.
    branch_seed: int | None = None
    # WHICH PUBLIC TRACE THIS RUN WAS FORKED FROM, and where it was cut. A fork is identical to
    # a fresh rollout of the same task on every other semantic field -- same suite, arm, seed,
    # corpus, prompts and code -- so without these, forking one task at two cut points produces
    # one run_id, one run directory, and `--resume` keeps only the first. Same argument as
    # branch_seed above, same failure mode.
    foreign_trace_sha: str | None = None
    foreign_prefix_k: int | None = None
    permutation_id: int | None = None
    prompt_variant_id: str = "v1"
    train_id_set_hash: str | None = None
    code_version: str = ""
    dirty: bool = False
    # WHICH FILES MADE IT DIRTY. `dirty` alone forces every dev- run into one bucket: it cannot
    # distinguish a run whose tree had `src/pinq_expt/policies/` uncommitted -- disqualifying,
    # the code under test changed -- from one whose tree had `paper/sections/mechanism.tex`
    # uncommitted, which cannot have altered a rollout. That distinction came up repeatedly on
    # 2026-09-17 and could not be made from any record: measured over 109,503 manifests under
    # runs/, 9,094 carry dirty=True and 0 carried a name, because this field did not exist and
    # `manifest_to_dict` wrote its key from a `getattr` default.
    #
    # DELIBERATELY NOT IN semantic_hash, and the one field where that is not merely prudent.
    # A repo-wide flag already picks up a CONCURRENT session's editor state -- that is how
    # 6,754 of 8,811 ok musique runs came out dev- -- so hashing the list would make a run's
    # identity depend on which files a colleague had open, and every finished dev- run would be
    # renamed by this commit. `dirty` itself stays the only tree fact in `run_id`, through the
    # dev- prefix. Same reason as canary_nonces_loaded and grid_sha256: launch environment,
    # not experiment.
    #
    # A MANIFEST WRITTEN BEFORE THIS COMMIT CANNOT HAVE IT, so an empty list is NOT evidence of
    # a clean tree -- read `dirty` for that, and see the module docstring of
    # tests/test_dirty_files_recorded.py for how to tell an old record from a genuinely clean
    # one. History is not backfilled: rewriting a provenance record after the fact is worse
    # than a known gap.
    dirty_files: tuple[str, ...] = ()
    canary_hit: bool = False
    firewall_ok: bool = True
    # HOW MANY NEEDLES LAYER 4 HAD LOADED when the arming check ran for this unit. The three
    # flags above are the same for an armed run that leaked nothing and a run whose tripwire was
    # never armed -- `scan` returns () either way -- so on their own they cannot support the
    # claim that all four firewall layers stayed alive on the runs we publish. None means no
    # arming check ran in this process and the manifest declines to state it; 0 means the check
    # RAN and the layer could not fire, which is the case worth finding. Outside `semantic_hash`
    # deliberately: it describes the launch environment, not the experiment.
    canary_nonces_loaded: int | None = None
    concurrency: int = 1
    # WHERE the corpus lived: the content-addressed DIRECTORY NAME under
    # data/corpora/<suite>/. NOT the same string as `corpus_hash`, which is the adapter's
    # order-independent hash over (doc_id, title, sha256(text)) triples -- a different
    # function over different inputs. Both are legitimate; what was not legitimate was
    # using one where the other was meant, which is exactly what
    # `pi_eval.score.load_questions` did: it opened
    # data/corpora/<suite>/<corpus_hash>/tasks.jsonl -- a path that never exists -- and so
    # returned ZERO questions for every run, on the path that feeds the judges.
    # Deliberately NOT in semantic_hash: `corpus_hash` is the identity, this is the pointer.
    corpus_dir: str = ""
    # WHICH GRID FILE produced this run, and its exact bytes. `pi_run.grids.load` has always
    # computed this and `cli.py` printed it to stdout, but it entered NO record -- so a run
    # could not be traced back to the grid text that produced it, and a grid edited between
    # two sweeps left no evidence of which version ran.
    #
    # DELIBERATELY NOT IN semantic_hash. Provenance, not identity: two grids can legitimately
    # produce the same run, and hashing the file would make a reworded YAML comment change
    # every run_id and evict a whole sweep. Same reason the per-call cache key and
    # semantic_hash are kept apart.
    grid_sha256: str = ""
    grid_name: str = ""
    # THE RETRIEVAL BUDGET. Inside semantic_hash, because it is a treatment parameter and not
    # an implementation detail: `tier1_confirmatory` pins k=5 for musique, 2 for strategyqa and
    # 3 for wiki2, and the probe that chose them was re-measured on 2026-08-24. Without it here
    # the SAME task at k=2 and k=8 produced the SAME run_id, so `--resume` skipped the second
    # as already done and a k sweep silently reported one k twice.
    k: int = 0
    # Identity of the RECORDED QUESTION LIST a replay or ceiling arm was seeded with. Empty for
    # every arm that generates its own. Inside semantic_hash for the same reason: seeding
    # `gold_evidence` from `--questions-from-file A` and then from B produced identical run ids,
    # so the second invocation resumed the first's results and the file was silently ignored --
    # which is precisely how both ceiling arms came to share one input.
    questions_hash: str = ""

    @property
    def model_pin_hash(self) -> str:
        return h("pins", canon({r: p.key for r, p in sorted(self.pins.items())}))

    @property
    def semantic_hash(self) -> str:
        from .ids import semantic_hash as _sem

        return _sem(
            {
                "suite_id": self.suite_id,
                "arm_id": self.arm_id,
                "seed": self.seed,
                "task_id": self.task_id,
                "corpus_hash": self.corpus_hash,
                "model_pin_hash": self.model_pin_hash,
                "prompt_hashes": dict(sorted(self.prompt_hashes.items())),
                "budget_cap": self.budget_cap,
                "max_turns": self.max_turns,
                "k": self.k,
                "questions_hash": self.questions_hash,
                "pilot_flag": self.pilot_flag,
                "exploratory": self.exploratory,
                "prompt_variant_id": self.prompt_variant_id,
                # A CANDIDATE BRANCH IS NOT ITS PARENT. `candidate_seed(s, 0) == s` by
                # construction -- candidate 0 is the on-policy continuation and reuses the
                # parent's seed -- so without these a candidate-0 branch agreed with its
                # parent on every semantic field there is, shared its run_id and therefore
                # its run DIRECTORY, and `--resume` skipped it as already done. Same argument
                # as pilot_flag and exploratory, same failure mode.
                "branch_of_run_id": self.branch_of_run_id,
                "branch_turn_idx": self.branch_turn_idx,
                "branch_seed": self.branch_seed,
                "foreign_trace_sha": self.foreign_trace_sha,
                "foreign_prefix_k": self.foreign_prefix_k,
                "code_version": self.code_version,
                "upstream_pins": dict(sorted(self.upstream_pins.items())),
            }
        )

    @property
    def run_id(self) -> RunId:
        """A dirty working tree yields a 'dev-' prefix, and dev- runs are mechanically
        excluded from every reported table by the aggregator."""
        core = h("runid", self.semantic_hash, self.counterfactual_kind, str(self.prefix_k))
        return ("dev-" if self.dirty else "") + core[:32]
