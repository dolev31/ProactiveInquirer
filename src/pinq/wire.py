"""The trained-policy seam. HTTP, not imports.

WHY THESE DATACLASSES LIVE IN `pinq` AND NOT IN `pi_run`. Both sides of the seam need the
same shapes: `pi_run.serve` writes them and `pinq_train` reads them. Putting them in
`pi_run` would force `pinq_train` to import a PAPER root, and a PAPER root imports
`pi_eval`, so the gold firewall would be one transitive edge away from broken. Putting them
here costs nothing: `pinq` is stdlib-only and both roots already depend on it.

WHY THERE IS A SEAM AT ALL. The trainer must never be able to read gold. Not "should not" —
must not, in the sense that no import path exists. So it asks a *server* for a number and
gets a number back. The server runs with `PI_GOLD_ROOT` set; the trainer runs with it unset.
The refusal is enforced on the server side, in `ScoreRequest.split_assert`, because a check
the trainer performs on itself is a check the trainer can skip.

WHY THE POLICY IS A `base_url`. `RolloutRequest.policy_base_url` addresses an
OpenAI-compatible endpoint. A vLLM server holding a LoRA adapter, a prompted open model and
a frontier API are then the same object to this code, which is precisely what makes the
three experiment families comparable: `ModelPin.base_url_sha` is the only field that differs
between them. The alternative — an `if trained:` branch — would put an untestable
difference in the middle of the comparison the paper is about.

WHY `/retrieve` IS SEARCH-R1 SHAPED. Search-R1, ReCall and Search-o1 all speak one small
retrieval protocol. Matching it byte for byte means those baselines can be pointed at OUR
frozen corpus with no code change, so "we compared against Search-R1" means the same
retriever and not merely the same name.

Everything here is a frozen dataclass with `to_dict`/`from_dict`, and nothing here validates
semantics: the server validates, because that is where refusing is load-bearing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Literal, Mapping, Sequence, TypeVar

# The exit code a scorer returns when it is asked to score a task outside `train`. Not 1:
# 1 is "something went wrong", and this is "you asked me to do the one thing that would
# void the experiment", which a CI job should be able to tell apart.
SPLIT_REFUSED_EXIT = 3

ScoreMode = Literal["terminal", "prefix_ladder", "loo", "stop_probe"]

T = TypeVar("T")


def to_dict(obj: Any) -> dict[str, Any]:
    if not is_dataclass(obj):
        raise TypeError(f"{type(obj).__name__} is not a wire dataclass")
    return asdict(obj)


def from_dict(cls: type[T], payload: Mapping[str, Any]) -> T:
    """Construct from a JSON body, ignoring unknown keys.

    Ignoring rather than raising is deliberate and it is the only forgiving thing in this
    file: a newer server that adds a response field must not break an older trainer mid-run.
    Unknown keys in a *request* are equally ignored, and the server then behaves as if they
    were never sent — which is visible in the echoed request, so nothing is silent.
    """
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in payload.items() if k in known})  # type: ignore[call-arg]


# --------------------------------------------------------------------------- POST /rollout


@dataclass(frozen=True, slots=True)
class RolloutRequest:
    """One (task, arm, seed) rollout, executed by a worker that cannot read gold.

    `sampling` is a free mapping rather than typed fields because it is passed through to an
    OpenAI-compatible endpoint verbatim and hashed into `ModelPin.sampling_sha`; typing it
    here would mean editing `pinq` every time a provider adds a knob.
    """

    task_id: str
    suite_id: str
    seed: int = 0
    max_turns: int = 16
    arm_id: str = "inquirer_prompted"
    policy_base_url: str = ""
    policy_model: str = ""
    policy_adapter: str | None = None
    sampling: Mapping[str, float | int | str] = field(default_factory=dict)
    k: int = 5
    budget_cap: int | None = None
    # A directory of candidate prompt templates that shadow the shipped ones for exactly
    # this rollout (pinq.promptlib.OVERLAY_ENV). Rung 0 optimises the Inquirer prompt and its
    # candidates have to be evaluated through the same loop the confirmatory arms run, or the
    # winner was selected under a harness the paper never uses. The template's bytes are
    # hashed into RunManifest.prompt_hashes, so a candidate rollout has its own run_id.
    prompt_overlay: str = ""


@dataclass(frozen=True, slots=True)
class RolloutResponse:
    run_id: str
    status: str
    suite_id: str
    task_id: str
    arm_id: str
    seed: int
    n_turns: int = 0
    n_asks: int = 0
    stop_reason: str = ""
    retrieval_calls: float = 0.0
    tok_total: int = 0
    usd: float = 0.0
    wall_ms: int = 0
    run_dir: str = ""
    error: str = ""


# --------------------------------------------------------------------------- POST /score


@dataclass(frozen=True, slots=True)
class ScoreRequest:
    """Ask the gold side what one recorded episode was worth.

    `split_assert` defaults to `"train"` and the server refuses anything else. It is a field
    rather than an implicit server policy so the refusal is visible in the request body a
    reviewer can read, and so a future eval-side caller has to *say* it is asking for
    something the trainer may not have.
    """

    episode_id: str
    mode: ScoreMode = "terminal"
    split_assert: Literal["train"] = "train"


@dataclass(frozen=True, slots=True)
class TurnScore:
    """One decision point, as measured — never as the policy described itself.

    `n_retrieved`/`n_new` come from `Turn.retrieved_uids`, the objective record of what the
    retriever returned. `phi_loo` is the REPORTED metric and is returned for diagnostics
    only; the trainer's reward is built from `potential`, because a leave-one-out value is
    not available online and pretending otherwise would train on a quantity the deployed
    policy can never see.
    """

    turn_idx: int
    n_retrieved: int = 0
    n_new: int = 0
    phi_loo: float | None = None
    qid: str = ""


@dataclass(frozen=True, slots=True)
class ScoreResponse:
    """The measurements. NOT the reward.

    The server returns components and the trainer applies weights, for two reasons. First,
    `pi_run` may not import `pinq_train` (import-linter contract 4), so the weights
    physically cannot live here. Second, re-weighting a finished rollout set must be
    arithmetic over stored numbers rather than a re-score, exactly as re-pricing a sweep is
    arithmetic over stored token counts.

    `potential` is Phi at each prefix k, k = 0..K, over RETRIEVED EvidenceUnits. That is the
    anti-matcher-gaming property: rewriting a question's text cannot move it.
    """

    episode_id: str
    split: str
    ok: bool = True
    q_terminal: float = 0.0
    q_ladder: tuple[float, ...] = ()
    potential: tuple[float, ...] = ()
    turns: tuple[TurnScore, ...] = ()
    stopped: bool = False
    stop_reason: str = ""
    n_ret: float = 0.0
    tok_total: int = 0
    wall_ms: int = 0
    n_malformed: int = 0
    scorer_hash: str = ""
    graph_version: str = ""
    supported: bool = True
    note: str = ""


# --------------------------------------------------------------------------- POST /retrieve


@dataclass(frozen=True, slots=True)
class RetrieveRequest:
    """Search-R1's request body, plus two optional fields.

    `queries`, `topk` and `return_scores` are the upstream contract verbatim. `suite_id` and
    `task_id` are additive: they select which frozen corpus answers, and a client that omits
    them gets the server's configured default, so an unmodified Search-R1 client works.
    """

    queries: Sequence[str] = ()
    topk: int = 3
    return_scores: bool = False
    suite_id: str | None = None
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievedDoc:
    """Search-R1 nests the text under `document.contents`; `uid` is additive.

    Carrying our uid alongside is what lets a retrieval made through this endpoint be joined
    back to `evidence.parquet`. Search-R1 ignores the extra key; we need it, because a doc id
    is not an EvidenceUnit identity — `(corpus_id, doc_id, span)` is.
    """

    document: Mapping[str, str]
    score: float = 0.0
    uid: str = ""


@dataclass(frozen=True, slots=True)
class RetrieveResponse:
    result: tuple[tuple[RetrievedDoc, ...], ...] = ()


# --------------------------------------------------------------------------- GET /healthz


@dataclass(frozen=True, slots=True)
class Health:
    """What a caller must know before trusting anything else this server says.

    `gold_root_set` is reported rather than hidden: a `/score` server with it unset will
    refuse every request, and a `/rollout` server with it SET is a firewall breach. One field
    answers both questions, and answering them is cheaper than debugging either.
    """

    ok: bool
    service: str = "pi_run.serve"
    code_version: str = ""
    gold_root_set: bool = False
    suites: tuple[str, ...] = ()
    endpoints: tuple[str, ...] = ("/rollout", "/score", "/retrieve", "/healthz")
