"""What every LLM-backed policy shares, and nothing else.

TWO RULES THIS FILE EXISTS TO HOLD:

1. `act(s)` TAKES ONLY STATE. No ledger, no context object, no cap, no turns-remaining, and
   no prompt field that could carry one. A budget-aware policy makes prefix-k of a B=16
   rollout non-exchangeable with a true B=k run and confounds STOP with the cap the
   experiment announced — and the main table is built on the prefix ladder, so that is not a
   detail. The policy holds a `Recorder`, which has exactly one method and returns nothing:
   it can report a malformed generation without being able to read a single number back.

2. A MALFORMED GENERATION IS DATA, NOT AN EXCEPTION. A model that emits prose where JSON was
   asked for yields STOP and increments `malformed`. Raising would abort a 30k-rollout sweep
   over one bad sample; silently retrying would hide a real difference between arms, because
   "how often does this policy fail to produce a well-formed decision" is itself a result.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pinq import promptlib
from pinq.protocols import LLM, Recorder
from pinq.types import Action, Ask, State, Stop, TaskView
from pinq_expt.components import LLMRequired, parse_json_object

MALFORMED = "malformed"

# Enough for one question plus a clause of rationale. Deliberately a constant: deriving it
# from anything budget-shaped would leak the cap into the request.
ACT_MAX_TOKENS = 400


class LLMPolicy:
    """Base for every Inquirer that calls a model.

    Subclasses implement `_prompt(s)` and `_decide(text, s)`. Everything about metering,
    malformed handling and budget-blindness is settled here, once.
    """

    # The provider role this component bills against. Collected by pi_run.worker to build
    # RunManifest.pins; see the note there for why an undeclared role is not a cosmetic gap.
    llm_role: ClassVar[str] = "inquirer"

    policy_id = "llm_policy"
    prompt_name = "inquirer_prompted"
    max_asks: int | None = None

    def __init__(
        self,
        llm: LLM | None = None,
        *,
        recorder: Recorder | None = None,
        max_asks: int | None = None,
        may_ask_user: bool = False,
    ) -> None:
        self._llm = llm
        self._recorder = recorder
        self._may_ask_user = may_ask_user
        if max_asks is not None:
            self.max_asks = max_asks
        self.malformed = 0
        self._view: TaskView | None = None
        self._seed = 0
        self._n_asks = 0

    # ------------------------------------------------------------------ identity

    @property
    def may_ask_user(self) -> bool:
        """PUBLIC because the RUNNER has to honour it, not just the parser.

        Two gates decide whether a question reaches the customer: this policy rewrites
        `target="user"` to "kb" unless it is set, and `pinq.loop.run_loop` rejects a non-kb
        target unless `allow_user_target`. `tau2_runner` hard-coded the second to False, so on
        the only suite with a user simulator the arm that exists to open the channel could not
        open it. Reading the loop gate off this attribute is what stops the two disagreeing.
        """
        return bool(self._may_ask_user)

    @property
    def prompt_hashes(self) -> dict[str, str]:
        return promptlib.hashes(self.prompt_name, self._user_channel_fragment())

    def _user_channel_fragment(self) -> str:
        """The real paragraph for the one arm that may address the user, a token-matched
        NON-DIRECTIVE placebo for every other arm. Opening the channel changes what the
        prompt says, never how long it is."""
        return "fragment_user_channel" if self._may_ask_user else "fragment_user_channel_placebo"

    # ------------------------------------------------------------------ lifecycle

    def reset(self, view: TaskView, seed: int) -> None:
        if self._llm is None:
            raise LLMRequired(
                f"{type(self).__name__} needs an LLM. Build the arm with "
                "pinq_expt.arms.build(arm, llm=...) rather than calling the factory bare."
            )
        self._view = view
        self._seed = seed
        self._n_asks = 0
        self.malformed = 0
        self._on_reset(view, seed)

    def _on_reset(self, view: TaskView, seed: int) -> None:
        return None

    # ------------------------------------------------------------------ the decision

    def act(self, s: State) -> Action:
        """Sees State only. There is no second parameter and there will not be one."""
        if self.max_asks is not None and self._n_asks >= self.max_asks:
            return Stop(reason="policy_stop")
        action = self._decide(self._complete(self._prompt(s)), s)
        if isinstance(action, Ask):
            self._n_asks += 1
        return action

    def _complete(self, prompt: str) -> str:
        assert self._llm is not None  # reset() has already refused a None llm
        text, _ = self._llm.complete(
            role="inquirer",
            messages=[{"role": "user", "content": prompt}],
            seed=self._seed + self._n_asks,
            max_tokens=ACT_MAX_TOKENS,
            actor="inquirer",
        )
        return text

    def _prompt(self, s: State) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def _decide(self, text: str, s: State) -> Action:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------ malformed

    def _malformed(self, why: str = "") -> Stop:
        """Count it, then STOP. Never raise, never retry.

        The count reaches the ledger through a write-only `Recorder`, so `malformed_rate`
        is reportable per arm without the policy being able to read anything back.
        """
        self.malformed += 1
        if self._recorder is not None:
            self._recorder.record(MALFORMED, 1)
        return Stop(reason="policy_stop")

    # ------------------------------------------------------------------ shared parsing

    def _action_from(self, obj: dict[str, Any] | None, s: State) -> Action | None:
        """The strict-JSON contract shared by the prompted policies. None means malformed."""
        if not isinstance(obj, dict):
            return None
        kind = str(obj.get("action", "")).strip().lower()
        if kind == "stop":
            return Stop(reason="policy_stop")
        if kind != "ask":
            return None
        question = str(obj.get("question", "")).strip()
        if not question:
            return None
        target = str(obj.get("target", "kb")).strip().lower()
        if target != "user" or not self._may_ask_user:
            target = "kb"
        raw = obj.get("parent_uids", ())
        cited = [str(r).strip() for r in raw] if isinstance(raw, (list, tuple)) else []
        # Self-reported and AUDITED, never scored (pinq.types.Ask). Resolved against the
        # evidence actually held — the prompt shows a uid prefix, so a prefix match is the
        # honest resolution — which stops a hallucinated citation from entering the audit
        # plot as a dependency edge that never existed.
        parents = tuple(
            u.uid for u in s.evidence.units if any(r and u.uid.startswith(r) for r in cited)
        )
        return Ask(
            text=question,
            rationale=str(obj.get("rationale", ""))[:400],
            parent_uids=parents,
            target=target,  # type: ignore[arg-type]
        )


def parse_or_none(text: str) -> dict[str, Any] | None:
    return parse_json_object(text)
