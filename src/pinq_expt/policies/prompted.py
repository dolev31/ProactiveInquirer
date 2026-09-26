"""The system, and the two ablations that can kill it.

`PromptedInquirer` sees `(x, D_t, E_t, H_t)`. The two policies below it are that policy with
one input removed, and the removal is the experiment:

  * `NoEvidenceInquirer` cannot see `E_t`. If it matches the system, then the system's
    questions were never conditioned on what it had read, VERTICAL PROACTIVITY DOES NOT
    EXIST, and the paper is a horizontal-facet paper. This is the most important row in the
    table and the reason `E_t` is threaded through `State` rather than kept in the policy.

  * `Depth1Inquirer` is restricted to what `x` alone names. If it matches the system, depth
    contributes nothing over breadth.

Both are prompt-matched to the system: same length, same output contract, same model pin.
The only difference is what the prompt is allowed to contain.
"""

from __future__ import annotations

from pinq import promptlib
from pinq.types import Action, Ask, State, Stop, TaskView
from pinq_expt.components import render_evidence, render_history
from pinq_expt.policies.base import LLMPolicy, parse_or_none


class PromptedInquirer(LLMPolicy):
    """The system: emits ASK(q) or STOP as strict JSON, conditioned on the full state."""

    policy_id = "inquirer_prompted"
    prompt_name = "inquirer_prompted"

    def _prompt(self, s: State) -> str:
        return promptlib.render(
            self.prompt_name,
            question=s.view.question,
            instructions=s.view.instructions,
            evidence=render_evidence(s.evidence),
            draft=(s.draft.text if s.draft else "(no draft yet)"),
            history=render_history(s.history),
            user_channel=promptlib.load(self._user_channel_fragment()).strip(),
        )

    def _decide(self, text: str, s: State) -> Action:
        return self._action_from(parse_or_none(text), s) or self._malformed("not an action")


class CapAwarePromptedInquirer(PromptedInquirer):
    """The comparator told its allowance, for testing whether a prefix understates it.

    WHY THIS CLASS EXISTS AT ALL, rather than the arm naming a prompt. `prompt_name` is a CLASS
    attribute and `_prompt` reads `self.prompt_name`, so an arm's `prompts={"inquirer": ...}` slot
    is NOT consulted by this policy: it feeds the parity and firewall guards and the recorded
    `prompt_hashes`, and nothing else. An arm that names a different inquirer prompt while pointing
    at `PromptedInquirer` therefore runs the BASE prompt and looks like a clean null. Measured, at a
    cost of 800 units and $3.57: both arms recorded identical `prompt_hashes` and an identical
    `tok_prompt` of 1228, and the paired coverage delta was exactly 0.00000 on 200 of 200 tasks with
    one distinct value. Overriding the attribute is what actually changes the request.
    """

    policy_id = "inquirer_prompted_capaware"
    prompt_name = "inquirer_prompted_capaware"


class NoEvidenceInquirer(LLMPolicy):
    """THE decisive ablation: `x` and its own question history, never `E_t`.

    `_prompt` takes `s` and reads `s.view` and the ASK texts in `s.history` — never
    `s.evidence` and never `s.draft`. The blindness is visible in three lines rather than
    asserted in a docstring, and tests/test_arms.py renders the prompt over a state whose
    evidence is present and checks that none of it appears.
    """

    policy_id = "inquirer_noevidence"
    prompt_name = "inquirer_noevidence"

    def _prompt(self, s: State) -> str:
        asked = [t.action.text for t in s.history if isinstance(t.action, Ask)]
        return promptlib.render(
            self.prompt_name,
            question=s.view.question,
            instructions=s.view.instructions,
            history="\n".join(f"Q{i + 1}: {q}" for i, q in enumerate(asked))
            or "(nothing asked yet)",
            user_channel=promptlib.load(self._user_channel_fragment()).strip(),
        )

    def _decide(self, text: str, s: State) -> Action:
        return self._action_from(parse_or_none(text), s) or self._malformed("not an action")


class Depth1Inquirer(LLMPolicy):
    """Breadth-only: the questions nameable from `x` alone, enumerated once and then walked.

    The single model call happens on the first `act()` and its prompt contains the view and
    nothing else, so on a suite whose retrieval is token-gated the policy CANNOT reach depth
    >= 1: the identifiers that unlock depth 1 exist only inside documents it has, by
    construction, never been shown. That is not a claim about how a model behaves — it is a
    property of what the prompt can contain, and tests/test_arms.py checks it exactly.
    """

    policy_id = "inquirer_depth1"
    prompt_name = "inquirer_depth1"

    def _on_reset(self, view: TaskView, seed: int) -> None:
        self._queue: list[str] | None = None

    def _prompt(self, s: State) -> str:
        """Deliberately a function of `s.view` ONLY. Adding `s.evidence` here would silently
        turn this ablation into the treatment and no other test would notice."""
        return promptlib.render(
            self.prompt_name,
            question=s.view.question,
            instructions=s.view.instructions,
            user_channel=promptlib.load(self._user_channel_fragment()).strip(),
        )

    def act(self, s: State) -> Action:
        if self._queue is None:
            self._queue = self._enumerate(s)
        if not self._queue:
            return Stop(reason="policy_stop")
        return Ask(text=self._queue.pop(0), rationale="breadth-only: named in the task")

    def _enumerate(self, s: State) -> list[str]:
        obj = parse_or_none(self._complete(self._prompt(s)))
        raw = (obj or {}).get("questions")
        if not isinstance(raw, list):
            self._malformed("not a question list")
            return []
        seen: set[str] = set()
        out: list[str] = []
        for q in raw:
            q = str(q).strip()
            if q and q not in seen:
                seen.add(q)
                out.append(q)
        return out if self.max_asks is None else out[: self.max_asks]

    def _decide(self, text: str, s: State) -> Action:  # pragma: no cover - act() is overridden
        raise NotImplementedError
