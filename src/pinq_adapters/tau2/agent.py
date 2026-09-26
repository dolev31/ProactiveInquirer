"""PinqTauAgent: the whole Inquirer<->Drafter dialogue, collapsed into one assistant turn.

THE DESIGN POINT OF THIS FILE
    The entire pinq run_loop executes inside ONE `generate_next_message` call. The
    Orchestrator and the user simulator therefore observe exactly one assistant message,
    and the internal questions never enter the tau2 transcript.

    This is not a convenience. It is what makes the claim honest:
      * If the internal turns leaked into the transcript, the user simulator would answer
        them, and the Inquirer would be *eliciting from the user* — the exact thing the
        paper claims it does not need to do.
      * tau2's `max_steps` counts orchestrator steps. An Inquirer that spent 12 internal
        turns would consume 12 steps of a budget every other arm spends on real dialogue,
        so the comparison against `drafter_only` would be confounded by step count rather
        than by policy.
    One call in, one AssistantMessage out. The internal trajectory is kept on the state
    object for logging and never rendered into a message.

WHY A CLASS FACTORY INSTEAD OF A `class PinqTauAgent(HalfDuplexAgent)` STATEMENT
    Subclassing at module scope would make `import pinq_adapters.tau2.agent` a hard
    dependency on tau2, and the default offline test run imports this package. The base
    class is resolved inside the factory, so the module stays importable with tau2 absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pinq.budget import BudgetLedger
from pinq.loop import run_loop
from pinq.types import TaskView, Trajectory


@dataclass
class PinqAgentState:
    """Agent-side state. `trajectories` accumulates one entry per assistant turn.

    Deliberately not a tau2 message list: the internal dialogue must never be reachable
    from anything the Orchestrator serializes into the transcript.
    """

    messages: list = field(default_factory=list)
    trajectories: list[Trajectory] = field(default_factory=list)


def make_pinq_agent_class() -> type:
    """Build the HalfDuplexAgent subclass. Imports tau2; call only when it is installed."""
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.data_model.message import AssistantMessage

    class PinqTauAgent(HalfDuplexAgent):
        """Runs a full pinq rollout per assistant turn and emits its answer as one message."""

        def __init__(
            self,
            tools: list,
            domain_policy: str,
            *,
            view: TaskView,
            build_loop: Callable[[], dict[str, Any]],
            max_turns: int = 16,
            k: int = 5,
            seed: int = 0,
        ) -> None:
            super().__init__(tools=tools, domain_policy=domain_policy)
            self._view = view
            self._build_loop = build_loop
            self._max_turns = max_turns
            self._k = k
            self._seed = seed

        def get_init_state(self, message_history: list | None = None) -> PinqAgentState:
            return PinqAgentState(messages=list(message_history or []))

        def generate_next_message(
            self, message: Any, state: PinqAgentState
        ) -> tuple[Any, PinqAgentState]:
            state.messages.append(message)

            parts = self._build_loop()
            ledger: BudgetLedger = parts["ledger"]
            traj = run_loop(
                view=self._view,
                inquirer=parts["inquirer"],
                retriever=parts["retriever"],
                drafter=parts["drafter"],
                answerer=parts["answerer"],
                ledger=ledger,
                actuator=parts.get("actuator"),
                max_turns=self._max_turns,
                k=self._k,
                seed=self._seed,
                # The user channel is closed: an Inquirer that could ask the user would be
                # testing elicitation, not autonomous discovery.
                allow_user_target=False,
            )
            state.trajectories.append(traj)

            answer = traj.outcome.answer
            out = AssistantMessage(role="assistant", content=answer.text if answer else "")
            state.messages.append(out)
            return out, state

    return PinqTauAgent


def last_trajectory(state: PinqAgentState) -> Trajectory | None:
    """The trajectory behind the most recent assistant message, for logging."""
    return state.trajectories[-1] if state.trajectories else None
