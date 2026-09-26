"""The three published methods this work has to beat on their own terms.

These are the closest ancestors, so they are implemented FAITHFULLY. A strawman here would
not save the paper; it would only move the objection from the reviewer's report into the
rebuttal, where it is worse. In particular `SelfAskInquirer` emits the literal `Follow up:` /
`Intermediate answer:` interleaving of Press et al., with the four canonical exemplars, and
`IRCoTInquirer` uses each generated reasoning SENTENCE verbatim as the next retrieval query
exactly as Trivedi et al. specify — no question mark is added and no question is extracted,
because the absence of an explicit self-question is the whole contrast being drawn.
"""

from __future__ import annotations

import re
from typing import Sequence

from pinq import promptlib
from pinq.types import Action, Ask, State, Stop, TaskView
from pinq_expt.components import render_evidence
from pinq_expt.policies.base import LLMPolicy, parse_or_none

FOLLOW_UP = "Follow up:"
INTERMEDIATE = "Intermediate answer:"
FINAL = "So the final answer is:"
IRCOT_FINAL = "So the answer is:"

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _last_response(s: State) -> str:
    """What came back for the most recent ask.

    run_loop stores the Drafter's reply on `Turn.response_text`; when a Drafter does not
    resolve, fall back to the text of the units that turn retrieved, so the interleaving
    still carries real content rather than an empty line.
    """
    for turn in reversed(s.history):
        if not isinstance(turn.action, Ask):
            continue
        if turn.response_text:
            return " ".join(turn.response_text.split())
        got = frozenset(turn.retrieved_uids)
        texts = [u.text for u in s.evidence.units if u.uid in got]
        return " ".join(" ".join(texts).split())[:600] or "(nothing was found for that question)"
    return "(nothing yet)"


class SelfAskInquirer(LLMPolicy):
    """Self-Ask (Press et al., 2022), verbatim.

    The scratchpad is the method: the prompt ends mid-sentence at "Are follow up questions
    needed here:" and the model continues it, so the format is elicited by continuation
    rather than by an instruction to produce JSON. Each `Follow up:` line becomes the ask;
    the answer that comes back is written into the scratchpad as `Intermediate answer:`
    before the next continuation, which is what makes the method sequential at all.
    """

    policy_id = "self_ask"
    prompt_name = "self_ask"

    def _on_reset(self, view: TaskView, seed: int) -> None:
        self.scratchpad = ""

    def _prompt(self, s: State) -> str:
        pad = self.scratchpad
        if pad.endswith(INTERMEDIATE):
            pad = f"{pad} {_last_response(s)}\n"
        self.scratchpad = pad
        return promptlib.render(self.prompt_name, question=s.view.question, scratchpad=pad)

    def _decide(self, text: str, s: State) -> Action:
        head = text.split(FINAL)[0]
        idx = head.find(FOLLOW_UP)
        if idx < 0:
            if FINAL in text:
                self.scratchpad += f" No.\n{FINAL}"
                return Stop(reason="policy_stop")
            return self._malformed("neither a follow-up nor a final answer")
        question = head[idx + len(FOLLOW_UP) :].splitlines()[0].strip()
        if not question:
            return self._malformed("empty follow-up")
        prefix = "" if self.scratchpad else " Yes.\n"
        self.scratchpad += f"{prefix}{FOLLOW_UP} {question}\n{INTERMEDIATE}"
        return Ask(text=question, rationale="self-ask follow up")


class SelfInquireInquirer(LLMPolicy):
    """ONE agent, handed the Inquirer prompt verbatim as a self-directive.

    THE ABLATION OF THE TWO-AGENT SPLIT. The inner prompt is not a paraphrase: it is
    `promptlib.render("inquirer_prompted", ...)`, the identical string the system's policy
    sends, wrapped in a header that says there is no second agent. The wrapper is the only
    difference, which is what makes the comparison an ablation instead of a prompt contest.

    The one behavioural difference follows from that header: the interleaved answers in this
    policy's history are the ones IT wrote in `self_answer`, never the Drafter's. Reading the
    Drafter's replies would quietly restore the second agent and make the arm meaningless.
    """

    policy_id = "self_inquire"
    prompt_name = "self_inquire"
    inner_prompt = "inquirer_prompted"

    @property
    def prompt_hashes(self) -> dict[str, str]:
        return promptlib.hashes(self.prompt_name, self.inner_prompt, self._user_channel_fragment())

    def _on_reset(self, view: TaskView, seed: int) -> None:
        self._pairs: list[tuple[str, str]] = []

    def _prompt(self, s: State) -> str:
        inner = promptlib.render(
            self.inner_prompt,
            question=s.view.question,
            instructions=s.view.instructions,
            evidence=render_evidence(s.evidence),
            draft=(s.draft.text if s.draft else "(no draft yet)"),
            history="\n".join(
                f"Q{i + 1}: {q}\nA{i + 1}: {a}" for i, (q, a) in enumerate(self._pairs)
            )
            or "(nothing asked yet)",
            user_channel=promptlib.load(self._user_channel_fragment()).strip(),
        )
        return promptlib.render(self.prompt_name, inquirer_prompt=inner)

    def _decide(self, text: str, s: State) -> Action:
        obj = parse_or_none(text)
        action = self._action_from(obj, s)
        if action is None:
            return self._malformed("not an action")
        if isinstance(action, Ask):
            self._pairs.append((action.text, str((obj or {}).get("self_answer", "")).strip()))
        return action


class IRCoTInquirer(LLMPolicy):
    """IRCoT (Trivedi et al., 2023): the reasoning sentence IS the query.

    There is no explicit self-question anywhere in this policy, and that is the point of the
    contrast: the Ask carries the generated sentence unaltered, so what the retriever sees is
    a declarative continuation of the chain rather than an interrogative about a named need.
    """

    policy_id = "ircot"
    prompt_name = "ircot"

    def _on_reset(self, view: TaskView, seed: int) -> None:
        self.cot: list[str] = []

    def _prompt(self, s: State) -> str:
        return promptlib.render(
            self.prompt_name,
            question=s.view.question,
            evidence=render_evidence(s.evidence),
            cot="\n".join(self.cot) or "(nothing yet)",
        )

    def _decide(self, text: str, s: State) -> Action:
        sentence = _first_sentence(text)
        if not sentence:
            return self._malformed("empty continuation")
        self.cot.append(sentence)
        if IRCOT_FINAL.lower() in sentence.lower():
            return Stop(reason="policy_stop")
        return Ask(text=sentence, rationale="ircot: this sentence is the query")


def _first_sentence(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            parts: Sequence[str] = _SENTENCE.split(line)
            return parts[0].strip()
    return ""
