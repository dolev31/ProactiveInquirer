"""The arms: policies, drafters, the frozen answerer, and the properties they must have.

EVERY TEST HERE IS OFFLINE. The only clients that appear are a scripted fake and
`pi_run.cache.ReplayClient`, which raises on a cache miss — so a test that reached the
network would fail on line one rather than quietly costing $40 and passing.

The tests are ordered by what they protect:

  * the protocol and the BUDGET BLINDNESS of `act`, which is what makes prefix-k of a long
    rollout exchangeable with a true short run;
  * malformed generations as DATA (a STOP and a counter), never an exception;
  * the two ablations whose whole point is what they cannot see;
  * `draft()` purity, without which phi_LOO and the prefix ladder are undefined;
  * prompt token parity, without which "this arm won" can mean "this arm's prompt was
    longer".
"""

from __future__ import annotations

import inspect
import json
import re

import pytest

from pi_eval.build.synth_build import build
from pi_run.cache import CachingClient, DiskCache, ReplayClient
from pinq import promptlib
from pinq.budget import BudgetLedger
from pinq.ids import h, request_sha
from pinq.loop import run_loop
from pinq.protocols import Inquirer, Recorder
from pinq.types import Ask, CallTelemetry, Evidence, EvidenceUnit, State, Stop
from pinq.view import make_view
from pinq_adapters.synth.suite import SynthSuite
from pinq_expt import arms as arm_table
from pinq_expt.components import (
    ComputeMatchedDrafter,
    FrozenLLMAnswerer,
    LLMDrafter,
    QueryExpansionDrafter,
    VerbosityDrafter,
    parse_json_object,
)
from pinq_expt.policies import (
    ChecklistInquirer,
    Depth1Inquirer,
    IRCoTInquirer,
    NoEvidenceInquirer,
    Par2RagInquirer,
    ParallelReplayInquirer,
    PromptedInquirer,
    RandomQInquirer,
    SelfAskInquirer,
    SelfInquireInquirer,
    recorded_questions,
)

TOKEN_RE = re.compile(r"\b[KD][0-9A-F]{8}\b")
DEPTH = 3
FACETS = 3

# A cap that cannot occur by accident anywhere else in a prompt: any appearance of this
# string in a rendered prompt is the budget leaking into the policy.
SENTINEL_CAP = 424242


# --------------------------------------------------------------------------- fakes


class ScriptLLM:
    """Deterministic, offline, and it KEEPS every prompt it was sent.

    Keeping the prompts is what lets the interesting assertions be about the request rather
    than about the reply: an ablation that cannot see the evidence is proved by the absence
    of the evidence in the bytes it sent, not by a docstring.
    """

    model = "stub/script"

    def __init__(self, responder=None, *, default: str = '{"action": "stop"}') -> None:
        self._responder = responder
        self._default = default
        self.prompts: list[tuple[str, str]] = []  # (role, prompt)
        self.calls = 0

    def request_payload(self, *, role, messages, seed, max_tokens=None, **kw):
        return {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "seed": seed,
            "max_tokens": max_tokens,
            **{k: v for k, v in sorted(kw.items())},
        }

    def complete(self, *, role, messages, seed, max_tokens=None, **kw):
        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        self.prompts.append((role, prompt))
        self.calls += 1
        text = self._responder(role, prompt) if self._responder else self._default
        payload = self.request_payload(
            role=role, messages=messages, seed=seed, max_tokens=max_tokens, **kw
        )
        tel = CallTelemetry(
            call_id=f"script-{self.calls}",
            actor=kw.get("actor", role),
            model=self.model,
            provider="stub",
            request_sha=request_sha(payload),
            response_sha="r" * 8,
            tok_prompt=len(prompt.split()),
            tok_completion=len(text.split()),
        )
        return text, tel

    def sent(self, role: str) -> list[str]:
        return [p for r, p in self.prompts if r == role]


class ExplodingLedger:
    """Raises on ANY attribute access, including `record`.

    Handed to a policy as its `Recorder`, it turns "the policy never consults the budget"
    from a claim into a test: a single read of `cap`, `spent`, `check` — or any other
    attribute — is an AttributeError inside `act`.
    """

    def __getattribute__(self, name):
        raise AssertionError(f"act() touched the ledger: .{name}")


def token_responder(role: str, prompt: str) -> str:
    """A competent policy, implemented with nothing but the prompt it was given.

    This is the fake that makes the depth-1 claim exact: it can only name a token that
    appears in the prompt, so a policy whose prompt cannot contain a depth-1 token cannot
    ask about one no matter how the model behaves.
    """
    if role == "drafter":
        return "Noted."
    if role == "answerer":
        return "final answer"
    asked = {
        t
        for line in prompt.splitlines()
        if line.startswith(("Q", "A", "Follow up:", "Intermediate answer:"))
        for t in TOKEN_RE.findall(line.upper())
    }
    fresh: list[str] = []
    for t in TOKEN_RE.findall(prompt.upper()):
        if t not in asked and t not in fresh and not t.startswith("D"):
            fresh.append(t)
    # Each branch below follows the OUTPUT CONTRACT stated in the prompt it was given, which
    # is the only way one fake can serve six policies without becoming six fakes.
    if '"questions"' in prompt:  # inquirer_depth1
        return json.dumps({"questions": [f"What does record {t} say?" for t in fresh]})
    if "Are follow up questions needed here:" in prompt:  # self_ask
        if not fresh:
            return " No.\nSo the final answer is: done"
        return f" Yes.\nFollow up: What does record {fresh[0]} say?\nIntermediate answer:"
    if "NEXT SENTENCE" in prompt:  # ircot
        if not fresh:
            return "So the answer is: done."
        return f"Record {fresh[0]} holds the next step of the chain."
    if not fresh:
        return json.dumps({"action": "stop", "rationale": "nothing left to ask"})
    return json.dumps(
        {"action": "ask", "question": f"What does record {fresh[0]} say?", "rationale": "chain"}
    )


class ConstantRetriever:
    """Returns k units for ANY query, so retrieval VOLUME is comparable across policies
    whose questions differ. The synthetic suite's token gate is the right tool for depth and
    the wrong tool for volume: there, a wrong question retrieves nothing at all."""

    corpus_id = "const"
    corpus_hash = "const_hash"

    def search(self, query: str, k: int):
        base = int(h("const", query)[:8], 16) % 100000  # stable across processes
        return [
            EvidenceUnit.make(
                corpus_id=self.corpus_id,
                doc_id=f"d{base + i}",
                span="0:10",
                title=f"doc {base + i}",
                text=f"content of doc {base + i}",
            )
            for i in range(k)
        ]


def a_view(**kw):
    base = dict(
        task_id="t1",
        suite_id="synth",
        question="Starting from records KAAAAAAAA, KBBBBBBBB, report the final value.",
        instructions="You may issue retrieval queries.",
        corpus_id="c",
        corpus_hash="h",
        word_cap=60,
    )
    base.update(kw)
    return make_view(**base)


def a_state(view=None, *, units=(), asked=()):
    from pinq.types import Turn

    history = tuple(
        Turn(turn_idx=i, action=Ask(text=q), response_text=f"answer to {q}")
        for i, q in enumerate(asked)
    )
    return State(view=view or a_view(), evidence=Evidence.of(units), history=history)


ALL_POLICIES = (
    PromptedInquirer,
    NoEvidenceInquirer,
    Depth1Inquirer,
    SelfAskInquirer,
    SelfInquireInquirer,
    IRCoTInquirer,
    ChecklistInquirer,
    RandomQInquirer,
    ParallelReplayInquirer,
    Par2RagInquirer,
)

LLM_POLICIES = ALL_POLICIES[:6]


def build_policy(cls, *, llm=None, recorder=None):
    kw = {}
    params = inspect.signature(cls).parameters
    if "llm" in params:
        kw["llm"] = llm
    if "recorder" in params:
        kw["recorder"] = recorder
    if "questions" in params:
        kw["questions"] = {"t1": ["q1", "q2"], "other": ["other q1", "other q2", "other q3"]}
    return cls(**kw)


# --------------------------------------------------------------------------- the protocol


@pytest.mark.parametrize("cls", ALL_POLICIES, ids=lambda c: c.policy_id)
def test_every_policy_satisfies_the_inquirer_protocol(cls):
    policy = build_policy(cls, llm=ScriptLLM())
    assert isinstance(policy, Inquirer)
    assert policy.policy_id


@pytest.mark.parametrize("cls", ALL_POLICIES, ids=lambda c: c.policy_id)
def test_act_takes_state_and_nothing_else(cls):
    """The signature IS the guarantee. A ledger parameter here would make prefix-k of a
    B=16 rollout non-exchangeable with a true B=k run and confound STOP with the cap."""
    assert list(inspect.signature(cls.act).parameters) == ["self", "s"]


@pytest.mark.parametrize("cls", ALL_POLICIES, ids=lambda c: c.policy_id)
def test_act_cannot_see_the_budget(cls):
    """Every policy runs a full episode holding a ledger that explodes on ANY attribute
    access. Nothing about the budget is readable from inside `act`, and the proof is that
    the object which would report a read is never touched."""
    llm = ScriptLLM(token_responder)
    policy = build_policy(cls, llm=llm, recorder=ExplodingLedger())
    view = a_view()
    policy.reset(view, seed=0)
    state = a_state(view)
    for _ in range(4):
        action = policy.act(state)  # would raise AssertionError on any ledger access
        assert isinstance(action, (Ask, Stop))
        if isinstance(action, Stop):
            break
        state = a_state(view, asked=[a.text for a in state.asks] + [action.text])


def test_the_recorder_seam_is_write_only():
    """`Recorder` has exactly one method and it returns nothing. That is the entire reason a
    policy can report a malformed generation without being able to read spend."""
    assert list(inspect.signature(Recorder.record).parameters) == ["self", "currency", "amount"]
    assert isinstance(BudgetLedger(cap=1), Recorder)
    assert not hasattr(Recorder, "cap")


# --------------------------------------------------------------------------- malformed


MALFORMED_SCRIPTS = {
    PromptedInquirer: "I think we should probably look into the second record.",
    NoEvidenceInquirer: "Sure! Here is my plan: first, consider the facets.",
    Depth1Inquirer: '{"not_questions": "oops"}',
    SelfInquireInquirer: "```\nno json here at all\n```",
    SelfAskInquirer: "Hmm, that is a hard one and I am not sure how to proceed.",
    IRCoTInquirer: "   \n  \n",
    # `ScriptLLM(default=...)` returns this SAME text for every call, stage 1 and stage 2
    # alike, so the fixture has to be chosen for what it does in BOTH: `{"questions": []}` is
    # a well-formed (if empty) stage-1 answer -- `_enumerate` accepts an empty list, no
    # malformed count -- which is exactly why stage 2 runs inside this same `act()` call. The
    # same text handed to stage 2's `_action_from` has no "action" key, which IS a contract
    # violation, so the single malformed count this test expects lands there, once, not twice.
    Par2RagInquirer: '{"questions": []}',
}


@pytest.mark.parametrize("cls", list(MALFORMED_SCRIPTS), ids=lambda c: c.policy_id)
def test_malformed_generation_yields_stop_and_is_counted(cls):
    """A model that ignores the output contract must cost one turn and one counter, never a
    stack trace: an exception here aborts a 30k-rollout sweep over one bad sample, and a
    silent retry hides a real difference between arms."""
    ledger = BudgetLedger(cap=8)
    policy = build_policy(cls, llm=ScriptLLM(default=MALFORMED_SCRIPTS[cls]), recorder=ledger)
    policy.reset(a_view(), seed=0)

    action = policy.act(a_state())

    assert isinstance(action, Stop)
    assert policy.malformed == 1
    assert ledger.spent["malformed"] == 1
    assert ledger.spent.get("retrieval_calls", 0) == 0


def test_a_malformed_generation_never_reaches_the_hard_cap():
    """The malformed counter is recorded, not charged: it must not consume the one currency
    that stops the loop, or a formatting failure would masquerade as a budget stop."""
    ledger = BudgetLedger(cap=2)
    policy = PromptedInquirer(ScriptLLM(default="not json"), recorder=ledger)
    policy.reset(a_view(), seed=0)
    for _ in range(5):
        assert isinstance(policy.act(a_state()), Stop)
    assert ledger.spent["malformed"] == 5
    assert "retrieval_calls" not in ledger.spent


def test_parse_json_object_survives_fences_and_prose():
    assert parse_json_object('```json\n{"action": "stop"}\n```') == {"action": "stop"}
    assert parse_json_object('Sure: {"action": "ask", "question": "why?"} hope that helps') == {
        "action": "ask",
        "question": "why?",
    }
    assert parse_json_object('{"a": "}"}') == {"a": "}"}  # a brace inside a string is not a brace
    assert parse_json_object("no object here") is None
    assert parse_json_object('{"unclosed": ') is None
    assert parse_json_object("[1, 2, 3]") is None  # a list is not an action


# --------------------------------------------------------------------------- the ablations


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth_arms")
    corpus, _gold, _chash = build(n_tasks=3, n_facets=FACETS, depth=DEPTH, seed=11, root=root)
    return SynthSuite(corpus.parent)


def _run(suite, tid, inquirer, llm, *, cap=32, drafter=None):
    ledger = BudgetLedger(cap=cap)
    traj = run_loop(
        view=suite.view(tid),
        inquirer=inquirer,
        retriever=suite.retriever(tid),
        drafter=drafter or LLMDrafter(llm),
        answerer=FrozenLLMAnswerer(llm),
        ledger=ledger,
        max_turns=16,
        k=5,
        seed=0,
    )
    return traj, ledger


def test_depth1_provably_cannot_reach_depth_ge_1(synth):
    """THE ablation, made exact by the suite's token gate.

    A record is retrievable only by quoting its identifier, and an identifier for depth d+1
    appears only inside a depth-d record. `Depth1Inquirer` renders `x` and never `E_t`, so no
    depth-1 identifier can appear in any prompt it sends — and the assertion below is on the
    PROMPTS, not on the retrieved set, because that is the part that holds for every model.
    """
    depth1_llm = ScriptLLM(token_responder)
    depth1, _ = _run(synth, "s0", Depth1Inquirer(depth1_llm), depth1_llm)

    deep_docs = {f"0:{f}:{d}" for f in range(FACETS) for d in range(1, DEPTH)}
    deep_tokens = {
        t
        for u in synth.retriever("s0")._units  # noqa: SLF001 - the suite's own corpus, in a test
        if u.doc_id in deep_docs
        for t in TOKEN_RE.findall(u.title.upper())
    }
    assert deep_tokens, "the fixture must actually have depth >= 1 records"

    for prompt in depth1_llm.sent("inquirer"):
        assert not (deep_tokens & set(TOKEN_RE.findall(prompt.upper()))), (
            "a depth-1 identifier reached the breadth-only policy's prompt"
        )
    assert depth1.evidence.doc_ids == {f"0:{f}:0" for f in range(FACETS)}

    # The load-bearing half: the prompt is a function of the VIEW ALONE. Checking only the
    # episode above would pass even if `_prompt` read `s.evidence`, because this policy
    # enumerates on turn zero when the evidence is empty anyway — so the mutation that
    # matters would slip through. Render it against a state carrying the deepest evidence
    # in the suite and require the same bytes.
    probe = Depth1Inquirer(ScriptLLM(token_responder))
    probe.reset(synth.view("s0"), 0)
    deep_units = [u for u in synth.retriever("s0")._units if u.doc_id in deep_docs]  # noqa: SLF001
    bare = probe._prompt(a_state(synth.view("s0")))  # noqa: SLF001
    loaded = probe._prompt(  # noqa: SLF001
        a_state(synth.view("s0"), units=deep_units, asked=["what about the deep record?"])
    )
    assert bare == loaded

    # And the contrast: the same fake model, given evidence, walks the whole chain.
    full_llm = ScriptLLM(token_responder)
    full, _ = _run(synth, "s0", PromptedInquirer(full_llm), full_llm)
    assert full.evidence.doc_ids == {f"0:{f}:{d}" for f in range(FACETS) for d in range(DEPTH)}
    assert depth1.evidence.uids < full.evidence.uids


def test_noevidence_policy_never_sends_the_evidence(synth):
    """The most important ablation is only an ablation if the evidence really is absent."""
    llm = ScriptLLM(token_responder)
    _traj, _ledger = _run(synth, "s0", NoEvidenceInquirer(llm), llm)
    units = synth.retriever("s0")._units  # noqa: SLF001
    bodies = {u.text for u in units}
    for prompt in llm.sent("inquirer"):
        assert not any(b in prompt for b in bodies)
        assert "[" + units[0].uid[:12] + "]" not in prompt


def test_checklist_is_not_conditioned_on_state():
    """Same task, wildly different states, byte-identical question sequence."""
    view = a_view()
    empty, loaded = ChecklistInquirer(), ChecklistInquirer()
    empty.reset(view, 0)
    loaded.reset(view, 0)
    rich = a_state(
        view,
        units=[
            EvidenceUnit.make(
                corpus_id="c", doc_id="d1", span="0:5", title="T", text="a decisive fact"
            )
        ],
        asked=["what about the fact?"],
    )
    a = [empty.act(a_state(view)) for _ in range(4)]
    b = [loaded.act(rich) for _ in range(4)]
    assert [x.text for x in a] == [x.text for x in b]
    assert all(view.question in x.text for x in a)


# --------------------------------------------------------------------------- matched volume


def test_random_q_retrieves_a_matched_volume():
    """The null for phi only works at matched volume: an arm that retrieves less would lose
    for a reason that has nothing to do with where its questions came from."""
    view = a_view(task_id="t1")
    retriever = ConstantRetriever()
    llm = ScriptLLM(token_responder)

    def go(policy):
        ledger = BudgetLedger(cap=16)
        traj = run_loop(
            view=view,
            inquirer=policy,
            retriever=retriever,
            drafter=LLMDrafter(llm),
            answerer=FrozenLLMAnswerer(llm),
            ledger=ledger,
            max_turns=8,
            k=5,
            seed=0,
        )
        return traj, ledger

    treatment, t_ledger = go(PromptedInquirer(llm))
    n_asks = treatment.n_asks
    assert n_asks >= 2, "the reference arm must actually ask something to be matched against"

    pool = {
        "t1": ["the treatment's own question"] * n_asks,
        "someone_else": [f"an unrelated question number {i}" for i in range(20)],
    }
    null, n_ledger = go(RandomQInquirer(questions=pool, ask_counts={"t1": n_asks}))

    assert null.n_asks == n_asks
    assert n_ledger.spent["retrieval_calls"] == t_ledger.spent["retrieval_calls"]
    assert sum(len(t.retrieved_uids) for t in null.turns) == sum(
        len(t.retrieved_uids) for t in treatment.turns
    )
    # ...and the questions really are the other task's, never this one's.
    asked = [t.action.text for t in null.turns if isinstance(t.action, Ask)]
    assert asked and all("unrelated question" in q for q in asked)


def test_random_q_refuses_to_draw_from_the_task_it_is_nulling():
    from pinq_expt.policies.controls import EmptyPool

    policy = RandomQInquirer(questions={"t1": ["only this task's questions"]})
    with pytest.raises(EmptyPool):
        policy.reset(a_view(task_id="t1"), 0)


def test_random_q_is_reproducible_from_the_manifest():
    pool = {"other": [f"q{i}" for i in range(20)]}
    a, b = RandomQInquirer(questions=pool, n_asks=5), RandomQInquirer(questions=pool, n_asks=5)
    a.reset(a_view(), 3)
    b.reset(a_view(), 3)
    assert [a.act(a_state()).text for _ in range(5)] == [b.act(a_state()).text for _ in range(5)]


# --------------------------------------------------------------------------- the ancestors


def test_self_ask_emits_the_literal_follow_up_format():
    """Self-Ask is the closest ancestor, so it is implemented as published: the scratchpad
    carries `Follow up:` / `Intermediate answer:` and the ask is the follow-up line."""
    llm = ScriptLLM(
        default=" Yes.\nFollow up: How old was Alan Turing when he died?\nIntermediate answer:"
    )
    policy = SelfAskInquirer(llm)
    policy.reset(a_view(), 0)

    first = policy.act(a_state())
    assert isinstance(first, Ask)
    assert first.text == "How old was Alan Turing when he died?"
    assert policy.scratchpad.startswith(" Yes.\nFollow up: ")
    assert policy.scratchpad.endswith("Intermediate answer:")

    sent = llm.sent("inquirer")[0]
    assert "Follow up:" in sent and "Intermediate answer:" in sent
    assert sent.rstrip().endswith("Are follow up questions needed here:")

    # The answer that came back is interleaved before the next continuation.
    policy.act(a_state(asked=[first.text]))
    assert f"Intermediate answer: answer to {first.text}" in llm.sent("inquirer")[1]


def test_self_ask_stops_on_the_final_answer():
    policy = SelfAskInquirer(ScriptLLM(default=" No.\nSo the final answer is: 41 years old"))
    policy.reset(a_view(), 0)
    assert isinstance(policy.act(a_state()), Stop)
    assert policy.malformed == 0


def test_self_inquire_sends_the_inquirer_prompt_verbatim():
    """The ablation of the two-agent split is only an ablation if the directive is the SAME
    directive. Not a paraphrase, not a summary: the identical rendered string."""
    llm = ScriptLLM(default='{"action": "ask", "question": "who?", "self_answer": "me"}')
    policy = SelfInquireInquirer(llm)
    view = a_view()
    policy.reset(view, 0)
    policy.act(a_state(view))

    sent = llm.sent("inquirer")[0]
    inner = promptlib.render(
        "inquirer_prompted",
        question=view.question,
        instructions=view.instructions,
        evidence="(nothing retrieved yet)",
        draft="(no draft yet)",
        history="(nothing asked yet)",
        user_channel=promptlib.load("fragment_user_channel_placebo").strip(),
    )
    assert inner in sent
    assert sent.index("ALONE") < sent.index(inner)


def test_self_inquire_interleaves_its_own_answers_not_the_drafters():
    """If it read the Drafter's replies the second agent would be back and the arm would be
    measuring nothing."""
    llm = ScriptLLM(default='{"action": "ask", "question": "who?", "self_answer": "I say Bob"}')
    policy = SelfInquireInquirer(llm)
    policy.reset(a_view(), 0)
    policy.act(a_state())
    policy.act(a_state(asked=["who?"]))  # the state carries "answer to who?" from the Drafter
    second = llm.sent("inquirer")[1]
    assert "I say Bob" in second
    assert "answer to who?" not in second


def test_ircot_uses_the_generated_sentence_verbatim_as_the_query():
    """No question is extracted and no question mark is added: the absence of an explicit
    self-question is the contrast IRCoT is here to draw."""
    sentence = "Alan Turing was a British mathematician who worked at Bletchley Park."
    policy = IRCoTInquirer(ScriptLLM(default=f"{sentence} He later moved to Manchester."))
    policy.reset(a_view(), 0)
    ask = policy.act(a_state())
    assert isinstance(ask, Ask)
    assert ask.text == sentence
    assert "?" not in ask.text
    assert policy.cot == [sentence]


def test_ircot_stops_on_its_own_conclusion():
    policy = IRCoTInquirer(ScriptLLM(default="So the answer is: 41."))
    policy.reset(a_view(), 0)
    assert isinstance(policy.act(a_state()), Stop)
    assert policy.malformed == 0


# --------------------------------------------------------------------------- replay arms


def test_parallel_replay_issues_the_recorded_questions_without_reading_state(tmp_path):
    run_dir = tmp_path / "runs" / "abc"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"task_id": "t1", "arm_id": "inquirer_prompted", "suite_id": "synth"})
    )
    (run_dir / "turns.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"action_kind": "ask", "question": "first recorded question"},
                {"action_kind": "ask", "question": "second recorded question"},
                {"action_kind": "stop", "question": ""},
            )
        )
    )
    pool = recorded_questions(tmp_path / "runs", arm_id="inquirer_prompted")
    assert pool == {"t1": ["first recorded question", "second recorded question"]}

    policy = ParallelReplayInquirer.from_runs(tmp_path / "runs")
    policy.reset(a_view(task_id="t1"), 0)
    # The state it is handed is irrelevant by construction: nothing informs anything.
    got = [policy.act(a_state(asked=["something else entirely"])) for _ in range(3)]
    assert [g.text for g in got[:2]] == pool["t1"]
    assert isinstance(got[2], Stop)


def test_a_scripted_arm_refuses_a_task_it_has_no_script_for():
    from pinq_expt.policies.controls import MissingScript

    policy = ParallelReplayInquirer(questions={"t9": ["q"]})
    with pytest.raises(MissingScript):
        policy.reset(a_view(task_id="t1"), 0)


# --------------------------------------------------------------------------- purity


def _evidence(n: int) -> Evidence:
    return Evidence.of(
        [
            EvidenceUnit.make(
                corpus_id="c",
                doc_id=f"d{i}",
                span="0:20",
                title=f"Doc {i}",
                text=f"the body of document {i}",
            )
            for i in range(n)
        ]
    )


def test_draft_is_a_pure_function_of_view_subset_hash_and_seed(tmp_path):
    """FIVE REPEATS, BYTE-IDENTICAL, through a client that RAISES on a cache miss.

    The replay client is the whole point: it looks up the request bytes and refuses to
    dispatch. Five hits and one underlying call therefore prove the drafter emitted the same
    request five times, which is the property phi_LOO and the prefix ladder rest on. A memo
    inside the drafter would have made this test tautological, so there is not one.
    """
    cache = DiskCache(tmp_path / "cache")
    ledger = BudgetLedger(cap=16)
    inner = ScriptLLM(default="a stable draft")
    view = a_view()
    units = list(_evidence(3).units)

    # Five DISTINCT Evidence objects with one subset_hash between them, built in five
    # different orders and all held alive at once. Reusing a single object would let an
    # implementation key on `id(ev)` — or on nothing at all — and still pass.
    subsets = [Evidence.of(units[i:] + units[:i]) for i in range(5)]
    assert len({e.subset_hash for e in subsets}) == 1
    assert len({id(e) for e in subsets}) == 5

    LLMDrafter(CachingClient(inner, cache)).draft(view, subsets[0], seed=0, ledger=ledger)
    assert inner.calls == 1

    replay = ReplayClient(inner, cache, ledger=ledger)
    # ONE drafter for all five, so per-instance hidden state (a call counter, a cached
    # prefix, a growing scratchpad) shows up as a different request and a CacheMiss.
    drafter = LLMDrafter(replay)
    drafts = [drafter.draft(view, ev, seed=0, ledger=ledger) for ev in subsets]

    assert {d.text for d in drafts} == {"a stable draft"}
    assert len({d.sha for d in drafts}) == 1
    assert replay.hits == 5
    assert inner.calls == 1, "a replayed draft must never reach the model"


def test_draft_depends_on_the_evidence_set_and_not_on_its_order(tmp_path):
    """`Evidence` is a canonically ordered SET, so two retrieval orders memoize together —
    which is exactly what `subset_hash` claims and what makes a re-draft over a leave-one-out
    subset a cache hit rather than a new sample."""
    inner = ScriptLLM(default="d")
    client = CachingClient(inner, DiskCache(tmp_path / "cache"))
    led = BudgetLedger(cap=16)
    view = a_view()
    units = list(_evidence(3).units)

    LLMDrafter(client).draft(view, Evidence.of(units), seed=0, ledger=led)
    LLMDrafter(client).draft(view, Evidence.of(list(reversed(units))), seed=0, ledger=led)
    assert inner.calls == 1  # same subset_hash, same bytes, one call

    LLMDrafter(client).draft(view, Evidence.of(units[:2]), seed=0, ledger=led)
    assert inner.calls == 2  # a different subset IS a different request
    LLMDrafter(client).draft(view, Evidence.of(units), seed=1, ledger=led)
    assert inner.calls == 3  # and so is a different seed


def test_compute_matched_drafter_is_pure_under_self_consistency():
    """n samples, deterministic winner. Purity is not weakened by sampling as long as the
    seeds are derived and the tie-break is total."""
    votes = iter(["A answer", "B answer", "A answer", "C answer", "A answer"] * 4)
    llm = ScriptLLM(lambda role, prompt: next(votes))
    led = BudgetLedger(cap=16)
    view, ev = a_view(), _evidence(2)
    first = ComputeMatchedDrafter(llm, n=5).draft(view, ev, seed=0, ledger=led)
    second = ComputeMatchedDrafter(llm, n=5).draft(view, ev, seed=0, ledger=led)
    assert first.text == second.text == "A answer"


def test_verbosity_drafter_is_the_drafter_alone_with_a_different_prompt():
    """THE KILL SWITCH must differ from the control in exactly one thing: the prompt."""
    llm = ScriptLLM(default="an exhaustive answer")
    led = BudgetLedger(cap=16)
    view, ev = a_view(), _evidence(2)
    VerbosityDrafter(llm).draft(view, ev, seed=0, ledger=led)
    assert promptlib.load("drafter_verbosity").split("{{")[0] in llm.sent("drafter")[0]
    # It never resolves and never retrieves: its retrieval count is zero, like drafter_only.
    response, out = VerbosityDrafter(llm).resolve(
        view, Ask(text="anything"), ev, seed=0, ledger=led
    )
    assert response == "" and out is ev
    assert led.spent.get("retrieval_calls", 0) == 0


def test_query_expansion_charges_every_expanded_query_to_the_shared_cap():
    """Equal retrieval calls is the whole claim of this arm, so every expansion is a
    `charge_retrieval` against the same hard currency the Inquirer spends questions from."""
    llm = ScriptLLM(default=json.dumps({"queries": ["q one", "q two", "q three"]}))
    led = BudgetLedger(cap=16)
    drafter = QueryExpansionDrafter(llm, retriever=ConstantRetriever(), n_queries=3, k=2)
    _text, ev = drafter.resolve(a_view(), Ask(text="the ask"), Evidence(), seed=0, ledger=led)
    assert led.spent["retrieval_calls"] == 3
    assert led.unique_docs == len(ev.doc_ids)  # what pi_run.worker.reconcile checks
    assert len(ev) == 6


def test_query_expansion_stops_at_the_cap_instead_of_raising():
    led = BudgetLedger(cap=2)
    llm = ScriptLLM(default=json.dumps({"queries": ["a", "b", "c"]}))
    drafter = QueryExpansionDrafter(llm, retriever=ConstantRetriever(), n_queries=3, k=1)
    drafter.resolve(a_view(), Ask(text="x"), Evidence(), seed=0, ledger=led)
    assert led.spent["retrieval_calls"] == 2
    assert led.spent["expansion_truncated"] == 1


def test_a_malformed_expansion_degrades_to_no_expansion():
    led = BudgetLedger(cap=8)
    drafter = QueryExpansionDrafter(
        ScriptLLM(default="sorry, no"), retriever=ConstantRetriever(), n_queries=3
    )
    _text, ev = drafter.resolve(a_view(), Ask(text="x"), Evidence(), seed=0, ledger=led)
    assert len(ev) == 0
    assert led.spent["malformed"] == 1
    assert "retrieval_calls" not in led.spent


# --------------------------------------------------------------------------- the answerer


def test_the_frozen_answerer_is_arm_blind_and_word_capped():
    """One prompt, one cap, no arm id anywhere in the signature. This is what removes
    answer-length bias by construction rather than by a post-hoc regression."""
    assert "arm" not in inspect.signature(FrozenLLMAnswerer.__init__).parameters
    assert "arm" not in inspect.signature(FrozenLLMAnswerer.answer).parameters

    llm = ScriptLLM(default=" ".join(f"word{i}" for i in range(500)))
    view = a_view()
    led = BudgetLedger(cap=8)
    a = FrozenLLMAnswerer(llm).answer(view, _evidence(2), None, seed=0, ledger=led)
    assert a.n_words == view.word_cap
    assert len(a.text.split()) == view.word_cap

    # Every arm that answers at all answers with the same prompt hash.
    hashes = {
        arm_table.get(aid).answerer().prompt_hash
        for aid in arm_table.arm_ids()
        if not arm_table.get(aid).llm_free
    }
    assert hashes == {promptlib.sha("answerer_frozen")}


# --------------------------------------------------------------------------- prompts


def test_no_prompt_can_render_a_budget():
    """Layer two of the budget firewall. `Inquirer.act` being budget-blind by type is worth
    nothing if the STRING handed to the model announces the cap."""
    for name in promptlib.names():
        holes = promptlib.placeholders(name)
        assert not (holes & promptlib.FORBIDDEN_PLACEHOLDERS), name
        assert "budget" not in promptlib.load(name).lower(), name


def test_the_rendered_prompt_never_contains_the_cap(synth):
    """Not the template — the BYTES that were actually sent, over a real episode run under a
    cap chosen so that any appearance of it is unambiguous."""
    llm = ScriptLLM(token_responder)
    ledger = BudgetLedger(cap=SENTINEL_CAP)
    run_loop(
        view=synth.view("s0"),
        inquirer=PromptedInquirer(llm, recorder=ledger),
        retriever=synth.retriever("s0"),
        drafter=LLMDrafter(llm),
        answerer=FrozenLLMAnswerer(llm),
        ledger=ledger,
        max_turns=12,
        k=5,
        seed=0,
    )
    assert llm.prompts, "the episode must actually have sent something"
    for _role, prompt in llm.prompts:
        assert str(SENTINEL_CAP) not in prompt
        assert "remaining" not in prompt.lower()


@pytest.mark.parametrize("family", sorted(promptlib.PARITY_FAMILIES))
def test_prompt_token_parity_within_a_slot(family):
    """Arms are compared slot by slot, so the templates in one slot are held to within 10%
    of each other. An arm that lacks a section carries an equal-length NON-DIRECTIVE placebo
    instead, which is why `drafter_draft` and `inquirer_noevidence` have RESERVED blocks."""
    p = promptlib.parity(family)
    assert p["ratio"] <= 1.10, p


def test_prompt_token_parity_across_the_arms_that_are_compared():
    """The same check, driven off the ARM TABLE rather than off the prompt directory, so a
    new arm with a longer prompt fails here even if it never joins a parity family."""
    by_slot: dict[str, dict[str, int]] = {}
    for aid in arm_table.arm_ids():
        for slot, name in arm_table.get(aid).prompts.items():
            by_slot.setdefault(slot, {})[aid] = promptlib.tokens(name)
    assert set(by_slot) <= set(arm_table.SLOTS)
    for slot, counts in by_slot.items():
        lo, hi = min(counts.values()), max(counts.values())
        assert hi / lo <= 1.10, (slot, counts)


def test_every_declared_prompt_exists_and_is_hashed_into_run_identity():
    from pinq_expt.components import prompt_hashes_of

    for aid in arm_table.arm_ids():
        arm = arm_table.get(aid)
        for name in arm.prompts.values():
            assert promptlib.sha(name)
    hashes = prompt_hashes_of(LLMDrafter(), FrozenLLMAnswerer(), PromptedInquirer())
    assert hashes["answerer_frozen"] == promptlib.sha("answerer_frozen")
    assert hashes["drafter_draft"] == promptlib.sha("drafter_draft")
    assert hashes["inquirer_prompted"] == promptlib.sha("inquirer_prompted")


def test_the_user_channel_paragraph_is_swapped_not_added():
    """Opening the user channel must change what the prompt SAYS, never how long it is."""
    open_p = PromptedInquirer(ScriptLLM(), may_ask_user=True)
    closed_p = PromptedInquirer(ScriptLLM())
    open_p.reset(a_view(), 0)
    closed_p.reset(a_view(), 0)
    a, b = open_p._prompt(a_state()), closed_p._prompt(a_state())  # noqa: SLF001
    real = promptlib.load("fragment_user_channel").strip()
    placebo = promptlib.load("fragment_user_channel_placebo").strip()
    assert real in a and placebo not in a
    assert placebo in b and real not in b
    assert abs(promptlib.count_tokens(a) - promptlib.count_tokens(b)) <= 5


# --------------------------------------------------------------------------- the table


def test_the_arm_table_covers_every_preregistered_arm():
    """A preregistered arm that does not exist in the table cannot be run, and a table that
    silently renames one produces a row nobody can join back to the prereg."""
    from pi_eval.prereg import default_stage1

    missing = sorted(set(default_stage1().arms) - set(arm_table.ARMS))
    assert missing == []


def test_build_supplies_only_what_a_component_declares():
    llm = ScriptLLM()
    comps = arm_table.build(arm_table.get("inquirer_prompted"), llm=llm)
    assert isinstance(comps.inquirer, PromptedInquirer)
    assert isinstance(comps.drafter, LLMDrafter)
    assert isinstance(comps.answerer, FrozenLLMAnswerer)

    with pytest.raises(ValueError):
        arm_table.build(arm_table.get("parallel_replay"), llm=llm)  # needs a question list

    replay = arm_table.build(arm_table.get("parallel_replay"), llm=llm, questions={"t1": ["q"]})
    assert isinstance(replay.inquirer, ParallelReplayInquirer)


def test_an_llm_arm_refuses_to_run_without_a_model():
    """Silently degrading to a no-op would contribute a row that says something false."""
    from pinq_expt.components import LLMRequired

    with pytest.raises(LLMRequired):
        arm_table.get("inquirer_prompted").inquirer().reset(a_view(), 0)


def test_every_confirmatory_arm_shares_one_budget_cap():
    """Budget parity is the precondition for the comparison. The ceilings are marked and
    excluded from every primary table, which is why they may differ."""
    caps = {
        arm.budget_cap
        for arm in arm_table.ARMS.values()
        if arm.kind != "ceiling" and not arm.llm_free
    }
    assert caps == {arm_table.DEFAULT_CAP}
    assert all(a.requires_gold for a in arm_table.ARMS.values() if a.kind == "ceiling")


def test_a_prompt_edit_changes_run_identity():
    """The reason prompts are files with hashes and not string constants.

    `semantic_hash` covers `prompt_hashes`, so two rows of one table can never have come
    from two different prompts without carrying two different run ids. This is the join key
    the whole provenance rule (CONTRIBUTING.md rule 1) rests on.
    """
    from pi_run.manifest import build_manifest
    from pinq_expt.components import prompt_hashes_of

    comps = arm_table.build(arm_table.get("inquirer_prompted"), llm=ScriptLLM())
    hashes = dict(prompt_hashes_of(comps.inquirer, comps.drafter, comps.answerer))
    assert set(hashes) == {
        "inquirer_prompted",
        "fragment_user_channel_placebo",
        "drafter_draft",
        "drafter_resolve",
        "answerer_frozen",
    }
    assert hashes["inquirer_prompted"] == promptlib.sha("inquirer_prompted")

    base = dict(
        suite_id="synth",
        task_id="s0",
        arm_id="inquirer_prompted",
        policy_id="inquirer_prompted",
        seed=0,
        corpus_hash="c",
        budget_cap=16,
        max_turns=16,
        word_cap=60,
        code_version="x",
        dirty=False,
    )
    same = build_manifest(**base, prompt_hashes=hashes)
    edited = build_manifest(**base, prompt_hashes={**hashes, "inquirer_prompted": "0" * 64})
    assert same.run_id != edited.run_id
    assert same.semantic_hash != edited.semantic_hash


def test_the_worker_records_the_prompts_each_arm_would_render(tmp_path):
    """The LLM-free arms render no template, so their prompt_hashes — and therefore their
    run ids — must not move when the template directory changes. Every other arm must carry
    the sha of every template it would render, or a prompt edit becomes invisible."""
    from pi_run.worker import UnitSpec, run_unit

    corpus, _gold, _chash = build(n_tasks=1, n_facets=2, depth=2, seed=3, root=tmp_path)
    res = run_unit(
        UnitSpec(
            suite_id="synth",
            corpus_dir=str(corpus.parent),
            task_id="s0",
            arm_id="fake_chain",
            seed=0,
            runs_root=str(tmp_path / "runs"),
            cache_root=str(tmp_path / "cache"),
            code_version="x",
            dirty=False,
        )
    )
    assert res["status"] == "ok"
    manifest = json.loads((tmp_path / "runs" / res["run_id"] / "manifest.json").read_text())
    assert manifest["prompt_hashes"] == {"answerer": "frozen-answerer-v1"}

    # An LLM arm through the same keyless worker: it FAILS, loudly, with the manifest still
    # on disk carrying every template it would have rendered. Both halves matter — a
    # keyless worker must not produce a row that looks like a result, and a manifest whose
    # prompt hashes are missing cannot tell two prompts apart afterwards.
    llm_res = run_unit(
        UnitSpec(
            suite_id="synth",
            corpus_dir=str(corpus.parent),
            task_id="s0",
            arm_id="inquirer_prompted",
            seed=0,
            runs_root=str(tmp_path / "runs"),
            cache_root=str(tmp_path / "cache"),
            code_version="x",
            dirty=False,
        )
    )
    assert llm_res["status"] == "error"
    # The worker now BUILDS the client, so the failure is configuration, not wiring: an arm
    # with no model pin is refused rather than silently defaulted, because defaulting would
    # let two arms run on different models and still be compared.
    assert "LLMRequired" not in llm_res["error"], llm_res["error"]
    assert "PI_MODEL_" in llm_res["error"], llm_res["error"]
    llm_manifest = json.loads((tmp_path / "runs" / llm_res["run_id"] / "manifest.json").read_text())
    assert llm_manifest["prompt_hashes"]["inquirer_prompted"] == promptlib.sha("inquirer_prompted")
    assert llm_manifest["prompt_hashes"]["drafter_resolve"] == promptlib.sha("drafter_resolve")
    assert llm_manifest["prompt_hashes"]["answerer_frozen"] == promptlib.sha("answerer_frozen")


def test_par2_rag_and_prompted_manifests_hash_different_prompt_keys(tmp_path):
    """THE CRITICAL LESSON this track was briefed with, made a test rather than a docstring.

    `prompt_name` is a CLASS attribute on a policy; an arm's `prompts={...}` slot in
    `pinq_expt.arms.ARMS` feeds the token-parity guard and the recorded hashes but is NEVER
    read by the policy itself. A policy mis-wired to render the base `inquirer_prompted`
    template underneath a differently-declared arm would pass every existing guard -- the
    parity check, the budget-cap check, the arm-table coverage check -- and produce a clean
    null result. The only thing that catches it is reading the manifest's `prompt_hashes`
    dict by its KEYS, as a set, never as a truncated JSON string: two prompt_hashes blobs of
    different length can share a long common prefix (both start with the same sorted
    `answerer_frozen`/`drafter_...` keys) and look identical to a comparison that stops
    early.

    Run (at least) 4 units -- 2 tasks x {par2_rag, inquirer_prompted} -- through the real
    keyless worker. It errors on the missing model pin, but only AFTER the manifest is
    written, exactly as `test_the_worker_records_the_prompts_each_arm_would_render`
    establishes; no LLM call happens here, so this is a pure wiring check.
    """
    from pi_run.worker import UnitSpec, run_unit

    corpus, _gold, _chash = build(n_tasks=2, n_facets=2, depth=2, seed=5, root=tmp_path)
    key_sets: dict[str, set[frozenset[str]]] = {"par2_rag": set(), "inquirer_prompted": set()}
    n_units = 0
    for arm_id in ("par2_rag", "inquirer_prompted"):
        for task_id in ("s0", "s1"):
            res = run_unit(
                UnitSpec(
                    suite_id="synth",
                    corpus_dir=str(corpus.parent),
                    task_id=task_id,
                    arm_id=arm_id,
                    seed=0,
                    runs_root=str(tmp_path / "runs"),
                    cache_root=str(tmp_path / "cache"),
                    code_version="x",
                    dirty=False,
                )
            )
            n_units += 1
            assert res["status"] == "error"  # keyless: config failure, not a real rollout
            manifest = json.loads((tmp_path / "runs" / res["run_id"] / "manifest.json").read_text())
            key_sets[arm_id].add(frozenset(manifest["prompt_hashes"]))

    assert n_units == 4
    # Both tasks of the same arm must declare the SAME set of prompt keys — the keys are a
    # property of the arm's wiring, not of which task happened to run.
    assert len(key_sets["par2_rag"]) == 1
    assert len(key_sets["inquirer_prompted"]) == 1
    (par2_keys,) = key_sets["par2_rag"]
    (prompted_keys,) = key_sets["inquirer_prompted"]

    assert par2_keys != prompted_keys  # THE assertion: different KEYS, not a string compare.
    assert {"par2_rag_plan", "par2_rag_refine"} <= par2_keys
    assert "inquirer_prompted" not in par2_keys
    assert "par2_rag_plan" not in prompted_keys
    assert "par2_rag_refine" not in prompted_keys
    assert "inquirer_prompted" in prompted_keys


def test_a_non_resolving_drafter_does_not_claim_the_resolve_prompt():
    """A manifest that names a prompt the run never rendered is a false provenance claim,
    and it is false in the direction that matters: it makes two runs look like they used the
    same scaffolding when one of them used less."""
    assert "drafter_resolve" in LLMDrafter().prompt_hashes
    assert "drafter_resolve" not in VerbosityDrafter().prompt_hashes
    assert "drafter_resolve" not in ComputeMatchedDrafter().prompt_hashes
    assert "drafter_resolve" not in LLMDrafter(agentic_resolve=False).prompt_hashes
    # ...and the arm table agrees with the components it names.
    for aid in arm_table.arm_ids():
        arm = arm_table.get(aid)
        if arm.llm_free:
            continue
        declared = set(arm.prompts.values())
        rendered = set(
            arm_table.build(arm, llm=ScriptLLM(), questions={"t": ["q"]}).drafter.prompt_hashes
        )
        assert rendered <= declared, aid


def test_query_expansion_without_a_retriever_fails_instead_of_degrading():
    """Silently becoming a one-ask arm would still write a row, and that row would sit in a
    table under a name claiming it expanded queries."""
    from pinq_expt.components import RetrieverRequired

    drafter = QueryExpansionDrafter(ScriptLLM(default='{"queries": ["a"]}'))
    with pytest.raises(RetrieverRequired):
        drafter.resolve(a_view(), Ask(text="x"), Evidence(), seed=0, ledger=BudgetLedger(cap=4))


def test_expects_asks_matches_the_policy_class():
    """`expects_asks` is DECLARED, not inferred, so it must not drift from the policy. An arm
    that never asks and says it asks would make `pi verify arms` fail forever; the reverse
    would make it blind to the failure it exists to catch."""
    from pinq_expt import arms as arm_table
    from pinq_expt.fakes import NeverAsk

    wrong = [k for k, a in arm_table.ARMS.items() if a.expects_asks != (a.inquirer is not NeverAsk)]
    assert not wrong, f"expects_asks disagrees with the policy class for: {wrong}"


def test_the_non_asking_arms_are_exactly_the_never_ask_family():
    from pinq_expt import arms as arm_table

    assert sorted(k for k, a in arm_table.ARMS.items() if not a.expects_asks) == [
        "compute_matched",
        "drafter_only",
        "fake_drafter_only",
        "pare_baseline",
        "pare_plus_verbosity",
        # tau2-bench's OWN agent loop, which has no Inquirer to ask with. It joins this list
        # by having no questioner at all rather than by having one that is switched off, so
        # the enumeration still names every arm for which `n_asks == 0` is correct.
        "tau2_stock",
        "verbosity",
    ]


def test_random_q_refuses_a_task_absent_from_the_pool_instead_of_asking_nothing():
    """MEASURED, 2026-09-18, on musique test at cap 8: 26 tasks were absent from the mined pool
    and this arm returned `ok` with 0 asks on 50 units, billing $0.0847 for trajectories that
    asked nothing -- `drafter_only` wearing `random_q`'s label, counted as a volume-matched null.

    The mechanism is that every guard misses by one step. `EmptyPool` inspects the questions of
    OTHER tasks, so it cannot fire for a task missing from the pool; the per-task volume falls
    through `self._counts` and `self._n_asks` to `len(self._pool.get(view.task_id, ()))`, which
    is 0; `rng.sample(others, k=0)` is legal and returns []; and `act` then STOPs on turn 0.

    `parallel_replay` IS THE CONTROL FOR THIS, in the same sweep and on the same task ids: it
    raised MissingScript 50 times on those same 26 tasks and billed $0.0000. One condition, two
    arms, opposite behaviour -- so the correct behaviour is not a matter of taste here, and the
    sibling already implements it. Volume-matching is UNDEFINED without a reference run for
    THIS task, and an undefined volume must refuse rather than silently become zero.
    """
    from pinq_expt.policies.controls import MissingScript, ParallelReplayInquirer

    pool = {"other1": ["q1", "q2"], "other2": ["q3"]}
    with pytest.raises(MissingScript):
        RandomQInquirer(questions=pool).reset(a_view(task_id="absent"), 0)
    # The sibling's behaviour on the identical condition, asserted so the parity cannot drift.
    with pytest.raises(MissingScript):
        ParallelReplayInquirer(questions=pool).reset(a_view(task_id="absent"), 0)


def test_random_q_still_volume_matches_when_the_task_IS_in_the_pool():
    """The refusal above must not break the path the arm exists for."""
    pool = {"t1": ["a", "b", "c"], "other": [f"q{i}" for i in range(20)]}
    p = RandomQInquirer(questions=pool)
    p.reset(a_view(task_id="t1"), 0)
    asked = [p.act(a_state()).text for _ in range(3)]
    assert len(asked) == 3 and all(q.startswith("q") for q in asked), asked


def test_an_explicit_ask_count_is_still_honoured_for_a_task_outside_the_pool():
    """`ask_counts` and `n_asks` are the CALLER stating the volume, which is the one case where
    a task absent from the pool is not an undefined volume. Refusing these too would break
    seeding a null from a count table rather than from a prior sweep."""
    pool = {"other": [f"q{i}" for i in range(20)]}
    p = RandomQInquirer(questions=pool, ask_counts={"absent": 2})
    p.reset(a_view(task_id="absent"), 0)
    assert len([p.act(a_state()).text for _ in range(2)]) == 2
    q = RandomQInquirer(questions=pool, n_asks=4)
    q.reset(a_view(task_id="absent"), 0)
    assert len([q.act(a_state()).text for _ in range(4)]) == 4
