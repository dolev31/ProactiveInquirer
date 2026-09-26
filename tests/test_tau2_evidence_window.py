"""The tau2 Drafter must see a whole record; the questioner's view must not move.

THE DEFECT. `pinq_expt.components.UNIT_CHARS` is 1200, sized for QA paragraphs, and
`render_evidence` cut every unit at it for the Drafter's resolve and draft prompts and for the
Answerer. tau2 records are long: every airline flight record is over 1200 characters (median
4418), and the cut removes the later dates and their `available` status -- on HAT110 the Drafter
saw 2024-05-01..05-10 and never the word "available", so no flight could ever be booked. Keyed
reads return the full record; the cut was the renderer.

THE FIX IS SCOPED TWICE. To tau2: the window is a constructor constant of the Drafter and the
Answerer, and only the tau2 driver passes a wider one -- QA arms are built without it and render
byte-identical prompts. To the Claude roles: the Inquirer (the Qwen questioner, 16384-token serve
window, TRAINED on the 1200-character view by `pi_run.cmd_train`) is built without it on tau2
too, so its prompt is byte-identical before and after.

Every test builds components the way the tau2 driver does, by capturing the keyword arguments
`run_tau2_unit` itself passes to `pinq_expt.arms.build` -- not by restating them here.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

# Skipped without tau2 -- unless PINQ_REQUIRE_TAU2_DATA=1, when a missing tau2 must FAIL
# (the imports below then raise at collection). See tests/_tau2_data.py.
if os.environ.get("PINQ_REQUIRE_TAU2_DATA") != "1":
    pytest.importorskip("tau2")

from pinq.budget import BudgetLedger  # noqa: E402
from pinq.types import Ask, Evidence, EvidenceUnit, State  # noqa: E402

QA_UNIT_CHARS = 1200


class _CaptureLLM:
    """Records every prompt; returns a fixed completion. Deterministic by construction."""

    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []

    def complete(self, *, role: str, messages, seed: int, max_tokens: int, **kw: Any):
        self.prompts.append((role, messages[0]["content"]))
        return "ok", None


def _suite(name: str):
    from tests._tau2_data import require_tau2_data

    require_tau2_data(name)
    if name == "airline":
        from pinq_adapters.tau2.airline_suite import Tau2AirlineSuite

        return Tau2AirlineSuite()
    from pinq_adapters.tau2.retail_suite import Tau2RetailSuite

    return Tau2RetailSuite()


def _tau2_build_kwargs(suite, tmp_path: Path, monkeypatch) -> dict[str, Any]:
    """What `run_tau2_unit` passes to `arms.build`, captured at its first call and nothing else.

    An llm-free arm, so no client is built; the capture raises to stop the unit there. The
    per-unit objects (llm, retriever, recorder, questions) are dropped: the test supplies its own.
    """
    from pi_run import worker
    from pi_run.stages import tau2_runner as T
    from pi_run.worker import UnitSpec
    from pinq_expt import arms as arm_table

    captured: dict[str, Any] = {}

    class _Captured(Exception):
        pass

    def spy(arm, **kw):
        captured.update(kw)
        raise _Captured

    real_build = arm_table.build
    monkeypatch.setattr(worker, "load_suite", lambda sid, cd: suite)
    monkeypatch.setattr(arm_table, "build", spy)
    spec = UnitSpec(
        suite_id=suite.suite_id,
        corpus_dir=str(tmp_path),
        task_id=suite.task_ids()[0],
        arm_id="fake_chain",
        seed=0,
        runs_root=str(tmp_path / "runs"),
        cache_root=str(tmp_path / "cache"),
        code_version="testsha",
        dirty=False,
    )
    status = T.run_tau2_unit(spec)
    monkeypatch.setattr(arm_table, "build", real_build)
    assert "_Captured" in status.get("error", ""), status
    return {
        k: v
        for k, v in captured.items()
        if k not in ("llm", "retriever", "recorder", "questions", "ask_counts")
    }


def _build(arm_id: str, llm: _CaptureLLM, **kw: Any):
    from pinq_expt import arms as arm_table

    return arm_table.build(arm_table.get(arm_id), llm=llm, **kw)


def _unit(suite, doc_id: str) -> EvidenceUnit:
    uid = suite.index.uids[doc_id]
    rec = suite.index.unit_by_uid(uid)
    return EvidenceUnit.make(
        corpus_id=str(rec["corpus_id"]),
        doc_id=str(rec["doc_id"]),
        span=str(rec["span"]),
        title=str(rec.get("title") or ""),
        text=str(rec["text"]),
    )


def _longest(suite, table: str) -> str:
    docs = [d for d in suite.index.uids if d.startswith(f"{table}:")]
    return max(docs, key=lambda d: len(suite.index.unit_by_uid(suite.index.uids[d])["text"]))


def _view(suite):
    from tau2.data_model.message import UserMessage

    from pi_run.stages.tau2_runner import dialogue_view

    return dialogue_view(suite, suite.task_ids()[0], [UserMessage(role="user", content="Help.")])


def _claude_prompts(suite, unit: EvidenceUnit, kw: dict[str, Any]) -> dict[str, str]:
    """The resolve, draft and answer prompts of the tau2 `inquirer_prompted` arm over `unit`."""
    llm = _CaptureLLM()
    c = _build("inquirer_prompted", llm, **kw)
    view, ev, led = _view(suite), Evidence.of([unit]), BudgetLedger(cap=16)
    c.drafter.resolve(view, Ask(text="which dates?"), ev, seed=0, ledger=led)
    draft = c.drafter.draft(view, ev, seed=0, ledger=led)
    c.answerer.answer(view, ev, draft, seed=0, ledger=led)
    roles = [r for r, _p in llm.prompts]
    assert roles == ["drafter", "drafter", "answerer"], roles
    return {"resolve": llm.prompts[0][1], "draft": llm.prompts[1][1], "answer": llm.prompts[2][1]}


# ------------------------------------------------------------------------------------ (a), (b)


@pytest.mark.parametrize("flight", ["HAT110", "longest"])
def test_a_the_tau2_drafter_sees_an_airline_flights_last_date_and_its_availability(
    flight, tmp_path, monkeypatch
):
    """HAT110 is the flight the defect was found on; `longest` is the longest airline record
    (flights:HAT002, 4527 characters, the maximum over both tau2 DBs). The record's LAST date
    and the word "available" must reach the resolve, draft AND answer prompts. Non-vacuous: at
    the QA window both are cut away."""
    suite = _suite("airline")
    doc = "flights:HAT110" if flight == "HAT110" else _longest(suite, "flights")
    unit = _unit(suite, doc)
    record = json.loads(unit.text)
    last_date = max(record["dates"])
    assert last_date in unit.text and "available" in unit.text
    assert last_date not in unit.text[:QA_UNIT_CHARS], "the QA window would not cut this record"

    prompts = _claude_prompts(suite, unit, _tau2_build_kwargs(suite, tmp_path, monkeypatch))
    for name, prompt in prompts.items():
        assert unit.text in prompt, f"the {name} prompt does not carry the whole record"
        assert last_date in prompt and "available" in prompt, name


def test_b_the_tau2_drafter_sees_the_longest_retail_product_whole(tmp_path, monkeypatch):
    suite = _suite("retail")
    unit = _unit(suite, _longest(suite, "products"))
    assert len(unit.text) > QA_UNIT_CHARS
    tail = unit.text[-200:]
    assert tail not in unit.text[:QA_UNIT_CHARS]

    prompts = _claude_prompts(suite, unit, _tau2_build_kwargs(suite, tmp_path, monkeypatch))
    for name, prompt in prompts.items():
        assert unit.text in prompt, f"the {name} prompt does not carry the whole record"


def test_the_window_covers_every_record_in_both_tau2_dbs(tmp_path, monkeypatch):
    """The constant is sized from the harness's own records, and every one fits it."""
    from pi_run.stages.tau2_runner import TAU2_EVIDENCE_CHARS

    longest = 0
    for name in ("airline", "retail"):
        suite = _suite(name)
        longest = max([longest, *(len(str(u["text"])) for u in suite.index.units.values())])
    assert longest == 4527, "the measured maximum recorded beside the constant has moved"
    assert TAU2_EVIDENCE_CHARS >= longest


# ------------------------------------------------------------------------------------ (c)
#
# Byte identity is pinned by sha256 of the rendered prompt, computed at f622eb19 (before the
# window existed) over a fixed synthetic 3000-character unit -- long enough that any change to
# the QA window would change the bytes.

_QA_UNIT = EvidenceUnit.make(
    corpus_id="qa",
    doc_id="doc:1",
    span="0:3000",
    title="A long paragraph",
    text="".join(f"{i:04d} " for i in range(600)),
)
QA_SHA = {
    "resolve": "1ed94cbed52f68fd6ad04c9a6ef86c22fc98413eced144a5802a0d7017dfc791",
    "draft": "33fd76f2402486153064bf91b3766f36d9b11b385e5f446d339cd3e1d32ef622",
    "answer": "749875656a0293e56c21513e3594cdfdd8de7b94b35d26ebb09e1991dbeac913",
}


def _qa_view():
    from pinq.view import make_view

    return make_view(
        task_id="t0",
        suite_id="synth",
        question="What does the paragraph say?",
        instructions="Answer from the evidence.",
        corpus_id="qa",
        corpus_hash="qa-hash",
        word_cap=180,
    )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_c_a_qa_arms_prompts_are_byte_identical():
    """QA builds its arms WITHOUT the window (`pi_run.worker.run_unit` passes none), so the
    Drafter's resolve and draft prompts and the Answerer's are the bytes they were."""
    import inspect

    from pi_run import worker

    assert "unit_chars" not in inspect.getsource(worker.run_unit)
    llm = _CaptureLLM()
    c = _build("inquirer_prompted", llm)
    view, ev, led = _qa_view(), Evidence.of([_QA_UNIT]), BudgetLedger(cap=16)
    c.drafter.resolve(view, Ask(text="which?"), ev, seed=0, ledger=led)
    draft = c.drafter.draft(view, ev, seed=0, ledger=led)
    c.answerer.answer(view, ev, draft, seed=0, ledger=led)
    got = dict(zip(("resolve", "draft", "answer"), (_sha(p) for _r, p in llm.prompts)))
    assert got == QA_SHA


# ------------------------------------------------------------------------------------ questioner

# Computed at f622eb19, before the window existed.
INQUIRER_SHA_TAU2 = "8ffd5da11afb8a4a8353bd662167b026f96537a659ad306adfcdf3af419f2d1c"


def test_the_questioners_tau2_prompt_is_byte_identical(tmp_path, monkeypatch):
    """The Qwen questioner renders evidence through the SAME `render_evidence`
    (`PromptedInquirer._prompt`), was trained on its 1200-character view (`pi_run.cmd_train`),
    and serves in a 16384-token window. Built exactly as the tau2 driver builds it, over the
    longest airline record, its prompt must be the bytes it was before the fix: the window is
    widened for the Claude roles only."""
    suite = _suite("airline")
    unit = _unit(suite, _longest(suite, "flights"))
    llm = _CaptureLLM()
    c = _build("inquirer_prompted", llm, **_tau2_build_kwargs(suite, tmp_path, monkeypatch))
    c.inquirer.reset(_view(suite), 0)
    prompt = c.inquirer._prompt(State(view=_view(suite), evidence=Evidence.of([unit])))
    assert unit.text[:QA_UNIT_CHARS] in prompt and unit.text not in prompt
    assert _sha(prompt) == INQUIRER_SHA_TAU2


# ------------------------------------------------------------------------------------ (d)


def test_d_the_widened_drafter_is_still_pure(tmp_path, monkeypatch):
    """`Drafter.draft()` must be pure in (view, evidence.subset_hash, seed): the window is a
    constructor constant, so two drafts over the same set -- supplied in either order -- send the
    same bytes, and a different set sends different ones."""
    suite = _suite("airline")
    kw = _tau2_build_kwargs(suite, tmp_path, monkeypatch)
    a = _unit(suite, "flights:HAT110")
    b = _unit(suite, _longest(suite, "flights"))
    llm = _CaptureLLM()
    c = _build("inquirer_prompted", llm, **kw)
    view, led = _view(suite), BudgetLedger(cap=16)
    one, two = Evidence.of([a, b]), Evidence.of([b, a])
    assert one.subset_hash == two.subset_hash
    c.drafter.draft(view, one, seed=3, ledger=led)
    c.drafter.draft(view, two, seed=3, ledger=led)
    c.drafter.draft(view, Evidence.of([a]), seed=3, ledger=led)
    p1, p2, p3 = (p for _r, p in llm.prompts)
    assert p1 == p2
    assert p1 != p3


# ------------------------------------------------------------------------------------ the merge


def test_an_env_unit_does_not_erase_the_text_the_drafter_read():
    """`_attach_env_evidence` merged the harvest's text-less units OVER the trajectory's, so a
    record the questioner retrieved (full text) and a tool call then re-read came out with
    n_chars 0 in evidence.jsonl. The trajectory's unit must win; the uid set, the doc set and
    `reconcile` must not move, and a uid only the harvest saw is still added."""
    from pi_run.stages.tau2_runner import _attach_env_evidence
    from pi_run.worker import evidence_rows, reconcile
    from pinq.types import Outcome, Trajectory, Usage

    suite = _suite("retail")
    users = sorted(suite.db["users"])
    read = _unit(suite, f"users:{users[0]}")  # retrieved by the questioner, with its text
    traj = Trajectory(
        view=_view(suite),
        turns=(),
        evidence=Evidence.of([read]),
        outcome=Outcome(),
        usage=Usage(),
        stop_reason="policy_stop",
    )
    ledger = BudgetLedger(cap=16)
    ledger.note_docs([read.doc_id])  # what run_loop noted for it
    calls = [
        {"tool_name": "get_user_details", "kwargs_json": json.dumps({"user_id": u}), "ok": True}
        for u in users[:2]
    ]
    out = _attach_env_evidence(traj, suite, ledger, calls, charge=False)

    other = suite.index.uids[f"users:{users[1]}"]
    assert out.evidence.uids == {read.uid, other}
    kept = {u.uid: u for u in out.evidence.units}
    assert kept[read.uid].text == read.text, "the harvest's text-less unit replaced the read one"
    assert kept[other].text == ""
    rows = {r["uid"]: r for r in evidence_rows(out)}
    assert rows[read.uid]["n_chars"] == len(read.text) > 0
    assert out.evidence.doc_ids == {read.doc_id, f"users:{users[1]}"}
    assert reconcile(out, ledger)["docs_ok"] is True


# ------------------------------------------------------------------------------------ (e), (f)
#
# The SECOND cut of the same class. `ToolBackedRetriever` stored at most 4000 characters of a
# tool result as unit text. A KEYED read never reached that cut -- its unit is minted from the
# index with the full record (tested below as a pin) -- but an UNKEYED read's unit IS the
# result, and `search_onestop_flight` results run to 23,936 characters: the itineraries past
# 4000 were never stored, so no window downstream could show them.

ONESTOP = {
    "tool": "search_onestop_flight",
    "args": {"origin": "ATL", "destination": "PHX", "date": "2024-05-25"},
}


class _Selector:
    """The retriever's tool-selector model, scripted to pick one read."""

    def __init__(self, selection: dict) -> None:
        self._s = json.dumps(selection)

    def complete(self, **kw: Any):
        return self._s, None


def _ask_units(suite, selection: dict):
    retriever = suite.retriever(
        suite.task_ids()[0], llm=_Selector(selection), llm_free=False, seed=0, recorder=None
    )
    return tuple(retriever.search("which flights?", 5))


def _serialized(raw: Any) -> str:
    """How `ToolBackedRetriever.search` turns a result into unit text."""
    return raw if isinstance(raw, str) else str(raw)


def test_e_an_asks_long_search_result_reaches_the_drafter_whole(tmp_path, monkeypatch):
    """(e) The ask unit for search_onestop_flight(ATL, PHX, 2024-05-25), the longest search
    result in the airline DB, through the REAL ToolRetriever on the REAL environment and
    rendered by the tau2 drafter window: it is the environment's own full result string, and
    every Claude prompt carries its LAST itinerary. Compared to the string, not to a length. The
    last itinerary lies past both earlier caps (4000 in the retriever, 6000 at db38dd0's window)."""
    suite = _suite("airline")
    env = suite.environment(suite.task_ids()[0])
    raw = env.make_tool_call(ONESTOP["tool"], requestor="assistant", **ONESTOP["args"])
    full = _serialized(raw)
    last = str(raw[-1])
    assert last in full and last not in full[:4000] and last not in full[:6000]

    (unit,) = _ask_units(suite, ONESTOP)
    assert unit.doc_id.startswith("call:search_onestop_flight:")
    assert unit.text == full, "the ask unit did not store the whole search result"

    prompts = _claude_prompts(suite, unit, _tau2_build_kwargs(suite, tmp_path, monkeypatch))
    for name, prompt in prompts.items():
        assert last in prompt, f"the {name} prompt does not carry the last itinerary"


def test_a_keyed_ask_read_was_never_cut_by_the_retriever():
    """A PIN, not a fix -- it passed before this commit too. A keyed flight read mints its unit
    from the INDEX with the full record, so the retriever's old 4000 cap never touched it:
    HAT110 comes back whole, with its last date and the word "available"."""
    suite = _suite("airline")
    rec = suite.index.unit_by_uid(suite.index.uids["flights:HAT110"])
    last_date = max(json.loads(rec["text"])["dates"])
    selection = {
        "tool": "get_flight_status",
        "args": {"flight_number": "HAT110", "date": last_date},
    }
    (unit,) = _ask_units(suite, selection)
    assert unit.doc_id == "flights:HAT110"
    assert unit.text == rec["text"] and len(unit.text) > 4000
    assert last_date in unit.text and "available" in unit.text


class _ListEnv:
    """A read-only environment whose one tool returns a result longer than the window."""

    def __init__(self, n: int) -> None:
        self.n = n

    def _is_mutating_tool(self, name: str) -> bool:
        return False

    def make_tool_call(self, name: str, requestor: str = "assistant", **kw: Any) -> str:
        return "".join(f"{i:06d}|" for i in range(self.n // 7 + 1))[: self.n]


def test_f_a_list_result_beyond_the_window_is_capped_at_it():
    """(f) THE RESIDUAL, documented: a list result longer than the window keeps only its first
    `TAU2_EVIDENCE_CHARS` characters. Over the airline DB that residual is empty -- see
    `test_the_window_exceeds_every_unkeyed_read_in_the_airline_db` -- so this pins the behaviour
    for a result the DB does not yet produce. The expected cap is the LITERAL 32000, so that
    before the constant existed this failed on the stored text rather than on a missing name."""
    from pinq_adapters.tau2.tool_retriever import ToolBackedRetriever

    env = _ListEnv(40000)
    retriever = ToolBackedRetriever(
        env,
        corpus_id="c",
        corpus_hash="h",
        tool_schemas=({"name": "search_all"},),
        llm=_Selector({"tool": "search_all", "args": {}}),
    )
    (unit,) = retriever.search("q", 5)
    full = env.make_tool_call("search_all")
    assert unit.text == full[:32000], f"stored {len(unit.text)} of {len(full)} characters"
    assert len(full) > 32000


def test_the_window_exceeds_every_unkeyed_read_in_the_airline_db():
    """Measured, not assumed: every search over every ordered airport pair and every date any
    flight carries, and `list_all_airports`, serialized exactly as the retriever serializes
    them. The maximum is recorded beside the constant; the share over the window is 0."""
    from pinq_adapters.tau2.tool_retriever import TAU2_EVIDENCE_CHARS

    suite = _suite("airline")
    env = suite.environment(suite.task_ids()[0])
    flights = suite.db["flights"].values()
    airports = sorted({f["origin"] for f in flights} | {f["destination"] for f in flights})
    dates = sorted({d for f in flights for d in f["dates"]})
    sizes = [len(_serialized(env.make_tool_call("list_all_airports", requestor="assistant")))]
    for tool in ("search_direct_flight", "search_onestop_flight"):
        for o in airports:
            for d in airports:
                if o != d:
                    for date in dates:
                        raw = env.make_tool_call(
                            tool, requestor="assistant", origin=o, destination=d, date=date
                        )
                        sizes.append(len(_serialized(raw)))
    assert len(sizes) == 1 + 2 * 20 * 19 * 30
    assert max(sizes) == 23936, "the measured maximum recorded beside the constant has moved"
    assert sum(s > TAU2_EVIDENCE_CHARS for s in sizes) == 0


def test_one_constant_sets_both_tau2_windows():
    """The retriever's unit cap and the Drafter/Answerer render window are ONE constant, so the
    two can never disagree: the driver imports the retriever's rather than defining its own."""
    import inspect

    from pi_run.stages import tau2_runner
    from pinq_adapters.tau2 import tool_retriever

    assert tool_retriever.TAU2_EVIDENCE_CHARS == 32000
    assert tau2_runner.TAU2_EVIDENCE_CHARS is tool_retriever.TAU2_EVIDENCE_CHARS
    src = inspect.getsource(tau2_runner)
    assert "from pinq_adapters.tau2.tool_retriever import TAU2_EVIDENCE_CHARS" in src
    assert "TAU2_EVIDENCE_CHARS =" not in src
    assert "TAU2_EVIDENCE_CHARS]" in inspect.getsource(tool_retriever.ToolBackedRetriever.search)


def test_a_long_search_units_uid_and_the_questioners_view_of_it_are_unchanged(
    tmp_path, monkeypatch
):
    """The wider cap stores more text; it must not move the unit's IDENTITY, and it must not move
    what the questioner sees. The uid of a request-addressed unit is minted over the span it was
    always minted over -- "0:4000" for a result that long -- and the questioner renders at 1200,
    so its prompt over this unit is the bytes db38dd0 sent (sha pinned there)."""
    from pinq.ids import evidence_uid

    suite = _suite("airline")
    (unit,) = _ask_units(suite, ONESTOP)
    assert unit.span == "0:4000"
    assert unit.uid == evidence_uid(unit.corpus_id, unit.doc_id, "0:4000")

    llm = _CaptureLLM()
    c = _build("inquirer_prompted", llm, **_tau2_build_kwargs(suite, tmp_path, monkeypatch))
    c.inquirer.reset(_view(suite), 0)
    prompt = c.inquirer._prompt(State(view=_view(suite), evidence=Evidence.of([unit])))
    assert _sha(prompt) == INQUIRER_SHA_SEARCH


# Computed at db38dd0, where the unit held 4000 characters and the same uid.
INQUIRER_SHA_SEARCH = "b7499ba54d6149dbb47c476379f84b9c86ef8e49006e23042921d6ba8531a5f9"


def test_the_fail_not_skip_guard(monkeypatch):
    """With PINQ_REQUIRE_TAU2_DATA=1 an unreachable tau2 domain FAILS; without it, it skips."""
    from tests import _tau2_data

    monkeypatch.setenv("TAU2_DATA_DIR", "/nonexistent-tau2-data-dir")
    monkeypatch.setenv(_tau2_data.REQUIRE_ENV, "1")
    with pytest.raises(pytest.fail.Exception, match="PINQ_REQUIRE_TAU2_DATA=1"):
        _tau2_data.require_tau2_data("airline")
    monkeypatch.delenv(_tau2_data.REQUIRE_ENV)
    with pytest.raises(pytest.skip.Exception):
        _tau2_data.require_tau2_data("airline")
