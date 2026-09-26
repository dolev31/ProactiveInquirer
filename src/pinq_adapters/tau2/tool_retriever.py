"""A retrieval channel for tau2 domains that have no free-text search.

WHAT IT REPLACES. Retail, airline and telecom expose typed, key-addressed tools and no
`KB_search`. `RetailNullRetriever` therefore returns `()` for every query, and on those suites
an `Ask` resolved to nothing at all: the Inquirer was asking questions of a corpus with no
query interface, and `retrieved_uids` came back empty however good the question was. This
module answers a question by selecting ONE read-only tool and executing it.

WHY A RETRIEVER AND NOT A NEW `Ask.target`. `Inquirer.act(s: State)` takes State and nothing
else, and `Ask.target` is `{"kb", "user"}`. A tool target would move tool choice inside the
Inquirer, where a tool-aware or budget-aware policy makes prefix-k of a long rollout
non-exchangeable with a true short run -- the reason `act` has the signature it has. The
Retriever protocol is already the "resolve this question against the world" seam, so the
question stays a question and the world stays behind the seam.

THREE INVARIANTS, EACH BECAUSE THE ALTERNATIVE IS SILENT.

  READ-ONLY, DERIVED FROM THE ENVIRONMENT. The selectable set is every advertised tool for
  which `Environment._is_mutating_tool` is false. Not a literal list: a literal goes stale the
  moment upstream adds a tool, and the failure is a write executed through the retrieval
  channel, which permanently disables tripwire #6 ("mutating EnvCalls outside
  evaluation_criteria.actions must be 0"). A model naming a write is data -- recorded as
  `retriever_rejected_tool` -- not an exception.

  NO GOLD, BY CONSTRUCTION. This class is built from an environment and tool SCHEMAS. It never
  receives the task record, so `evaluation_criteria` (the answer key) and `user_scenario` (the
  user-private partition the ceiling is defined over) cannot reach the selector prompt. That is
  a property of the constructor signature, which is why a test asserts the signature.

  CHARGED EXACTLY ONCE. `run_loop` charges retrieval before calling `search`. The runner
  separately meters the env calls it finds in the TRANSCRIPT. These calls go through
  `env.make_tool_call`, which bypasses the Orchestrator, so they never enter `sim.messages` and
  the two paths cannot both see them. The runner adds `retriever.env_calls` to the outcome
  explicitly.

EVIDENCE FROM ARGUMENTS WHERE THERE IS A KEY, FROM THE REQUEST WHERE THERE IS NOT. A keyed
read (`get_reservation_details(reservation_id=...)`) mints the uid of the record it named,
through the domain index -- arguments only, never the result, exactly as `retail_units` does.
A read whose arguments name no record (`search_direct_flight(origin, destination, date)`) still
told the agent something, so it mints ONE unit addressed by the REQUEST: the doc_id is the tool
name and the canonical arguments. Nothing about the uid comes from the payload, which is the
rule `EvidenceUnit.uid` obeys everywhere else; the gold builders never emit such a doc_id, so
these units are evidence that can never be mistaken for a satisfied gold node.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from pinq import promptlib
from pinq.ids import canon
from pinq.ids import h as _h
from pinq.types import EnvCall, EvidenceUnit

SELECT_TEMPLATE = "retriever_select"


class RetrieverNotWired(RuntimeError):
    """A tool-backed retriever was asked to search with no model and no model-free declaration.

    IT USED TO FAIL OPEN, and that is the whole reason this exists. `search` returned `()` when
    `self._llm is None`, so a runner that built the retriever before constructing its client
    produced runs where every Ask resolved to nothing while the run reported success with a real
    graded reward -- 1,228 of 1,228 asks over 87 fork units, and one withdrawn finding. The
    published-era runner carried a factory that passed the client and a docstring warning of
    exactly this; a refactor deleted both, and nothing objected for four revisions because a
    silent empty channel is indistinguishable from a policy asking badly.
    """


# THE tau2 EVIDENCE WINDOW -- ONE constant for both places a tau2 unit's text can be cut, so the
# two can never disagree: how much of an UNKEYED read's result this retriever stores as unit
# text, and (imported by `pi_run.stages.tau2_runner`) how much of any unit the Drafter's and
# Answerer's prompts render. A window the renderer widened could not show text the unit never
# stored, and that is what happened: this was 4000 here while the renderer went to 6000.
#
# A KEYED read never meets it: `_units_for` mints the unit from the index with the full record
# (max 4527 characters over both tau2 DBs, airline flights:HAT002). An UNKEYED read's unit IS
# the result. MEASURED over every airline search -- both search tools x every ordered pair of
# the 20 airports x every one of the 30 dates any flight carries -- plus list_all_airports,
# serialized exactly as `search` serializes them (`str(result)`): max 23,936 characters
# (search_onestop_flight ATL->PHX 2024-05-25); 15.1% of search_onestop_flight calls over
# 4000 and 10.2% over 6000 (most pairs return `[]`); search_direct_flight max 2833. Retail's
# largest unkeyed read (list_all_product_types) is 1478. 32000 covers 23,936 with margin; the
# share of results over it is 0 (tests/test_tau2_evidence_window.py measures it).
#
# THE RESIDUAL is a list result longer than 32000: its first 32000 characters are kept. Moving
# this constant moves `result_digest` of this retriever's EnvCalls for results longer than the
# old cap -- expected, and it moves with `code_version`; those EnvCalls are not stored.
TAU2_EVIDENCE_CHARS = 32000

# THE ADDRESSING SPAN of a request-addressed unit, FROZEN at the cap that minted every recorded
# `call:` uid. A uid is (corpus_id, doc_id, span); `span` was `0:<len(text)>` with the text cut
# at 4000, so a result that long was addressed `0:4000`. Keeping that span when the stored text
# grows is what leaves every uid -- and the uid tag the questioner's prompt prints beside each
# unit -- exactly as before. For such a unit the span therefore names the addressed prefix,
# not the stored length; `len(unit.text)` is the stored length.
CALL_UNIT_SPAN_CHARS = 4000


def _schema_line(schema: Mapping[str, Any]) -> str:
    params = schema.get("parameters") or {}
    props = params.get("properties") if isinstance(params, Mapping) else None
    names = ", ".join(sorted(props)) if isinstance(props, Mapping) and props else ""
    desc = str(schema.get("description") or "").strip().splitlines()
    head = desc[0] if desc else ""
    return f"- {schema.get('name')}({names}): {head}".rstrip(": ").rstrip()


def parse_selection(raw: str) -> tuple[str | None, dict[str, Any]]:
    """`(tool_name | None, args)` from the selector's reply. Never raises.

    A malformed answer is a measurement of the policy, not a crash: the rollout continues with
    no evidence for that turn, which is the honest accounting.
    """
    if not raw:
        return None, {}
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None, {}
    try:
        obj = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None, {}
    if not isinstance(obj, Mapping):
        return None, {}
    name = obj.get("tool")
    args = obj.get("args")
    if not isinstance(name, str) or not name.strip():
        return None, {}
    return name.strip(), dict(args) if isinstance(args, Mapping) else {}


class ToolBackedRetriever:
    """Satisfies `pinq.protocols.Retriever` over a tau2 Environment with no search tool."""

    def __init__(
        self,
        env: Any,
        *,
        corpus_id: str,
        corpus_hash: str,
        index: Any = None,
        uids_for_call: Any = None,
        tool_schemas: Sequence[Mapping[str, Any]] = (),
        llm: Any = None,
        llm_free: bool = False,
        seed: int = 0,
        recorder: Any = None,
    ) -> None:
        self._env = env
        # DECLARED, not inferred. The exemption keys on the ARM saying it runs without a model
        # (`fake_chain`, `fake_depth1`, `fake_drafter_only`), never on the client happening to be
        # absent -- which is the condition that hid the defect.
        self._llm_free = bool(llm_free)
        self.corpus_id = corpus_id
        self.corpus_hash = corpus_hash
        self._index = index
        self._uids_for_call = uids_for_call
        self._llm = llm
        self._seed = seed
        self._recorder = recorder
        self._calls: list[EnvCall] = []
        self._seq = 0
        self.turn_idx = 0
        self.units_seen: dict[str, EvidenceUnit] = {}
        # Read-only, asked of the environment rather than declared here.
        self._readable = tuple(
            sorted(
                str(s.get("name"))
                for s in tool_schemas
                if s.get("name") and not env._is_mutating_tool(str(s.get("name")))
            )
        )
        self._menu = "\n".join(
            _schema_line(s) for s in tool_schemas if str(s.get("name")) in set(self._readable)
        )

    # ---------------------------------------------------------------- Retriever protocol

    @property
    def env_calls(self) -> tuple[EnvCall, ...]:
        return tuple(self._calls)

    @property
    def prompt_hashes(self) -> dict[str, str]:
        return {SELECT_TEMPLATE: promptlib.sha(SELECT_TEMPLATE)}

    def _note(self, key: str) -> None:
        if self._recorder is not None:
            self._recorder.record(key, 1)

    def search(self, query: str, k: int) -> Sequence[EvidenceUnit]:
        # A DECLARED model-free arm gets an empty channel; anything else REFUSES. The llm-free
        # ablations must stay runnable and comparable, and they say so on the arm. An UNDECLARED
        # missing client is a wiring defect, and returning () for it is what made that defect
        # invisible for four revisions.
        if self._llm is None:
            if self._llm_free:
                return ()
            raise RetrieverNotWired(
                "tool-backed retriever has no LLM client and the arm did not declare itself "
                "model-free, so it cannot turn a question into a read. Every Ask would resolve "
                "to nothing and the run would still report success. Pass llm= at the "
                "construction site (see tests/test_retriever_is_wired.py)."
            )
        if not self._readable:
            return ()
        prompt = promptlib.render(SELECT_TEMPLATE, question=query, tools=self._menu)
        raw, _tel = self._llm.complete(
            role="drafter",
            messages=[{"role": "user", "content": prompt}],
            seed=self._seed,
        )
        name, args = parse_selection(raw if isinstance(raw, str) else str(raw))
        if name is None:
            self._note("retriever_declined")
            return ()
        if name not in self._readable:
            # Either a write or a hallucinated name. Both execute nothing.
            self._note("retriever_rejected_tool")
            return ()

        ok = True
        try:
            raw_result = self._env.make_tool_call(name, requestor="assistant", **args)
        except Exception as exc:  # a failed read is data, not a crash
            ok, raw_result = False, f"ERROR: {exc}"
        serialized = raw_result if isinstance(raw_result, str) else str(raw_result)
        text = serialized[:TAU2_EVIDENCE_CHARS]

        self._seq += 1
        self._calls.append(
            EnvCall(
                seq=self._seq,
                turn_idx=self.turn_idx,
                requestor="assistant",
                tool_name=name,
                kwargs_json=canon(args),
                ok=ok,
                result_digest=_h("tool_result", text)[:16],
                mutating=False,
            )
        )
        if not ok:
            return ()

        units = self._units_for(name, args, text)
        for u in units:
            self.units_seen.setdefault(u.uid, u)
        return units[:k]

    # ---------------------------------------------------------------- evidence

    def _units_for(self, name: str, args: Mapping[str, Any], text: str) -> tuple[EvidenceUnit, ...]:
        """Keyed reads mint the records they named; unkeyed reads mint one request-addressed
        unit. Both carry the result as TEXT, and neither derives its uid from the result."""
        uids: tuple[str, ...] = ()
        if self._index is not None and self._uids_for_call is not None:
            uids = tuple(self._uids_for_call(self._index, name, dict(args)))
        if uids:
            by_uid = getattr(self._index, "unit_by_uid", None)
            if callable(by_uid):
                # The index stores units as DICTS -- the shape `units_from_db` and the gold
                # builders share -- and `run_loop` reads `.uid` off whatever `search` returns.
                # Handing the dict straight back killed the first live airline rollout with
                # `AttributeError: 'dict' object has no attribute 'uid'`, after the LLM call
                # had already been paid for.
                out = [
                    EvidenceUnit.make(
                        corpus_id=str(rec["corpus_id"]),
                        doc_id=str(rec["doc_id"]),
                        span=str(rec["span"]),
                        title=str(rec.get("title") or ""),
                        text=str(rec.get("text") or ""),
                    )
                    for rec in (by_uid(uid) for uid in uids)
                    if rec is not None
                ]
                if out:
                    return tuple(out)
        doc_id = f"call:{name}:{_h('tool_call', canon(dict(args)))[:16]}"
        return (
            EvidenceUnit.make(
                corpus_id=self.corpus_id,
                doc_id=doc_id,
                span=f"0:{min(len(text), CALL_UNIT_SPAN_CHARS)}",
                title=name,
                text=text,
            ),
        )
