"""Consensus labels -> training rows for the Inquirer.

WHAT A ROW IS. One decision point of one real conversation: the prompt the policy would be
shown at that moment, and the action two strong raters independently agreed was right there.
Nothing else in this repository has that shape, because nothing else has a person in it.

THE MAPPING, ARM BY ARM, AND WHY EACH POINTS WHERE IT DOES.

    B1 necessary                   ASK, carrying the question the agent actually asked.
    B1 answer_in_state             STOP. The honest lesson from a question that bought nothing
       answer_inferable            is "do not ask" -- the same rule `export_sft` applies when a
       default_existed             margin fails to clear the judge noise floor.
    B3 should_have_asked           ASK, carrying the question the RATER supplied. This is the
                                   only training target whose text comes from an annotator
                                   rather than from the log, and it is why B3 refuses that
                                   verdict when the question is missing.
    B3 stop_was_right              STOP.
    B3 should_have_continued       NO ROW. "Keep working" is the tool loop's decision, not the
                                   Inquirer's, and writing a STOP there would teach the exact
                                   opposite of what the rater said.
    B3 should_have_stopped_earlier NO ROW. It says this state was already too late, so the
                                   right action belongs to an earlier state that is its own
                                   decision point with its own label.
    B2 anything                    NO ROW. B2 measures a MISS, and a miss names no action the
                                   policy could have taken instead. It stays an eval metric.
    anything cant_tell             NO ROW.

UNANIMITY ONLY, AND FOILS NEVER. A split item is not a weak label; it is two competent readers
disagreeing, measured at 22-35% of items in the pilot. `annotate.consensus` already resolves
only unanimous units, so this consumes its output rather than re-deciding. A foil is a question
the agent never asked, rewritten so its answer sits in the state: its verdict is about a state
the log never contained, so it can never become a row.

ONE ROW PER DECISION POINT. B1 and B3 can both resolve at one state, and two rows there would
write a coin flip into the dataset. B3 wins the tie, because its verdict is about the whole
stop-or-ask decision while B1's is about a question that was already asked.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from pinq.actions import STOP_ACTION_JSON
from pinq.splitting import split_of

MATCHER_ID = "human_b1b3"

# THE ASK TARGET IS A QUESTION, NOT THE MESSAGE THAT CONTAINED IT. A `yield` decision's
# `asked_question` is the agent's ENTIRE final message, which merely happens to end in a
# question mark; measured on the first full export the median ASK target was 2,368 characters
# and the longest 6,622. Training on that teaches a model to emit a status report where a
# question belongs.
#
# THE EXTRACTION HAPPENS HERE AND NOT AT PARSE TIME. The rater was shown the whole block and
# judged whether asking was necessary, so the label is about that block; pulling the question
# out earlier would desynchronise the target from what the annotator actually read.
_SENTENCE = re.compile(r"[^.!?\n]*\?")
_EMPHASIS = " \t\n*_`\"'"


def extract_question(text: str) -> str:
    """The last interrogative sentence, or "" when there is none.

    "" is not a failure to be papered over: a `necessary` verdict on a message that asks
    nothing has no ASK target, and inventing one would put the agent's own report in the
    place where its question belongs.
    """
    found = _SENTENCE.findall(text or "")
    if not found:
        return ""
    return found[-1].strip(_EMPHASIS)


# THE ONE STOP. This used to carry `"rationale": "report"`, a clause no annotator wrote and
# no model emitted -- a fabricated target that taught the policy to produce prose it had no
# reason for, in a shape that differed from the benchmark exporter's. `pinq.actions` owns it.
_STOP_JSON = STOP_ACTION_JSON

_B1_STOP = frozenset({"answer_in_state", "answer_inferable", "default_existed"})
# B3 outranks B1 at a shared decision point -- see the module docstring.
_KIND_RANK = {"B3": 0, "B1": 1}


@dataclass(frozen=True, slots=True)
class ExportedRow:
    suite_id: str
    task_id: str
    run_id: str
    turn_idx: int
    state_text: str
    action_json: str
    value: float
    is_stop: bool
    scorer_hash: str
    graph_version: str
    matcher_id: str
    split: str
    template_id: str
    # Carried, not filtered: one looping session was 20% of every decision point mined, and a
    # consumer that cannot see which rows came from it cannot weight them down.
    repeats_earlier_turn: bool
    # WHICH INSTRUMENT AND VERDICT PRODUCED THIS ROW. Without it a reader cannot tell an ASK
    # whose text came from the log from one whose text came from an annotator.
    instrument: str
    verdict: str
    reaction: str
    # HOW STRONGLY THE RATERS AGREED. Unanimity of two and a two-of-three majority are
    # different evidence, and a shard that mixes them without saying which is which cannot be
    # filtered by whoever trains on it. `annotate.consensus` refuses a majority below three
    # raters -- a majority of two is one person outvoting nobody -- so the choice exists only
    # at three, and it belongs to the consumer rather than to this exporter.
    #
    # 0/0/False when the unit carried no vote record: absent is not agreement, and defaulting
    # `unanimous` to True would silently promote every such row to the stronger standard.
    n_agreeing: int = 0
    n_raters: int = 0
    unanimous: bool = False

    def as_example_kwargs(self) -> dict[str, Any]:
        """The subset `pinq_train.export.dataset.Example` accepts.

        `phi_tilde` is left at its NaN default deliberately: there is no potential function
        over a conversation, and writing 0.0 would be a measurement nobody took.
        """
        d = asdict(self)
        for k in (
            "repeats_earlier_turn",
            "instrument",
            "verdict",
            "reaction",
            "n_agreeing",
            "n_raters",
            "unanimous",
        ):
            d.pop(k)
        return d


def _action_for(kind: str, label: str, entry: Mapping[str, Any]) -> str | None:
    if kind == "B1":
        if label == "necessary":
            q = extract_question(str(entry.get("asked_question") or ""))
            return _ask_json(q) if q else None
        return _STOP_JSON if label in _B1_STOP else None
    if kind == "B3":
        if label == "stop_was_right":
            return _STOP_JSON
        if label == "should_have_asked":
            # The annotator wrote this one as a question already, so it is taken whole rather
            # than passed through the extractor, which would drop a question phrased without a
            # question mark.
            q = str((entry.get("_question") or "")).strip(_EMPHASIS)
            return _ask_json(q) if q else None
    return None


def _ask_json(question: str) -> str:
    return json.dumps(
        {"action": "ASK", "question": question, "rationale": "", "target": "user"},
        sort_keys=True,
    )


def rows_from_consensus(
    units: Iterable[Mapping[str, Any]],
    key: Mapping[str, Mapping[str, Any]],
    *,
    scorer_hash: str = "",
    graph_version: str = "convlog-v1",
) -> list[ExportedRow]:
    """Resolved units + the bundle key -> one row per labelled decision point.

    `units` are `annotate.ResolvedUnit`-shaped mappings: `kind`, `label`, `item_id` and a
    `provenance` mapping that carries B3's supplied question. A unit whose item is not in the
    key raises rather than being skipped -- a key and a record set that disagree about which
    items exist is a mismatch between a campaign and its own bundle, and silently dropping the
    difference hides it.
    """
    best: dict[tuple[str, int], tuple[int, ExportedRow]] = {}
    for u in units:
        kind, label = str(u.get("kind")), str(u.get("label"))
        item_id = str(u.get("item_id") or u.get("unit_id"))
        entry = key[item_id]  # KeyError on purpose; see the docstring.
        if entry.get("foil"):
            continue
        votes = dict(u.get("votes") or {})
        n_rate = len(votes)
        n_agree = sum(1 for v in votes.values() if v == label)
        prov = dict(u.get("provenance") or {})
        merged = dict(entry, _question=prov.get("question"))
        action = _action_for(kind, label, merged)
        if action is None:
            continue
        session, dp = str(entry["session_id"]), int(entry["dp_index"])
        task_id = f"{session}:{dp}"
        row = ExportedRow(
            suite_id="convlog",
            task_id=task_id,
            run_id=f"convlog-{session}",
            turn_idx=dp,
            state_text=str(entry["state_text"]),
            action_json=action,
            # 1.0 because the label is a human-equivalent consensus rather than a measured
            # gain: there is no potential function over a conversation to compare it against.
            value=1.0,
            is_stop=action == _STOP_JSON,
            scorer_hash=scorer_hash,
            graph_version=graph_version,
            matcher_id=MATCHER_ID,
            split=split_of("convlog", task_id, session),
            template_id=session,
            repeats_earlier_turn=bool(entry.get("repeats_earlier_turn")),
            instrument=kind,
            verdict=label,
            reaction=str(entry.get("reaction") or ""),
            n_agreeing=n_agree,
            n_raters=n_rate,
            unanimous=bool(n_rate) and n_agree == n_rate,
        )
        rank = _KIND_RANK.get(kind, 9)
        prev = best.get((session, dp))
        if prev is None or rank < prev[0]:
            best[(session, dp)] = (rank, row)
    return [r for _rank, r in sorted(best.values(), key=lambda x: (x[1].task_id,))]


def write_shard(
    units: Iterable[Mapping[str, Any]],
    key: Mapping[str, Mapping[str, Any]],
    *,
    out_dir: Any,
    scorer_hash: str = "",
    graph_version: str = "convlog-v1",
    disagreements: Iterable[Mapping[str, Any]] = (),
    gate: Mapping[str, Any] | None = None,
    gate_override_reason: str | None = None,
) -> dict[str, Any]:
    """Write `sft.jsonl`, `adjudication.jsonl` and `manifest.json`, and return the manifest.

    THE MANIFEST COUNTS WHAT DID NOT BECOME A ROW, by reason. A guard that drops rows without
    saying how many is a guard nobody can audit, which is the same argument `ExportManifest`
    makes for the benchmark exporter -- and here the drops are the majority: every B2 verdict,
    every `should_have_continued`, every split item.

    ITS OWN SHARD, NEVER THE PUBLIC ONES. `docs/DECISIONS.md` D16 keeps this suite unreleased,
    so its rows must not land beside musique's or strategyqa's where a per-suite licence would
    then be wrong about them.

    THE MANIFEST NAMES THE GATE, PASS OR FAIL. `gate` is the Gate B report these labels were
    built under and it is recorded unconditionally, because a field that appears only on
    failure is a field readers learn to skip. On 2026-09-02 the v2 gate recorded
    `passed: false` and this shard was written from the same records at the same timestamp
    with nothing in it saying so; the labels' agreement was measured, printed, and then not
    carried anywhere a consumer could see. `gate_override_reason` is the sentence a person
    wrote to justify exporting anyway -- it is required by the caller when the gate failed,
    and it is stored here so the justification travels with the rows rather than living in a
    shell history.
    """
    import collections
    import hashlib
    from pathlib import Path

    units = list(units)
    rows = rows_from_consensus(units, key, scorer_hash=scorer_hash, graph_version=graph_version)

    reasons: collections.Counter[str] = collections.Counter()
    for u in units:
        kind, label = str(u.get("kind")), str(u.get("label"))
        item_id = str(u.get("item_id") or u.get("unit_id"))
        entry = key.get(item_id) or {}
        if entry.get("foil"):
            reasons["n_foil_dropped"] += 1
        elif kind == "B2":
            reasons["n_b2_no_action"] += 1
        elif label == "cant_tell":
            reasons["n_cant_tell"] += 1
        elif (
            kind == "B3"
            and label == "should_have_asked"
            and not (u.get("provenance") or {}).get("question")
        ):
            reasons["n_b3_missing_question"] += 1
        elif _action_for(kind, label, dict(entry, _question=None)) is None:
            reasons["n_verdict_no_action"] += 1

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sft.jsonl").write_text(
        "".join(json.dumps(asdict(r), sort_keys=True) + "\n" for r in rows)
    )
    # THE SPLITS ARE AN ARTIFACT, NOT A LOSS. Two strong readers disagreeing about one state
    # names exactly the items a person should read next, and the state travels with the votes
    # because a verdict pair without it cannot be adjudicated.
    adj = []
    for d in disagreements:
        item_id = str(d.get("item_id") or str(d.get("unit_id", "")).split("/")[0])
        e = key.get(item_id) or {}
        adj.append(
            {
                "item_id": item_id,
                "unit_id": str(d.get("unit_id") or item_id),
                "kind": str(d.get("kind") or ""),
                "votes": dict(d.get("votes") or {}),
                "state_text": str(e.get("state_text") or ""),
                "asked_question": str(e.get("asked_question") or ""),
                "final_assistant_text": str(e.get("final_assistant_text") or ""),
                "session_id": str(e.get("session_id") or ""),
                "dp_index": e.get("dp_index"),
            }
        )
    (out / "adjudication.jsonl").write_text(
        "".join(json.dumps(a, sort_keys=True) + "\n" for a in adj)
    )

    train_ids = sorted(r.task_id for r in rows if r.split == "train")
    manifest: dict[str, Any] = {
        "n_units_in": len(units),
        "n_rows": len(rows),
        "counts_by_split": dict(collections.Counter(r.split for r in rows)),
        "counts_by_instrument": dict(collections.Counter(r.instrument for r in rows)),
        "counts_by_verdict": dict(collections.Counter(r.verdict for r in rows)),
        "n_stop_rows": sum(1 for r in rows if r.is_stop),
        "n_ask_rows": sum(1 for r in rows if not r.is_stop),
        # ON THE MANIFEST, not left for whoever trains on it to find out.
        "ask_share": sum(1 for r in rows if not r.is_stop) / len(rows) if rows else float("nan"),
        "n_adjudication": len(adj),
        "n_unanimous_rows": sum(1 for r in rows if r.unanimous),
        "n_majority_rows": sum(1 for r in rows if r.n_raters and not r.unanimous),
        "n_repeated_prompt_rows": sum(1 for r in rows if r.repeats_earlier_turn),
        "n_sessions": len({r.template_id for r in rows}),
        "scorer_hash": scorer_hash,
        "graph_version": graph_version,
        "matcher_id": MATCHER_ID,
        # THE AGREEMENT THESE LABELS WERE BUILT UNDER. Recorded even when it passed, and
        # `None` only when no gate was supplied at all -- which the caller refuses to allow.
        "gate_passed": (None if gate is None else bool(gate.get("passed"))),
        "gate_alpha": (None if gate is None else dict(gate.get("alpha") or {})),
        "gate_override_reason": gate_override_reason,
        # The ids a checkpoint saw, so a later reader can prove which conversations trained it.
        "train_id_set_hash": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        **{
            k: reasons.get(k, 0)
            for k in (
                "n_foil_dropped",
                "n_b2_no_action",
                "n_cant_tell",
                "n_b3_missing_question",
                "n_verdict_no_action",
            )
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest
