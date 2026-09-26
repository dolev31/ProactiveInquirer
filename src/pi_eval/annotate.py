"""Human annotation: bundles out, records in, and the fields that turn ADR on.

WHY THIS EXISTS. `gold_human_asked` is a human annotation and is None on every node this
repository has ever built, so `metrics.human.anticipated_discovery_rate` refuses to compute
and `score` writes no `adr` row. Four preregistered gates print NOT RUN for the same reason:
their input is a measurement no code in this repository can take. This module is the one
place where a measurement taken by people becomes a gold field, and every rule below exists
to stop that transfer from inventing anything.

THREE FILES, NOT ONE. An annotation campaign has a bundle (what the annotators see), a key
(what would give the answer away) and records (what came back). They are separate files
because the bundle is embedded verbatim in a web page an annotator can read:

  * the BUNDLE carries the question, the candidate texts, the evidence — nothing that says
    which answer the pipeline already believes;
  * the KEY carries the unblinding side — which candidate the automatic margin preferred,
    what the matcher decided, what the mechanical latency label says — and never ships;
  * the RECORDS carry one response per (item, annotator).

An A2 preference whose bundle announced the pipeline's own choice would measure agreement
with a hint, not agreement with a judgment. So a shown A2 slot has no canonical meaning
until it is read back through the key, and `consensus` emits no unit for an A2 item whose
key entry is missing rather than guessing an orientation.

WHAT IS NEVER WRITTEN.
  * A node nobody was SHOWN keeps `gold_human_asked = None`. False there would enter ADR's
    numerator on the strength of a question never asked — the same absent-is-not-zero error
    the metric's own docstring was written to refuse.
  * A unit two annotators DISAGREED on writes nothing and goes to the adjudication queue.
  * An A3 verdict never edits `gold_partition` or `gold_discoverability`. A human "not a
    need" on a mined-required node is a PRECISION datum for G-M1; rewriting the partition
    would move every coverage metric's universe mid-campaign, so the graph records that a
    human looked (`gold_human_adjudicated`) and the verdict is carried in the gate numbers.
  * An A6 ranking never writes ANYTHING gold-side either, for the same reason A5 does not: it
    is a preference between two RUNS' next questions, not a claim about the task's graph.
    `merge_into_graphs` has no branch for `kind == "A6"` (see `test_a6_never_writes_a_gold_field`).
  * An A5 verdict never writes ANYTHING gold-side. It is a judgment about a RUN's stop, not
    about the task's graph -- `merge_into_graphs` has no branch for `kind == "A5"` or
    `"A5_missing"`, on purpose, not an oversight (see `test_a5_never_writes_a_gold_field`).
    What it produces instead is training data: a `should_have_asked_more` verdict paired with
    the SPECIFIC still-open need (`missing`) is a supervised example of a question the policy
    should have asked from that exact state -- the reason this instrument exists at all.
  * Free-text missing needs are never consensus'd. Two people describing the same gap in
    different words cannot be matched mechanically, so `node_recall` is NaN until someone
    adjudicates the merge and says how many distinct needs were named.

WHY A6 EXISTS, AND WHY IT IS NOT A SECOND A2. `pi train sample-candidates` forks one state
into k=5 candidate continuations. A2 shows an annotator TWO of them and buys one pairwise
comparison; ranking all five buys C(5,2) = 10, from the same state, at the same sampling cost
and for roughly the same annotation effort -- a ~10x gain in signal density on exactly the
preference data rung-2 DPO consumes. That density is the whole justification for the
instrument, and everything else about A6 is subordinate to it:

  * the judgment is TIERS, not a strict order (`{"c0": 1, "c1": 2, "c2": 2, ...}`, lower is
    better). EQUAL TIERS ARE A GENUINE TIE. Forcing a strict order on candidates a person
    considers equivalent manufactures a preference that was never there, and A6's derived
    pairs feed DPO training directly, so a manufactured preference is not a rounding error --
    it is a training example of a distinction nobody drew;
  * the UNIT is one unordered candidate PAIR, never the whole ranking. Two annotators who
    agree on 9 of 10 pairs and swap the bottom two must not score as total disagreement,
    which is precisely what a ranking-equality comparison would report;
  * the shipped `candidate_id` is item-local (`c0..c4`, assigned AFTER the shuffle). A run id
    in the bundle would let an annotator correlate candidates across items, and it is exactly
    the sort of identifier the key exists to hold.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from pi_eval.build.common import as_dict
from pi_eval.gold import GoldGraph
from pi_eval.judges.paired import krippendorff_alpha_nominal
from pi_eval.metrics.convlog import B1_LABELS as _B1_LABELS
from pi_eval.metrics.convlog import B2_LABELS as _B2_LABELS
from pi_eval.metrics.convlog import B3_LABELS as _B3_LABELS

TOOL_VERSION = "pi_annotate/1"

# Item types are what an annotator is shown. UNIT KINDS are what an agreement statistic is
# computed over, and they are not the same list: one A3_node item asks two questions drawn
# from two different label vocabularies, and pooling them into one alpha would report
# agreement over a mixture with no unit.
TASK_TYPES = (
    "A1",
    "A2",
    "A3_node",
    "A3_edge",
    "A3_match",
    "A3_missing",
    "A4",
    "A5",
    "A6",
    "A7",
    # THE B FAMILY judges a real conversation instead of a rollout against a gold graph. It is
    # a separate family, not more A instruments, because its ground truth is different in kind:
    # what the person actually did next, with no graph to be right or wrong about. Its unit
    # kinds therefore never reach `merge_into_graphs`, exactly like A5's and A6's.
    "B1",
    # B1b IS B1 MINUS ITS CONTESTED OPTION, kept as its own task type so the two are a
    # controlled pair rather than a before-and-after nobody can compare. Its units are emitted
    # under kind "B1" so both land in the same column.
    "B1b",
    "B2",
    "B3",
    # B4 PROPOSES rather than recognises. B3 asked "should it have asked something first?" and
    # three strong raters answered yes, unanimously, ONCE in 848 units -- a recognition task
    # with a base rate near zero. B4 asks a strong model for the question itself, and the
    # proposal is then validated through B1 unchanged by different raters.
    "B4",
)

# Below three candidates there is nothing A2 does not already measure, and the same state
# shipped as both instruments would spend two annotation slots on one comparison and enter it
# into the preference table twice. Enforced at the SAMPLER (which never builds such an item)
# and here (which refuses one that reached a bundle anyway).
A6_MIN_CANDIDATES = 3

_LABELS: dict[str, tuple[str, ...]] = {
    "A1": ("asked", "not_asked"),
    "A2": ("chosen", "rejected", "tie", "both_bad"),
    "A3_node": ("required", "optional", "not_a_need"),
    "A3_node_disc": ("kb", "user_private", "cant_tell"),
    "A3_edge": ("holds", "does_not_hold", "unsure"),
    "A3_match": ("addresses", "does_not_address"),
    "A4": ("stated_in_task", "evidence_only", "unsure"),
    "A4_parents": ("plausible", "not_plausible"),
    # A5: the STOP judgment. UNIT KINDS deliberately split the same response two ways -- see
    # `_units` -- because "was stopping right" and "was candidate X still needed" are answered
    # from two different label vocabularies and pooling them into one alpha would be a mixture
    # with no unit, same reasoning as A3_node/A3_node_disc above.
    "A5": (
        "stopping_was_right",
        "should_have_asked_more",
        "should_have_stopped_earlier",
        "cant_tell",
    ),
    "A5_missing": ("still_needed", "resolved_enough"),
    # A6: read off a TIER MAP, one label per unordered candidate pair, oriented by the same
    # lexicographic (a, b) ordering the unit id uses -- see `a6_pair_labels`.
    "A6": ("a_better", "b_better", "tie"),
    # A7: one response, TWO unit kinds, never pooled -- "does this candidate reach for
    # something the task statement never names" and "which candidate is the better
    # anticipatory move" are answered from two different label vocabularies, and one alpha
    # over both would be a mixture with no unit, same reasoning as A3_node/A5 above. Both are
    # CANONICAL (chosen/rejected), derived through the key's shown-order coin flip like A2's.
    "A7": ("reaches_unstated", "stays_stated", "cant_tell"),
    "A7_pref": ("chosen", "rejected", "tie", "both_bad"),
    # The B vocabularies are DEFINED in pi_eval.metrics.convlog and imported, not restated.
    # The metric decides which labels sit in its numerator, so a label added here and forgotten
    # there would silently leave the new verdict out of every rate.
    "B1": _B1_LABELS,
    "B1b": tuple(x for x in _B1_LABELS if x != "default_existed"),
    # The proposer must be able to DECLINE. Without `none_needed` a generator asked for a
    # question returns one every time, and the downstream acceptance rate would measure the
    # validators' patience rather than this model's judgement.
    "B4": ("question_needed", "none_needed", "cant_tell"),
    "B2": _B2_LABELS,
    "B3": _B3_LABELS,
}

_A2_CHOICES = ("a", "b", "tie", "both_bad")
_USEFULNESS_RANGE = (1.0, 5.0)

# A RATIONALE is explanatory metadata, never a label: it sits beside `response`, not inside
# it, and every consensus/gate computation below must be blind to whether it is present (see
# `validate_records` and the tests that pin that down). Capped short on purpose -- a REASON,
# not an essay: long free text costs tokens on every future read, and the only reader
# (`pi annotate review`'s disagreement sample) has no use for more than a sentence or two.
RATIONALE_MAX_CHARS = 400

# A2's closed-vocabulary "what made the winner better". Closed, not free text, so it is
# aggregatable across hundreds of items the way prose is not; "other" exists so a genuine
# reason outside this list is not forced into a bad-fitting bucket.
A2_BASIS_VALUES = (
    "targets_an_unresolved_need",
    "names_its_entities",
    "less_redundant",
    "better_scoped",
    "other",
)

# A7's "what made the winner the better ANTICIPATORY move". Its own vocabulary, never
# A2's: A2's basis has no anticipation option, which is the hole A7 exists to close, and a
# shared list would let one instrument's buckets absorb the other's judgment.
A7_BASIS_VALUES = ("anticipation", "hygiene", "no_difference")

# WHO IS ALLOWED TO BE V_human.
#
# A model may annotate. It may never BE the annotation. `adr` is a claim about what a PERSON
# would have thought to ask, and a model's prediction of that is a different quantity wearing
# the same shape -- plausible, well-formed, and not the measurement. Writing it to
# `gold_human_asked` would manufacture the exact artifact this repository refuses, under the
# one metric whose whole purpose is to support "exceeds what the user thought to ask".
#
# An LLM annotator id is prefixed, and the prefix and the `annotator_kind` field must agree.
# The redundancy is the point: a forgotten field cannot smuggle a model in as a person, and
# neither can a borrowed name.
RATER_KINDS = ("human", "llm")
LLM_ID_PREFIX = "llm:"
HUMAN_ONLY: tuple[str, ...] = ("human",)


class LLMAnnotationNotGold(RuntimeError):
    """A consensus that counted a model was routed at the gold write-back path."""


def rater_kind(record: Mapping[str, Any]) -> str:
    """Absent means human: every record written before models could annotate was one, and a
    default of "llm" would retroactively disqualify them."""
    return str(record.get("annotator_kind") or "human")


# --------------------------------------------------------------------------- identity


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def item_id(task_type: str, provenance: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    """Content-addressed, so an item cannot be edited without becoming a different item."""
    return hashlib.sha256(_canon([task_type, provenance, payload]).encode()).hexdigest()[:16]


def item_set_hash(items: Sequence[Mapping[str, Any]]) -> str:
    """Over the item ids alone, which is sufficient: each id already hashes that item's whole
    payload, so an edited item changes its id and therefore this."""
    ids = sorted(str(i["item_id"]) for i in items)
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- units


def _nodes_of(item: Mapping[str, Any]) -> tuple[dict, ...]:
    return tuple((item.get("payload") or {}).get("nodes") or ())


def _candidates_of(item: Mapping[str, Any]) -> tuple[dict, ...]:
    return tuple((item.get("payload") or {}).get("candidates") or ())


def _a6_candidate_ids(item: Mapping[str, Any]) -> list[str]:
    """Sorted, because that sort IS the canonical (a, b) orientation of every derived pair --
    the unit id, the label and the judgment row's `run_id_a`/`run_id_b` all read it."""
    return sorted(str(c.get("candidate_id")) for c in _candidates_of(item))


def a6_pair_labels(
    item: Mapping[str, Any], response: Mapping[str, Any]
) -> list[tuple[str, str, str]]:
    """(a, b, label) for every unordered candidate pair, a < b lexicographically.

    ONE ENTRY PER PAIR, not one per ranking: two annotators who agree on 9 of 10 pairs and swap
    the bottom two would score as total disagreement under a whole-ranking equality comparison
    (see `test_a6_agreement_is_per_pair_not_per_ranking`). EQUAL TIERS EMIT "tie" and nothing
    breaks the tie -- see the module docstring on why a manufactured order is worse here than a
    missing one.

    An incomplete or unreadable tier map yields NOTHING rather than a partial ranking: a
    candidate with no tier has no pairwise reading against anything, and defaulting one would
    invent the preference `validate_records` refuses the record for.
    """
    tiers = (response or {}).get("tiers")
    if not isinstance(tiers, Mapping):
        return []
    ids = _a6_candidate_ids(item)
    if len(ids) < A6_MIN_CANDIDATES:
        return []
    ranks: dict[str, float] = {}
    for cid in ids:
        t = tiers.get(cid)
        if isinstance(t, bool) or not isinstance(t, (int, float)):
            return []
        ranks[cid] = float(t)
    out: list[tuple[str, str, str]] = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if ranks[a] == ranks[b]:
                label = "tie"
            else:
                label = "a_better" if ranks[a] < ranks[b] else "b_better"
            out.append((a, b, label))
    return out


def _units(
    item: Mapping[str, Any],
    response: Mapping[str, Any],
    key_item: Mapping[str, Any] | None,
) -> list[tuple[str, str, str, dict, float | None]]:
    """(unit_id, unit_kind, label, provenance, usefulness) for one annotator's response."""
    tt = item["task_type"]
    iid = str(item["item_id"])
    prov = dict(item.get("provenance") or {})
    ki = dict(key_item or {})

    if tt == "A1":
        ticked = {str(x) for x in (response.get("ticked") or ())}
        useful = response.get("usefulness") or {}
        out = []
        for n in _nodes_of(item):
            nid = str(n["node_id"])
            p = dict(prov, gold_node_id=nid)
            rating = useful.get(nid)
            out.append(
                (
                    f"{iid}/{nid}",
                    "A1",
                    "asked" if nid in ticked else "not_asked",
                    p,
                    float(rating) if rating is not None else None,
                )
            )
        return out

    if tt == "A2":
        order = ki.get("order")
        if order not in ("ab", "ba"):
            # Without the key the shown slot has no canonical meaning. Silence, not a guess.
            return []
        choice = response.get("choice")
        if choice in ("tie", "both_bad"):
            label = str(choice)
        elif choice in ("a", "b"):
            label = "chosen" if (choice == "a") == (order == "ab") else "rejected"
        else:
            return []
        p = dict(prov)
        p.update({k: v for k, v in ki.items() if k != "order"})
        return [(iid, "A2", label, p, None)]

    if tt == "A3_node":
        out = [(iid, "A3_node", str(response.get("verdict")), dict(prov), None)]
        disc = response.get("discoverability")
        if disc:
            out.append((f"{iid}/disc", "A3_node_disc", str(disc), dict(prov), None))
        return out

    if tt == "A3_edge":
        holds = response.get("holds")
        label = {True: "holds", False: "does_not_hold"}.get(holds, "unsure")
        return [(iid, "A3_edge", label, dict(prov), None)]

    if tt == "A3_match":
        addresses = bool(response.get("addresses"))
        return [
            (iid, "A3_match", "addresses" if addresses else "does_not_address", dict(prov), None)
        ]

    if tt == "A4":
        out = [(iid, "A4", str(response.get("latency")), dict(prov), None)]
        ok = response.get("parent_uids_ok")
        if ok is not None:
            label = "plausible" if ok else "not_plausible"
            out.append((f"{iid}/parents", "A4_parents", label, dict(prov), None))
        return out

    if tt == "A5":
        verdict = str(response.get("verdict"))
        out = [(iid, "A5", verdict, dict(prov), None)]
        missing = {str(x) for x in (response.get("missing") or ())}
        # ONE UNIT PER CANDIDATE, not one unit over the whole `missing` set: a set-equality
        # comparison would score two annotators who agree on 3 of 4 open needs as total
        # disagreement (see test_a5_agreement_is_per_candidate_not_per_set). Each candidate's
        # label is read off the SAME `missing` list every candidate on this item shares, so an
        # annotator who names none of them still produces one "resolved_enough" unit per
        # candidate rather than nothing.
        for c in _candidates_of(item):
            nid = str(c["node_id"])
            p = dict(prov, gold_node_id=nid)
            label = "still_needed" if nid in missing else "resolved_enough"
            out.append((f"{iid}/{nid}", "A5_missing", label, p, None))
        return out

    if tt == "A6":
        # No key is consulted, deliberately, and this is the ONE blinded task type where that
        # is correct: A2's shown slot has no canonical meaning without the key because the
        # pipeline's preferred candidate hides behind it, whereas an A6 label is oriented by
        # candidate ids that are item-local and already on the shipped item. The key holds the
        # run ids those ids stand for, which is provenance, not orientation.
        return [
            (
                f"{iid}/{a}|{b}",
                "A6",
                label,
                dict(prov, candidate_a=a, candidate_b=b),
                None,
            )
            for a, b, label in a6_pair_labels(item, response)
        ]

    if tt == "A7":
        order = ki.get("order")
        if order not in ("ab", "ba"):
            # Without the key the shown slot has no canonical meaning. Silence, not a guess.
            return []
        p = dict(prov)
        p.update({k: v for k, v in ki.items() if k != "order"})
        out = []
        a_reaches, b_reaches = response.get("a_reaches"), response.get("b_reaches")
        chosen, rejected = (a_reaches, b_reaches) if order == "ab" else (b_reaches, a_reaches)
        if str(chosen) in _LABELS["A7"]:
            out.append((f"{iid}/chosen", "A7", str(chosen), dict(p), None))
        if str(rejected) in _LABELS["A7"]:
            out.append((f"{iid}/rejected", "A7", str(rejected), dict(p), None))
        pref = response.get("preference")
        if pref in ("tie", "both_bad"):
            out.append((f"{iid}/pref", "A7_pref", str(pref), dict(p), None))
        elif pref in ("a", "b"):
            label = "chosen" if (pref == "a") == (order == "ab") else "rejected"
            out.append((f"{iid}/pref", "A7_pref", label, dict(p), None))
        return out

    if tt in ("B1", "B1b"):
        verdict = str(response.get("verdict"))
        p = dict(prov)
        if response.get("default_action"):
            p["default_action"] = str(response["default_action"])
        return [(iid, "B1", verdict, p, None)]

    if tt == "B2":
        # ONE UNIT PER ITEM, never one over the set -- the same reasoning as A5_missing: two
        # annotators who agree on three of four items must not score as total disagreement.
        # An item the annotator did not mention yields NOTHING: silence is not a verdict, and
        # reading it as `new_task` would move the miss rate on the strength of a skipped row.
        verdicts = response.get("verdicts") or {}
        out = []
        for entry in (item.get("payload") or {}).get("items") or ():
            k = str(entry.get("item_key"))
            if k in verdicts:
                out.append((f"{iid}/{k}", "B2", str(verdicts[k]), dict(prov, item_key=k), None))
        return out

    if tt == "B4":
        verdict = str(response.get("verdict"))
        p = dict(prov)
        # The QUESTION rides in provenance, never in the label. `response` is what consensus
        # and alpha compare for equality, and free text inside it would sit inside that
        # comparison -- the same rule B3's supplied question follows.
        if response.get("question"):
            p["question"] = str(response["question"])
        return [(iid, "B4", verdict, p, None)]

    if tt == "B3":
        verdict = str(response.get("verdict"))
        p = dict(prov)
        # THE SUPERVISED TARGET rides in the provenance, not the label. "should_have_asked" is
        # the verdict two annotators must agree on for a consensus; the question text is what
        # the training row is built FROM, and free text can never be consensus'd (see the
        # module docstring on A3_missing).
        if response.get("question"):
            p["question"] = str(response["question"])
        return [(iid, "B3", verdict, p, None)]

    return []  # A3_missing is free text and has no unit


# --------------------------------------------------------------------------- validation


def _response_errors(item: Mapping[str, Any], response: Mapping[str, Any], where: str) -> list[str]:
    tt = item["task_type"]
    errs: list[str] = []
    if tt == "A1":
        ids = {str(n["node_id"]) for n in _nodes_of(item)}
        for nid in response.get("ticked") or ():
            if str(nid) not in ids:
                errs.append(f"{where}: ticked node {nid} is not on this item")
        for nid, rating in (response.get("usefulness") or {}).items():
            if str(nid) not in ids:
                errs.append(f"{where}: usefulness rating for node {nid} is not on this item")
            elif not (_USEFULNESS_RANGE[0] <= float(rating) <= _USEFULNESS_RANGE[1]):
                errs.append(f"{where}: usefulness {rating} outside 1..5")
    elif tt == "A2":
        if response.get("choice") not in _A2_CHOICES:
            errs.append(f"{where}: choice {response.get('choice')!r} not in {_A2_CHOICES}")
    elif tt == "A3_node":
        if str(response.get("verdict")) not in _LABELS["A3_node"]:
            errs.append(f"{where}: verdict {response.get('verdict')!r} not in {_LABELS['A3_node']}")
        disc = response.get("discoverability")
        if disc and str(disc) not in _LABELS["A3_node_disc"]:
            errs.append(f"{where}: discoverability {disc!r} not in {_LABELS['A3_node_disc']}")
    elif tt == "A3_edge":
        if response.get("holds") not in (True, False, "unsure"):
            errs.append(f"{where}: holds {response.get('holds')!r} not in (True, False, 'unsure')")
    elif tt == "A3_match":
        if not isinstance(response.get("addresses"), bool):
            errs.append(f"{where}: addresses must be a bool")
    elif tt == "A3_missing":
        needs = response.get("missing_needs")
        if needs is not None and not isinstance(needs, (list, tuple)):
            errs.append(f"{where}: missing_needs must be a list")
    elif tt in ("B1", "B1b"):
        verdict = str(response.get("verdict"))
        if verdict not in _LABELS[tt]:
            errs.append(f"{where}: verdict {verdict!r} not in {_LABELS[tt]}")
        # `default_existed` is the claim "you did not need to ask, the obvious choice was X".
        # With no X it is unfalsifiable AND it lands in the wasted-ask numerator, so it is
        # refused rather than counted -- the same requirement A7 puts on reaches_unstated.
        if verdict == "default_existed" and not str(response.get("default_action") or "").strip():
            errs.append(f"{where}: default_existed must name the default it says existed")
    elif tt == "B2":
        keys = {str(e.get("item_key")) for e in (item.get("payload") or {}).get("items") or ()}
        verdicts = response.get("verdicts")
        if verdicts is not None and not isinstance(verdicts, Mapping):
            errs.append(f"{where}: verdicts must be an object")
            verdicts = {}
        for k, v in (verdicts or {}).items():
            if str(k) not in keys:
                errs.append(f"{where}: verdict for item {k!r} which is not on this item")
            if str(v) not in _LABELS["B2"]:
                errs.append(f"{where}: label {v!r} not in {_LABELS['B2']}")
    elif tt == "B4":
        verdict = str(response.get("verdict"))
        q = str(response.get("question") or "").strip()
        if verdict not in _LABELS["B4"]:
            errs.append(f"{where}: verdict {verdict!r} not in {_LABELS['B4']}")
        if verdict == "question_needed" and not q:
            errs.append(f"{where}: question_needed must carry the question it proposes")
        if verdict != "question_needed" and q:
            errs.append(f"{where}: a question was proposed but the verdict is {verdict!r}")
    elif tt == "B3":
        verdict = str(response.get("verdict"))
        if verdict not in _LABELS["B3"]:
            errs.append(f"{where}: verdict {verdict!r} not in {_LABELS['B3']}")
        # This verdict IS the supervised target: it becomes an ASK row whose question is the
        # annotator's own text. Without the text there is no training example, only a complaint.
        if verdict == "should_have_asked" and not str(response.get("question") or "").strip():
            errs.append(
                f"{where}: should_have_asked must name the question that should have been asked"
            )
    elif tt == "A4":
        if str(response.get("latency")) not in _LABELS["A4"]:
            errs.append(f"{where}: latency {response.get('latency')!r} not in {_LABELS['A4']}")
    elif tt == "A5":
        verdict = response.get("verdict")
        if str(verdict) not in _LABELS["A5"]:
            errs.append(f"{where}: verdict {verdict!r} not in {_LABELS['A5']}")
        node_ids = {str(c.get("node_id")) for c in _candidates_of(item)}
        missing = response.get("missing")
        if missing is None:
            missing = []
        if not isinstance(missing, (list, tuple)):
            errs.append(f"{where}: missing must be a list")
        else:
            missing_set = {str(x) for x in missing}
            if not missing_set <= node_ids:
                errs.append(
                    f"{where}: missing {sorted(missing_set - node_ids)} are not candidates on "
                    "this item"
                )
            # REQUIRED iff should_have_asked_more, else it must be empty -- `missing` is what
            # makes an A5 verdict training data rather than a satisfaction score, so a verdict
            # that claims something was missed without naming it is not a smaller version of a
            # valid answer, it is an unusable one.
            if verdict == "should_have_asked_more" and not missing_set:
                errs.append(
                    f"{where}: verdict should_have_asked_more requires a non-empty missing list"
                )
            if verdict != "should_have_asked_more" and missing_set:
                errs.append(
                    f"{where}: missing must be empty unless verdict is should_have_asked_more"
                )
    elif tt == "A6":
        ids = _a6_candidate_ids(item)
        tiers = response.get("tiers")
        if not isinstance(tiers, Mapping):
            errs.append(
                f"{where}: tiers must be an object mapping every candidate_id to a positive "
                "integer (lower is better)"
            )
        else:
            # EVERY CANDIDATE EXACTLY ONCE. A candidate with no tier has no pairwise reading
            # against any other, so `a6_pair_labels` emits nothing at all for the item -- a
            # partial ranking is silently no ranking, which is why it is refused loudly here.
            for cid in ids:
                if cid not in tiers:
                    errs.append(f"{where}: no tier for candidate {cid}")
            for cid, t in tiers.items():
                if str(cid) not in ids:
                    errs.append(f"{where}: tier for {cid} is not a candidate on this item")
                elif isinstance(t, bool) or not isinstance(t, int) or t < 1:
                    errs.append(
                        f"{where}: tier {t!r} for {cid} must be a positive integer. Lower is "
                        "better, gaps are allowed, and EQUAL TIERS ARE A GENUINE TIE"
                    )
    elif tt == "A7":
        for side in ("a", "b"):
            reaches = response.get(f"{side}_reaches")
            if str(reaches) not in _LABELS["A7"]:
                errs.append(f"{where}: {side}_reaches {reaches!r} not in {_LABELS['A7']}")
            need = response.get(f"unstated_need_{side}")
            if need is not None and not isinstance(need, str):
                errs.append(f"{where}: unstated_need_{side} must be a string")
                continue
            # REQUIRED iff reaches_unstated, else it must be absent -- the named need is what
            # makes the label auditable rather than a vibe, so a side judged reaches_unstated
            # without naming what it reaches for is not a smaller version of a valid answer,
            # it is an unusable one; a need named for a stays_stated side is an invented one.
            if reaches == "reaches_unstated" and not (need or "").strip():
                errs.append(
                    f"{where}: {side}_reaches reaches_unstated requires a non-empty "
                    f"unstated_need_{side}"
                )
            if reaches != "reaches_unstated" and (need or "").strip():
                errs.append(
                    f"{where}: unstated_need_{side} must be absent unless {side}_reaches is "
                    "reaches_unstated"
                )
        if response.get("preference") not in _A2_CHOICES:
            errs.append(f"{where}: preference {response.get('preference')!r} not in {_A2_CHOICES}")
    return errs


_CONTEXT_REQUIRED: dict[str, tuple[str, ...]] = {
    "A1": ("question",),
    "A2": ("question", "evidence", "history", "draft", "state_text"),
    "A3_node": ("question", "node_text", "evidence"),
    "A3_edge": ("question", "src_text", "dst_text"),
    "A3_match": ("asked_question", "node_text"),
    "A3_missing": ("question", "node_list"),
    "A4": ("question", "asked_question"),
    "A5": ("question", "evidence", "history", "answer"),
    # Identical to A2's, and that is the requirement, not a coincidence: both instruments judge
    # continuations of ONE state, so both must show the same state, built by the same replay.
    "A6": ("question", "evidence", "history", "draft", "state_text"),
    "A7": ("question", "evidence", "history", "draft", "state_text"),
    # THE B FAMILY SHOWS A CONVERSATION STATE, not a rollout: the task as the person stated it,
    # what has been said since, and what the agent has already done. Every B judgment is
    # "given this state", so an item that omits part of the state is not a harder item -- it is
    # a different question.
    "B1": ("task", "prior_turns", "tool_trace", "asked_question"),
    "B1b": ("task", "prior_turns", "tool_trace", "asked_question"),
    # B2 alone is shown the next turn, because its unit IS an item of that turn. B1 and B3 are
    # not, and the unexpected-key check below is what enforces it: B1 asks whether the question
    # was necessary BEFORE the person replied, so showing the reply is showing the answer.
    "B2": ("task", "prior_turns", "tool_trace", "next_turn"),
    "B3": ("task", "prior_turns", "tool_trace", "final_assistant_text"),
    # Identical to B3's, and that is the requirement rather than a coincidence: both judge the
    # same moment, so both must be shown the same state. Neither sees the reply -- for B4 that
    # would turn proposing into copying, and every proposal would be trivially necessary.
    "B4": ("task", "prior_turns", "tool_trace", "final_assistant_text"),
}
_CONTEXT_OPTIONAL: dict[str, tuple[str, ...]] = {"A4": ("parent_units",)}
_CONTEXT_NONEMPTY_STR: dict[str, tuple[str, ...]] = {
    "A1": ("question",),
    "A2": ("question", "state_text"),
    "A3_node": ("question", "node_text"),
    "A3_edge": ("question", "src_text", "dst_text"),
    "A3_match": ("asked_question", "node_text"),
    "A3_missing": ("question",),
    "A4": ("question", "asked_question"),
    "A5": ("question", "answer"),
    "A6": ("question", "state_text"),
    "A7": ("question", "state_text"),
    "B1": ("task", "asked_question"),
    "B1b": ("task", "asked_question"),
    "B2": ("task", "next_turn"),
    "B3": ("task", "final_assistant_text"),
    "B4": ("task", "final_assistant_text"),
}
# Keys a payload MAY carry beyond its required set. `blind` marks an A5 item built by
# `--a5-blind`, which deliberately ships NO `candidates`: that list is the unresolved gold
# frontier, and showing it is what decides the verdict (see `pi_eval.annotate_llm._build_a5`).
_PAYLOAD_OPTIONAL: dict[str, tuple[str, ...]] = {
    "A5": ("blind",),
}

_PAYLOAD_REQUIRED: dict[str, tuple[str, ...]] = {
    "A1": ("nodes",),
    "A2": ("option_a", "option_b"),
    "A5": ("candidates",),
    "A6": ("candidates",),
    "A7": ("option_a", "option_b"),
    "B2": ("items",),
}


def _uid_entry_errors(where: str, entry: Any, keys: tuple[str, ...]) -> list[str]:
    if not isinstance(entry, Mapping):
        return [f"{where}: entry is not an object"]
    return [f"{where}: {k!r} must be a string" for k in keys if not isinstance(entry.get(k), str)]


# `#N` is MuSiQue's mechanical decomposition placeholder (see
# `pi_eval.build.musique_build._graph` / `resolve_placeholders`), and by construction it never
# belongs in text a human is shown: the exporter must resolve it before an item ships, or refuse
# the item. This is the STRUCTURAL half of that fix -- the resolver can be bypassed by a future
# sampler bug the same way the canary firewall's registry can be bypassed by a new leak site, so
# `export` is refused on this exactly like it is refused on a canary hit. It fires on every
# suite, not only musique: real text can coincidentally contain "#1" (a chart position, a rank),
# and refusing that one item is the safe direction -- shipping an unresolved decomposition
# placeholder is the failure this exists to close.
_PLACEHOLDER_RE = re.compile(r"#\d+")


def _iter_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)


# WHICH STRINGS THE `#N` GATE POLICES, AND WHY NOT ALL OF THEM.
#
# `#N` is a MuSiQue decomposition reference wherever it appears in GOLD node text, and an
# unresolved one there is the defect this gate exists to close. Everywhere else it is just a
# number: corpus prose, a task question and a model's own answer may all legitimately contain
# "#1". Scanning them cost a real ten-minute export, which died on an evidence paragraph
# describing a radio station as "Connecticut's #1 Rock Station".
#
# Refusing ONE item over a false positive would have been the safe direction; refusing the
# whole export was not, and that text was never gold to begin with. So the gate follows
# provenance rather than shape: only fields carrying gold node text are scanned.
_GOLD_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "A3_node": ("node_text",),
    "A3_edge": ("src_text", "dst_text"),
    "A3_match": ("node_text",),
    "A3_missing": ("node_list",),
}


def _gold_derived_strings(
    task_type: str, context: Mapping[str, Any], payload: Mapping[str, Any]
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for field in _GOLD_TEXT_FIELDS.get(task_type, ()):
        v = context.get(field)
        if isinstance(v, str):
            out.append((f"context[{field!r}]", v))
        elif isinstance(v, (list, tuple)):
            out.extend((f"context[{field!r}]", str(x)) for x in v if isinstance(x, str))
    # A1's checklist and A5's candidate list are gold node text living in `payload`, because
    # there the text IS the choice being offered.
    for field in ("nodes", "candidates"):
        for entry in payload.get(field) or ():
            if isinstance(entry, Mapping) and isinstance(entry.get("text"), str):
                out.append((f"payload[{field!r}]", entry["text"]))
    return out


def edge_is_self_referential(context: Mapping[str, Any]) -> bool:
    """Does this edge item's DST contain its own SRC verbatim?

    MuSiQue needs are `#N`-referential, so DST has to name its prerequisite somehow to be
    readable at all. Rendering it as a bracketed reference puts SRC's exact text inside DST,
    and the item degenerates into "does <X> >> y require X?" -- yes, by construction, with no
    information about the graph in the answer. Such an item cannot contribute to
    edge_precision; see tests/test_annotate_edge_selfref.py for the pair of opposite
    degeneracies the two obvious renderings produce.
    """
    src = str(context.get("src_text") or "").strip()
    dst = str(context.get("dst_text") or "").strip()
    return bool(src) and src in dst


def bundle_shape_errors(bundle: Mapping[str, Any]) -> list[str]:
    """Every way one exported item's context/payload violates the annotator-tool contract:
    `context` is what a human reads, `payload` is ids and choices, and nothing before this
    checked an export against that split. An A2 item shipping only a raw rendered prompt, or
    an A1 item carrying evidence, builds and imports without error and either fails to render
    in the tool or silently blinds a judgment that was supposed to be blind. Returns a LIST,
    like `validate_records`, so `export` can refuse on every problem in one pass rather than
    the first.
    """
    errs: list[str] = []
    seen_ids: dict[str, int] = {}
    for item in bundle.get("items") or ():
        iid = str(item.get("item_id") or "")
        seen_ids[iid] = seen_ids.get(iid, 0) + 1
    for iid, n in sorted(seen_ids.items()):
        if n > 1:
            # `item_id` hashes (task_type, provenance, payload), so a collision means the two
            # items are the SAME item -- one question asked twice, two annotation slots spent,
            # and two records that will claim one `record_id`. `item_set_hash` cannot catch it:
            # it is computed over a SET of ids and dedupes silently.
            errs.append(f"item_id {iid} appears {n} times: the same item was sampled twice")

    for item in bundle.get("items") or ():
        iid = str(item.get("item_id") or "<no id>")
        tt = str(item.get("task_type") or "")
        where = f"{iid}/{tt}"
        if tt not in TASK_TYPES:
            errs.append(f"{where}: unknown task_type {tt!r}")
            continue

        if tt == "A3_edge" and isinstance(item.get("context"), Mapping):
            if edge_is_self_referential(item["context"]):
                errs.append(
                    f"{where}: self-referential edge -- dst_text contains src_text verbatim, "
                    "so 'does dst depend on src' is yes by construction and the item cannot "
                    "contribute to edge_precision"
                )

        context = item.get("context")
        if not isinstance(context, Mapping):
            errs.append(f"{where}: context is not an object")
            context = {}
        payload = item.get("payload")
        if not isinstance(payload, Mapping):
            errs.append(f"{where}: payload is not an object")
            payload = {}

        required = _CONTEXT_REQUIRED.get(tt, ())
        allowed = set(required) | set(_CONTEXT_OPTIONAL.get(tt, ()))
        for k in required:
            # A blind A5 item withholds the final answer as well as the frontier, so `answer`
            # is required only for the sighted variant.
            if tt == "A5" and k == "answer" and bool((item.get("payload") or {}).get("blind")):
                continue
            if k not in context:
                errs.append(f"{where}: context missing required key {k!r}")
        for k in context:
            if k not in allowed:
                errs.append(f"{where}: context carries unexpected key {k!r}")
        for k in _CONTEXT_NONEMPTY_STR.get(tt, ()):
            v = context.get(k)
            if k in context and (not isinstance(v, str) or not v.strip()):
                errs.append(f"{where}: context[{k!r}] must be a non-empty string")

        preq = _PAYLOAD_REQUIRED.get(tt, ())
        popt = _PAYLOAD_OPTIONAL.get(tt, ())
        blind_item = tt == "A5" and bool(payload.get("blind"))
        for k in preq:
            # A BLIND A5 ITEM HAS NO `candidates` BY DESIGN, and requiring them would refuse
            # the whole instrument -- the first blind bundle died with three errors per item.
            if blind_item and k == "candidates":
                continue
            if k not in payload:
                errs.append(f"{where}: payload missing required key {k!r}")
        for k in payload:
            if k not in preq and k not in popt:
                errs.append(f"{where}: payload carries unexpected key {k!r}")
        if blind_item and "candidates" in payload:
            errs.append(
                f"{where}: payload is marked blind and still carries 'candidates'. The claim "
                "of the blind variant is that the item does not hold the gold frontier; a "
                "bundle whose name says blind and whose bytes say otherwise would silently "
                "make the sighted measurement."
            )

        if tt == "A1":
            nodes = payload.get("nodes")
            if not isinstance(nodes, (list, tuple)):
                errs.append(f"{where}: payload['nodes'] must be a list")
            else:
                for i, n in enumerate(nodes):
                    errs.extend(_uid_entry_errors(f"{where}/nodes[{i}]", n, ("node_id", "text")))

        if tt == "A2":
            for i, e in enumerate(context.get("evidence") or ()):
                errs.extend(
                    _uid_entry_errors(f"{where}/evidence[{i}]", e, ("uid", "title", "text"))
                )
            for i, h in enumerate(context.get("history") or ()):
                errs.extend(_uid_entry_errors(f"{where}/history[{i}]", h, ("q", "a")))
            if not isinstance(context.get("draft"), str):
                errs.append(f"{where}: context['draft'] must be a string")
            for slot in ("option_a", "option_b"):
                opt = payload.get(slot)
                if not isinstance(opt, Mapping) or not str(opt.get("question") or "").strip():
                    errs.append(f"{where}: payload[{slot!r}] must carry a non-empty 'question'")

        if tt == "A3_node":
            for i, e in enumerate(context.get("evidence") or ()):
                errs.extend(
                    _uid_entry_errors(f"{where}/evidence[{i}]", e, ("uid", "title", "text"))
                )

        if tt == "A3_missing":
            nl = context.get("node_list")
            if not isinstance(nl, (list, tuple)) or not all(isinstance(x, str) for x in nl):
                errs.append(f"{where}: context['node_list'] must be a list of strings")

        if tt == "A4" and "parent_units" in context:
            for i, u in enumerate(context.get("parent_units") or ()):
                errs.extend(
                    _uid_entry_errors(f"{where}/parent_units[{i}]", u, ("uid", "title", "text"))
                )

        if tt == "A5":
            for i, e in enumerate(context.get("evidence") or ()):
                errs.extend(
                    _uid_entry_errors(f"{where}/evidence[{i}]", e, ("uid", "title", "text"))
                )
            for i, h in enumerate(context.get("history") or ()):
                errs.extend(_uid_entry_errors(f"{where}/history[{i}]", h, ("q", "a")))
            if not blind_item and not isinstance(context.get("answer"), str):
                errs.append(f"{where}: context['answer'] must be a string")
            if blind_item and "answer" in context:
                errs.append(f"{where}: payload is marked blind and context still carries 'answer'")
            candidates = payload.get("candidates")
            if isinstance(candidates, (list, tuple)):
                for i, c in enumerate(candidates):
                    errs.extend(
                        _uid_entry_errors(f"{where}/candidates[{i}]", c, ("node_id", "text"))
                    )
            elif not blind_item:
                # A blind item has none by design; anything else must carry a list. The old
                # `else` iterated `candidates` unconditionally and raised TypeError on None.
                errs.append(f"{where}: payload['candidates'] must be a list")

        if tt == "A6":
            for i, e in enumerate(context.get("evidence") or ()):
                errs.extend(
                    _uid_entry_errors(f"{where}/evidence[{i}]", e, ("uid", "title", "text"))
                )
            for i, h in enumerate(context.get("history") or ()):
                errs.extend(_uid_entry_errors(f"{where}/history[{i}]", h, ("q", "a")))
            if not isinstance(context.get("draft"), str):
                errs.append(f"{where}: context['draft'] must be a string")
            candidates = payload.get("candidates")
            if not isinstance(candidates, (list, tuple)):
                errs.append(f"{where}: payload['candidates'] must be a list")
            elif len(candidates) < A6_MIN_CANDIDATES:
                errs.append(
                    f"{where}: payload['candidates'] has {len(candidates)}; an A6 item needs at "
                    f"least {A6_MIN_CANDIDATES}. Below that it is an A2 item, and shipping one "
                    "state as both instruments asks the same comparison twice"
                )
            else:
                seen_cids: set[str] = set()
                for i, c in enumerate(candidates):
                    errs.extend(_uid_entry_errors(f"{where}/candidates[{i}]", c, ("candidate_id",)))
                    if not isinstance(c, Mapping):
                        continue
                    if not str(c.get("question") or "").strip():
                        errs.append(
                            f"{where}/candidates[{i}]: 'question' must be a non-empty string"
                        )
                    cid = str(c.get("candidate_id"))
                    if cid in seen_cids:
                        # Two slots under one id collapse into one unit and silently drop a
                        # comparison the annotator actually made.
                        errs.append(f"{where}: candidate_id {cid!r} appears twice")
                    seen_cids.add(cid)

        if tt == "A7":
            for i, e in enumerate(context.get("evidence") or ()):
                errs.extend(
                    _uid_entry_errors(f"{where}/evidence[{i}]", e, ("uid", "title", "text"))
                )
            for i, h in enumerate(context.get("history") or ()):
                errs.extend(_uid_entry_errors(f"{where}/history[{i}]", h, ("q", "a")))
            if not isinstance(context.get("draft"), str):
                errs.append(f"{where}: context['draft'] must be a string")
            for slot in ("option_a", "option_b"):
                opt = payload.get(slot)
                if not isinstance(opt, Mapping) or not str(opt.get("question") or "").strip():
                    errs.append(f"{where}: payload[{slot!r}] must carry a non-empty 'question'")

        for label, s in _gold_derived_strings(tt, context, payload):
            if _PLACEHOLDER_RE.search(s):
                errs.append(f"{where}: {label} carries an unresolved '#N' placeholder: {s!r}")
    return errs


def validate_records(bundle: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every reason these records may not be imported. Empty means importable.

    Returns a LIST rather than raising on the first problem: an annotator whose client wrote
    forty malformed rows should learn all forty at once.
    """
    errs: list[str] = []
    manifest = bundle.get("manifest") or {}
    items = {str(i["item_id"]): i for i in bundle.get("items") or ()}

    declared = str(manifest.get("item_set_hash") or "")
    actual = item_set_hash(list(items.values()))
    if declared and declared != actual:
        errs.append(
            f"bundle manifest item_set_hash {declared} does not match its own items ({actual}): "
            "these records were collected against a different build of this bundle"
        )

    bundle_id = str(manifest.get("bundle_id") or "")
    seen: set[tuple[str, str]] = set()
    for r in records:
        iid = str(r.get("item_id") or "")
        ann = str(r.get("annotator_id") or "")
        where = f"{iid}/{ann or '<no annotator>'}"
        if not ann:
            errs.append(f"{where}: record carries no annotator_id")
        kind = rater_kind(r)
        if kind not in RATER_KINDS:
            errs.append(f"{where}: annotator_kind {kind!r} not in {RATER_KINDS}")
        elif ann.startswith(LLM_ID_PREFIX) != (kind == "llm"):
            errs.append(
                f"{where}: annotator_kind {kind!r} disagrees with the id. A model annotator's "
                f"id starts with {LLM_ID_PREFIX!r} and its annotator_kind is 'llm'; both, or "
                "neither"
            )
        if kind == "llm" and not str(r.get("model_pin") or "").strip():
            errs.append(
                f"{where}: an llm record needs a model_pin. Two models under one annotator_id "
                "make one agreement number out of two different raters"
            )
        rationale = r.get("rationale")
        if rationale is not None and not isinstance(rationale, str):
            errs.append(f"{where}: rationale must be a string")
        elif isinstance(rationale, str) and len(rationale) > RATIONALE_MAX_CHARS:
            errs.append(
                f"{where}: rationale is {len(rationale)} chars, over the "
                f"{RATIONALE_MAX_CHARS}-char cap"
            )
        if kind == "llm" and not (isinstance(rationale, str) and rationale.strip()):
            errs.append(
                f"{where}: an llm record needs a non-empty rationale. A model verdict with no "
                "stated reason is the one hardest to audit later, and it is never defaulted -- "
                "see AnnotationParseError in pi_eval.annotate_llm"
            )
        basis = r.get("basis")
        # Each preference instrument's basis is validated against ITS vocabulary -- an A2
        # basis on an A7 record would answer in a vocabulary with no anticipation option,
        # which is the hole A7 exists to close.
        basis_vocab = A7_BASIS_VALUES if str(r.get("task_type")) == "A7" else A2_BASIS_VALUES
        if basis is not None and basis not in basis_vocab:
            errs.append(f"{where}: basis {basis!r} not in {basis_vocab}")
        if bundle_id and str(r.get("bundle_id") or "") != bundle_id:
            errs.append(f"{where}: record bundle_id {r.get('bundle_id')!r} is not {bundle_id!r}")
        item = items.get(iid)
        if item is None:
            errs.append(f"{where}: unknown item {iid!r} is not in this bundle")
            continue
        if str(r.get("task_type") or "") != item["task_type"]:
            errs.append(
                f"{where}: record task_type {r.get('task_type')!r} is not the item's "
                f"{item['task_type']!r}"
            )
            continue
        if (iid, ann) in seen:
            errs.append(f"{where}: duplicate record for this (item, annotator)")
        seen.add((iid, ann))
        elapsed = r.get("elapsed_ms")
        if elapsed is not None and (not isinstance(elapsed, int) or elapsed < 0):
            errs.append(f"{where}: elapsed_ms {elapsed!r} is not a non-negative integer")
        errs.extend(_response_errors(item, r.get("response") or {}, where))
    return errs


# --------------------------------------------------------------------------- consensus


@dataclass(frozen=True, slots=True)
class ResolvedUnit:
    unit_id: str
    kind: str
    label: str
    n_raters: int
    provenance: Mapping[str, Any]
    usefulness: float | None = None


@dataclass(frozen=True, slots=True)
class Consensus:
    """What the annotators agreed on, what they did not, and the raw votes behind both.

    `votes` is kept because agreement is not derivable from a consensus: resolution discards
    exactly the disagreements an alpha is computed from, so a kappa built off `units` would
    report 1.0 by construction.
    """

    units: tuple[ResolvedUnit, ...] = ()
    disagreements: tuple[Mapping[str, Any], ...] = ()
    missing: tuple[Mapping[str, Any], ...] = ()
    # Which rater classes were counted. Carried rather than assumed so the write-back path
    # can refuse a consensus that a model helped form without having to re-derive it.
    rater_kinds: tuple[str, ...] = HUMAN_ONLY
    votes: Mapping[str, Mapping[str, str]] = None  # type: ignore[assignment]
    kinds: Mapping[str, str] = None  # type: ignore[assignment]
    provenance: Mapping[str, Mapping[str, Any]] = None  # type: ignore[assignment]

    def by_kind(self, kind: str) -> tuple[ResolvedUnit, ...]:
        return tuple(u for u in self.units if u.kind == kind)


def _collect(
    bundle: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    key: Mapping[str, Any] | None,
    rater_kinds: Sequence[str] = HUMAN_ONLY,
):
    items = {str(i["item_id"]): i for i in bundle.get("items") or ()}
    key_items = (key or {}).get("items") or {}
    votes: dict[str, dict[str, str]] = defaultdict(dict)
    ratings: dict[str, dict[str, float]] = defaultdict(dict)
    kinds: dict[str, str] = {}
    provs: dict[str, dict] = {}
    missing: list[dict] = []

    for r in sorted(records, key=lambda r: (str(r.get("item_id")), str(r.get("annotator_id")))):
        item = items.get(str(r.get("item_id") or ""))
        if item is None:
            continue
        if rater_kind(r) not in rater_kinds:
            continue
        # A planted attention check (see `pi_run.cmd_annotate.sample_attention_items`) must
        # never enter a measurement -- that is the HARD CONSTRAINT that lets one exist at all
        # without contaminating a published number. Its key entry is the only place that says
        # so (the item itself carries no tell), so a response to one is dropped here, before it
        # can become a vote, a missing-needs text, or anything else `consensus`/`iaa_report`/
        # `gate_numbers` read off `_collect`'s output.
        if (key_items.get(str(item["item_id"])) or {}).get("attention_check"):
            continue
        ann = str(r.get("annotator_id") or "")
        resp = r.get("response") or {}
        if item["task_type"] == "A3_missing":
            for text in resp.get("missing_needs") or ():
                missing.append(
                    {
                        "item_id": str(item["item_id"]),
                        "annotator_id": ann,
                        "text": str(text),
                        "provenance": dict(item.get("provenance") or {}),
                    }
                )
            continue
        for unit, kind, label, prov, rating in _units(
            item, resp, key_items.get(str(item["item_id"]))
        ):
            votes[unit][ann] = label
            kinds[unit] = kind
            provs[unit] = prov
            if rating is not None:
                ratings[unit][ann] = rating
    return votes, ratings, kinds, provs, missing


def consensus(
    bundle: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    key: Mapping[str, Any] | None = None,
    min_annotators: int = 2,
    allow_majority: bool = False,
    rater_kinds: Sequence[str] = HUMAN_ONLY,
) -> Consensus:
    """Unanimity resolves; anything else queues for adjudication.

    `allow_majority` needs three raters on purpose: a "majority" of two is one person
    outvoting nobody.

    HUMAN RATERS ONLY, by default and in every caller that writes gold. A model's records are
    dropped here rather than filtered downstream, so a model can neither form a consensus of
    its own nor complete a lone human into one. Widening `rater_kinds` is possible for
    analysis, and `merge_into_graphs` then refuses the result.
    """
    votes, ratings, kinds, provs, missing = _collect(bundle, records, key, rater_kinds)

    units: list[ResolvedUnit] = []
    disagreements: list[dict] = []
    for unit in sorted(votes):
        per_ann = votes[unit]
        labels = set(per_ann.values())
        rated = ratings.get(unit) or {}
        mean_rating = sum(rated.values()) / len(rated) if rated else None
        if len(per_ann) < min_annotators:
            disagreements.append(
                {
                    "unit_id": unit,
                    "kind": kinds[unit],
                    "reason": "insufficient_raters",
                    "votes": dict(per_ann),
                    "provenance": provs[unit],
                }
            )
            continue
        label: str | None = None
        if len(labels) == 1:
            label = next(iter(labels))
        elif allow_majority and len(per_ann) >= 3:
            counts = defaultdict(int)
            for v in per_ann.values():
                counts[v] += 1
            top = max(counts.values())
            winners = [k for k, c in counts.items() if c == top]
            if len(winners) == 1 and top * 2 > len(per_ann):
                label = winners[0]
        if label is None:
            disagreements.append(
                {
                    "unit_id": unit,
                    "kind": kinds[unit],
                    "reason": "no_consensus",
                    "votes": dict(per_ann),
                    "provenance": provs[unit],
                }
            )
            continue
        units.append(ResolvedUnit(unit, kinds[unit], label, len(per_ann), provs[unit], mean_rating))

    return Consensus(
        rater_kinds=tuple(rater_kinds),
        units=tuple(units),
        disagreements=tuple(disagreements),
        missing=tuple(missing),
        votes={k: dict(v) for k, v in votes.items()},
        kinds=dict(kinds),
        provenance=provs,
    )


# --------------------------------------------------------------------------- agreement


def iaa_report(
    bundle: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    key: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Krippendorff's nominal alpha per unit kind, plus the ordinal usefulness scale apart.

    A usefulness rating is ORDINAL and is deliberately not pushed through a nominal alpha,
    which would score 4-vs-5 and 1-vs-5 as the same disagreement. Exact agreement and mean
    absolute delta say what a reader of a 1..5 scale actually wants to know.
    """
    # Humans only: the reported figure is agreement BETWEEN ANNOTATORS, and a model dropped
    # into the pool would move a number the paper attributes to people.
    votes, ratings, kinds, _provs, missing = _collect(bundle, records, key, HUMAN_ONLY)

    by_kind: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for unit, per_ann in votes.items():
        by_kind[kinds[unit]][unit] = per_ann

    out: dict[str, dict[str, float]] = {}
    for kind in sorted(by_kind):
        per_unit = by_kind[kind]
        multi = {u: v for u, v in per_unit.items() if len(v) > 1}
        out[kind] = {
            "alpha": krippendorff_alpha_nominal(multi) if multi else float("nan"),
            "n_units": len(per_unit),
            "n_multi_rated": len(multi),
            "n_raters": len({a for v in per_unit.values() for a in v}),
        }

    multi_rated = {u: v for u, v in ratings.items() if len(v) > 1}
    if multi_rated:
        deltas = [max(v.values()) - min(v.values()) for v in multi_rated.values()]
        out["usefulness"] = {
            "exact_agreement": sum(1 for d in deltas if d == 0) / len(deltas),
            "mean_abs_delta": sum(deltas) / len(deltas),
            "n": len(deltas),
        }
    out["missing_needs"] = {"n_texts": len(missing)}
    return out


# --------------------------------------------------------------------------- model raters


def llm_agreement(
    bundle: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    key: Mapping[str, Any] | None = None,
    min_annotators: int = 2,
) -> dict[str, Any]:
    """How well the model tracked the people, scored ONLY where the people resolved a unit.

    This is what an automatic pass is for and the only number it may produce: a model answer
    on a unit no two humans reached is not evidence about the model, it is an answer with
    nothing to check it against. So the human consensus is the reference and units without one
    are excluded rather than counted as agreement (they would inflate) or as disagreement
    (they would deflate).

    Reported, never gated. G-M2 asks whether the MATCHER agrees with a human; substituting a
    model there would measure two machines agreeing about a third.
    """
    human = consensus(bundle, records, key=key, min_annotators=min_annotators)
    reference = {u.unit_id: u.label for u in human.units}
    kinds = dict(human.kinds or {})

    items = {str(i["item_id"]): i for i in bundle.get("items") or ()}
    key_items = (key or {}).get("items") or {}

    by_kind: dict[str, list[bool]] = defaultdict(list)
    by_model: dict[str, list[bool]] = defaultdict(list)
    disagreements: list[dict] = []

    for r in sorted(records, key=lambda r: (str(r.get("item_id")), str(r.get("annotator_id")))):
        if rater_kind(r) != "llm":
            continue
        item = items.get(str(r.get("item_id") or ""))
        if item is None:
            continue
        who = str(r.get("annotator_id") or "")
        for unit, kind, label, _prov, _rating in _units(
            item, r.get("response") or {}, key_items.get(str(item["item_id"]))
        ):
            if unit not in reference:
                continue
            hit = label == reference[unit]
            by_kind[kinds.get(unit, kind)].append(hit)
            by_model[who].append(hit)
            if not hit:
                disagreements.append(
                    {
                        "unit_id": unit,
                        "kind": kinds.get(unit, kind),
                        "model": who,
                        "model_label": label,
                        "human_label": reference[unit],
                        "model_pin": str(r.get("model_pin") or ""),
                    }
                )

    def _score(hits: Sequence[bool]) -> dict[str, float]:
        return {
            "n": len(hits),
            "agreement": (sum(hits) / len(hits)) if hits else float("nan"),
        }

    everything = [h for v in by_model.values() for h in v]
    return {
        "by_kind": {k: _score(v) for k, v in sorted(by_kind.items())},
        "by_model": {k: _score(v) for k, v in sorted(by_model.items())},
        "overall": _score(everything),
        "disagreements": disagreements,
        "n_human_resolved_units": len(reference),
    }


# --------------------------------------------------------------------------- write-back


def merge_into_graphs(
    graphs: Mapping[str, GoldGraph], cons: Consensus, *, out_version: str
) -> list[dict]:
    """Consensus -> serialised gold rows, under a NEW graph version.

    Never in place. `score.graph_hash` fingerprints exactly the gold that was read and feeds
    `scorer_hash`, so editing v1's bytes would make one version string denote two different
    graphs and orphan the provenance of every row already scored against it. A new version
    leaves those rows meaning what they meant and gives the re-score an honest identity.
    """
    if tuple(cons.rater_kinds) != HUMAN_ONLY:
        raise LLMAnnotationNotGold(
            f"this consensus counted {sorted(set(cons.rater_kinds) - set(HUMAN_ONLY))} raters. "
            "`gold_human_asked` is what a PERSON would have thought to ask; a model's "
            "prediction of it is a different quantity with the same shape, and adr computed "
            "over it would be a published number from a measurement nobody took"
        )

    asked: dict[tuple[str, str], bool] = {}
    useful: dict[tuple[str, str], float] = {}
    seen_nodes: set[tuple[str, str]] = set()
    seen_edges: set[tuple[str, str, str]] = set()

    for u in cons.units:
        p = u.provenance
        tk = str(p.get("task_key") or p.get("task_id") or "")
        if u.kind == "A1":
            nid = str(p.get("gold_node_id") or "")
            asked[(tk, nid)] = u.label == "asked"
            if u.usefulness is not None:
                useful[(tk, nid)] = u.usefulness
        elif u.kind in ("A3_node", "A3_node_disc"):
            seen_nodes.add((tk, str(p.get("gold_node_id") or "")))
        elif u.kind == "A3_edge":
            seen_edges.add(
                (tk, str(p.get("gold_src_node_id") or ""), str(p.get("gold_dst_node_id") or ""))
            )

    rows: list[dict] = []
    for task_key in sorted(graphs):
        g = graphs[task_key]
        nodes = []
        for n in g.gold_nodes:
            k = (task_key, n.gold_node_id)
            nodes.append(
                as_dict(
                    replace(
                        n,
                        gold_human_asked=asked.get(k, n.gold_human_asked),
                        gold_usefulness_rating=useful.get(k, n.gold_usefulness_rating),
                        gold_human_adjudicated=n.gold_human_adjudicated or k in seen_nodes,
                        gold_graph_version=out_version,
                    )
                )
            )
        edges = []
        for e in g.gold_edges:
            k3 = (task_key, e.gold_src_node_id, e.gold_dst_node_id)
            edges.append(
                as_dict(
                    replace(
                        e,
                        gold_human_adjudicated=e.gold_human_adjudicated or k3 in seen_edges,
                        gold_graph_version=out_version,
                    )
                )
            )
        rows.append(
            {
                "gold_suite": g.gold_suite,
                "gold_task_key": g.gold_task_key,
                "gold_nodes": nodes,
                "gold_edges": edges,
                "gold_facets": [as_dict(f) for f in g.gold_facets],
                "gold_seed_node_ids": list(g.gold_seed_node_ids),
                "gold_graph_version": out_version,
                "gold_answer": g.gold_answer,
                "gold_aliases": list(g.gold_aliases),
                # Carried, not re-minted: `write_graphs` is idempotent on a row that already
                # has one, so the re-serialised graph keeps the nonce firewall layer 4 scans
                # for instead of arming a second one nothing was ever written under.
                "gold_canary": g.gold_canary,
                "gold_corpus_hash": g.gold_corpus_hash,
            }
        )
    return rows


# --------------------------------------------------------------------------- gates


def gate_numbers(
    cons: Consensus,
    key: Mapping[str, Any] | None,
    *,
    bundle: Mapping[str, Any] | None = None,
    n_missing_adjudicated: int | None = None,
) -> dict[str, float]:
    """The three G-M1/G-M2 inputs `pi gold validate` takes as flags.

    Each is NaN rather than 0.0 when its measurement was not taken, and carries the reason:
    `gate_report` compares with `>=`, so a NaN handed to it as a float prints FAIL, which is
    a different claim from NOT RUN and the wrong one. The caller passes None, not NaN.
    """
    key_items = (key or {}).get("items") or {}

    confirmed = sum(1 for u in cons.by_kind("A3_node") if u.label in ("required", "optional"))
    rejected = sum(1 for u in cons.by_kind("A3_node") if u.label == "not_a_need")

    # An edge item whose DST names its own SRC answers "holds" by construction, so counting
    # it would report the exporter's string formatting as edge_precision. Excluded and counted
    # separately rather than dropped silently: a caller has to be able to see that the gate's
    # denominator shrank, and why.
    self_ref: set[str] = set()
    for item in (bundle or {}).get("items") or ():
        if item.get("task_type") == "A3_edge" and edge_is_self_referential(
            item.get("context") or {}
        ):
            self_ref.add(str(item.get("item_id")))
    edges = [u for u in cons.by_kind("A3_edge") if u.unit_id not in self_ref]
    holds = sum(1 for u in edges if u.label == "holds")
    not_holds = sum(1 for u in edges if u.label == "does_not_hold")
    unsure = sum(1 for u in edges if u.label == "unsure")

    ratings: dict[str, dict[str, str]] = {}
    for unit, per_ann in (cons.votes or {}).items():
        if (cons.kinds or {}).get(unit) != "A3_match":
            continue
        decision = (key_items.get(unit) or {}).get("matcher_addresses")
        if decision is None:
            continue
        ratings[unit] = dict(per_ann, matcher="addresses" if decision else "does_not_address")

    out: dict[str, Any] = {
        "node_precision": confirmed / (confirmed + rejected)
        if (confirmed + rejected)
        else float("nan"),
        "n_node_verdicts": confirmed + rejected,
        "n_node_confirmed": confirmed,
        "edge_precision": holds / (holds + not_holds) if (holds + not_holds) else float("nan"),
        "n_edge_precision": holds + not_holds,
        "n_edge_unsure": unsure,
        "n_edge_self_referential": len(
            [u for u in cons.by_kind("A3_edge") if u.unit_id in self_ref]
        ),
        "matcher_kappa": krippendorff_alpha_nominal(ratings) if ratings else float("nan"),
        "n_matcher_kappa": len(ratings),
        "n_missing_texts": len(cons.missing),
    }

    if n_missing_adjudicated is None:
        out["node_recall"] = float("nan")
        out["node_recall_absent_because"] = (
            "missing needs are free text and were not adjudicated: two annotators describing "
            "one gap in different words cannot be merged mechanically, and a recall computed "
            "over an unmerged list counts the same need twice"
        )
    else:
        den = confirmed + int(n_missing_adjudicated)
        out["node_recall"] = confirmed / den if den else float("nan")
        out["n_missing_adjudicated"] = int(n_missing_adjudicated)
    return out


# --------------------------------------------------------------------------- judgments


def _words(text: str) -> int:
    return len(str(text).split())


def human_judgment_rows(
    bundle: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    key: Mapping[str, Any] | None,
) -> list[dict]:
    """Preference responses as `judgments.parquet` rows: one per (item, annotator) for A2, and
    one per DERIVED PAIR for A6.

    Into the existing table rather than a new one: `judge_family` is the column that table
    was designed to discriminate raters on, and the sigma_J, position-bias and
    length-explanation tooling already reads it. A human is `judge_family='human'` with the
    annotator in `judge_model`; a model rater carries its own family, so a model's preference
    can never be pooled with a person's by a reader of the table who trusted that column.

    `pref_sign` is from A's point of view, where A is the slot the annotator SAW FIRST — not
    the pipeline's preferred candidate. That is what makes `order` a usable position-bias
    control instead of a restatement of the sign.

    ONE FUNCTION, TWO ITEM SHAPES, deliberately: A6 exists to feed the SAME preference path A2
    feeds (`criterion="human_preference"`), so its rows must be minted by the code that mints
    A2's or the two instruments drift into two tables that only look alike.
    """
    items = {str(i["item_id"]): i for i in bundle.get("items") or ()}
    key_items = (key or {}).get("items") or {}
    bundle_id = str((bundle.get("manifest") or {}).get("bundle_id") or "")

    rows: list[dict] = []
    for r in sorted(records, key=lambda r: (str(r.get("item_id")), str(r.get("annotator_id")))):
        item = items.get(str(r.get("item_id") or ""))
        if item is None or item["task_type"] not in ("A2", "A6", "A7"):
            continue
        ki = key_items.get(str(item["item_id"])) or {}
        prov = item.get("provenance") or {}
        record_id = str(
            r.get("record_id") or f"{bundle_id}/{item['item_id']}/{r.get('annotator_id')}"
        )
        common = {
            "suite_id": str(prov.get("suite") or prov.get("suite_id") or ""),
            "task_id": str(prov.get("task_id") or ""),
            "criterion": "human_preference",
            "judge_family": rater_kind(r),
            "judge_model": str(r.get("annotator_id") or ""),
            # The item id IS a hash of exactly what was shown, which is what a prompt sha is for.
            "judge_prompt_sha": str(item["item_id"]),
            "temperature": None,
            "magnitude": None,
            "score": None,
            "paraphrase_id": "",
            "judge_pin": bundle_id,
            "response_sha": hashlib.sha256(_canon(r.get("response") or {}).encode()).hexdigest()[
                :16
            ],
        }

        if item["task_type"] == "A2":
            order = ki.get("order")
            if order not in ("ab", "ba"):
                continue
            payload = item.get("payload") or {}
            a_text = str((payload.get("option_a") or {}).get("question") or "")
            b_text = str((payload.get("option_b") or {}).get("question") or "")
            chosen, rejected_run = (
                str(ki.get("chosen_run_id") or ""),
                str(ki.get("rejected_run_id") or ""),
            )
            run_a, run_b = (chosen, rejected_run) if order == "ab" else (rejected_run, chosen)

            choice = (r.get("response") or {}).get("choice")
            rows.append(
                {
                    **common,
                    "judgment_id": hashlib.sha256(record_id.encode()).hexdigest()[:16],
                    "run_id_a": run_a,
                    "run_id_b": run_b,
                    "order": str(order),
                    "pref_sign": {"a": 1, "b": -1}.get(choice, 0),
                    "label": str(choice) if choice in ("tie", "both_bad") else "",
                    # The unit two raters are compared ON, so the existing alpha tooling groups
                    # the same pair across annotators.
                    "key_point_id": str(item["item_id"]),
                    "len_a_words": _words(a_text),
                    "len_b_words": _words(b_text),
                    "retest_group_id": str(ki.get("pair_id") or ""),
                }
            )
            continue

        if item["task_type"] == "A7":
            # ONE ROW PER RECORD WITH A REAL PREFERENCE. A tie/both_bad names no winner, and
            # the per-candidate reaches judgments are latency relabels, not preferences -- they
            # live in A7's own units, never in this table.
            order = ki.get("order")
            if order not in ("ab", "ba"):
                continue
            preference = (r.get("response") or {}).get("preference")
            if preference not in ("a", "b"):
                continue
            payload = item.get("payload") or {}
            a_text = str((payload.get("option_a") or {}).get("question") or "")
            b_text = str((payload.get("option_b") or {}).get("question") or "")
            chosen, rejected_run = (
                str(ki.get("chosen_run_id") or ""),
                str(ki.get("rejected_run_id") or ""),
            )
            run_a, run_b = (chosen, rejected_run) if order == "ab" else (rejected_run, chosen)
            rows.append(
                {
                    **common,
                    "judgment_id": hashlib.sha256(record_id.encode()).hexdigest()[:16],
                    "run_id_a": run_a,
                    "run_id_b": run_b,
                    "order": str(order),
                    "pref_sign": {"a": 1, "b": -1}[preference],
                    # The basis, when named, rides where A2's tie/both_bad ride: the winner is
                    # already in pref_sign, and "anticipation" vs "hygiene" is exactly what a
                    # reader of an A7 preference row needs that a sign cannot carry.
                    "label": str(r.get("basis") or ""),
                    "key_point_id": str(item["item_id"]),
                    "len_a_words": _words(a_text),
                    "len_b_words": _words(b_text),
                    "retest_group_id": str(ki.get("pair_id") or ""),
                }
            )
            continue

        # ---- A6: one row per derived pair.
        runs = ki.get("candidate_runs")
        if not isinstance(runs, Mapping):
            # No key entry, no rows. Same rule A2 follows for a missing `order`: a run id
            # guessed from the shown slot would be fabricated provenance on a published
            # preference.
            continue
        texts = {
            str(c.get("candidate_id")): str(c.get("question") or "") for c in _candidates_of(item)
        }
        for a, b, label in a6_pair_labels(item, r.get("response") or {}):
            run_a, run_b = str(runs.get(a) or ""), str(runs.get(b) or "")
            if not run_a or not run_b:
                continue
            rows.append(
                {
                    **common,
                    "judgment_id": hashlib.sha256(f"{record_id}/{a}|{b}".encode()).hexdigest()[:16],
                    "run_id_a": run_a,
                    "run_id_b": run_b,
                    # Always "ab": `candidate_id` is assigned after the shuffle, so the pair's A
                    # is BY CONSTRUCTION the candidate shown earlier. Claiming a "ba" here would
                    # be inventing a second presentation order that was never shown.
                    "order": "ab",
                    "pref_sign": {"a_better": 1, "b_better": -1}.get(label, 0),
                    "label": "tie" if label == "tie" else "",
                    # The PAIR, not the item: an A6 item carries C(k,2) of these, and grouping
                    # two raters on the item would pool ten different comparisons into one unit.
                    "key_point_id": f"{item['item_id']}/{a}|{b}",
                    "len_a_words": _words(texts.get(a, "")),
                    "len_b_words": _words(texts.get(b, "")),
                    # The RANKING every one of these pairs came from -- the retest grouping A2
                    # gets from its source `pair_id`.
                    "retest_group_id": str(item["item_id"]),
                }
            )
    return rows


def is_absent(value: float) -> bool:
    """A gate number that was not measured. Named because `math.isnan` at three call sites
    reads as arithmetic rather than as the absent-is-not-zero rule it enforces."""
    return isinstance(value, float) and math.isnan(value)
