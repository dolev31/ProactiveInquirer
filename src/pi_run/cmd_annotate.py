"""`pi annotate export|import|review` -- the CLI shell around the pure logic in
`pi_eval.annotate`. This module owns three things that module does not: WHICH items get
sampled from a suite's gold and recorded runs, HOW a bundle reaches disk without a canary in
it, and the argparse wiring. Every rule about blinding, consensus and gate arithmetic lives in
`pi_eval.annotate` and is reused here, never reimplemented.

WHY THE SAMPLERS ARE FREE FUNCTIONS, NOT METHODS ON A CLASS. Each one is the exact unit a
determinism test exercises: `sampler(inputs, rng, ...)` twice at one seed must produce one
`item_set_hash`, and at two seeds two different ones. A class would let state leak between
calls through `self`; a free function taking `rng` explicitly cannot.

WHY SOME SAMPLERS RETURN `(items, key_entries)` AND OTHERS RETURN JUST `items`. A1 and
A3_missing have no unblinding side at all -- there is nothing an A1 checklist or a free-text
missing-needs box could leak by construction. A2, A3 (its match rung), A4 and A5 each hide a
fact from the annotator (which slot the pipeline preferred, what the matcher decided, what
the mechanical latency label says, whether the run's own answer was actually correct), and
that fact has to travel to the KEY file and nowhere else. Returning it as a second value keyed
by item_id is what keeps it out of `items` even by accident -- a caller that only ever reads
the first element of the tuple cannot leak it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import exp, lgamma, log, log1p
from pathlib import Path
from typing import Any, Mapping, Sequence

from pi_eval.annotate import (
    A6_MIN_CANDIDATES,
    TASK_TYPES,
    TOOL_VERSION,
    a6_pair_labels,
    bundle_shape_errors,
    consensus,
    gate_numbers,
    human_judgment_rows,
    iaa_report,
    is_absent,
    item_id,
    item_set_hash,
    llm_agreement,
    merge_into_graphs,
    rater_kind,
    validate_records,
)
from pi_run.cache import cache_root
from pi_run.cmd_annotate_args import DEFAULT_OUT_VERSION, register  # noqa: F401
from pi_run.cmd_train import (
    StateMismatch,
    SuiteCache,
    _read_json,
    _read_jsonl,
    _require_gold,
    _train,
    collect_rows,
    render_state,
)

# A nonce minted but not yet registered still matches this shape; `canary.scan_text` only
# catches REGISTERED ones, and a nonce is exactly as dangerous the instant before it is
# registered as the instant after. See `_canary_hits`.
_CANARY_RE = re.compile(r"PINQCANARY_[0-9A-F]{16}")


class CanaryInBundle(RuntimeError):
    """A registered or nonce-shaped PINQCANARY token reached a bundle or key before it shipped.

    Raised, not returned, because the caller's ONLY correct response is "print it, write
    nothing, exit 2" -- there is no return value that means anything else.
    """


# --------------------------------------------------------------------------- shared item shape


def _prov(suite: str, task_key: str, graph_version: str, **extra: Any) -> dict:
    p = {"suite": suite, "task_id": task_key, "task_key": task_key, "graph_version": graph_version}
    p.update(extra)
    return p


def _make_item(
    task_type: str, prov: Mapping[str, Any], payload: Mapping[str, Any], context: Mapping[str, Any]
) -> dict:
    return {
        "item_id": item_id(task_type, prov, payload),
        "task_type": task_type,
        "context": dict(context),
        "payload": dict(payload),
        "provenance": dict(prov),
    }


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _action_question(raw_json: Any) -> str | None:
    """The `question` field out of an `action_json`/`chosen_json`/`rejected_json` string, or
    None for a STOP action or unparseable text -- neither names a question to show."""
    try:
        obj = json.loads(raw_json)
    except (TypeError, ValueError):
        return None
    q = obj.get("question") if isinstance(obj, Mapping) else None
    return str(q) if q else None


# --------------------------------------------------------------------------- A1: what a user would ask


def a1_node_projection(node: Any, answers: Mapping[str, str] | None = None) -> dict:
    """Exactly what an annotator may see: the id and the text. `gold_partition`,
    `gold_discoverability`, `gold_depth`, `gold_answer` and `gold_canary` must appear nowhere
    on an A1 item -- that is what makes A1 a question about what a user would have thought to
    ask, not a quiz on the pipeline's own labels.

    `node_id` is passed through UNCHANGED -- resolution only ever touches the copy of the text
    an annotator reads, never the id an annotation maps back to. `answers` may raise
    `UnresolvedPlaceholder`; the caller decides what refusing this node means for the item."""
    from pi_eval.build.musique_build import resolve_placeholders

    return {
        "node_id": node.gold_node_id,
        "text": resolve_placeholders(node.gold_text, answers or {}),
    }


def _a1_universe(graph: Any) -> list:
    """The ADR universe, the exact filter `pi_eval.metrics.human.anticipated_discovery_rate`
    uses. An A1 checklist over any other node population would not be the population ADR's
    denominator draws from, and the annotation would answer a different question than the
    metric asks."""
    return [
        n
        for n in graph.gold_nodes
        if n.gold_partition in ("required", "optional") and n.gold_discoverability == "kb"
    ]


def sample_a1_items(
    graphs: Mapping[str, Any],
    rng: random.Random,
    *,
    n_tasks: int,
    cache: Any | None = None,
    answers: Mapping[str, Mapping[str, str]] | None = None,
    stats: dict[str, int] | None = None,
) -> list[dict]:
    """`answers` is `{task_key: {node_id: sub_answer}}` (musique only; `{}` elsewhere, which
    makes resolution a no-op on text with no `#N`). A task whose checklist has EVEN ONE
    unresolvable node is refused WHOLE -- not shipped with that one node silently missing,
    which would change the node population a real A1 checklist is over without saying so."""
    from pi_eval.build.musique_build import UnresolvedPlaceholder

    answers = answers or {}
    stats = stats if stats is not None else {}
    candidates = sorted(tk for tk, g in graphs.items() if _a1_universe(g))
    picked = sorted(rng.sample(candidates, min(n_tasks, len(candidates))))
    items: list[dict] = []
    for task_key in picked:
        g = graphs[task_key]
        nodes = list(_a1_universe(g))
        rng.shuffle(nodes)
        task_answers = answers.get(task_key, {})
        try:
            projected = [a1_node_projection(n, task_answers) for n in nodes]
        except UnresolvedPlaceholder:
            stats["a1_unresolved_placeholder"] = stats.get("a1_unresolved_placeholder", 0) + 1
            continue
        question = ""
        if cache is not None:
            question = str(cache.view(g.gold_suite, task_key).question)
        prov = _prov(g.gold_suite, task_key, g.gold_graph_version)
        payload = {"nodes": projected}
        items.append(_make_item("A1", prov, payload, {"question": question}))
    return items


# --------------------------------------------------------------------------- A2: paired preference


def _pairs_margin_threshold(pairs_path: Path) -> float | None:
    sidecar = pairs_path.parent / f"{pairs_path.stem}.manifest.json"
    if not sidecar.exists():
        return None
    try:
        return float(json.loads(sidecar.read_text()).get("margin_threshold"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _a2_replay(cache: Any, row: Mapping[str, Any], runs_root: Path | None) -> tuple[dict, bool]:
    """(context, used_fallback). `render_state` reconstructs its prompt from the parent run's
    turns by replaying them into held evidence units, a Q/A history and a draft -- it HAS all
    three in hand before it ever renders text. This reads off those SAME objects rather than
    parsing the rendered prompt back apart, which would be lossy where this is exact. Falls
    back to the row's OWN recorded `state_text` (with empty evidence/history/draft) only when
    the parent run cannot be replayed; `state_text` itself is shipped VERBATIM either way as
    the escape hatch a collapsed panel can show."""
    run_id = str(row.get("run_id") or "")
    suite_id = str(row.get("suite_id", ""))
    task_id = str(row.get("task_id", ""))
    turn_idx = int(row.get("turn_idx", 0))
    question = ""
    if suite_id and task_id:
        try:
            question = str(cache.view(suite_id, task_id).question)
        except Exception:
            question = ""
    if runs_root is not None and run_id:
        turns_path = Path(runs_root) / run_id / "turns.jsonl"
        if turns_path.exists():
            turns = sorted(_load_jsonl(turns_path), key=lambda t: int(t["turn_idx"]))
            try:
                state_text = render_state(cache, suite_id, task_id, turns, turn_idx, verify=True)
            except (StateMismatch, KeyError, IndexError):
                pass
            else:
                from pinq.types import Evidence

                by_uid = cache.units(suite_id, task_id)
                held: list = []
                history: list[dict] = []
                for prior in turns[:turn_idx]:
                    held.extend(by_uid[u] for u in prior.get("retrieved_uids") or () if u in by_uid)
                    history.append(
                        {
                            "q": str(prior.get("question", "")),
                            "a": str(prior.get("response_text") or ""),
                        }
                    )
                evidence = [
                    {"uid": u.uid, "title": getattr(u, "title", ""), "text": u.text}
                    for u in Evidence.of(tuple(held)).units
                ]
                prev = turns[turn_idx - 1] if turn_idx > 0 else {}
                draft = str(prev.get("draft_text") or "")
                return (
                    {
                        "question": question,
                        "evidence": evidence,
                        "history": history,
                        "draft": draft,
                        "state_text": state_text,
                    },
                    False,
                )
    return (
        {
            "question": question,
            "evidence": [],
            "history": [],
            "draft": "",
            "state_text": str(row.get("state_text") or ""),
        },
        True,
    )


def sample_a2_items(
    pairs_rows: Sequence[Mapping[str, Any]],
    cache: Any,
    rng: random.Random,
    *,
    n: int,
    runs_root: Path | None = None,
    margin_threshold: float | None = None,
    graph_version: str = "",
    stats: dict[str, int] | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    """A2 preference items. The bundle never says which slot the pipeline preferred -- that
    orientation, and the candidate run ids it names, live only in the returned key, keyed by
    item_id, matching `pi_eval.annotate`'s "an A2 slot has no canonical meaning until read
    back through the key" rule."""
    stats = stats if stats is not None else {}
    usable: list[tuple[Mapping[str, Any], str, str]] = []
    for r in sorted(pairs_rows, key=lambda r: str(r.get("pair_id"))):
        qa, qb = (
            _action_question(r.get("chosen_json", "")),
            _action_question(r.get("rejected_json", "")),
        )
        if qa is None or qb is None:
            stats["a2_no_question"] = stats.get("a2_no_question", 0) + 1
            continue
        usable.append((r, qa, qb))
    if not usable:
        return [], {}

    tau = margin_threshold
    if tau is None:
        tau = statistics.median(float(r.get("margin", 0.0)) for r, _, _ in usable)

    # Oversample near tau (the pairs the automatic threshold is least confident about) and
    # spread across is_latent, by sorting on (is_latent, distance-to-tau) and taking a prefix
    # of a list that is itself sorted first on a tie-breaking id -- deterministic under seed.
    ordered = sorted(
        usable,
        key=lambda t: (
            bool(t[0].get("is_latent", False)),
            abs(float(t[0].get("margin", 0.0)) - tau),
            str(t[0].get("pair_id")),
        ),
    )
    picked = sorted(ordered[:n], key=lambda t: str(t[0].get("pair_id")))

    items: list[dict] = []
    key_entries: dict[str, dict] = {}
    for r, qa, qb in picked:
        order = "ab" if rng.random() < 0.5 else "ba"
        context, fell_back = _a2_replay(cache, r, runs_root)
        if fell_back:
            stats["a2_state_text_fallback"] = stats.get("a2_state_text_fallback", 0) + 1
        first, second = (qa, qb) if order == "ab" else (qb, qa)
        payload = {"option_a": {"question": first}, "option_b": {"question": second}}
        prov = _prov(
            str(r.get("suite_id", "")),
            str(r.get("task_id", "")),
            graph_version,
            run_id=str(r.get("run_id", "")),
            turn_idx=int(r.get("turn_idx", 0)),
        )
        it = _make_item("A2", prov, payload, context)
        items.append(it)
        key_entries[it["item_id"]] = {
            "order": order,
            "chosen_run_id": str(r.get("chosen_run_id", "")),
            "rejected_run_id": str(r.get("rejected_run_id", "")),
            "margin": float(r.get("margin", 0.0)),
            "pair_id": str(r.get("pair_id", "")),
        }
    return items, key_entries


# --------------------------------------------------------------------------- A6: rank all k candidates

# The cap on how many candidates one A6 item may show. `sample-candidates` forks k=5, but a
# STATE accumulates candidates across every pair that names it: measured on the live
# `data/rl/pairs.jsonl` (1,611 pairs, 172 states), the candidate-count histogram is
# {2: 14, 3: 15, 4: 7, 5: 8, 6: 14, 7: 24, 8: 34, 9: 54, 17: 2}. Seventeen candidates is
# C(17,2) = 136 pairwise judgments riding on one screen of attention -- past the point where
# the density argument holds, because the ranking a tired annotator produces over seventeen
# items is not ten times better evidence than one over five, it is worse evidence about more
# pairs. Five keeps an item at C(5,2) = 10 and matches the k the forker actually samples.
A6_MAX_CANDIDATES = 5


def _a6_norm_question(text: str) -> str:
    return " ".join(str(text).split()).strip().casefold()


def _a6_states(
    pairs_rows: Sequence[Mapping[str, Any]], stats: dict[str, int]
) -> list[tuple[tuple[str, str, str, int], Mapping[str, Any], dict[str, str], list[dict]]]:
    """Group `pairs.jsonl` into STATES: `(suite_id, task_id, run_id, turn_idx)` with the union
    of `chosen_run_id`/`rejected_run_id` as that state's candidate set.

    Two candidates whose question text is IDENTICAL are collapsed into one. Measured on the
    live file: 95 of the 158 otherwise-eligible states carry two candidate run ids with
    byte-identical question text (the forker resampled and got the same question twice).
    Asking a person which of two identical questions is better manufactures a coin flip, and
    the pair it derives enters DPO as a preference between two copies of one question --
    exactly the fabricated preference the tier vocabulary exists to avoid.
    """
    by_state: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for r in pairs_rows:
        key = (
            str(r.get("suite_id", "")),
            str(r.get("task_id", "")),
            str(r.get("run_id", "")),
            int(r.get("turn_idx", 0)),
        )
        by_state[key].append(r)

    out = []
    for state in sorted(by_state):
        rows = sorted(by_state[state], key=lambda r: str(r.get("pair_id")))
        texts: dict[str, str] = {}
        for r in rows:
            for side in ("chosen", "rejected"):
                run_id = str(r.get(f"{side}_run_id", ""))
                q = _action_question(r.get(f"{side}_json", ""))
                if not run_id or q is None:
                    stats["a6_no_question"] = stats.get("a6_no_question", 0) + 1
                    continue
                texts.setdefault(run_id, q)
        by_text: dict[str, str] = {}
        for run_id in sorted(texts):
            norm = _a6_norm_question(texts[run_id])
            if norm in by_text:
                stats["a6_duplicate_question"] = stats.get("a6_duplicate_question", 0) + 1
                continue
            by_text[norm] = run_id
        candidates = {run_id: texts[run_id] for run_id in sorted(by_text.values())}
        if len(candidates) < A6_MIN_CANDIDATES:
            stats["a6_too_few_candidates"] = stats.get("a6_too_few_candidates", 0) + 1
            continue
        out.append((state, rows[0], candidates, list(rows)))
    return out


def _a6_auto(rows: Sequence[Mapping[str, Any]], run_of: Mapping[str, str]) -> dict[str, Any]:
    """The pipeline's own view of this state, for the KEY only.

    `auto_pairs` is the comparisons the pipeline ACTUALLY made -- measured, a 5-candidate state
    carries 4 pairs, not 10, because `sample-candidates` scores every branch against the winner
    rather than against each other. `auto_order` is a full ordering DERIVED from the mean signed
    margin, which is why the review's agreement statistic reads `auto_pairs` and not
    `auto_order`: a preference between two candidates the pipeline never compared is not a
    disagreement to score, it is a comparison nobody made.
    """
    cid_of = {run_id: cid for cid, run_id in run_of.items()}
    scores: dict[str, list[float]] = defaultdict(list)
    auto_pairs: dict[str, str] = {}
    margins: dict[str, float] = {}
    for r in rows:
        win, lose = (
            cid_of.get(str(r.get("chosen_run_id", ""))),
            cid_of.get(str(r.get("rejected_run_id", ""))),
        )
        margin = float(r.get("margin", 0.0))
        if win is not None:
            scores[win].append(margin)
        if lose is not None:
            scores[lose].append(-margin)
        if win is None or lose is None or win == lose:
            continue
        a, b = (win, lose) if win < lose else (lose, win)
        auto_pairs[f"{a}|{b}"] = "a_better" if a == win else "b_better"
        margins[f"{a}|{b}"] = margin
    mean = {cid: (sum(v) / len(v) if v else 0.0) for cid, v in scores.items()}
    for cid in run_of:
        mean.setdefault(cid, 0.0)
    return {
        "auto_scores": {cid: mean[cid] for cid in sorted(mean)},
        "auto_order": sorted(mean, key=lambda cid: (-mean[cid], cid)),
        "auto_pairs": dict(sorted(auto_pairs.items())),
        "margins": dict(sorted(margins.items())),
    }


def sample_a6_items(
    pairs_rows: Sequence[Mapping[str, Any]],
    cache: Any,
    rng: random.Random,
    *,
    n: int,
    runs_root: Path | None = None,
    graph_version: str = "",
    stats: dict[str, int] | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    """A6 ranking items: A2's generalisation from two candidates to all of them.

    WHY THIS EXISTS. `pi train sample-candidates` forks a state into k continuations; A2 shows
    an annotator two of them and buys ONE pairwise comparison. Ranking all five buys
    C(5,2) = 10, from the same states, at no extra sampling cost and for roughly the same
    annotation effort -- a ~10x gain in signal density on exactly the preference data rung-2
    DPO consumes. That density is the whole justification for the instrument.

    Built on A2's machinery throughout rather than beside it: the same `pairs.jsonl` rows, the
    same `_a2_replay` context (so both instruments show the SAME state), the same
    (items, key_entries) return shape, and `human_judgment_rows` mints its rows with the same
    `criterion` so they pool with A2's rather than starting a second preference table.

    A state with fewer than `A6_MIN_CANDIDATES` distinct candidates is NOT shipped: at two it
    IS an A2 item, and one state shipped as both instruments asks one comparison twice.

    The candidate list is SHUFFLED and the `candidate_id`s (`c0..c4`) are assigned afterwards,
    so `c0` is not the pipeline's favourite; the `candidate_id -> run_id` map and the automatic
    ordering live in the key and never ship.
    """
    stats = stats if stats is not None else {}
    states = _a6_states(pairs_rows, stats)
    if not states:
        return [], {}

    # RANKED BY HOW MANY PAIRS THE ITEM WOULD YIELD, capped, then by the state key. A
    # 5-candidate item is 10 derived pairs for one annotation slot and a 3-candidate one is 3,
    # so when the quota binds the slots go where the density argument is strongest. The
    # ordering never consults `margin`: preferring states the automatic scorer felt strongly
    # about would make the shown population a function of the very quantity the human judgment
    # is there to check.
    ordered = sorted(
        states,
        key=lambda s: (-min(len(s[2]), A6_MAX_CANDIDATES), s[0][0], s[0][1], s[0][2], s[0][3]),
    )
    picked = ordered[: max(0, n)]

    items: list[dict] = []
    key_entries: dict[str, dict] = {}
    for state, row, candidates, rows in picked:
        run_ids = sorted(candidates)
        if len(run_ids) > A6_MAX_CANDIDATES:
            # A UNIFORM draw, not the top of the automatic ordering: taking the candidates the
            # pipeline scored highest would show the annotator a set selected by the thing being
            # validated.
            run_ids = sorted(rng.sample(run_ids, A6_MAX_CANDIDATES))
            stats["a6_candidates_capped"] = stats.get("a6_candidates_capped", 0) + 1
        rng.shuffle(run_ids)
        run_of = {f"c{i}": run_id for i, run_id in enumerate(run_ids)}

        context, fell_back = _a2_replay(cache, row, runs_root)
        if fell_back:
            stats["a6_state_text_fallback"] = stats.get("a6_state_text_fallback", 0) + 1
        payload = {
            "candidates": [
                {"candidate_id": cid, "question": candidates[run_of[cid]]} for cid in sorted(run_of)
            ]
        }
        prov = _prov(
            state[0],
            state[1],
            graph_version,
            run_id=state[2],
            turn_idx=state[3],
        )
        it = _make_item("A6", prov, payload, context)
        items.append(it)
        key_entries[it["item_id"]] = {
            "candidate_runs": run_of,
            **_a6_auto(rows, run_of),
            "pair_ids": sorted({str(r.get("pair_id", "")) for r in rows}),
            "is_latent": bool(row.get("is_latent", False)),
            "n_candidates": len(run_of),
        }
    return items, key_entries


# --------------------------------------------------------------------------- A7: reaches-unstated + anticipation preference


def sample_a7_items(
    cache: Any,
    pairs_rows: Sequence[Mapping[str, Any]],
    runs_root: Path | None = None,
    *,
    n: int,
    rng: random.Random,
    stats: dict[str, int] | None = None,
    graph_version: str = "",
) -> tuple[list[dict], dict[str, dict]]:
    """A7 decision-point items: A2's pairs, re-asked as an ANTICIPATION judgment.

    Same blinding contract as A2 -- the bundle never says which slot the pipeline preferred,
    and the shown-order coin flip, the candidate run ids, the margin and the mechanical
    `is_latent` flag live only in the returned key, keyed by item_id.

    STRATIFIED BY `is_latent`, half and half: the per-candidate reaches judgment doubles as a
    relabel of the mechanical flag that selects the trainset, and a relabel measured on one
    side of the flag can only ever confirm it. The two pools are drawn from alternately, so a
    balanced pool yields a balanced sample and a one-sided pool degrades to what exists.
    """
    stats = stats if stats is not None else {}
    usable: list[tuple[Mapping[str, Any], str, str]] = []
    for r in sorted(pairs_rows, key=lambda r: str(r.get("pair_id"))):
        qa, qb = (
            _action_question(r.get("chosen_json", "")),
            _action_question(r.get("rejected_json", "")),
        )
        if qa is None or qb is None:
            stats["a7_no_question"] = stats.get("a7_no_question", 0) + 1
            continue
        usable.append((r, qa, qb))
    if not usable:
        return [], {}

    pools = [
        [t for t in usable if bool(t[0].get("is_latent", False))],
        [t for t in usable if not bool(t[0].get("is_latent", False))],
    ]
    for pool in pools:
        rng.shuffle(pool)  # each pool is already pair_id-sorted, so one seed is one draw
    target = min(max(0, n), len(usable))
    picked: list[tuple[Mapping[str, Any], str, str]] = []
    while len(picked) < target:
        for pool in pools:
            if len(picked) >= target:
                break
            if pool:
                picked.append(pool.pop())
    picked = sorted(picked, key=lambda t: str(t[0].get("pair_id")))

    items: list[dict] = []
    key_entries: dict[str, dict] = {}
    for r, qa, qb in picked:
        order = "ab" if rng.random() < 0.5 else "ba"
        context, fell_back = _a2_replay(cache, r, runs_root)
        if fell_back:
            stats["a7_state_text_fallback"] = stats.get("a7_state_text_fallback", 0) + 1
        first, second = (qa, qb) if order == "ab" else (qb, qa)
        payload = {"option_a": {"question": first}, "option_b": {"question": second}}
        prov = _prov(
            str(r.get("suite_id", "")),
            str(r.get("task_id", "")),
            graph_version,
            run_id=str(r.get("run_id", "")),
            turn_idx=int(r.get("turn_idx", 0)),
        )
        it = _make_item("A7", prov, payload, context)
        items.append(it)
        key_entries[it["item_id"]] = {
            "order": order,
            "chosen_run_id": str(r.get("chosen_run_id", "")),
            "rejected_run_id": str(r.get("rejected_run_id", "")),
            "margin": float(r.get("margin", 0.0)),
            "pair_id": str(r.get("pair_id", "")),
            "is_latent": bool(r.get("is_latent", False)),
            "latent_depth": r.get("latent_depth"),
        }
    return items, key_entries


def sample_a7_foils(
    cache: Any,
    pairs_rows: Sequence[Mapping[str, Any]],
    *,
    n: int,
    rng: random.Random,
    graph_version: str = "",
) -> tuple[list[dict], dict[str, dict]]:
    """Planted A7 items whose coin-flipped side IS the task question verbatim, so the honest
    reaches judgment for that side is always "stays_stated" -- they exist to catch an
    annotator, human or model, who is not reading. Same hard constraint as
    `sample_attention_items`: the expected answer lives ONLY in the returned key, and
    `pi_eval.annotate._collect` drops any record whose key entry carries `attention_check`
    before it can become a vote. The entry still carries an `order`, so `_units` on such a
    record forms units for the attention report to compare, and nothing else ever reads them.
    """
    usable: list[tuple[Mapping[str, Any], str, str]] = []
    for r in sorted(pairs_rows, key=lambda r: str(r.get("pair_id"))):
        q = _action_question(r.get("chosen_json", ""))
        state_text = str(r.get("state_text") or "")
        suite_id, task_id = str(r.get("suite_id", "")), str(r.get("task_id", ""))
        if q is None or not state_text or not suite_id or not task_id:
            continue
        try:
            task_q = str(cache.view(suite_id, task_id).question)
        except Exception:
            continue
        if not task_q.strip() or _a6_norm_question(task_q) == _a6_norm_question(q):
            continue
        usable.append((r, task_q, q))
    if not usable:
        return [], {}

    items: list[dict] = []
    key_entries: dict[str, dict] = {}
    for r, task_q, real_q in rng.sample(usable, min(max(0, n), len(usable))):
        side = "a" if rng.random() < 0.5 else "b"
        first, second = (task_q, real_q) if side == "a" else (real_q, task_q)
        payload = {"option_a": {"question": first}, "option_b": {"question": second}}
        context = {
            "question": task_q,
            "evidence": [],
            "history": [],
            "draft": "",
            "state_text": str(r.get("state_text") or ""),
        }
        prov = _prov(
            str(r.get("suite_id", "")),
            str(r.get("task_id", "")),
            graph_version,
            run_id=str(r.get("run_id", "")),
            turn_idx=int(r.get("turn_idx", 0)),
        )
        it = _make_item("A7", prov, payload, context)
        items.append(it)
        key_entries[it["item_id"]] = {
            "order": "ab" if rng.random() < 0.5 else "ba",
            "attention_check": {"expected": {f"{side}_reaches": "stays_stated"}},
        }
    return items, key_entries


# --------------------------------------------------------------------------- A3: node / edge / match / missing


def _a3_node_context(
    question: str, node: Any, units: Mapping[str, Any], answers: Mapping[str, str]
) -> dict:
    from pi_eval.build.musique_build import resolve_placeholders

    evidence = []
    for uid in node.gold_ev_uids:
        u = units.get(uid)
        evidence.append(
            {"uid": uid, "title": getattr(u, "title", ""), "text": getattr(u, "text", "")}
        )
    return {
        "question": question,
        "node_text": resolve_placeholders(node.gold_text, answers),
        "evidence": evidence,
    }


# The fan-out cap on A3_match near-misses. A `match_kind == "none"` row has no single matched
# turn to point at, so it is shown as one item PER (candidate question, node) pair rather than
# one item listing every question in the run -- the unit `matcher_kappa` is computed over is a
# single (ask, node) judgment, and a multi-question item cannot produce one. Capped per run so
# one long run's turns cannot dominate the sample; what the cap drops is counted in `stats`.
_A3_MATCH_FANOUT_CAP = 3


def _asked_questions(
    runs_root: Path | None, run_id: str, matched_turn_idx: Any
) -> list[tuple[int, str]]:
    """The (turn_idx, question) pair(s) to show an A3_match annotator. The specific turn when
    the matcher named one (ask/resolve/use); every question asked in the run when it did not
    (match_kind 'none') -- the annotator is judging whether ANYTHING asked addresses the need,
    which is exactly what a near-miss item needs shown, one question at a time."""
    if runs_root is None or not run_id:
        return []
    turns = sorted(
        _load_jsonl(Path(runs_root) / run_id / "turns.jsonl"),
        key=lambda t: int(t.get("turn_idx", 0)),
    )
    if not turns:
        return []
    if matched_turn_idx is not None:
        for t in turns:
            if int(t.get("turn_idx", -1)) == int(matched_turn_idx):
                return [(int(t["turn_idx"]), str(t.get("question", "")))]
        return []
    return [(int(t["turn_idx"]), str(t["question"])) for t in turns if t.get("question")]


def _a3_match_items(
    row: Mapping[str, Any],
    graphs: Mapping[str, Any],
    runs_root: Path | None,
    *,
    per_run_used: dict[str, int],
    fanout_cap: int,
    stats: dict[str, int],
    answers: Mapping[str, str] | None = None,
) -> list[tuple[dict, dict]]:
    from pi_eval.build.musique_build import UnresolvedPlaceholder, resolve_placeholders

    answers = answers or {}
    task_key = str(row.get("task_id", ""))
    g = graphs.get(task_key)
    node_id = str(row.get("node_id", ""))
    node = (
        next((n for n in g.gold_nodes if n.gold_node_id == node_id), None)
        if g is not None
        else None
    )
    run_id = str(row.get("run_id", ""))
    asked = _asked_questions(runs_root, run_id, row.get("matched_turn_idx"))
    if node is None or not asked:
        return []
    try:
        node_text = resolve_placeholders(node.gold_text, answers)
    except UnresolvedPlaceholder:
        stats["a3_match_unresolved_placeholder"] = (
            stats.get("a3_match_unresolved_placeholder", 0) + 1
        )
        return []
    matcher_addresses = str(row.get("match_kind")) != "none"
    out: list[tuple[dict, dict]] = []
    for turn_idx, question in asked:
        used = per_run_used.get(run_id, 0)
        if used >= fanout_cap:
            stats["a3_match_fanout_capped"] = stats.get("a3_match_fanout_capped", 0) + 1
            continue
        per_run_used[run_id] = used + 1
        context = {"asked_question": question, "node_text": node_text}
        prov = _prov(
            str(row.get("suite_id", "")),
            task_key,
            str(row.get("graph_version", "")),
            gold_node_id=node_id,
            run_id=run_id,
            turn_idx=turn_idx,
        )
        it = _make_item("A3_match", prov, {}, context)
        # WHICH matcher ruled, not just what it said. `matches.parquet` is keyed on
        # `(run_id, node_id, matcher_id, graph_version)` because more than one matcher exists;
        # it currently holds three, disagreeing on 475 (run, node) pairs. G-M2 asks whether
        # "the matcher" agrees with a human, and a kappa that cannot name which one it
        # validated is a number about an unidentified instrument.
        out.append(
            (
                it,
                {
                    "matcher_addresses": matcher_addresses,
                    "matcher_id": str(row.get("matcher_id") or ""),
                    "match_kind": str(row.get("match_kind") or ""),
                },
            )
        )
    return out


_TRIPLE = " >> "


def mask_edge_subject(src_text: str, dst_text: str) -> str:
    """Hide dst's subject behind a reference to src, for relation-triple node text.

    `reference_placeholders` fixes the musique case, where the dependency is spelled `#N` and
    can be rewritten. It does nothing for wiki2, whose node text arrives ALREADY resolved:

        src: "Anumodhanam >> director"
        dst: "I. V. Sasi >> date of death"      <- "I. V. Sasi" IS the director

    and an annotator reading that dst answers, correctly, that it needs nothing from src --
    the wrong verdict about the dependency A3_edge exists to judge. `wiki2_build` creates a
    prerequisite edge exactly when `objects[i] == subjects[j]` (wiki2_build.py:251), so dst's
    subject IS src's answer BY CONSTRUCTION. That is what makes this a substitution rather
    than a guess, and it is why the rewrite is scoped to a dst that actually has a subject to
    replace: a dst phrased as a question carries its dependency in prose, and overwriting its
    first words would corrupt a need instead of masking one.
    """
    if _TRIPLE not in dst_text or _TRIPLE not in src_text:
        return dst_text
    if dst_text.startswith("\u27e8"):
        return dst_text  # already masked; a second pass must not nest another reference
    _subject, _, rest = dst_text.partition(_TRIPLE)
    return f"\u27e8{src_text}\u27e9{_TRIPLE}{rest}"


def sample_a3_items(
    graphs: Mapping[str, Any],
    matches_rows: Sequence[Mapping[str, Any]],
    cache: Any,
    rng: random.Random,
    *,
    n_node: int,
    n_edge: int,
    n_match: int,
    runs_root: Path | None = None,
    stats: dict[str, int] | None = None,
    match_fanout_cap: int = _A3_MATCH_FANOUT_CAP,
    answers: Mapping[str, Mapping[str, str]] | None = None,
    match_graphs: Mapping[str, Any] | None = None,
    match_answers: Mapping[str, Mapping[str, str]] | None = None,
    match_suite: str | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    """`answers` is `{task_key: {node_id: sub_answer}}` (musique only; empty elsewhere). A
    node/edge whose text cannot be resolved is refused individually -- each is its own item, so
    unlike A1's checklist there is no "whole item" to fall back to shrinking."""
    from pi_eval.build.musique_build import UnresolvedPlaceholder, resolve_placeholders

    answers = answers or {}
    stats = stats if stats is not None else {}
    items: list[dict] = []
    key_entries: dict[str, dict] = {}

    # ---- nodes: mostly confirmed candidates, ~15% planted `dropped` FOILS so a
    # confirm-everything annotator is detectable. `gold_partition` never reaches the ITEM --
    # it is exactly what A3_node exists to ask about -- but it DOES reach the key, which is
    # what lets `review` compute the foil confirmation rate without ever showing it.
    primary = sorted(
        (tk, n.gold_node_id)
        for tk, g in graphs.items()
        for n in g.gold_nodes
        if n.gold_partition in ("required", "optional")
    )
    foils = sorted(
        (tk, n.gold_node_id)
        for tk, g in graphs.items()
        for n in g.gold_nodes
        if n.gold_partition == "dropped"
    )
    n_foil = min(len(foils), round(n_node * 0.15))
    n_primary = max(0, n_node - n_foil)

    # Within the non-foil primary pool, OVERSAMPLE `gold_discoverability == "unknown"` nodes,
    # targeting roughly a third of it. `unknown` directly softens `ceiling_private_share`,
    # which this project's own framing calls the single most important number, and a human
    # verdict on one of these nodes is immediately useful precisely because the pipeline could
    # not classify it -- yet a plain uniform draw over `primary` would show them in proportion
    # to their (often small) share of the graph. wiki2 has 1,022 of them, strategyqa 1,247.
    disc_of = {
        (tk, n.gold_node_id): n.gold_discoverability
        for tk, g in graphs.items()
        for n in g.gold_nodes
    }
    unknown_pool = sorted(x for x in primary if disc_of.get(x) == "unknown")
    rest_pool = sorted(x for x in primary if disc_of.get(x) != "unknown")
    n_unknown_target = round(n_primary / 3)
    n_rest_target = n_primary - n_unknown_target
    picked_unknown = rng.sample(unknown_pool, min(n_unknown_target, len(unknown_pool)))
    picked_rest = rng.sample(rest_pool, min(n_rest_target, len(rest_pool)))
    shortfall_unknown = n_unknown_target - len(picked_unknown)
    shortfall_rest = n_rest_target - len(picked_rest)
    # Fall back gracefully when one side runs short, backfilling from the other side's
    # remaining supply so the requested `n_node` is still shipped, and count what could not be
    # honored rather than silently reverting to an unstratified draw.
    if shortfall_unknown > 0:
        stats["a3_node_unknown_discoverability_shortfall"] = (
            stats.get("a3_node_unknown_discoverability_shortfall", 0) + shortfall_unknown
        )
        remaining_rest = [x for x in rest_pool if x not in picked_rest]
        picked_rest = picked_rest + rng.sample(
            remaining_rest, min(shortfall_unknown, len(remaining_rest))
        )
    if shortfall_rest > 0:
        remaining_unknown = [x for x in unknown_pool if x not in picked_unknown]
        picked_unknown = picked_unknown + rng.sample(
            remaining_unknown, min(shortfall_rest, len(remaining_unknown))
        )
    stats["a3_node_unknown_discoverability_sampled"] = len(picked_unknown)

    picked_nodes = sorted(picked_unknown + picked_rest + rng.sample(foils, n_foil))
    for tk, nid in picked_nodes:
        g = graphs[tk]
        node = next(x for x in g.gold_nodes if x.gold_node_id == nid)
        units = cache.units(g.gold_suite, tk) if cache is not None else {}
        question = str(cache.view(g.gold_suite, tk).question) if cache is not None else ""
        try:
            context = _a3_node_context(question, node, units, answers.get(tk, {}))
        except UnresolvedPlaceholder:
            stats["a3_node_unresolved_placeholder"] = (
                stats.get("a3_node_unresolved_placeholder", 0) + 1
            )
            continue
        prov = _prov(g.gold_suite, tk, g.gold_graph_version, gold_node_id=nid)
        it = _make_item("A3_node", prov, {}, context)
        items.append(it)
        key_entries[it["item_id"]] = {"gold_partition": node.gold_partition}

    # ---- edges
    #
    # dst_text is rendered with `reference_placeholders`, NOT `resolve_placeholders`, and only
    # here: an A3_edge annotator is judging whether dst DEPENDS ON src, and printing src's
    # ANSWER inside dst's text (what `resolve_placeholders` does, correctly, everywhere else)
    # makes dst independently answerable from what is on screen -- the honest verdict about
    # THAT text is "no edge", which is the wrong instrument for the dependency being judged.
    # See `reference_placeholders`'s docstring for the measured evidence (6/6 true edges
    # rejected). src_text keeps `resolve_placeholders`: it is shown as the need being judged
    # AS DEPENDENT, not as a referent, so it should read as a real, answerable question.
    from pi_eval.build.musique_build import REF_RE, reference_placeholders

    edge_pool = sorted(
        (tk, e.gold_src_node_id, e.gold_dst_node_id)
        for tk, g in graphs.items()
        for e in g.gold_edges
    )
    for tk, src, dst in rng.sample(edge_pool, min(n_edge, len(edge_pool))):
        g = graphs[tk]
        by_id = {n.gold_node_id: n for n in g.gold_nodes}
        question = str(cache.view(g.gold_suite, tk).question) if cache is not None else ""
        task_answers = answers.get(tk, {})
        try:
            src_text = (
                resolve_placeholders(by_id[src].gold_text, task_answers) if src in by_id else ""
            )
            dst_node = by_id.get(dst)
            if dst_node is None:
                dst_text = ""
            else:
                # Only the nodes dst's OWN text actually references need resolving here --
                # resolving every node in the graph would refuse this item over placeholders
                # unrelated to the edge being shown.
                refs = sorted({int(m.group(1)) for m in REF_RE.finditer(dst_node.gold_text)})
                referent_texts = {
                    f"s{n}": resolve_placeholders(by_id[f"s{n}"].gold_text, task_answers)
                    for n in refs
                    if f"s{n}" in by_id
                }
                dst_text = reference_placeholders(dst_node.gold_text, referent_texts)
            dst_text = mask_edge_subject(src_text, dst_text)
            context = {"question": question, "src_text": src_text, "dst_text": dst_text}
        except UnresolvedPlaceholder:
            stats["a3_edge_unresolved_placeholder"] = (
                stats.get("a3_edge_unresolved_placeholder", 0) + 1
            )
            continue
        prov = _prov(
            g.gold_suite, tk, g.gold_graph_version, gold_src_node_id=src, gold_dst_node_id=dst
        )
        items.append(_make_item("A3_edge", prov, {}, context))

    # A3_match judges whether an ASKED QUESTION addresses a need, so it needs RUNS -- and
    # runs belong to the suite the rollouts were made on, not to whichever suite `--a3-suite`
    # borrowed graphs from for the node/edge judgments. Measured: matches.parquet holds 3,148
    # musique rows over 466 run directories against 12 for wiki2, so following `--a3-suite`
    # produced ZERO match items and starved matcher kappa -- the one preregistered gate human
    # annotation can still open, G-M1 being blocked upstream on a miner that has admitted
    # nothing. The matcher is one instrument on every suite, so its agreement with a person is
    # measured where the rollouts actually are.
    _match_graphs = graphs if match_graphs is None else match_graphs
    _match_answers = answers if match_answers is None else match_answers

    # ---- match: stratified ACROSS match_kind, near-miss `none` rows included on purpose --
    # they are what makes a matcher-generosity kappa meaningful rather than a tautology over
    # rows the matcher already called a hit.
    # FILTERED BEFORE STRATIFYING, not after drawing. `matches.parquet` holds every suite's
    # rows in one table -- measured, musique is 3,148 of 21,994 and drgym 16,704 -- so a draw
    # over the whole table spends six samples in seven on rows whose graphs were never loaded.
    # Each of those is counted as unresolvable, which is honest about the failure and silent
    # about the cause: asking for 10 match items returned 2. Excluding them from the POOL is
    # what makes the quota a draw from the population the instrument is about.
    by_kind: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in matches_rows:
        if match_suite and str(row.get("suite_id") or "") != match_suite:
            continue
        # A `dev-` prefix means the tree was dirty when the run was made. `report.ELIGIBLE`
        # excludes those mechanically (`run_id LIKE 'dev-%'`), and D9 makes that a rule rather
        # than a preference, so a judgment collected about one cannot back a gate. A4 and A5
        # already refuse them; this sampler did not, and a real export drew 127 of 211
        # A3_match items from dev- runs -- 60% of the evidence for matcher kappa, the one
        # preregistered gate annotation can still open.
        if str(row.get("run_id") or "").startswith("dev-"):
            stats["a3_match_dev_run"] = stats.get("a3_match_dev_run", 0) + 1
            continue
        by_kind[str(row.get("match_kind"))].append(row)
    kinds = sorted(k for k in by_kind if by_kind[k])
    quota = {k: n_match // len(kinds) for k in kinds} if kinds else {}
    for k in kinds[: n_match - sum(quota.values())]:
        quota[k] += 1
    per_run_used: dict[str, int] = {}
    for kind in kinds:
        pool = sorted(by_kind[kind], key=lambda r: (str(r.get("run_id")), str(r.get("node_id"))))
        for row in rng.sample(pool, min(quota.get(kind, 0), len(pool))):
            made = _a3_match_items(
                row,
                _match_graphs,
                runs_root,
                per_run_used=per_run_used,
                fanout_cap=match_fanout_cap,
                stats=stats,
                answers=_match_answers.get(str(row.get("task_id", "")), {}),
            )
            if not made:
                stats["a3_match_unresolvable"] = stats.get("a3_match_unresolvable", 0) + 1
                continue
            for it, key_entry in made:
                items.append(it)
                key_entries[it["item_id"]] = key_entry

    return items, key_entries


def sample_a3_missing_items(
    graphs: Mapping[str, Any],
    cache: Any,
    task_keys: Sequence[str],
    *,
    answers: Mapping[str, Mapping[str, str]] | None = None,
    stats: dict[str, int] | None = None,
) -> list[dict]:
    """One item per task: the full mined-node text list, read-only, plus a free-text box. The
    ONLY instrument that makes `node_recall` computable -- see `pi_eval.annotate.gate_numbers`.

    Like A1, this is one item per TASK carrying every node's text at once, so one unresolvable
    node refuses the whole item rather than silently shortening the list an annotator is meant
    to treat as exhaustive."""
    from pi_eval.build.musique_build import UnresolvedPlaceholder, resolve_placeholders

    answers = answers or {}
    stats = stats if stats is not None else {}
    items = []
    for tk in sorted(set(task_keys)):
        g = graphs.get(tk)
        if g is None:
            continue
        question = str(cache.view(g.gold_suite, tk).question) if cache is not None else ""
        task_answers = answers.get(tk, {})
        try:
            node_list = [resolve_placeholders(n.gold_text, task_answers) for n in g.gold_nodes]
        except UnresolvedPlaceholder:
            stats["a3_missing_unresolved_placeholder"] = (
                stats.get("a3_missing_unresolved_placeholder", 0) + 1
            )
            continue
        context = {"question": question, "node_list": node_list}
        prov = _prov(g.gold_suite, tk, g.gold_graph_version)
        items.append(_make_item("A3_missing", prov, {}, context))
    return items


# --------------------------------------------------------------------------- planted attention checks


def sample_attention_items(
    graphs: Mapping[str, Any],
    rng: random.Random,
    *,
    n: int,
    cache: Any | None = None,
    answers: Mapping[str, Mapping[str, str]] | None = None,
    stats: dict[str, int] | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    """`n` A3_node-shaped items pairing task A's question with a candidate need mined for an
    UNRELATED task B, so the honest verdict is always "not_a_need" -- they exist to catch an
    annotator, human or model, who is not reading.

    HARD CONSTRAINT: an attention-check item may never enter a measurement. The expected
    answer is recorded ONLY in the returned key (`{"attention_check": {"expected":
    "not_a_need"}}`), never on the item itself -- a shipped tell would let a careful annotator
    read the answer off the item's shape rather than its content, defeating the point, and
    `pi_eval.annotate._collect` is what enforces the other half: it skips any unit whose key
    entry carries `attention_check`, so these items form no consensus unit, no alpha
    contribution and no gate number no matter what an annotator answers.
    """
    from pi_eval.build.musique_build import UnresolvedPlaceholder, resolve_placeholders

    answers = answers or {}
    stats = stats if stats is not None else {}
    candidates = sorted(tk for tk, g in graphs.items() if g.gold_nodes)
    if len(candidates) < 2:
        return [], {}

    order = list(candidates)
    rng.shuffle(order)
    # Pair each task with its neighbour in the shuffled order (wrapping around), which
    # guarantees every pair is genuinely two DIFFERENT tasks with no extra bookkeeping.
    pairs = [(order[i], order[(i + 1) % len(order)]) for i in range(len(order))][:n]

    items: list[dict] = []
    key_entries: dict[str, dict] = {}
    for task_key, other_key in pairs:
        g_task = graphs[task_key]
        g_other = graphs[other_key]
        foreign_node = rng.choice(list(g_other.gold_nodes))
        question = (
            str(cache.view(g_task.gold_suite, task_key).question) if cache is not None else ""
        )
        try:
            node_text = resolve_placeholders(foreign_node.gold_text, answers.get(other_key, {}))
        except UnresolvedPlaceholder:
            stats["attention_unresolved_placeholder"] = (
                stats.get("attention_unresolved_placeholder", 0) + 1
            )
            continue
        context = {"question": question, "node_text": node_text, "evidence": []}
        prov = _prov(
            g_task.gold_suite,
            task_key,
            g_task.gold_graph_version,
            gold_node_id=f"ATTN:{other_key}/{foreign_node.gold_node_id}",
        )
        it = _make_item("A3_node", prov, {}, context)
        items.append(it)
        key_entries[it["item_id"]] = {"attention_check": {"expected": "not_a_need"}}
    return items, key_entries


# --------------------------------------------------------------------------- A4: latent need


def sample_a4_items(
    rows: Sequence[Mapping[str, Any]],
    graphs: Mapping[str, Any],
    rng: random.Random,
    *,
    n: int,
    cache: Any | None = None,
    stats: dict[str, int] | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    stats = stats if stats is not None else {}
    ask_rows: list[tuple[Mapping[str, Any], str]] = []
    for r in rows:
        q = _action_question(r.get("action_json", ""))
        if q is None:
            stats["a4_stop_turn"] = stats.get("a4_stop_turn", 0) + 1
            continue
        ask_rows.append((r, q))

    # A MIX, not a latent-first ranking. Measured: the old code sorted by (is_latent,
    # latent_depth, ...) reverse=True and took the top n -- deliberately latent-first, so an
    # 8-item pilot came back 8/8 mechanically latent. An agreement statistic computed against a
    # constant is not a measurement (the model agrees with "latent" 8/8 times, but there was
    # never a "stated" case for it to have gotten wrong), and the latent/stated CONTRAST is the
    # whole construct A4 exists to validate. So the two populations are drawn separately, each
    # targeting half of `n`, and only WITHIN the latent half is depth still used to prefer the
    # deep, rare decision points the campaign cares about -- see the comment on `latent_pool`.
    n_latent_target = n // 2
    n_stated_target = n - n_latent_target

    latent_pool = sorted(
        (t for t in ask_rows if bool(t[0].get("is_latent", False))),
        key=lambda t: (
            int(t[0].get("latent_depth", -1)),
            str(t[0].get("run_id")),
            int(t[0].get("turn_idx", 0)),
        ),
        reverse=True,
    )
    stated_pool = sorted(
        (t for t in ask_rows if not bool(t[0].get("is_latent", False))),
        key=lambda t: (str(t[0].get("run_id")), int(t[0].get("turn_idx", 0))),
    )

    picked_latent = latent_pool[:n_latent_target]
    picked_stated = stated_pool[:n_stated_target]
    shortfall_latent = n_latent_target - len(picked_latent)
    shortfall_stated = n_stated_target - len(picked_stated)
    # Fall back gracefully when one side is short, backfilling from the other side's remaining
    # supply rather than shipping fewer than `n` items -- and count what could not be honored,
    # so a campaign that quietly ran out of latent (or stated) decision points is visible in
    # the manifest rather than silently reverting to the old latent-first skew.
    if shortfall_latent > 0:
        stats["a4_latent_shortfall"] = stats.get("a4_latent_shortfall", 0) + shortfall_latent
        picked_stated = (
            picked_stated + stated_pool[n_stated_target : n_stated_target + shortfall_latent]
        )
    if shortfall_stated > 0:
        stats["a4_stated_shortfall"] = stats.get("a4_stated_shortfall", 0) + shortfall_stated
        picked_latent = (
            picked_latent + latent_pool[n_latent_target : n_latent_target + shortfall_stated]
        )

    picked = sorted(
        (picked_latent + picked_stated)[:n],
        key=lambda t: (str(t[0].get("run_id")), int(t[0].get("turn_idx", 0))),
    )

    items: list[dict] = []
    key_entries: dict[str, dict] = {}
    for r, question in picked:
        task_key = str(r.get("task_id", ""))
        g = graphs.get(task_key)
        task_question = ""
        units: Mapping[str, Any] = {}
        if cache is not None and g is not None:
            task_question = str(cache.view(g.gold_suite, task_key).question)
            units = cache.units(g.gold_suite, task_key)
        context = {"question": task_question, "asked_question": question}
        parent_uids = r.get("parent_uids") or ()
        if parent_uids:
            # ONLY when the decision point self-reported parent_uids: this is what makes the
            # `parent_uids_ok` sub-judgment answerable, and its absence IS the datum for a
            # depth-0 (stated-in-task) turn, so it must not appear as an empty list there.
            context["parent_units"] = [
                {
                    "uid": u,
                    "title": getattr(units.get(u), "title", ""),
                    "text": getattr(units.get(u), "text", ""),
                }
                for u in parent_uids
            ]
        prov = _prov(
            str(r.get("suite_id", "")),
            task_key,
            str(r.get("graph_version", "")),
            run_id=str(r.get("run_id", "")),
            turn_idx=int(r.get("turn_idx", 0)),
        )
        it = _make_item("A4", prov, {}, context)
        items.append(it)
        key_entries[it["item_id"]] = {
            "mechanical_is_latent": bool(r.get("is_latent", False)),
            "mechanical_latent_depth": int(r.get("latent_depth", -1)),
        }
    return items, key_entries


# --------------------------------------------------------------------------- A5: was the stop right?


def sample_a5_items(
    graphs: Mapping[str, Any],
    matches_rows: Sequence[Mapping[str, Any]],
    runs_root: Path | None,
    cache: Any | None,
    rng: random.Random,
    *,
    n: int,
    answers: Mapping[str, Mapping[str, str]] | None = None,
    stats: dict[str, int] | None = None,
    blind: bool = False,
) -> tuple[list[dict], dict[str, dict]]:
    """A5: was stopping here the right call? -- `pinq` treats STOP as a decision on the same
    footing as asking, and `stop_overshoot`/`stop_undershoot` report it in question units, but
    nothing before this asked a PERSON. Built from the run's END STATE, not a turn row: every
    recorded turn in this repository is an `ask` (see `pinq.loop`), so there is no per-turn
    "stop" row to sample the way A2/A4 sample turns.jsonl rows -- this samples RUN DIRECTORIES
    instead, one item per run.

    ONLY `stop_reason == "policy_stop"` runs. `budget` and `max_turns` are CAPS, not choices --
    there is no decision to judge where the loop was cut off rather than stopped on its own
    terms, and shipping one with a caveat nobody would read is worse than excluding it outright.
    Measured: of 1,817 runs, 988 ended `policy_stop` against 535 `budget` and 280 `max_turns`.

    Drawn from `graphs`/`runs_root` as given -- the CALLER passes the `--suite` (never
    `--a3-suite`) graphs and runs, because a run's own stop only exists on the suite that
    produced it.

    Candidates are the REQUIRED gold needs still unresolved when the run stopped: a need whose
    `matches.parquet` row for this run never reached `resolve`/`use` rank. `#N` placeholders are
    resolved via `resolve_placeholders` before a candidate ships -- a need nobody can read
    cannot be judged -- and, like A1's checklist, ONE unresolvable candidate refuses the WHOLE
    item rather than silently shortening the candidate list an annotator is meant to treat as
    exhaustive.
    """
    from pi_eval.build.musique_build import UnresolvedPlaceholder, resolve_placeholders
    from pi_eval.metrics import discovery, quality
    from pi_run.cmd_train import GOLD_EXPOSED_ARMS

    answers = answers or {}
    stats = stats if stats is not None else {}
    items: list[dict] = []
    key_entries: dict[str, dict] = {}
    if runs_root is None or not Path(runs_root).exists():
        return items, key_entries

    resolved_by_run: dict[str, set[str]] = defaultdict(set)
    for row in matches_rows:
        if str(row.get("match_kind")) in ("resolve", "use"):
            resolved_by_run[str(row.get("run_id"))].add(str(row.get("node_id")))

    pool: list[tuple[str, Mapping[str, Any], list[dict], str, str, Any, str]] = []
    for d in sorted(p for p in Path(runs_root).iterdir() if (p / "manifest.json").exists()):
        run_id = d.name
        if run_id.startswith("dev-"):
            stats["a5_dev_run"] = stats.get("a5_dev_run", 0) + 1
            continue
        try:
            manifest = _read_json(d / "manifest.json")
        except (OSError, ValueError):
            stats["a5_unreadable_manifest"] = stats.get("a5_unreadable_manifest", 0) + 1
            continue
        if str(manifest.get("arm_id", "")) in GOLD_EXPOSED_ARMS:
            stats["a5_gold_exposed"] = stats.get("a5_gold_exposed", 0) + 1
            continue
        status: dict = {}
        if (d / "status.json").exists():
            try:
                status = _read_json(d / "status.json")
            except (OSError, ValueError):
                status = {}
        stop_reason = str(status.get("stop_reason") or "")
        if stop_reason != "policy_stop":
            stats["a5_not_policy_stop"] = stats.get("a5_not_policy_stop", 0) + 1
            continue
        task_key = str(manifest.get("task_id") or "")
        graph = graphs.get(task_key)
        if graph is None:
            stats["a5_no_gold_graph"] = stats.get("a5_no_gold_graph", 0) + 1
            continue
        turns = sorted(_load_jsonl(d / "turns.jsonl"), key=lambda t: int(t["turn_idx"]))
        if not turns:
            stats["a5_no_turns"] = stats.get("a5_no_turns", 0) + 1
            continue
        outcome: dict = {}
        if (d / "outcome.json").exists():
            try:
                outcome = _read_json(d / "outcome.json")
            except (OSError, ValueError):
                outcome = {}
        ans = outcome.get("answer") if isinstance(outcome, Mapping) else None
        answer_text = str((ans or {}).get("text") or "") if isinstance(ans, Mapping) else ""
        if not answer_text:
            stats["a5_no_answer"] = stats.get("a5_no_answer", 0) + 1
            continue
        pool.append((run_id, manifest, turns, answer_text, task_key, graph, stop_reason))

    picked = sorted(rng.sample(pool, min(n, len(pool))), key=lambda t: t[0])

    for run_id, manifest, turns, answer_text, task_key, graph, stop_reason in picked:
        suite = str(manifest.get("suite_id") or "")
        task_answers = answers.get(task_key, {})
        try:
            candidates = [
                {
                    "node_id": nd.gold_node_id,
                    "text": resolve_placeholders(nd.gold_text, task_answers),
                }
                for nd in graph.required()
                if nd.gold_node_id not in resolved_by_run.get(run_id, set())
            ]
        except UnresolvedPlaceholder:
            stats["a5_unresolved_placeholder"] = stats.get("a5_unresolved_placeholder", 0) + 1
            continue
        rng.shuffle(candidates)

        question = str(cache.view(suite, task_key).question) if cache is not None else ""
        units = cache.units(suite, task_key) if cache is not None else {}
        retrieved_uids = sorted({u for t in turns for u in (t.get("retrieved_uids") or ())})
        evidence = [
            {
                "uid": u,
                "title": getattr(units.get(u), "title", ""),
                "text": getattr(units.get(u), "text", ""),
            }
            for u in retrieved_uids
        ]
        history = [
            {"q": str(t.get("question", "")), "a": str(t.get("response_text") or "")} for t in turns
        ]

        gold_uids = frozenset(u for nd in graph.required() for u in nd.gold_ev_uids)
        cov = discovery.evidence_coverage(frozenset(retrieved_uids), gold_uids)
        correct = (
            quality.contains_answer(answer_text, graph.answer, graph.gold_aliases)
            if graph.answer
            else None
        )

        context = {
            "question": question,
            "evidence": evidence,
            "history": history,
        }
        # BLIND: the candidate list is the gold frontier and showing it decides the verdict
        # (see `_build_a5`), and the final answer lets a rater judge the stop by whether the
        # answer looks right. BOTH are dropped from the shipped ITEM, not merely from the
        # prompt, so neither can be recovered by anything downstream. They move together:
        # dropping one and keeping the other produces a bundle whose name says blind and whose
        # bytes are half sighted, which `bundle_shape_errors` now refuses outright.
        if blind:
            payload = {"blind": True}
        else:
            payload = {"candidates": candidates}
            context["answer"] = answer_text
        prov = _prov(suite, task_key, graph.gold_graph_version, run_id=run_id)
        it = _make_item("A5", prov, payload, context)
        items.append(it)
        # THE BLINDING. None of these four facts may reach `context`/`payload` -- an
        # annotator told the answer was wrong, or shown how many needs the mechanical pipeline
        # already knows are unresolved, will find something missing; that is anchoring, not
        # judgment. See test_a5_item_hides_whether_the_answer_was_correct.
        key_entries[it["item_id"]] = {
            "answer_correct": correct,
            "evidence_coverage": cov,
            "stop_reason": stop_reason,
            "n_unresolved": len(candidates),
        }

    return items, key_entries


# --------------------------------------------------------------------------- canary firewall


def _canary_hits(bundle: Mapping[str, Any], key: Mapping[str, Any], root: Path) -> list[str]:
    from pi_eval import canary as canary_mod

    known = canary_mod.load(root)
    hits: list[str] = []
    for label, obj in (("bundle", bundle), ("key", key)):
        text = json.dumps(obj, sort_keys=True, ensure_ascii=False)
        for h in canary_mod.scan_text(text, known, where=label):
            hits.append(f"{label}: a REGISTERED canary reached the {label} near {h.excerpt!r}")
        unregistered = sorted(set(_CANARY_RE.findall(text)) - known)
        for m in unregistered:
            hits.append(f"{label}: an unregistered canary-shaped token {m} reached the {label}")
    return hits


def _assert_no_canary(bundle: Mapping[str, Any], key: Mapping[str, Any], root: Path) -> None:
    hits = _canary_hits(bundle, key, root)
    if hits:
        raise CanaryInBundle("\n".join(hits))


# --------------------------------------------------------------------------- pi annotate export


def _read_matches(path: Path) -> list[dict]:
    if not path.exists():
        return []
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _collect_a4_rows(
    runs_root: Path, root: Path, *, graph_version: str
) -> tuple[list[dict], dict[str, int]]:
    if not runs_root.exists():
        return [], {}
    weights = _train("reward").RewardWeights()
    rows, skipped = collect_rows(
        runs_root, root, graph_version=graph_version, weights=weights, verify_state=True
    )
    skipped = dict(skipped)
    dev_rows = [r for r in rows if str(r["run_id"]).startswith("dev-")]
    if dev_rows:
        skipped["dev_run"] = skipped.get("dev_run", 0) + len(dev_rows)
    rows = [r for r in rows if not str(r["run_id"]).startswith("dev-")]
    return rows, skipped


def cmd_annotate_export(a: argparse.Namespace) -> int:
    rc = _require_gold(a)
    if rc is not None:
        return rc
    from pi_eval.gold import gold_root, load_graphs
    from pi_run.manifest import git_info, repo_root

    root = Path(a.root).resolve() if a.root else repo_root()
    runs_root = Path(a.runs_root).resolve() if a.runs_root else root / "runs"

    graphs = load_graphs(a.suite, a.graph_version)
    if not graphs:
        print(
            f"no gold graphs for suite={a.suite!r} graph_version={a.graph_version!r}",
            file=sys.stderr,
        )
        return 1

    # `--a3-suite` lets the A3 family (A3_node/A3_edge/A3_match/A3_missing, plus the attention
    # checks planted alongside A3_node) be drawn from a DIFFERENT suite than A1/A2/A4. Measured:
    # every one of musique's 2,660 nodes is `gold_partition == "required"` -- there are no
    # `dropped` nodes, so A3_node's planted-foil check can never fire and node precision can
    # only ever be 1.0. wiki2 (30,098 required / 1,022 optional) and strategyqa (5,473 / 1,247)
    # carry both partitions and make the judgment non-trivial. Reuses `graphs` when the two
    # suites are the same, rather than loading gold twice.
    a3_suite = a.a3_suite or a.suite
    a3_graphs = graphs if a3_suite == a.suite else load_graphs(a3_suite, a.graph_version)
    if not a3_graphs:
        print(
            f"no gold graphs for a3_suite={a3_suite!r} graph_version={a.graph_version!r}",
            file=sys.stderr,
        )
        return 1

    cache = SuiteCache(root)
    rng = random.Random(a.seed)
    stats: dict[str, int] = {}

    # MuSiQue-only, and read once for the whole export: `gold_text` on any node past depth 0
    # carries a mechanical `#N` (see `pi_eval.build.musique_build._graph`), and every sampler
    # below resolves it before an annotator ever sees the text. Suites other than musique pass
    # an empty map, under which `resolve_placeholders` is a no-op. Computed once per suite, so
    # `--suite musique --a3-suite musique` (the default) does not read the raw files twice.
    def _musique_answers(suite: str, task_keys: Any) -> dict[str, dict[str, str]]:
        if suite != "musique":
            return {}
        from pi_eval.build.musique_build import load_subanswers

        return load_subanswers(root / "data" / "raw" / "musique", task_keys)

    answers = _musique_answers(a.suite, graphs.keys())
    a3_answers = answers if a3_suite == a.suite else _musique_answers(a3_suite, a3_graphs.keys())

    a1_items = sample_a1_items(
        graphs, rng, n_tasks=a.n_a1, cache=cache, answers=answers, stats=stats
    )

    pairs_path = Path(a.pairs).resolve() if a.pairs else root / "data" / "rl" / "pairs.jsonl"
    pairs_rows = _load_jsonl(pairs_path)
    a2_items, a2_key = sample_a2_items(
        pairs_rows,
        cache,
        rng,
        n=a.n_a2,
        runs_root=runs_root,
        margin_threshold=_pairs_margin_threshold(pairs_path),
        graph_version=a.graph_version,
        stats=stats,
    )

    parquet_dir = Path(a.parquet).resolve() if a.parquet else root / "scores" / "parquet"
    matches_rows = _read_matches(parquet_dir / "matches.parquet")
    a3_items, a3_key = sample_a3_items(
        a3_graphs,
        matches_rows,
        cache,
        rng,
        n_node=a.n_a3_node,
        n_edge=a.n_a3_edge,
        n_match=a.n_a3_match,
        runs_root=runs_root,
        stats=stats,
        answers=a3_answers,
        match_graphs=graphs,
        match_answers=answers,
        match_suite=a.suite,
    )
    missing_task_keys = sorted(
        {it["provenance"]["task_key"] for it in a3_items if it["task_type"] == "A3_node"}
    )
    a3_missing_items = sample_a3_missing_items(
        a3_graphs, cache, missing_task_keys, answers=a3_answers, stats=stats
    )

    a4_rows, refused = _collect_a4_rows(runs_root, root, graph_version=a.graph_version)
    a4_items, a4_key = sample_a4_items(a4_rows, graphs, rng, n=a.n_a4, cache=cache, stats=stats)

    # A5 draws from `graphs`/`runs_root` -- the RUN-BEARING `--suite`, never `--a3-suite` --
    # because a run's own stop only exists on the suite that produced it.
    a5_items, a5_key = sample_a5_items(
        graphs,
        matches_rows,
        runs_root,
        cache,
        rng,
        n=a.n_a5,
        answers=answers,
        stats=stats,
        blind=bool(getattr(a, "a5_blind", False)),
    )

    attention_items, attention_key = sample_attention_items(
        a3_graphs, rng, n=a.n_attention, cache=cache, answers=a3_answers, stats=stats
    )

    # LAST, and from the SAME `pairs_rows` A2 was drawn from -- A6 is A2's generalisation, not a
    # second source. Called after every pre-existing sampler so that adding this instrument does
    # not shift the shared `rng` stream underneath them: a re-export at one seed still produces
    # exactly the A1/A2/A3/A4/A5 items it did before, with A6's on top. A7 extends the same
    # rule: it runs after A6 (and its foils after it), so A1..A6 are seed-stable under any
    # `--n-a7`/`--n-a7-foils`.
    a6_items, a6_key = sample_a6_items(
        pairs_rows,
        cache,
        rng,
        n=a.n_a6,
        runs_root=runs_root,
        graph_version=a.graph_version,
        stats=stats,
    )

    a7_items, a7_key = sample_a7_items(
        cache,
        pairs_rows,
        runs_root,
        n=a.n_a7,
        rng=rng,
        stats=stats,
        graph_version=a.graph_version,
    )
    a7_foil_items, a7_foil_key = sample_a7_foils(
        cache, pairs_rows, n=a.n_a7_foils, rng=rng, graph_version=a.graph_version
    )

    all_items = (
        a1_items
        + a2_items
        + a3_items
        + a3_missing_items
        + a4_items
        + a5_items
        + a6_items
        + a7_items
        + a7_foil_items
        + attention_items
    )
    key_items = {
        **a2_key,
        **a3_key,
        **a4_key,
        **a5_key,
        **a6_key,
        **a7_key,
        **a7_foil_key,
        **attention_key,
    }

    # Two samplers can independently draw the same item -- measured, an export produced 90
    # items over 89 distinct ids because two A2 pairs rendered identically. A repeat is
    # BENIGN (the id is a content hash, so they are the same question) and dropping it is the
    # right response, but doing it silently is not: an annotation slot was spent on a question
    # already asked, and the count belongs beside every other refusal. `item_set_hash` cannot
    # notice, being computed over a set; `bundle_shape_errors` refuses on a duplicate, so
    # this must run BEFORE the shape gate or a benign repeat would fail the whole export.
    deduped: list[dict] = []
    seen_item_ids: set[str] = set()
    for it in all_items:
        if it["item_id"] in seen_item_ids:
            stats["duplicate_item_dropped"] = stats.get("duplicate_item_dropped", 0) + 1
            continue
        seen_item_ids.add(it["item_id"])
        deduped.append(it)
    all_items = deduped

    ish = item_set_hash(all_items)
    bundle_id = f"{a.suite}-{a.seed}-{ish[:12]}"
    counts = {tt: sum(1 for it in all_items if it["task_type"] == tt) for tt in TASK_TYPES}
    refused_all = {**refused, **stats}
    gi = git_info(str(root))
    manifest = {
        "bundle_id": bundle_id,
        "tool_version": TOOL_VERSION,
        "suite": a.suite,
        "a3_suite": a3_suite,
        "graph_version": a.graph_version,
        "gold_corpus_hash": next(iter(graphs.values())).gold_corpus_hash if graphs else "",
        "code_version": gi.sha,
        "seed": a.seed,
        "counts": counts,
        "item_set_hash": ish,
        "refused": refused_all,
    }
    bundle = {"manifest": manifest, "items": all_items}
    key = {"manifest": {"bundle_id": bundle_id, "item_set_hash": ish}, "items": key_items}

    shape_errs = bundle_shape_errors(bundle)
    if shape_errs:
        print(
            "REFUSED: an item violates the context/payload contract. Nothing written.",
            file=sys.stderr,
        )
        for e in shape_errs:
            print(e, file=sys.stderr)
        return 2

    try:
        _assert_no_canary(bundle, key, root)
    except CanaryInBundle as exc:
        print("REFUSED: a canary reached the bundle or key. Nothing written.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2

    out_dir = Path(a.out).resolve() if a.out else gold_root() / "human" / a.suite / "bundles"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / f"{bundle_id}.json"
    key_path = out_dir / f"{bundle_id}.key.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True))
    key_path.write_text(json.dumps(key, indent=2, sort_keys=True))

    print(f"bundle {bundle_path}  items={len(all_items)}  {counts}")
    print(f"key    {key_path}  (never ships)")
    if refused_all:
        print(f"  refused/fallbacks: {refused_all}")
    return 0


# --------------------------------------------------------------------------- pi annotate import


_GATE_FLAGS: tuple[tuple[str, str], ...] = (
    ("node_recall", "--node-recall"),
    ("edge_precision", "--edge-precision"),
    ("matcher_kappa", "--matcher-kappa"),
)


_N_OF = {"edge_precision": "n_edge_precision", "matcher_kappa": "n_matcher_kappa"}


def _gate_n(key: str, gates: Mapping[str, Any]) -> int | None:
    """Recall's denominator is a sum -- the confirmed mined needs plus the adjudicated ones
    they missed -- so it has no single count field to read."""
    if key == "node_recall":
        confirmed, missing = gates.get("n_node_confirmed"), gates.get("n_missing_adjudicated")
        if confirmed is None or missing is None:
            return None
        return int(confirmed) + int(missing)
    n = gates.get(_N_OF.get(key, ""))
    return None if n is None else int(n)


def _validate_line(suite: str, gates: Mapping[str, Any]) -> str:
    """The ready-to-paste `pi gold validate` invocation, each number carrying its n.

    A NaN gate is OMITTED, never printed as a number: `gate_report` compares with `>=` and
    `nan >= 0.75` is False, so a NaN handed to it as a flag would print FAIL where the truth
    is NOT RUN.

    The counts ride along because a gate value carries no power information and this line is
    pasted straight into a command that answers PASS or FAIL. A matcher kappa of -0.25 over
    two adjudicated pairs is nearer to not-measured than to failed, and a bare
    `--matcher-kappa -0.2500` cannot say which. No threshold is imposed here -- picking one
    would be inventing a constant -- the reader is simply not allowed to miss the denominator.
    """
    parts = [f"pi gold validate --suite {suite}"]
    notes: list[str] = []
    for key, flag in _GATE_FLAGS:
        v = gates.get(key)
        if v is None or is_absent(v):
            continue
        parts.append(f"{flag} {float(v):.4f}")
        n = _gate_n(key, gates)
        if n is not None:
            notes.append(f"{flag.lstrip('-')} n={n}")
    line = " ".join(parts)
    return f"{line}\n  # {', '.join(notes)}" if notes else line


def _item_id_of(unit_id: str) -> str:
    return unit_id.split("/")[0]


def _unit_row(u: Any) -> dict:
    return {
        "unit_id": u.unit_id,
        "kind": u.kind,
        "label": u.label,
        "n_raters": u.n_raters,
        "provenance": dict(u.provenance),
        "usefulness": u.usefulness,
    }


def _disagreement_row(d: Mapping[str, Any], items: Mapping[str, Mapping[str, Any]]) -> dict:
    """A disagreement plus enough of the ITEM it came from that a third adjudication pass
    needs no repo access -- the queue is meant to be handed to someone, not to a checkout."""
    row = dict(d)
    item = items.get(_item_id_of(str(d["unit_id"])))
    if item is not None:
        row["item_context"] = item.get("context")
        row["item_payload"] = item.get("payload")
        row["item_task_type"] = item.get("task_type")
    return row


def _latent_adjudication_rows(cons: Any, key: Mapping[str, Any]) -> list[dict]:
    """A4 consensus vs the key's mechanical label. COMPARISON ONLY: never written back as a
    gold field, matching `pi_eval.annotate`'s own rule for A3 verdicts."""
    key_items = (key or {}).get("items") or {}
    out = []
    for u in cons.by_kind("A4"):
        ki = key_items.get(u.unit_id) or {}
        mech = ki.get("mechanical_is_latent")
        agree = None
        if mech is not None and u.label in ("stated_in_task", "evidence_only"):
            agree = (u.label == "evidence_only") == bool(mech)
        out.append(
            {
                "item_id": u.unit_id,
                "consensus_label": u.label,
                "mechanical_is_latent": mech,
                "mechanical_latent_depth": ki.get("mechanical_latent_depth"),
                "agree": agree,
            }
        )
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True, default=str) for r in rows) + ("\n" if rows else "")
    )


def cmd_annotate_import(a: argparse.Namespace) -> int:
    from pi_eval.score import _append, _judgment_key

    bundle = _read_json(Path(a.bundle))
    key = _read_json(Path(a.key))
    records: list[dict] = []
    for p in a.records:
        records.extend(_read_jsonl(Path(p)))
    # Before validation, because zero records are trivially VALID. Importing nothing would
    # otherwise mint an out-version whose gold is byte-identical to its input: a new
    # `scorer_hash`, and therefore a new provenance identity, asserting a measurement nobody
    # took. `_read_jsonl` returns [] for a missing path, so a typo lands here.
    if not records:
        print(
            f"no records read from {', '.join(str(p) for p in a.records)}. Importing zero "
            "annotations would mint a graph version carrying none; nothing written.",
            file=sys.stderr,
        )
        return 2

    errs = validate_records(bundle, records)
    if errs:
        for e in errs:
            print(f"INVALID: {e}", file=sys.stderr)
        print(f"{len(errs)} problem(s); nothing written.", file=sys.stderr)
        return 1

    rc = _require_gold(a)
    if rc is not None:
        return rc
    from pi_eval.build.common import write_graphs
    from pi_eval.gold import gold_root, load_graphs
    from pi_run.manifest import git_info, repo_root

    cons = consensus(
        bundle, records, key=key, min_annotators=a.min_annotators, allow_majority=a.allow_majority
    )
    iaa = iaa_report(bundle, records, key=key)
    gates = gate_numbers(cons, key, n_missing_adjudicated=a.n_missing_adjudicated)

    print(
        f"units resolved: {len(cons.units)}  disagreements: {len(cons.disagreements)}  missing_texts: {len(cons.missing)}"
    )
    for kind in sorted(iaa):
        r = iaa[kind]
        if "alpha" in r:
            a_val = "NaN" if is_absent(r["alpha"]) else f"{r['alpha']:.4f}"
            print(
                f"  alpha[{kind}] = {a_val}  (n_units={r['n_units']} n_multi_rated={r['n_multi_rated']})"
            )
    print()
    for gk, v in sorted(gates.items()):
        print(f"  {gk} = {'NaN (not run)' if isinstance(v, float) and is_absent(v) else v}")

    graph_version = str((bundle.get("manifest") or {}).get("graph_version") or "v1")
    root = Path(a.root).resolve() if a.root else repo_root()

    if a.dry_run:
        print("\n--dry-run: nothing written.")
        print(_validate_line(a.suite, gates))
        return 0

    # `load_graphs` resolves against PI_GOLD_ROOT; `write_graphs` resolves against `root` and
    # appends data/gold itself. Nothing connects the two, so a PI_GOLD_ROOT pointed at a
    # scratch tree reads that tree and writes the out-version into the REPOSITORY's gold --
    # both paths exist, neither call can see the other's, and the only symptom is a graph
    # version appearing somewhere nobody was looking. The out-version belongs in the tree its
    # base version came from.
    if gold_root().resolve() != (root / "data" / "gold").resolve():
        print(
            "gold is read from one tree and would be written to another; they must be the "
            f"same tree.\n  PI_GOLD_ROOT reads : {gold_root().resolve()}\n"
            f"  --root would write : {(root / 'data' / 'gold').resolve()}\n"
            "Nothing written. Pass --root so that <root>/data/gold IS the gold root.",
            file=sys.stderr,
        )
        return 2

    graphs = load_graphs(a.suite, graph_version)
    rows = merge_into_graphs(graphs, cons, out_version=a.out_version)
    out_path = write_graphs(root, a.suite, a.out_version, rows)
    print(f"\nwrote {out_path}")

    parquet_dir = Path(a.parquet).resolve() if a.parquet else root / "scores" / "parquet"
    j_rows = human_judgment_rows(bundle, records, key=key)
    if j_rows:
        parquet_dir.mkdir(parents=True, exist_ok=True)
        total, added = _append(
            parquet_dir / "judgments.parquet", "judgments", j_rows, _judgment_key
        )
        print(f"judgments.parquet: +{added} rows (now {total})")

    human_dir = gold_root() / "human" / a.suite
    human_dir.mkdir(parents=True, exist_ok=True)
    items = {str(i["item_id"]): i for i in bundle.get("items") or ()}
    _write_jsonl(human_dir / "annotations.jsonl", [_unit_row(u) for u in cons.units])
    _write_jsonl(
        human_dir / "disagreements.jsonl", [_disagreement_row(d, items) for d in cons.disagreements]
    )
    _write_jsonl(human_dir / "latent_adjudication.jsonl", _latent_adjudication_rows(cons, key))

    gates_out = {
        **gates,
        "bundle_id": (bundle.get("manifest") or {}).get("bundle_id"),
        "graph_version": graph_version,
        "out_version": a.out_version,
        "code_version": git_info(str(root)).sha,
    }
    (human_dir / "gates.json").write_text(
        json.dumps(gates_out, indent=2, sort_keys=True, default=str)
    )

    line = _validate_line(a.suite, gates)
    print(f"\n{line}")
    return 0


# --------------------------------------------------------------------------- pi annotate review


def _response_label(r: Mapping[str, Any]) -> Any:
    resp = r.get("response") or {}
    tt = str(r.get("task_type"))
    if tt == "A1":
        return tuple(sorted(str(x) for x in resp.get("ticked") or ()))
    if tt == "A2":
        return resp.get("choice")
    if tt == "A3_node":
        return resp.get("verdict")
    if tt == "A3_edge":
        return resp.get("holds")
    if tt == "A3_match":
        return resp.get("addresses")
    if tt == "A4":
        return resp.get("latency")
    if tt == "A5":
        return resp.get("verdict")
    if tt == "A6":
        # The whole tier map, canonicalised -- so `_straight_line_runs` flags a rater who
        # answers every ranking identically (all-one-tier, or the same c0..c4 order every time).
        tiers = resp.get("tiers")
        if not isinstance(tiers, Mapping):
            return None
        return tuple(sorted((str(k), v) for k, v in tiers.items()))
    if tt == "A7":
        # The preference, not the reaches pair: like A2's `choice`, it is the one field a
        # straight-line rater repeats across items.
        return resp.get("preference")
    return None


def _per_annotator_agreement(cons: Any) -> dict[str, dict]:
    """Agreement with the OTHERS, leaving the scored annotator out of the comparison.

    A majority computed over ALL raters, the annotator being scored included, is 1.0 by
    construction at two raters -- which is the pilot's configuration. They agree, and both
    match the majority they jointly formed; or they disagree, the unit has no majority and is
    dropped. There is no third case, so the statistic cannot take any other value, and it
    prints as a per-person quality score that is really a count of nothing.

    A unit where the OTHERS are split with no plurality is skipped rather than counted as a
    miss: there is nothing there to have agreed with.
    """
    hits: dict[str, list[bool]] = defaultdict(list)
    for votes in cons.votes.values():
        for ann, label in votes.items():
            others = [v for a, v in votes.items() if a != ann]
            if not others:
                continue
            counts = Counter(others)
            top, n_top = counts.most_common(1)[0]
            if list(counts.values()).count(n_top) != 1:
                continue
            hits[ann].append(label == top)
    return {
        ann: {"n": len(v), "agreement": sum(v) / len(v) if v else float("nan")}
        for ann, v in sorted(hits.items())
    }


def _pairwise_agreement(cons: Any) -> dict[str, dict]:
    pair_hits: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for votes in cons.votes.values():
        anns = sorted(votes)
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                x, y = anns[i], anns[j]
                pair_hits[(x, y)].append(votes[x] == votes[y])
    return {
        f"{x}:{y}": {"n": len(v), "agreement": sum(v) / len(v)}
        for (x, y), v in sorted(pair_hits.items())
        if v
    }


def _timing_report(records: Sequence[Mapping[str, Any]]) -> dict:
    """How long each annotator spent, and who spent suspiciously little.

    HUMANS ONLY. `elapsed_ms` on a person is attention; on a model it is a round trip, and a
    cached reply legitimately reads 0ms. Measured on a real pass, that put a `click_through`
    flag on every cache hit -- and a check that fires on normal operation is one a reader
    learns to skip past, taking the genuine click-throughs with it. Model pace is reported
    separately by the pass itself, which knows what it paid for.
    """
    records = [r for r in records if rater_kind(r) == "human"]
    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        by_key[(str(r.get("task_type")), str(r.get("annotator_id")))].append(
            float(r.get("elapsed_ms") or 0)
        )
    stats = {}
    for (tt, ann), vals in sorted(by_key.items()):
        med = statistics.median(vals)
        mad = statistics.median([abs(v - med) for v in vals]) if vals else 0.0
        stats[f"{tt}/{ann}"] = {"median_ms": med, "mad_ms": mad, "n": len(vals)}

    flagged = []
    for r in records:
        tt, ann = str(r.get("task_type")), str(r.get("annotator_id"))
        med = stats.get(f"{tt}/{ann}", {}).get("median_ms")
        v = float(r.get("elapsed_ms") or 0)
        if med:
            if v < 0.2 * med:
                flagged.append(
                    {
                        "record_id": r.get("record_id"),
                        "reason": "click_through",
                        "elapsed_ms": v,
                        "median_ms": med,
                    }
                )
            elif v > 10 * med:
                flagged.append(
                    {
                        "record_id": r.get("record_id"),
                        "reason": "outlier_slow",
                        "elapsed_ms": v,
                        "median_ms": med,
                    }
                )
    return {"per_task_annotator": stats, "flagged": flagged}


def _straight_line_runs(
    records: Sequence[Mapping[str, Any]], *, min_run: int = 5
) -> dict[str, int]:
    by_ann: dict[str, list] = defaultdict(list)
    for r in sorted(records, key=lambda r: (str(r.get("annotator_id")), str(r.get("ts") or ""))):
        by_ann[str(r.get("annotator_id"))].append(_response_label(r))
    out = {}
    for ann, labels in by_ann.items():
        best = cur = 1
        for i in range(1, len(labels)):
            cur = cur + 1 if labels[i] is not None and labels[i] == labels[i - 1] else 1
            best = max(best, cur)
        if best >= min_run:
            out[ann] = best
    return out


def _degenerate_report(
    bundle: Mapping[str, Any], records: Sequence[Mapping[str, Any]], key: Mapping[str, Any] | None
) -> dict:
    items = {str(i["item_id"]): i for i in bundle.get("items") or ()}
    key_items = (key or {}).get("items") or {}

    a1_rates: dict[str, list[float]] = defaultdict(list)
    for r in records:
        item = items.get(str(r.get("item_id")))
        if item is None or item["task_type"] != "A1":
            continue
        nodes = (item.get("payload") or {}).get("nodes") or ()
        ticked = {str(x) for x in (r.get("response") or {}).get("ticked") or ()}
        if nodes:
            a1_rates[str(r.get("annotator_id"))].append(len(ticked) / len(nodes))
    a1_tick_rate = {
        ann: {
            "mean_rate": sum(v) / len(v),
            "n": len(v),
            "flag": (sum(v) / len(v)) <= 0.02 or (sum(v) / len(v)) >= 0.98,
        }
        for ann, v in sorted(a1_rates.items())
        if v
    }

    foil_hits: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        item = items.get(str(r.get("item_id")))
        if item is None or item["task_type"] != "A3_node":
            continue
        if (key_items.get(str(item["item_id"])) or {}).get("gold_partition") != "dropped":
            continue
        verdict = (r.get("response") or {}).get("verdict")
        foil_hits[str(r.get("annotator_id"))].append(verdict in ("required", "optional"))
    foil_confirmation_rate = {
        ann: {"rate": sum(v) / len(v), "n": len(v)} for ann, v in sorted(foil_hits.items()) if v
    }

    return {
        "a1_tick_rate": a1_tick_rate,
        "a3_node_foil_confirmation_rate": foil_confirmation_rate,
        "straight_line_runs": _straight_line_runs(records),
    }


def _binomial_two_sided_p(k: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial p, computed in LOG space.

    `comb(n, i)` is an exact int and overflows the float conversion around n = 1000; the
    helper only ever saw A2's small samples until a 3,574-observation slot-bias check hit it.
    Working through `lgamma` keeps the same value at small n and simply does not overflow.
    """

    def log_pmf(i: int) -> float:
        return (
            lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * log(p0) + (n - i) * log1p(-p0)
        )

    obs = log_pmf(k)
    # Sum the probabilities no larger than the observed one, shifted by the max log-term so
    # the exponentials stay in range.
    terms = [log_pmf(i) for i in range(n + 1) if log_pmf(i) <= obs + 1e-9]
    if not terms:
        return 1.0
    m = max(terms)
    return min(1.0, exp(m) * sum(exp(t - m) for t in terms))


def _slot_bias(
    bundle: Mapping[str, Any], records: Sequence[Mapping[str, Any]], task_type: str
) -> dict:
    """How often a rater's label lands on the FIRST-SHOWN option, per judged field.

    `order` is a per-item coin flip, so a rater reading content splits evenly between the
    slots; a rater reading position does not. Measured on a real A7 pass, gpt-5.6-terra put
    `reaches_unstated` in slot B 466 times against slot A's 285 (binomial p ~ 4e-11) while
    catching 99 of 99 planted foils -- reading the task text correctly and still unable to
    judge two candidates independently of their order.

    PER-CANDIDATE FIELDS AS WELL AS THE PREFERENCE, because there the bias was larger, and
    because canonicalising through `order` HIDES it: the coin flip averages a slot effect
    across chosen and rejected, so the aggregate that gets reported looks healthy (chosen
    390 vs rejected 361) while every individual label carries a position component.

    SCOPED TO WHAT EACH INSTRUMENT CAN ACTUALLY EXPRESS -- this used to return `{}` for every
    task type but A7, which a caller cannot tell apart from "checked, clean". A6 shows up to
    `A6_MAX_CANDIDATES` candidates in one fixed list order (no per-rater re-randomisation), so
    its analogous statistic is `_a6_slot_bias` below: whether a rater's TOP TIER contains the
    first-LISTED candidate, against the TIE-AWARE null (that response's own top-tier size
    over that item's own shown k, not a single fixed p0 the way A7's 1/2 is -- see
    `_a6_slot_bias` for why the naive 1/k reading it also returns as `p_naive` runs hot). A5
    and A3_node show one item and one verdict -- there is no second slot to read a position
    effect against at all -- and say so explicitly rather than returning the same empty dict
    an unimplemented task type would.
    """
    if task_type in ("A5", "A3_node"):
        return {"applicable": False, "reason": "single-item instrument"}
    items = {str(i["item_id"]): i for i in bundle.get("items") or ()}
    if task_type == "A6":
        return {"top_tier": _a6_slot_bias(items, records)}
    fields = {"A7": {"reaches": ("a_reaches", "b_reaches"), "preference": None}}.get(task_type)
    if not fields:
        return {}
    out: dict[str, dict] = {}
    for name, pair in fields.items():
        n = n_first = 0
        for r in records:
            item = items.get(str(r.get("item_id")))
            if item is None or item["task_type"] != task_type:
                continue
            resp = r.get("response") or {}
            if pair is None:
                if resp.get("preference") not in ("a", "b"):
                    continue
                n += 1
                n_first += resp["preference"] == "a"
            else:
                # One observation per SIDE: the question is where a label lands, not which
                # side won, so both sides of every item count.
                for slot, field in zip(("a", "b"), pair):
                    if resp.get(field) is None:
                        continue
                    if resp[field] == "cant_tell":
                        continue
                    n += 1
                    n_first += slot == "a" and resp[field] == "reaches_unstated"
                    n_first += slot == "b" and resp[field] == "stays_stated"
        out[name] = (
            {"n": 0, "p_first": float("nan"), "binomial_p": float("nan")}
            if n == 0
            else {
                "n": n,
                "n_first": n_first,
                "p_first": n_first / n,
                "binomial_p": _binomial_two_sided_p(n_first, n),
            }
        )
    return out


def _a6_slot_bias(
    items: Mapping[str, Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> dict:
    """P(a rater's top tier contains the first-LISTED candidate) against the TIE-AWARE null:
    given that response's own top-tier size t (how many candidates it tied for the top) and
    that item's own shown count k, the no-position-effect rate is t/k, not 1/k.

    A6 has no A/B slot to coin-flip the way A7 does -- `payload["candidates"]` is one fixed
    list order, set once at build time (see `sample_a6_items`: shuffled, then never
    re-randomised per rater) -- so "first-shown" here means index 0 of that list, not any
    particular `candidate_id` spelling (`c0` is only ever first because the builder also
    happens to sort by it; this reads list order directly and does not assume that).

    WHY t/k AND NOT 1/k. 1/k is the chance rate for a UNIQUE winner; A6 permits ties (equal
    tiers are a real judgment, not a forced order -- see `a6_pair_labels`), and MEASURED
    2026-09-18 on ~/pi-corpus-backup/annotations-20260914 (a6/, a6b/, fixed1/, fixed2/,
    growth1/, 4,363 deduplicated A6 records), a tie at the top is the MAJORITY case: 58.8% of
    responses, mean top-tier size 2.05 of the 3-5 candidates shown. A rater who reads content
    perfectly and is fully blind to position lands a uniformly random position in the top
    tier at rate t/k, not 1/k, whenever the response ties t>1 candidates for the top, so 1/k
    understates the true chance rate on the majority of real responses and flags a clean
    rater. `expected` is the mean of t_i/k_i over the items this rater actually tiered, and
    `binomial_p` runs the same exact two-sided test A7 uses against that mean.

    `p_naive` reruns the SAME test against the OLD, MISSPECIFIED null (mean of 1/k_i,
    ignoring each response's own tie size) -- kept for comparison only, not as a second,
    equally valid reading. Pooled per rater, `p_naive` clears alpha=0.001 for all three raters
    in the corpus above (p in 1.7e-62..8.7e-142); `binomial_p` (this function's actual
    default) clears it for only one (gpt-oss-120b p=3.4e-05; gemini-3.1-pro p=0.11 and
    claude-sonnet-5 p=0.55 do not) -- agreeing with appendix_validity.tex's independently-
    derived pairwise statistic on the same instrument (the same rater at p=3.3e-13, the other
    two at chance). Do not re-derive a table from `p_naive`.

    A TIE FOR TOP COUNTS AS A HIT if the first-listed candidate is one of the tied-lowest
    tiers -- "top tier contains X", not "X is the unique top" -- matching how `a6_pair_labels`
    already treats equal tiers as a real judgment rather than noise.

    Requires the tier map to be complete enough to name a top tier AND to say where the
    first-listed candidate sits within it; a record silent on the first-listed candidate
    (or carrying no readable tiers at all) contributes nothing rather than a guess.
    """
    n = n_first = 0
    p0_tie_sum = 0.0
    p0_naive_sum = 0.0
    for r in records:
        item = items.get(str(r.get("item_id")))
        if item is None or item["task_type"] != "A6":
            continue
        candidates = (item.get("payload") or {}).get("candidates") or ()
        if len(candidates) < A6_MIN_CANDIDATES:
            continue
        first_id = str(candidates[0].get("candidate_id"))
        shown = {str(c.get("candidate_id")) for c in candidates}
        tiers = (r.get("response") or {}).get("tiers")
        if not isinstance(tiers, Mapping) or first_id not in tiers:
            continue
        rated = {
            str(cid): t
            for cid, t in tiers.items()
            if str(cid) in shown and not isinstance(t, bool) and isinstance(t, (int, float))
        }
        if first_id not in rated:
            continue
        top = min(rated.values())
        top_size = sum(1 for t in rated.values() if t == top)
        k = len(candidates)
        n += 1
        n_first += rated[first_id] == top
        p0_tie_sum += top_size / k
        p0_naive_sum += 1.0 / k
    if n == 0:
        return {
            "n": 0,
            "p_first": float("nan"),
            "expected": float("nan"),
            "binomial_p": float("nan"),
            "p_naive": float("nan"),
        }
    expected = p0_tie_sum / n
    expected_naive = p0_naive_sum / n
    return {
        "n": n,
        "n_first": n_first,
        "p_first": n_first / n,
        "expected": expected,
        "binomial_p": _binomial_two_sided_p(n_first, n, expected),
        "p_naive": _binomial_two_sided_p(n_first, n, expected_naive),
    }


def _a2_position_bias(bundle: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict:
    items = {str(i["item_id"]): i for i in bundle.get("items") or ()}
    n = n_first = 0
    for r in records:
        item = items.get(str(r.get("item_id")))
        if item is None or item["task_type"] != "A2":
            continue
        choice = (r.get("response") or {}).get("choice")
        if choice not in ("a", "b"):
            continue
        n += 1
        n_first += choice == "a"
    if n == 0:
        return {"n": 0, "p_first": float("nan"), "binomial_p": float("nan")}
    return {
        "n": n,
        "n_first": n_first,
        "p_first": n_first / n,
        "binomial_p": _binomial_two_sided_p(n_first, n),
    }


def _a2_sign_agreement(
    bundle: Mapping[str, Any], records: Sequence[Mapping[str, Any]], key: Mapping[str, Any] | None
) -> dict:
    """Human choice vs the AUTOMATIC margin's sign, overall and by margin bucket. The
    threshold splitting 'near' from 'clear' is a review-time diagnostic choice, not a gate
    input, so it is a fixed, documented constant rather than a CLI flag nobody would tune."""
    _NEAR_MARGIN = 0.2
    items = {str(i["item_id"]): i for i in bundle.get("items") or ()}
    key_items = (key or {}).get("items") or {}
    overall: list[bool] = []
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        item = items.get(str(r.get("item_id")))
        if item is None or item["task_type"] != "A2":
            continue
        ki = key_items.get(str(item["item_id"])) or {}
        order = ki.get("order")
        choice = (r.get("response") or {}).get("choice")
        if order not in ("ab", "ba") or choice not in ("a", "b"):
            continue
        auto_slot = "a" if order == "ab" else "b"  # the slot the automatic margin preferred
        agree = choice == auto_slot
        overall.append(agree)
        bucket = "near_threshold" if abs(float(ki.get("margin", 0.0))) < _NEAR_MARGIN else "clear"
        buckets[bucket].append(agree)

    def _rate(v: list[bool]) -> float:
        return sum(v) / len(v) if v else float("nan")

    return {
        "overall": {"n": len(overall), "agreement": _rate(overall)},
        "by_margin_bucket": {
            b: {"n": len(v), "agreement": _rate(v)} for b, v in sorted(buckets.items())
        },
    }


# An annotator whose tier maps are degenerate at this rate is flagged. Same constant and same
# reasoning as `_degenerate_report`'s A1 tick-rate bound (0.02/0.98): a threshold that fires on
# ordinary variation is one a reviewer learns to skip past, taking the real cases with it.
_A6_ONE_TIER_FLAG = 0.98


def _a6_tier_spread(
    bundle: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, dict]:
    """How many distinct tiers each annotator actually used, per item, averaged.

    An annotator who puts every candidate in ONE tier is the A1 tick-rate saturation failure in
    a new place, and it is invisible in an alpha: every pair they produce is "tie", two such
    annotators agree perfectly, and the campaign buys nothing. Reported as a mean spread AND as
    a one-tier rate because a rater who collapses SOME rankings is a different (and much more
    likely) case than one who collapses all of them.
    """
    items = {str(i["item_id"]): i for i in bundle.get("items") or ()}
    per_ann: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in records:
        item = items.get(str(r.get("item_id") or ""))
        if item is None or item["task_type"] != "A6":
            continue
        tiers = (r.get("response") or {}).get("tiers")
        if not isinstance(tiers, Mapping) or not tiers:
            continue
        per_ann[str(r.get("annotator_id"))].append(
            (len({str(v) for v in tiers.values()}), len(tiers))
        )
    out: dict[str, dict] = {}
    for ann, seen in sorted(per_ann.items()):
        one_tier = sum(1 for distinct, _n in seen if distinct == 1) / len(seen)
        out[ann] = {
            "n_items": len(seen),
            "mean_distinct_tiers": sum(d for d, _n in seen) / len(seen),
            "mean_candidates": sum(n for _d, n in seen) / len(seen),
            "one_tier_rate": one_tier,
            "flag": one_tier >= _A6_ONE_TIER_FLAG,
        }
    return out


def _a6_auto_agreement(
    bundle: Mapping[str, Any], records: Sequence[Mapping[str, Any]], key: Mapping[str, Any] | None
) -> dict:
    """Human tier order vs the AUTOMATIC margin, per derived pair -- A2's `_a2_sign_agreement`
    at A6's unit.

    Scored ONLY over pairs the pipeline ACTUALLY compared (`auto_pairs` in the key). A
    5-candidate state carries 4 pairs, not 10, because `sample-candidates` scores every branch
    against the winner; deriving the missing six from a summary score and calling a mismatch a
    disagreement would score the human against a comparison nobody made.

    A human TIE against a strict automatic preference is counted apart rather than folded in:
    the automatic side has no tie to express, so counting it as disagreement would penalise
    exactly the judgment the tier vocabulary exists to permit.
    """
    items = {str(i["item_id"]): i for i in bundle.get("items") or ()}
    key_items = (key or {}).get("items") or {}
    hits: list[bool] = []
    n_tie = 0
    n_uncompared = 0
    for r in sorted(records, key=lambda r: (str(r.get("item_id")), str(r.get("annotator_id")))):
        item = items.get(str(r.get("item_id") or ""))
        if item is None or item["task_type"] != "A6":
            continue
        auto = (key_items.get(str(item["item_id"])) or {}).get("auto_pairs") or {}
        for a, b, label in a6_pair_labels(item, r.get("response") or {}):
            auto_label = auto.get(f"{a}|{b}")
            if auto_label is None:
                n_uncompared += 1
                continue
            if label == "tie":
                n_tie += 1
                continue
            hits.append(label == auto_label)
    return {
        "overall": {
            "n": len(hits),
            "agreement": (sum(hits) / len(hits)) if hits else float("nan"),
        },
        "n_human_tie_where_auto_strict": n_tie,
        "n_pairs_auto_never_compared": n_uncompared,
    }


_RATIONALE_SAMPLE_LIMIT = 3


def _rationale_sample(
    cons: Any, records: Sequence[Mapping[str, Any]], *, limit: int = _RATIONALE_SAMPLE_LIMIT
) -> dict[str, list[dict]]:
    """Up to `limit` rationales per unit kind, drawn ONLY from records behind a unit `consensus`
    could not resolve to one label (`reason == "no_consensus"`; "insufficient_raters" is
    under-annotation, not disagreement -- a lone rater trivially agrees with themselves). A
    rationale attached to a unanimous item explains nothing anyone was unsure about; these are
    the ones worth a reviewer's time.

    Reads `rationale` off whatever RECORD cast that vote -- human or model alike, since both may
    carry one -- and skips a vote whose record has none (a human is allowed to omit it).
    """
    by_item_ann: dict[tuple[str, str], Mapping[str, Any]] = {}
    for r in records:
        by_item_ann[(str(r.get("item_id")), str(r.get("annotator_id")))] = r

    out: dict[str, list[dict]] = defaultdict(list)
    for d in sorted(cons.disagreements, key=lambda d: str(d["unit_id"])):
        if d.get("reason") != "no_consensus":
            continue
        kind = str(d["kind"])
        if len(out[kind]) >= limit:
            continue
        item_id = _item_id_of(str(d["unit_id"]))
        for ann in sorted((d.get("votes") or {}).keys()):
            if len(out[kind]) >= limit:
                break
            rec = by_item_ann.get((item_id, ann))
            rationale = (rec or {}).get("rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                continue
            out[kind].append(
                {
                    "unit_id": d["unit_id"],
                    "annotator_id": ann,
                    "label": (d.get("votes") or {}).get(ann),
                    "rationale": rationale,
                }
            )
    return dict(out)


def _a2_basis_distribution(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """How often each closed-vocabulary A2 `basis` was named, across every record that carries
    one (tie/both_bad items may have none -- see `pi_eval.annotate_llm.parse_reply`)."""
    counts: Counter[str] = Counter()
    for r in records:
        if str(r.get("task_type")) != "A2":
            continue
        basis = r.get("basis")
        if isinstance(basis, str) and basis:
            counts[basis] += 1
    return dict(sorted(counts.items()))


def _a5_verdict_distribution(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """A5 verdict counts per annotator -- the STOP judgment's only human validation, and the
    one place a reviewer can see WHICH WAY a rater's answers skew, not just whether they
    repeat. An annotator who answers `stopping_was_right` to every item is the A1 tick-rate
    saturation failure (`_degenerate_report`'s `a1_tick_rate`) in a new place; unlike that
    flag this reports raw counts across FOUR labels rather than one rate, because a skew
    toward `cant_tell` and a skew toward `stopping_was_right` are different failures a
    reviewer needs to tell apart. `_straight_line_runs` (fed by `_response_label`, which now
    reads A5's `verdict`) is what actually FLAGS a rater who repeats one answer -- this table
    is the context a reviewer reads alongside that flag, not a replacement for it.
    """
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for r in records:
        if str(r.get("task_type")) != "A5":
            continue
        verdict = (r.get("response") or {}).get("verdict")
        if isinstance(verdict, str) and verdict:
            counts[str(r.get("annotator_id"))][verdict] += 1
    return {ann: dict(sorted(c.items())) for ann, c in sorted(counts.items())}


def _attention_check_report(
    bundle: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    key: Mapping[str, Any] | None,
) -> dict[str, dict]:
    """Pass rate per annotator on planted attention checks -- read straight off the RAW
    records, never through `consensus`, which (by design, see `pi_eval.annotate._collect`)
    drops these units entirely. This is the one place their answers are allowed to be looked
    at: to flag an annotator who is not reading, not to feed any measurement."""
    items = {str(i["item_id"]): i for i in bundle.get("items") or ()}
    key_items = (key or {}).get("items") or {}
    hits: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        item = items.get(str(r.get("item_id") or ""))
        if item is None:
            continue
        attn = (key_items.get(str(item["item_id"])) or {}).get("attention_check")
        if not attn:
            continue
        expected = attn.get("expected")
        if isinstance(expected, Mapping):
            # A partial spec over `response`: an A7 foil pins one side's reaches judgment
            # ("the task question verbatim stays stated") and says nothing about the rest.
            resp = r.get("response") or {}
            hits[str(r.get("annotator_id"))].append(
                all(resp.get(k) == v for k, v in expected.items())
            )
            continue
        hits[str(r.get("annotator_id"))].append(_response_label(r) == expected)
    return {
        ann: {"n": len(v), "pass_rate": sum(v) / len(v) if v else float("nan")}
        for ann, v in sorted(hits.items())
    }


def review_report(
    bundle: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    key: Mapping[str, Any] | None = None,
) -> dict:
    # `consensus` here keeps its HUMAN_ONLY default, so a model dropped into the records file
    # cannot move any section above -- every one of them describes people. The model gets its
    # own section instead, computed by `llm_agreement` against the SAME human-only consensus,
    # and is added only when there is something to report: an "llm" key on a campaign with no
    # model records would read as "the model agreed with nobody" rather than "not run".
    cons = consensus(bundle, records, key=key, min_annotators=1)
    report = {
        "alpha": iaa_report(bundle, records, key=key),
        "per_annotator": _per_annotator_agreement(cons),
        "pairwise": _pairwise_agreement(cons),
        "timing": _timing_report(records),
        "degenerate": _degenerate_report(bundle, records, key),
        "a2_position_bias": _a2_position_bias(bundle, records),
        "a7_slot_bias": _slot_bias(bundle, records, "A7"),
        "a2_sign_agreement": _a2_sign_agreement(bundle, records, key),
        # Alongside A2's, not instead of it: A6 is A2's generalisation and a campaign carrying
        # both should let a reader compare the same diagnostic at both units.
        "a6_tier_spread": _a6_tier_spread(bundle, records),
        "a6_auto_agreement": _a6_auto_agreement(bundle, records, key),
        "rationale_sample": _rationale_sample(cons, records),
    }
    basis_dist = _a2_basis_distribution(records)
    if basis_dist:
        report["a2_basis_distribution"] = basis_dist
    a5_dist = _a5_verdict_distribution(records)
    if a5_dist:
        report["a5_verdict_distribution"] = a5_dist
    attention = _attention_check_report(bundle, records, key)
    if attention:
        report["attention_checks"] = attention
    if any(rater_kind(r) == "llm" for r in records):
        report["llm"] = llm_agreement(bundle, records, key=key)
    return report


def _print_review(report: Mapping[str, Any]) -> None:
    print("alpha per unit kind")
    for kind, r in sorted(report["alpha"].items()):
        if "alpha" in r:
            av = "NaN" if is_absent(r["alpha"]) else f"{r['alpha']:.4f}"
            print(
                f"  {kind:<16} alpha={av}  n_units={r['n_units']} n_multi_rated={r['n_multi_rated']}"
            )
    print("\nper-annotator agreement vs consensus")
    for ann, r in sorted(report["per_annotator"].items()):
        print(
            f"  {ann:<16} n={r['n']:<5} agreement={r['agreement']:.4f}"
            if r["n"]
            else f"  {ann}: n=0"
        )
    print("\npairwise agreement")
    for pair, r in sorted(report["pairwise"].items()):
        print(f"  {pair:<24} n={r['n']:<5} agreement={r['agreement']:.4f}")
    print("\ntiming outliers")
    for f in report["timing"]["flagged"]:
        print(
            f"  {f['reason']:<14} {f['record_id']}  elapsed={f['elapsed_ms']:.0f}ms  median={f['median_ms']:.0f}ms"
        )
    print("\ndegenerate patterns")
    for ann, r in sorted(report["degenerate"]["a1_tick_rate"].items()):
        flag = "  FLAG" if r["flag"] else ""
        print(f"  A1 tick-rate {ann:<16} {r['mean_rate']:.3f} (n={r['n']}){flag}")
    for ann, r in sorted(report["degenerate"]["a3_node_foil_confirmation_rate"].items()):
        print(f"  A3_node foil confirmation {ann:<16} {r['rate']:.3f} (n={r['n']})")
    for ann, run in sorted(report["degenerate"]["straight_line_runs"].items()):
        print(f"  straight-line run {ann:<16} {run}")
    sb = report.get("a7_slot_bias") or {}
    for _name, _st in sorted(sb.items()):
        if _st.get("n"):
            _flag = "  <-- READS THE SLOT, NOT THE CONTENT" if _st["binomial_p"] < 0.001 else ""
            print(
                f"\nA7 slot bias [{_name}]: P(first-shown)={_st['p_first']:.4f}  "
                f"n={_st['n']}  binomial p={_st['binomial_p']:.4g}{_flag}"
            )
    pb = report["a2_position_bias"]
    if pb["n"]:
        print(
            f"\nA2 position bias: P(first)={pb['p_first']:.4f}  n={pb['n']}  binomial p={pb['binomial_p']:.4g}"
        )
    sa = report["a2_sign_agreement"]["overall"]
    if sa["n"]:
        print(f"A2 human-vs-automatic sign agreement: {sa['agreement']:.4f}  n={sa['n']}")
    spread = report.get("a6_tier_spread") or {}
    if spread:
        print("\nA6 tier spread per annotator (one tier for every candidate is saturation)")
        for ann, r in sorted(spread.items()):
            flag = "  FLAG" if r["flag"] else ""
            print(
                f"  {ann:<16} mean_distinct_tiers={r['mean_distinct_tiers']:.2f}"
                f"/{r['mean_candidates']:.2f}  one_tier_rate={r['one_tier_rate']:.3f}"
                f"  n={r['n_items']}{flag}"
            )
    aa = (report.get("a6_auto_agreement") or {}).get("overall") or {}
    if aa.get("n"):
        print(
            f"A6 human-vs-automatic pair agreement: {aa['agreement']:.4f}  n={aa['n']}"
            f"  (human ties where the automatic side was strict: "
            f"{report['a6_auto_agreement']['n_human_tie_where_auto_strict']}; pairs the "
            f"pipeline never compared: "
            f"{report['a6_auto_agreement']['n_pairs_auto_never_compared']})"
        )
    if report.get("rationale_sample"):
        print("\nrationale sample (from disagreements only)")
        for kind, items in sorted(report["rationale_sample"].items()):
            print(f"  {kind}:")
            for it in items:
                print(f"    [{it['annotator_id']}={it['label']}] {it['rationale']}")
    if report.get("a2_basis_distribution"):
        print("\nA2 basis distribution")
        for basis, n in report["a2_basis_distribution"].items():
            print(f"  {basis:<28} {n}")
    if report.get("a5_verdict_distribution"):
        print("\nA5 verdict distribution per annotator")
        for ann, dist in sorted(report["a5_verdict_distribution"].items()):
            print(f"  {ann:<16} " + ", ".join(f"{v}={c}" for v, c in dist.items()))
    if report.get("attention_checks"):
        print("\nattention-check pass rate per annotator (never gated)")
        for ann, r in sorted(report["attention_checks"].items()):
            pv = "NaN" if is_absent(r["pass_rate"]) else f"{r['pass_rate']:.4f}"
            print(f"  {ann:<16} pass_rate={pv}  n={r['n']}")
    if "llm" in report:
        print("\nLLM agreement vs. human consensus (reported, NOT gated)")
        for kind, r in sorted(report["llm"]["by_kind"].items()):
            av = "NaN" if is_absent(r["agreement"]) else f"{r['agreement']:.4f}"
            print(f"  by_kind  {kind:<16} agreement={av}  n={r['n']}")
        for model, r in sorted(report["llm"]["by_model"].items()):
            av = "NaN" if is_absent(r["agreement"]) else f"{r['agreement']:.4f}"
            print(f"  by_model {model:<24} agreement={av}  n={r['n']}")
        ov = report["llm"]["overall"]
        ov_av = "NaN" if is_absent(ov["agreement"]) else f"{ov['agreement']:.4f}"
        print(f"  overall  agreement={ov_av}  n={ov['n']}")


def cmd_annotate_review(a: argparse.Namespace) -> int:
    bundle = _read_json(Path(a.bundle))
    key = _read_json(Path(a.key)) if a.key else None
    records: list[dict] = []
    for p in a.records:
        records.extend(_read_jsonl(Path(p)))
    # `_read_jsonl` returns [] for a path that does not exist, and a review over zero records
    # renders every section empty -- which reads as "nothing to flag" rather than "nothing was
    # read". A typo in --records would otherwise return a clean bill of health for a campaign
    # nobody has looked at.
    if not records:
        print(
            f"no records read from {', '.join(str(p) for p in a.records)}. A review of zero "
            "records is not a clean review; check the path.",
            file=sys.stderr,
        )
        return 2

    report = review_report(bundle, records, key=key)
    if a.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        _print_review(report)
    return 0


# --------------------------------------------------------------------------- pi annotate llm


def _model_key_check(model: str, env: Mapping[str, str]) -> str | None:
    """None if a plausible API key is already in the environment; otherwise the message to
    print. Named the variables rather than raising, so a missing `.env` is a one-line fix
    instead of a traceback out of `litellm.completion`."""
    from pinq_adapters.llm.litellm_client import PROVIDER_KEY_ENV, _provider_of

    base_url = str(env.get("LITELLM_BASE_URL") or "").strip() or None
    provider = _provider_of(model, base_url)
    if provider == "litellm_proxy":
        if env.get("LITELLM_API_KEY"):
            return None
        return (
            f"model {model!r} routes through LITELLM_BASE_URL, but LITELLM_API_KEY is not "
            "set. Set LITELLM_API_KEY (see .env)."
        )
    key_var = PROVIDER_KEY_ENV.get(provider)
    if key_var is None or env.get(key_var):
        return None  # unrecognised provider prefix: nothing here to check against
    return (
        f"no API key for provider {provider!r} (model {model!r}): set {key_var}, or set "
        "LITELLM_BASE_URL + LITELLM_API_KEY to route through a proxy instead."
    )


def _annotator_client(model: str, *, cache_root_path: Path, env: Mapping[str, str]):
    """Built the way `pi_run.judge_client.judge()` builds one: metered, cached,
    temperature-pinned. `models={"annotator": model}` is what lets `MeteredClient.model_for`
    resolve the role WITHOUT ever consulting `PI_MODEL_JUDGE` -- annotating and judging are
    different measurements with different pins, by the same reasoning `ROLE_ENV` gives for
    keeping every role's model swappable independently.
    """
    from pi_eval.annotate_llm import ANNOTATOR_TEMPERATURE
    from pi_run.cache import CachingClient, DiskCache
    from pinq.budget import BudgetLedger
    from pinq_adapters.llm.litellm_client import MeteredClient

    # An unreachable hard cap: `BudgetLedger.cap` gates `charge_retrieval`, and an annotation
    # pass issues no retrieval calls at all. The real spend limit is `--max-usd`, read from
    # `ledger.spent["usd"]` by the caller.
    ledger = BudgetLedger(cap=10**9)
    inner = MeteredClient(
        ledger, temperature=ANNOTATOR_TEMPERATURE, env=dict(env), models={"annotator": model}
    )
    client = CachingClient(inner, DiskCache(cache_root_path), ledger=ledger)
    return client, ledger


def _existing_llm_keys(out_path: Path, annotator_id: str) -> set[tuple[str, str]]:
    """(item_id, prompt_sha) already on disk for this annotator_id -- what `--resume` skips.
    Keyed on the PAIR, not the item alone: a record from a different prompt_sha is a different
    rater and must be re-annotated, matching `annotate_llm.annotate_item`'s own reasoning for
    carrying `prompt_sha` on every record."""
    keys: set[tuple[str, str]] = set()
    for r in _read_jsonl(out_path):
        if str(r.get("annotator_id")) == annotator_id:
            keys.add((str(r.get("item_id")), str(r.get("prompt_sha") or "")))
    return keys


def _resort_llm_out_by_item_id(out_path: Path) -> None:
    """Rewrite `out_path`, sorted by (item_id, annotator_id) -- see `cmd_annotate_llm`'s call
    site and `_run_llm_pass_concurrent`'s docstring for why a `--concurrency > 1` pass needs
    this and a sequential one does not. Reads and rewrites the WHOLE file, including records
    written by an earlier `--resume`d pass, so the file's order is fully normalized regardless
    of how many prior passes (sequential or concurrent) produced it."""
    rows = _read_jsonl(out_path)
    rows.sort(key=lambda r: (str(r.get("item_id", "")), str(r.get("annotator_id", ""))))
    out_path.write_text(
        "".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows),
        encoding="utf-8",
    )


def _run_llm_pass(
    client: Any,
    ledger: Any,
    items: Sequence[Mapping[str, Any]],
    *,
    bundle_id: str,
    model_pin: str,
    annotator_id: str,
    seed: int,
    max_usd: float | None,
    already: set[tuple[str, str]],
    on_record: Any | None = None,
    concurrency: int = 1,
) -> dict[str, Any]:
    """The pass loop, taking an ALREADY-BUILT client and ledger rather than resolving a model
    from the environment -- what lets a test drive the resume-skip and the spend cap with a
    FakeLLM and a stub ledger, with no network and no filesystem.

    No file I/O here directly: `on_record`, when given, is called with each record the instant
    it is produced, and the CALLER (`cmd_annotate_llm`) is what turns that into a flushed write
    to `--out`. Keeping the write itself out of this function is what lets a test drive it with
    a plain list-appending callback and assert on `result["records"]` with no filesystem at
    all -- the same reason the docstring above gives for taking `client`/`ledger` as already
    built. `on_record` is what makes a killed or capped pass leave a resumable file: see
    `cmd_annotate_llm`, which a 700-item, ~3-hour pass used to lose entirely on a crash,
    silently defeating `--resume`, which reads that same file to learn what is already done.

    `concurrency <= 1` is this EXACT loop, byte-for-byte, unchanged -- every test written
    against it before `--concurrency` existed still exercises this branch and nothing about it
    moved. `concurrency > 1` delegates to `_run_llm_pass_concurrent`; see that function's
    docstring for why a thread pool and not the process pool `pi_run.sweep` uses for rollouts.
    """
    from pi_eval.annotate_llm import AnnotationParseError, annotate_item, build_prompt, prompt_sha

    if concurrency > 1:
        return _run_llm_pass_concurrent(
            client,
            ledger,
            items,
            bundle_id=bundle_id,
            model_pin=model_pin,
            annotator_id=annotator_id,
            seed=seed,
            max_usd=max_usd,
            already=already,
            on_record=on_record,
            concurrency=concurrency,
        )

    records: list[dict] = []
    n_resumed = 0
    n_unparsed = 0
    n_seen = 0
    stopped_for_spend = False
    for item in items:
        if max_usd is not None and float((ledger.spent or {}).get("usd", 0.0)) >= max_usd:
            stopped_for_spend = True
            break
        n_seen += 1
        this_sha = prompt_sha(build_prompt(item))
        if (str(item["item_id"]), this_sha) in already:
            n_resumed += 1
            continue
        try:
            rec = annotate_item(
                client,
                item,
                bundle_id=bundle_id,
                seed=seed,
                model_pin=model_pin,
                annotator_id=annotator_id,
            )
        except AnnotationParseError as exc:
            n_unparsed += 1
            print(
                f"  UNPARSED {item.get('item_id')}/{item.get('task_type')}: {exc}",
                file=sys.stderr,
            )
            continue
        if on_record is not None:
            on_record(rec)
        records.append(rec)
    return {
        "records": records,
        "n_resumed": n_resumed,
        "n_unparsed": n_unparsed,
        "n_remaining": len(items) - n_seen,
        "stopped_for_spend": stopped_for_spend,
    }


def _run_llm_pass_concurrent(
    client: Any,
    ledger: Any,
    items: Sequence[Mapping[str, Any]],
    *,
    bundle_id: str,
    model_pin: str,
    annotator_id: str,
    seed: int,
    max_usd: float | None,
    already: set[tuple[str, str]],
    on_record: Any | None,
    concurrency: int,
) -> dict[str, Any]:
    """The `--concurrency > 1` path: a THREAD pool, not the process pool `pi_run.sweep` uses
    for a rollout sweep.

    WHY THREADS, NOT PROCESSES. `pi_run.sweep.run_sweep` fans out (task, arm, seed) UNITS, each
    of which owns a whole `BudgetLedger` and drives a full `pinq` loop with gold-adjacent state
    -- "nothing to share and nothing to lock" ONLY because the process boundary makes it true.
    An annotation pass is the opposite shape: one blocking HTTP call per item against a single
    endpoint, sharing one request cache and one spend total ON PURPOSE. Measured on a real
    48-item pass: mean latency 15.7s against a 2.7s median and a 117s tail -- wall time is
    dominated by a few slow calls while everything else sits idle waiting on a socket, exactly
    the shape a thread pool overlaps (Python releases the GIL for the duration of the network
    wait) with none of a process pool's spawn cost or serialization overhead per item.

    WHY BOOKKEEPING NEEDS NO LOCK. Each submitted task (`_work`, below) does ONLY the network
    call and its own parse; it touches no counter and calls no callback. `records`, `n_unparsed`
    and `on_record` (the file write) are all touched ONLY as `as_completed` hands a finished
    future back to THIS function, which runs in the thread that called it -- never a worker
    thread. That is a stronger guarantee than "a lock protects the shared state": there is only
    ever one thread that CAN touch it, so nothing needs to be excluded from anything. That
    single thread is also what makes `--out` have one writer, satisfying "appends must be
    serialised" without an explicit `threading.Lock` anywhere in this module.

    WHAT IS DELIBERATELY NOT SERIALIZED, AND WHY THAT GAP IS DOCUMENTED RATHER THAN CLOSED.
    `client`/`ledger` ARE shared across worker threads -- that sharing is the whole point, since
    it is what lets every item hit the same on-disk request cache and accumulate one spend
    total. `pinq.budget.BudgetLedger`'s own docstring says it is built for "one ledger per
    process...so it needs no locking at all"; `MeteredClient.complete` charges it
    (`litellm_client.py`) from inside a worker thread here, and that charge -- like
    `CachingClient`'s hit/miss/race counters -- is a plain read-modify-write with no lock
    around it. Closing that gap would mean adding locking inside `pinq_adapters`/`pi_run.cache`,
    both shared with the untouched rollout path this task was explicitly scoped away from, for
    a benefit this task does not need: the annotation pass's own spend cap is a soft stop, and
    the ONE documented, ACCEPTED consequence is that it can overshoot by up to `concurrency - 1`
    calls, because the cap check below reads a ledger that up to `concurrency - 1` other
    in-flight calls have not charged yet. (The on-disk request cache itself, `DiskCache.put`,
    already IS race-free -- `os.link`, first-writer-wins -- because `pi_run.sweep` already
    writes it from a process pool; nothing here relies on it being any safer than that.)

    WHY THE CAP IS CHECKED ONLY AT SUBMISSION, IN ONE THREAD. Submission (both the initial
    priming and each refill after a completion) happens only in this function's own thread, one
    call at a time, so that decision cannot race with itself even though the charges it is
    reading can lag behind reality by up to `concurrency - 1` calls, as above. As soon as the
    cap reads as reached, no NEW work is submitted; work already in flight is not cancelled --
    matching `pi_run.sweep.run_sweep`'s "a cap is a stop-loss, not a ceiling" contract.

    WHY OUTPUT ORDER IS RESORTED, NOT LEFT IN COMPLETION ORDER. Completion order depends on
    which worker happened to finish first, which depends on thread scheduling -- exactly the
    dependency the spec this was built against forbids ("a pass whose output bytes differ
    run-to-run purely from thread timing is a pass whose --resume cannot be reasoned about").
    `item_id` does not depend on scheduling, so `cmd_annotate_llm` re-sorts the WHOLE `--out`
    file by it once this pass returns normally; see that function. This function's own
    `records` return value is sorted the same way, for the same reason and so a caller (a test,
    or a future one) never has to re-derive it.
    """
    from pi_eval.annotate_llm import AnnotationParseError, annotate_item, build_prompt, prompt_sha

    n_resumed = 0
    to_run: list[Mapping[str, Any]] = []
    for item in items:
        this_sha = prompt_sha(build_prompt(item))
        if (str(item["item_id"]), this_sha) in already:
            n_resumed += 1
            continue
        to_run.append(item)

    def _work(item: Mapping[str, Any]) -> tuple[str, Any]:
        try:
            rec = annotate_item(
                client,
                item,
                bundle_id=bundle_id,
                seed=seed,
                model_pin=model_pin,
                annotator_id=annotator_id,
            )
            return ("ok", rec)
        except AnnotationParseError as exc:
            return ("unparsed", exc)

    # Spend is totalled from the RECORDS, not read off the ledger. `pinq.budget` states the
    # ledger "is per (task, arm, seed) process, so it needs no locking at all" -- true for a
    # rollout worker, false here, where several threads share one. A lock-free counter feeding
    # the one guard that stops an overspend can undercount, and the failure is silent. Each
    # record carries its own `CallTelemetry.usd`, and this sum runs only in the dispatching
    # thread, which is the same thread that does every other piece of bookkeeping.
    spent = {"usd": 0.0}

    def _cap_reached() -> bool:
        return max_usd is not None and spent["usd"] >= max_usd

    records: list[dict] = []
    n_unparsed = 0
    n_completed = 0
    stopped_for_spend = False
    remaining = iter(to_run)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures: dict[Any, Mapping[str, Any]] = {}

        for _ in range(concurrency):
            if _cap_reached():
                stopped_for_spend = True
                break
            nxt = next(remaining, None)
            if nxt is None:
                break
            futures[ex.submit(_work, nxt)] = nxt

        while futures:
            done = next(as_completed(futures))
            item = futures.pop(done)
            kind, payload = done.result()  # a non-AnnotationParseError exception propagates here
            n_completed += 1
            if kind == "ok":
                records.append(payload)
                spent["usd"] += float(payload.get("usd") or 0.0)
                if on_record is not None:
                    on_record(payload)
            else:
                n_unparsed += 1
                print(
                    f"  UNPARSED {item.get('item_id')}/{item.get('task_type')}: {payload}",
                    file=sys.stderr,
                )
            if not stopped_for_spend:
                if _cap_reached():
                    stopped_for_spend = True
                else:
                    nxt = next(remaining, None)
                    if nxt is not None:
                        futures[ex.submit(_work, nxt)] = nxt

    records.sort(key=lambda r: (str(r.get("item_id", "")), str(r.get("annotator_id", ""))))
    return {
        "records": records,
        "n_resumed": n_resumed,
        "n_unparsed": n_unparsed,
        "n_remaining": len(items) - n_resumed - n_completed,
        "stopped_for_spend": stopped_for_spend,
    }


def cmd_annotate_llm(a: argparse.Namespace) -> int:
    """No `--key`, and PI_GOLD_ROOT is never touched.

    The bundle is already blinded for a human annotator -- the model must see exactly what a
    person would see, nothing the pipeline itself preferred, and nothing gold. Reading the key
    here would let the A2 prompt announce the automatic margin's own pick, which is precisely
    the hint `sample_a2_items` exists to keep out of the bundle; and this pass has no
    legitimate reason to open gold at all, since every gold-facing gate stays human-only by
    construction in `pi_eval.annotate`.
    """
    from pi_eval.annotate_llm import ANNOTATOR_TEMPERATURE

    out_path = Path(a.out)
    # Checked FIRST, before the model is even resolved: `--out` used to be truncated by
    # `_write_jsonl` on every run that did not pass `--resume`, so running this pass twice --
    # the ordinary way to extend a partial pass, or just a mistake -- silently discarded the
    # first run's records with no error and no trace. A campaign's whole point is not losing
    # measurements, so this refuses outright rather than guessing which run was meant to win.
    if not a.resume and not a.overwrite and out_path.exists() and _read_jsonl(out_path):
        n_existing = len(_read_jsonl(out_path))
        print(
            f"{out_path} already has {n_existing} record(s) and neither --resume nor "
            "--overwrite was passed. Running this pass again would silently discard them. "
            "Pass --resume to continue this campaign (skipping records already annotated at "
            "the same prompt_sha), or --overwrite to intentionally start it over.",
            file=sys.stderr,
        )
        return 2

    bundle = _read_json(Path(a.bundle))
    items = list(bundle.get("items") or ())
    if a.task_type:
        wanted = set(a.task_type)
        items = [it for it in items if str(it.get("task_type")) in wanted]
    if a.limit is not None:
        items = items[: a.limit]

    env = dict(os.environ)
    model = str(a.model or env.get("PI_MODEL_ANNOTATOR", "")).strip()
    if not model:
        print(
            "no model to annotate with: pass --model or set PI_MODEL_ANNOTATOR. Annotating is "
            "its own role with its own pin, deliberately separate from PI_MODEL_JUDGE -- "
            "borrowing the judge's pin would let a judge-model change silently move an "
            "annotation agreement number with nothing in the record explaining why.",
            file=sys.stderr,
        )
        return 2
    problem = _model_key_check(model, env)
    if problem:
        print(problem, file=sys.stderr)
        return 2

    bundle_id = str((bundle.get("manifest") or {}).get("bundle_id") or "")
    annotator_id = f"llm:{model}"
    model_pin = f"{model}@t{ANNOTATOR_TEMPERATURE}"

    n_existing = len(_read_jsonl(out_path)) if a.resume else 0
    already = _existing_llm_keys(out_path, annotator_id) if a.resume else set()

    cache_root_path = (
        Path(a.cache_root).resolve() if a.cache_root else cache_root(env.get("PI_CACHE_ROOT"))
    )
    client, ledger = _annotator_client(model, cache_root_path=cache_root_path, env=env)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Appended and flushed PER RECORD, not written once at the end. A 700-item pass is ~3
    # hours; a crash used to lose every record produced so far -- both for its own sake, and
    # because `--resume` reads this same file to learn what is already done, so a truncated
    # write silently defeated the one thing meant to make a partial pass recoverable. `--resume`
    # appends onto the file it is resuming (its prior records are already ON DISK and must stay
    # there); every other case (`--overwrite`, or a fresh `--out`) truncates once up front and
    # then only ever appends, so `--overwrite` still discards a prior run's records rather than
    # accumulating onto them.
    mode = "a" if a.resume else "w"
    with out_path.open(mode, encoding="utf-8") as fh:

        def _persist(rec: dict) -> None:
            fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
            fh.flush()

        result = _run_llm_pass(
            client,
            ledger,
            items,
            bundle_id=bundle_id,
            model_pin=model_pin,
            annotator_id=annotator_id,
            seed=a.seed,
            max_usd=a.max_usd,
            already=already,
            on_record=_persist,
            concurrency=a.concurrency,
        )

    if a.concurrency > 1:
        # Output-order stability (see `_run_llm_pass_concurrent`'s docstring): re-sort the
        # WHOLE file by item_id now that the pass has returned NORMALLY (a spend-cap stop
        # counts as normal; an exception propagating out of `_run_llm_pass` above does not
        # reach this line at all). A crash leaves the file in completion order instead --
        # still one well-formed JSON object per line, and `--resume` reads every line
        # regardless of order, so a partial file is exactly as resumable as before, just not
        # yet re-sorted. concurrency == 1 never reaches here, so its output is untouched.
        _resort_llm_out_by_item_id(out_path)

    new_records = result["records"]
    n_total = n_existing + len(new_records) if a.resume else len(new_records)

    n_attempted = len(new_records) + result["n_unparsed"]
    unparsed_rate = result["n_unparsed"] / n_attempted if n_attempted else float("nan")
    spend = float((ledger.spent or {}).get("usd", 0.0))
    print(f"annotated {len(new_records)} item(s) as {annotator_id}  ({model_pin})")
    if a.resume:
        print(f"  resumed: {result['n_resumed']} already annotated at this prompt_sha")
    print(f"  unparsed: {result['n_unparsed']}/{n_attempted}  (rate={unparsed_rate:.4f})")
    print(
        f"  remaining: {result['n_remaining']}"
        + ("  (stopped: --max-usd reached)" if result["stopped_for_spend"] else "")
    )
    print(f"  spend: ${spend:.6f}")
    print(f"wrote {out_path}  ({n_total} record(s) total)")
    return 0
