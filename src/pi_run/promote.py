"""Re-attach every rater verdict to a freshly exported pairs file. Idempotent, by identity.

WHY THIS EXISTS AS CODE AND NOT AS A SCRIPT ANYONE RUNS BY HAND.

`pi train export` OVERWRITES `data/rl/pairs.jsonl` and carries no annotation forward. Three
passes of A2 verdicts were lost that way, and the fix each time was a one-off script that
lived in a transcript. The verdicts themselves are safe -- the immutable records under the
annotation root are the system of record, and `data/rl` never is -- but "safe" is worth
nothing if re-attaching them is a bespoke act each time.

WHY THE JOIN IS ON IDENTITY AND NOT ON `pair_id`. MEASURED 2026-09-04: a re-export moved
`pairs.jsonl` from 3,648 to 27,266 rows and changed EVERY `pair_id` -- 0 of 27,266 matched a
previously rated set, because the id hashes the pair kind and the state key, both of which
moved. Joining on (state, {the two candidate run ids}) instead recovered 354 of 435. So the
identity here is:

    (suite_id, task_id, run_id, turn_idx, frozenset{chosen_run_id, rejected_run_id})

UNORDERED on purpose. `pairs.rater.jsonl` may order a pair the other way round from
`pairs.jsonl`, and a verdict is about the two candidates, not about which one the exporter put
first. `*_agrees_with_label` is then computed against the row's OWN direction, so the same
verdict yields opposite agreement on the two artifacts -- which is correct, and is the whole
point of keeping the control separate.

WHY THE QUARANTINE IS A FILE AND NOT A JUDGEMENT MADE HERE. Two raters were dropped for slot
bias, and re-running `_slot_bias` over the records shows the decision was not applied
symmetrically: `claude-opus-5` on the round-1 bundle is significant on the reach axis
(p = 1.1e-4) and not on preference (p = 0.46), yet BOTH its axes were dropped; `claude-sonnet-5`
on the round-2 bundle is significant on reach (p = 1.0e-4) and not on preference (p = 0.95),
and its preference votes were KEPT. Same statistics, two treatments. `conf/annotations/
quarantine.json` names the decision per (bundle, rater, axis) and `check_quarantine` refuses to
run if the records do not support what the file claims -- so the asymmetry is a visible,
reviewable choice rather than something buried in a session transcript.

WHY A LOSS GUARD. The target file is the one thing here that is not reproducible: if a promote
runs against a root that is missing a campaign, it would silently write a file with fewer
verdicts than the one it replaced. `annotation_loss` counts, per field, the rows that would
lose an annotation they currently have, and refuses unless the caller says otherwise.

NO TIMESTAMP ANYWHERE. Re-running promote over unchanged inputs must produce byte-identical
output, or "did anything change?" is not a question the artifact can answer.
"""

from __future__ import annotations

import collections
import hashlib
import importlib
import itertools
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _distinctness(chosen_json: str, rejected_json: str) -> tuple[float, str]:
    """`pinq_train.export.distinctness`, loaded BY NAME.

    `pi_run` may not import `pinq_train` statically -- an import-linter contract forbids it,
    so that the CLI keeps working without the training extras installed. Same lazy pattern as
    `cmd_train._train`.
    """
    try:
        mod = importlib.import_module("pinq_train.export.distinctness")
    except ModuleNotFoundError as exc:  # pragma: no cover - only without an editable install
        raise SystemExit(
            f"cannot load pinq_train.export.distinctness: {exc}. Install the repo editable "
            '(`uv pip install -e ".[dev]"`).'
        ) from None
    return mod.distinctness(chosen_json, rejected_json)


# Every field this module attaches. The loss guard compares exactly these between the file on
# disk and the file it is about to write; a field not named here is not protected.
ANNOTATION_PREFIXES = ("a7_", "a6_", "a5_", "a5blind_", "a2_", "q_")

# `scripts/a7_artifacts.py`'s abort threshold, restated so the check here cannot drift from it.
SLOT_BIAS_ALPHA = 0.001

# A bundle whose items carry no state provenance cannot be joined to a pair. The `suggest`
# campaign is the whole of this category: its A7 items are proposals, and their provenance is
# `{"origin": "suggest_on_disagreement"}` -- deliberately, since a proposal has no rollout.
UNJOINABLE_ORIGINS = ("suggest_on_disagreement",)


class AnnotationLoss(RuntimeError):
    """Writing would drop verdicts the target currently carries."""


class QuarantineDisagreement(RuntimeError):
    """The quarantine file claims something the records do not support."""


def rater_of(annotator_id: str) -> str:
    """`llm:openai/gcp/gemini-3.1-pro-preview` -> `gemini-3.1-pro`.

    A rule, not an alias table: strip the `llm:` kind prefix, take the last path segment, drop
    a `-preview` suffix. An alias table would need an entry the day a new rater is used, and
    the missing entry would silently key its votes under a different name.
    """
    s = str(annotator_id or "")
    s = s.split(":", 1)[1] if ":" in s else s
    s = s.rsplit("/", 1)[-1]
    return s[: -len("-preview")] if s.endswith("-preview") else s


def pair_identity(row: Mapping[str, Any]) -> tuple:
    """The join key. See the module docstring for why it is not `pair_id`."""
    return (
        str(row.get("suite_id") or ""),
        str(row.get("task_id") or ""),
        str(row.get("run_id") or ""),
        int(row.get("turn_idx") or 0),
        frozenset({str(row.get("chosen_run_id") or ""), str(row.get("rejected_run_id") or "")}),
    )


def majority(votes: Mapping[str, str]) -> tuple[str, bool]:
    """(label, unanimous) over one item's votes, or ("split", False) with no strict majority.

    A STRICT majority of the raters who actually answered, and a unique winner. Two raters
    therefore resolve only when they agree -- "a majority of two is one person outvoting
    nobody" (`pi_eval.annotate.consensus`, whose rule this mirrors).
    """
    # ONE RATER IS NOT A PANEL. n=1 satisfies "strict majority with a unique winner" (1*2 > 1)
    # and "everyone who answered agreed", so a single opinion came back as a unanimous verdict.
    # A7 never exposed this because every A7 item has two raters or more; 21 A5 runs were
    # judged by one, and 30 shipped pairs carried the result flagged `unanimous`,
    # indistinguishable from 3-0. The docstring below already refuses a "majority" of two
    # outvoting nobody; this is the same rule with the degenerate case closed.
    if len(votes) < 2:
        return "split", False
    counts = collections.Counter(votes.values())
    top, n = counts.most_common(1)[0]
    winners = [k for k, v in counts.items() if v == n]
    if len(winners) == 1 and n * 2 > len(votes):
        return top, n == len(votes)
    return "split", False


# --------------------------------------------------------------------------- the record root


@dataclass(frozen=True, slots=True)
class Item:
    """One bundle item, joined to its key entry."""

    item_id: str
    task_type: str
    bundle_id: str
    provenance: dict
    key: dict
    # A5 only: this item withheld the unresolved-need list and the final answer. Read off the
    # shipped payload, so the partition follows what the rater was actually shown.
    blind: bool = False


@dataclass
class Annotations:
    # KEYED ON (bundle_id, item_id), NOT item_id. Item ids are NOT unique across bundles: the
    # round-1 and round-2 bundles share them, so a dict keyed on the id alone silently resolved
    # every round-1 record to the round-2 item. MEASURED 2026-09-07: that gave quarantined
    # raters' votes the round-2 quarantine rules -- claude-opus-5 and gpt-5.6-sol, both dropped
    # from the shipped panel, were being counted back in -- and joined 310 pairs to the wrong
    # state. A record is looked up by ITS OWN `bundle_id`, which every record carries.
    items: dict[tuple[str, str], Item] = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)
    preferences: list[dict] = field(default_factory=list)
    bundle_order: list[str] = field(default_factory=list)
    n_unjoinable_items: int = 0
    n_retest_duplicates: int = 0

    def item_for(self, record: Mapping[str, Any]) -> Item | None:
        """The item a record judged, resolved through the record's OWN bundle."""
        return self.items.get((str(record.get("bundle_id") or ""), str(record.get("item_id"))))

    def round_of(self, bundle_id: str) -> str:
        """`r1`, `r2`, ... by sorted bundle id -- the order the campaigns were built in."""
        try:
            return f"r{self.bundle_order.index(bundle_id) + 1}"
        except ValueError:
            return ""


def load_root(root: str | Path) -> Annotations:
    """Read every bundle, key and record under an annotation root."""
    root = Path(root)
    ann = Annotations()
    joinable_bundles: set[str] = set()
    for bf in sorted((root / "bundles").glob("*.json")):
        if bf.name.endswith(".key.json"):
            continue
        bundle = json.loads(bf.read_text())
        keyf = bf.parent / f"{bf.stem}.key.json"
        keys = json.loads(keyf.read_text())["items"] if keyf.exists() else {}
        bid = str(bundle.get("bundle_id") or bf.stem)
        for it in bundle.get("items", []):
            prov = it.get("provenance") or {}
            if str(prov.get("origin") or "") in UNJOINABLE_ORIGINS or not prov.get("suite"):
                ann.n_unjoinable_items += 1
                continue
            ann.items[bid, str(it["item_id"])] = Item(
                item_id=str(it["item_id"]),
                task_type=str(it.get("task_type") or ""),
                bundle_id=bid,
                provenance=prov,
                key=keys.get(str(it["item_id"])) or {},
                blind=bool((it.get("payload") or {}).get("blind")),
            )
            joinable_bundles.add(bid)
    ann.bundle_order = sorted(joinable_bundles)

    seen: set[tuple[str, str, str]] = set()
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "bundles"):
        for rf in sorted(d.glob("*.jsonl")):
            if rf.stem == "preferences":
                # EXTEND, never assign. A root holds one `preferences.jsonl` per A6 campaign
                # (a6/, a6b/, ...), and assigning kept only the last one the sorted walk
                # reached -- 241 of 2,429 -- so the pairs the other campaign had ordered
                # arrived with no verdict attached to audit them with.
                ann.preferences.extend(
                    json.loads(x) for x in rf.read_text().splitlines() if x.strip()
                )
                continue
            if rf.stem == "proposals":
                continue  # generated questions, not verdicts on our pairs
            for line in rf.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                k = (
                    str(r.get("bundle_id") or ""),
                    str(r.get("item_id")),
                    rater_of(r.get("annotator_id", "")),
                )
                if k in seen:
                    # A planted retest: the same rater asked the same item twice, on purpose.
                    # First wins, so the attach is deterministic; the count is reported.
                    ann.n_retest_duplicates += 1
                    continue
                seen.add(k)
                ann.records.append(r)
    return ann


def load_quarantine(path: str | Path) -> dict[str, dict[str, list[str]]]:
    return json.loads(Path(path).read_text())


def _is_quarantined(q: Mapping[str, Any], bundle_id: str, rater: str, axis: str) -> bool:
    return axis in (q.get(bundle_id) or {}).get(rater, [])


def check_quarantine(ann: Annotations, q: Mapping[str, Any]) -> dict[str, Any]:
    """Re-measure slot bias and check it against what the quarantine file claims.

    THE TWO DIRECTIONS ARE NOT SYMMETRIC, and treating them as though they were is what made
    an earlier version of this function refuse to run on the shipped data.

      * A rater NOT quarantined that reads position on `preference` is FATAL. That is the axis
        that decides a pair, so a biased vote there enters the majority and the label.
      * A rater quarantined on an axis whose p sits ABOVE alpha is over-strict, not wrong: it
        discards evidence rather than admitting bad evidence. It is recorded as
        `conservative`, and the promote continues. `claude-opus-5` on the round-1 bundle is
        exactly this case (p = 0.00144 against alpha = 0.001 on the deduplicated records,
        where the paper reports 1.1e-4), and lifting it would add a fifth rater to every
        round-1 majority -- a research decision, not one a re-attach step may take.

    A file nobody re-checks is a file that outlives its reason, so the measured p-values ride
    into the manifest whichever way they come out.
    """
    from pi_run.cmd_annotate import _slot_bias

    by_bundle: dict[str, list[dict]] = collections.defaultdict(list)
    for r in ann.records:
        if r.get("task_type") == "A7":
            by_bundle[str(r.get("bundle_id") or "")].append(r)

    report: dict[str, Any] = {}
    for bid, recs in sorted(by_bundle.items()):
        items = [
            {"item_id": i.item_id, "task_type": "A7"}
            for i in ann.items.values()
            if i.bundle_id == bid and i.task_type == "A7"
        ]
        # PER RATER. `_slot_bias` aggregates over every record it is given, so calling it
        # once per bundle would average a biased rater into a clean panel and report health.
        per_rater: dict[str, list[dict]] = collections.defaultdict(list)
        for r in recs:
            per_rater[rater_of(r.get("annotator_id", ""))].append(r)
        stats = {
            rater: _slot_bias({"items": items}, rs, "A7") for rater, rs in sorted(per_rater.items())
        }
        report[bid] = stats
        for rater, axes in stats.items():
            for axis, st in (axes or {}).items():
                pv = (st or {}).get("binomial_p")
                # NaN means the rater judged nothing in this bundle -- not evidence of health
                # and not evidence of bias. `nan < alpha` is False, but say so explicitly.
                p = 1.0 if pv is None or pv != pv else float(pv)
                claimed = _is_quarantined(q, bid, rater, axis)
                if claimed and p >= SLOT_BIAS_ALPHA:
                    st["conservative"] = True
                if not claimed and axis == "preference" and p < SLOT_BIAS_ALPHA:
                    raise QuarantineDisagreement(
                        f"{bid}/{rater}/preference: p={p:.3g} < {SLOT_BIAS_ALPHA} and the file "
                        "does not quarantine it. `preference` is the axis that decides a pair; "
                        "a biased rater may not vote on it."
                    )
    return report


def _a6_payload_by_item(root: str | Path) -> dict[tuple[str, str], dict]:
    """(bundle_id, item_id) -> payload, read straight from the bundle files.

    `Item` (this module's own joined record) does not carry `payload` -- nothing before this
    needed it, and it stays that way here rather than widening a frozen, slotted dataclass
    every other caller shares for the one report that needs it. `_slot_bias`'s A6 branch reads
    `payload["candidates"]` to find the shown order, so this reads it directly from disk.
    Bundle id resolution mirrors `load_root` exactly (`bundle.get("bundle_id") or bf.stem`) so
    a lookup by an `Item`'s own `bundle_id` cannot miss through a spelling difference.
    """
    out: dict[tuple[str, str], dict] = {}
    for bf in sorted(Path(root).glob("bundles/*.json")):
        if bf.name.endswith(".key.json"):
            continue
        bundle = json.loads(bf.read_text())
        bid = str(bundle.get("bundle_id") or bf.stem)
        for it in bundle.get("items", []):
            out[bid, str(it.get("item_id"))] = it.get("payload") or {}
    return out


def a6_slot_bias_report(ann: Annotations, root: str | Path) -> dict[str, Any]:
    """Per (bundle, rater), the A6 top-tier statistic from `_slot_bias` -- RECORD ONLY.

    Unlike `check_quarantine`'s A7 pass, this never raises and never marks anything
    `conservative`: it does not compare the measured p against the quarantine file at all, so
    there is nothing here for a rater to be over-strict or under-strict against. A6 gets no
    quarantine-and-refuse path on this axis -- gating a build on it, the way A7's `preference`
    axis is gated, is a research decision this function does not make, because withdrawing a
    rater here would move majorities the paper already quotes. It does not touch
    `a6_verdicts`, `attach`, or the pairs file `promote` writes -- see
    `test_a6_slot_bias_report_does_not_change_the_promoted_pairs_file`.

    Field names match what a manifest reader needs, not `_slot_bias`'s internal ones:
    `p_first_top` is its `p_first`, `expected_tie_aware` is its `expected` (the tie-aware null
    is the default there), `p` is its `binomial_p`, and `p_naive` passes through the naive
    1/k reading unchanged -- see `_a6_slot_bias`'s docstring for why that reading runs hot
    when tiers are common.
    """
    from pi_run.cmd_annotate import _slot_bias

    payload_by_item = _a6_payload_by_item(root)
    by_bundle: dict[str, list[dict]] = collections.defaultdict(list)
    for r in ann.records:
        if r.get("task_type") == "A6":
            by_bundle[str(r.get("bundle_id") or "")].append(r)

    report: dict[str, Any] = {}
    for bid, recs in sorted(by_bundle.items()):
        items = [
            {
                "item_id": i.item_id,
                "task_type": "A6",
                "payload": payload_by_item.get((bid, i.item_id), {}),
            }
            for i in ann.items.values()
            if i.bundle_id == bid and i.task_type == "A6"
        ]
        per_rater: dict[str, list[dict]] = collections.defaultdict(list)
        for r in recs:
            per_rater[rater_of(r.get("annotator_id", ""))].append(r)
        stats: dict[str, dict] = {}
        for rater, rs in sorted(per_rater.items()):
            st = _slot_bias({"items": items}, rs, "A6")["top_tier"]
            stats[rater] = {
                "n": st["n"],
                "p_first_top": st["p_first"],
                "expected_tie_aware": st["expected"],
                "p": st["binomial_p"],
                "p_naive": st["p_naive"],
            }
        report[bid] = stats
    return report


# --------------------------------------------------------------------------- verdicts


def a7_verdicts(ann: Annotations, q: Mapping[str, Any]) -> dict[tuple, dict]:
    """One entry per rated pair, in RUN-ID space.

    Not in chosen/rejected space. A verdict is about two candidates; "chosen" and "rejected"
    are a property of the ARTIFACT that was exported, and `pairs.rater.jsonl` may order the
    same pair the other way round. Storing the preferred run id and translating at attach time
    is what keeps a verdict meaning the same thing on every artifact -- and stops
    `a7_agrees_with_label` from reading `True` on a row whose direction is the opposite of the
    one the raters were shown.
    """
    votes: dict[tuple, dict[str, str]] = collections.defaultdict(dict)
    reaches: dict[tuple, dict[str, dict]] = collections.defaultdict(dict)
    meta: dict[tuple, tuple[str, str]] = {}
    for r in ann.records:
        if r.get("task_type") != "A7":
            continue
        it = ann.item_for(r)
        if it is None:
            continue
        key, prov = it.key, it.provenance
        chosen, rejected = key.get("chosen_run_id"), key.get("rejected_run_id")
        if not chosen or not rejected:
            continue
        ident = (
            str(prov.get("suite") or ""),
            str(prov.get("task_id") or ""),
            str(prov.get("run_id") or ""),
            int(prov.get("turn_idx") or 0),
            frozenset({str(chosen), str(rejected)}),
        )
        meta[ident] = (it.bundle_id, ann.round_of(it.bundle_id))
        rater = rater_of(r.get("annotator_id", ""))
        resp = r.get("response") or {}
        # `order` says which SLOT held the chosen side. Mapping through it is what makes the
        # coin flip a blind, rather than a relabelling of the same bias.
        slot_of_chosen = "a" if str(key.get("order") or "ab") == "ab" else "b"
        pref = str(resp.get("preference") or "")
        if not _is_quarantined(q, it.bundle_id, rater, "preference"):
            if pref in ("a", "b"):
                votes[ident][rater] = str(chosen) if pref == slot_of_chosen else str(rejected)
            elif pref in ("tie", "both_bad"):
                votes[ident][rater] = pref
        if not _is_quarantined(q, it.bundle_id, rater, "reaches"):
            a, b = resp.get("a_reaches"), resp.get("b_reaches")
            if a or b:
                reaches[ident][rater] = {
                    str(chosen): a if slot_of_chosen == "a" else b,
                    str(rejected): b if slot_of_chosen == "a" else a,
                }

    out: dict[tuple, dict] = {}
    for ident, v in votes.items():
        if not v:
            continue
        lab, unan = majority(v)
        bundle_id, rnd = meta[ident]
        out[ident] = {
            "_votes_by_run": dict(sorted(v.items())),
            "_reaches_by_run": {k: dict(x) for k, x in sorted(reaches[ident].items())},
            "_majority_run": lab,
            "a7_unanimous": unan,
            "a7_n_raters": len(v),
            "a7_round": rnd,
            "a7_rater_panel": sorted(v),
            "a7_bundle_id": bundle_id,
        }
    return out


class QuarantineUnenforceable(RuntimeError):
    """A quarantine names a rater on an axis this root cannot re-derive, so honouring it is
    impossible and IGNORING it would silently admit withdrawn votes."""


def _a6_tiers(ann: Annotations, q: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, dict]]:
    """(bundle, item) -> rater -> tier map, from the RAW A6 records, quarantine applied.

    PER RATER AND PER AXIS, exactly as `a7_verdicts` and `a5_verdicts` do it. The rater key is
    `rater_of(annotator_id)`, so the quarantine matches the same canonical spelling on every
    campaign -- the pre-aggregated `preferences.jsonl` files disagree about this (`a6b/`,
    `fixed1/` and `fixed2/` write `oss`/`gemini`/`sonnet` where `a6/` and `growth1/` write
    `gpt-oss-120b`/`gemini-3.1-pro`/`claude-sonnet-5`), and matching a quarantine entry against
    the short spelling by substring would collide `oss` with `gpt-oss-120b`. Going through the
    records removes the question instead of answering it.
    """
    out: dict[tuple[str, str], dict[str, dict]] = collections.defaultdict(dict)
    for r in ann.records:
        if r.get("task_type") != "A6":
            continue
        it = ann.item_for(r)
        if it is None:
            continue
        rater = rater_of(r.get("annotator_id", ""))
        if _is_quarantined(q, it.bundle_id, rater, "a6"):
            continue
        tiers = (r.get("response") or {}).get("tiers") or {}
        if tiers:
            out[it.bundle_id, it.item_id][rater] = {str(k): v for k, v in tiers.items()}
    return out


def a6_verdicts(ann: Annotations, q: Mapping[str, Any]) -> dict[tuple, dict]:
    """One entry per candidate pair a rater majority ordered.

    TAKES A QUARANTINE. It did not, and that was a type defect with a measured consequence:
    the a6 axis is the one that ORDERS `pairs.rater.jsonl`, and a panel decision expressed as
    a quarantine entry -- the only way this repository expresses one -- could not reach it. The
    other three axes each take `q` and test it per rater per axis; this one took no parameter,
    so there was nothing to bypass and nothing to audit. MEASURED 2026-09-17: withdrawing
    `gpt-oss-120b` moved the reaches ordering (1,426 -> 1,818 preferences, digest
    c15251f7ecb6cf5e -> 64e95b7439253f67) while the rater ordering stayed byte-identical, with
    4,476 of its 4,610 preference rows still carrying the withdrawn rater.

    DERIVED FROM THE RAW TIER RECORDS, not from the campaigns' pre-aggregated
    `preferences.jsonl`. A6 asks for a TIER RANKING over the candidates, and the pairwise
    majority is an expansion of it; the aggregate keeps only `raters`, `n_votes` and
    `unanimous`, which does not say WHO voted for what, so a quarantine could at best drop
    every non-unanimous row containing the withdrawn rater. On the shipped instrument that is
    2,125 of 4,610 rows thrown away for want of information the records still hold.

    VALIDATED AGAINST THE AGGREGATE, so this is not a second implementation left to drift.
    `test_a6_rederivation_reproduces_a_preaggregated_campaign` pins the expansion rule (lower
    tier wins, equal tier abstains, strict majority) against a hand-written aggregate. MEASURED
    2026-09-17 on `~/pi-corpus-backup/annotations-20260914` with an empty quarantine: identical
    (item, winner, loser) sets, `n_votes` and `unanimous` on `a6b`, `fixed1`, `fixed2` and
    `growth1`, and a superset of 2,200 against 2,188 on `a6` -- the 12 extra are the same 12
    that separate `derive_a6`'s 6,758 from the shipped 6,746. Only the rater SPELLING differs,
    and that is this function canonicalising it.

    THE PRE-AGGREGATED FILES REMAIN THE FALLBACK for an item with no raw records, because
    `load_root` reads roots that hold only `preferences.jsonl`. That path cannot honour a
    quarantine, so it REFUSES rather than ignoring one: silently admitting a withdrawn rater's
    votes is the failure this function was fixed to stop.
    """
    out: dict[tuple, dict] = {}
    a6_bundles = {i.bundle_id for i in ann.items.values() if i.task_type == "A6"}
    tiers_by_item = _a6_tiers(ann, q)

    for (bid, iid), by_rater in sorted(tiers_by_item.items()):
        it = ann.items.get((bid, iid))
        if it is None:
            continue
        runs = (it.key or {}).get("candidate_runs") or {}
        prov = it.provenance
        for x, y in itertools.combinations(sorted(runs), 2):
            votes: dict[str, str] = {}
            for rater, tiers in by_rater.items():
                if x not in tiers or y not in tiers or tiers[x] == tiers[y]:
                    continue  # an unranked or same-tier pair is an abstention, not a vote
                votes[rater] = x if tiers[x] < tiers[y] else y
            if not votes:
                continue
            lab, unan = majority(votes)
            if lab == "split":
                continue
            los = y if lab == x else x
            ident = (
                str(prov.get("suite") or ""),
                str(prov.get("task_id") or ""),
                str(prov.get("run_id") or ""),
                int(prov.get("turn_idx") or 0),
                frozenset({str(runs[lab]), str(runs[los])}),
            )
            out[ident] = {
                "a6_winner_run": str(runs[lab]),
                "a6_n_votes": sum(1 for v in votes.values() if v == lab),
                "a6_unanimous": unan,
                "a6_raters": sorted(votes),
                "a6_bundle_id": bid,
            }

    for p in ann.preferences:
        # `preferences.jsonl` is derived per campaign and carries no bundle id, so resolve the
        # item within the A6 bundles only -- never against the whole (colliding) id space.
        it = next(
            (
                ann.items[b, str(p.get("item_id"))]
                for b in sorted(a6_bundles)
                if (b, str(p.get("item_id"))) in ann.items
            ),
            None,
        )
        w, lo = p.get("winner_run"), p.get("loser_run")
        if it is None or not w or not lo:
            continue
        if (it.bundle_id, it.item_id) in tiers_by_item:
            continue  # the raw records decided this item; the aggregate is the fallback only
        named = sorted(r for r, axes in (q.get(it.bundle_id) or {}).items() if "a6" in (axes or []))
        if named:
            # REFUSE, do not ignore. This item has no raw A6 records, so the pre-aggregated
            # row cannot be re-tallied without the quarantined rater: it records `raters`,
            # `n_votes` and `unanimous`, never who voted for what. Dropping through would
            # count a withdrawn rater's votes back into the ordering while the manifest said
            # the panel excluded them -- the exact failure this parameter exists to stop.
            raise QuarantineUnenforceable(
                f"{it.bundle_id}/{it.item_id}: the quarantine withdraws {named} from the `a6` "
                "axis, but this root holds no raw A6 records for the item -- only a campaign's "
                "pre-aggregated preferences.jsonl, which does not say which rater voted for "
                "which candidate. Re-derive from the records, or drop the entry."
            )
        prov = it.provenance
        ident = (
            str(prov.get("suite") or ""),
            str(prov.get("task_id") or ""),
            str(prov.get("run_id") or ""),
            int(prov.get("turn_idx") or 0),
            frozenset({str(w), str(lo)}),
        )
        out[ident] = {
            "a6_winner_run": str(w),
            "a6_n_votes": int(p.get("n_votes") or 0),
            "a6_unanimous": bool(p.get("unanimous")),
            "a6_raters": sorted(p.get("raters") or []),
            "a6_bundle_id": it.bundle_id,
        }
    return out


def a5_verdicts(ann: Annotations, q: Mapping[str, Any]) -> dict[tuple[bool, str], dict]:
    """One entry per (instrument, judged episode), keyed on `blind` and the run id.

    NOT ON THE RUN ALONE. A5 exists in two variants that deliberately show the rater different
    things: the sighted item prints the unresolved gold nodes and the final answer, the blind
    one prints neither, and their verdicts differ systematically because of it -- which is the
    entire point of the blinded campaign. Keyed on the run alone, 34 runs on the live corpus
    had votes from both merged into a single verdict, which destroys the comparison and
    silently mixes an instrument we trust with one we showed is reading its own answer.

    The partition is the ITEM's own `blind` flag, never a list of bundle names, so a later
    campaign cannot be mis-filed by being called something new.
    """
    votes: dict[tuple[bool, str], dict[str, str]] = collections.defaultdict(dict)
    for r in ann.records:
        if r.get("task_type") != "A5":
            continue
        it = ann.item_for(r)
        if it is None:
            continue
        rater = rater_of(r.get("annotator_id", ""))
        if _is_quarantined(q, it.bundle_id, rater, "a5"):
            continue
        v = (r.get("response") or {}).get("verdict")
        run = str(it.provenance.get("run_id") or "")
        if v and run:
            votes[it.blind, run][rater] = str(v)
    out: dict[tuple[bool, str], dict] = {}
    for key, v in votes.items():
        lab, unan = majority(v)
        if lab == "split":
            continue
        out[key] = {"verdict": lab, "unanimous": unan, "n_votes": len(v)}
    return out


# --------------------------------------------------------------------------- attach


def strip_annotations(row: Mapping[str, Any]) -> dict:
    return {k: v for k, v in row.items() if not k.startswith(ANNOTATION_PREFIXES)}


def attach(
    rows: Iterable[Mapping[str, Any]],
    a7: Mapping[tuple, dict],
    a6: Mapping[tuple, dict],
    a5: Mapping[str, dict],
) -> tuple[list[dict], dict[str, int]]:
    """Re-attach every verdict. Existing annotation fields are REPLACED, never merged.

    Merging would let a verdict survive a promote that no longer has a record behind it, which
    is exactly the state `data/rl` must never be in: the file is a projection of the immutable
    records, and anything in it that they cannot reproduce is unprovenanced.
    """
    out: list[dict] = []
    counts: collections.Counter = collections.Counter()
    for row in rows:
        d = strip_annotations(row)
        ident = pair_identity(d)
        chosen = str(d.get("chosen_run_id") or "")

        v7 = a7.get(ident)
        if v7:
            # TRANSLATE INTO THIS ROW'S FRAME. The verdict names a run; the row decides which
            # of the two runs is "chosen". On `pairs.rater.jsonl` that may be the opposite of
            # the artifact the raters were shown, and an untranslated verdict would report
            # agreement exactly backwards.
            def side(run: str, _chosen: str = chosen) -> str:
                return "chosen" if run == _chosen else "rejected"

            d.update({k: v for k, v in v7.items() if not k.startswith("_")})
            d["a7_votes"] = {
                r: (v if v in ("tie", "both_bad") else side(v))
                for r, v in v7["_votes_by_run"].items()
            }
            d["a7_reaches"] = {
                r: {side(run): lbl for run, lbl in per.items()}
                for r, per in v7["_reaches_by_run"].items()
            }
            mrun = v7["_majority_run"]
            d["a7_majority"] = mrun if mrun in ("split", "tie", "both_bad") else side(mrun)
            d["a7_agrees_with_label"] = (
                (mrun == chosen) if mrun not in ("split", "tie", "both_bad") else None
            )
            counts["a7"] += 1

        v6 = a6.get(ident)
        if v6:
            d.update(v6)
            d["a6_agrees_with_label"] = v6["a6_winner_run"] == chosen
            counts["a6"] += 1

        hit5 = False
        for side in ("chosen", "rejected"):
            run = str(d.get(f"{side}_run_id") or "")
            for blind in (False, True):
                m = a5.get((blind, run))
                if not m:
                    continue
                pre = "a5blind" if blind else "a5"
                d[f"{pre}_{side}_verdict"] = m["verdict"]
                d[f"{pre}_{side}_unanimous"] = m["unanimous"]
                d[f"{pre}_{side}_n_votes"] = m["n_votes"]
                hit5 = True
        counts["a5"] += hit5

        j, bucket = _distinctness(
            str(d.get("chosen_json") or ""), str(d.get("rejected_json") or "")
        )
        d["q_jaccard"] = j
        d["q_distinctness"] = bucket
        counts[f"q_{bucket}"] += 1
        out.append(d)
    return out, dict(counts)


def annotation_loss(
    old: Sequence[Mapping[str, Any]], new: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """Per field, how many rows would lose an annotation they currently have."""
    new_by_id = {pair_identity(r): r for r in new}
    lost: collections.Counter = collections.Counter()
    for o in old:
        n = new_by_id.get(pair_identity(o))
        for k in o:
            if not k.startswith(ANNOTATION_PREFIXES):
                continue
            if n is None:
                lost["row_absent"] += 1
                break
            if k not in n:
                lost[k] += 1
    return dict(sorted(lost.items()))


# --------------------------------------------------------------------------- the entry point


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def promote(
    pairs: str | Path,
    annotations: str | Path,
    quarantine: str | Path,
    *,
    out: str | Path | None = None,
    allow_loss: bool = False,
) -> dict[str, Any]:
    """Attach every verdict to `pairs`, refusing to lose one. Returns the manifest blocks."""
    src = Path(pairs)
    dst = Path(out) if out else src
    rows = _read_jsonl(src)
    ann = load_root(annotations)
    q = load_quarantine(quarantine)
    slot_bias = check_quarantine(ann, q)
    a6_slot_bias = a6_slot_bias_report(ann, annotations)

    a7 = a7_verdicts(ann, q)
    a6 = a6_verdicts(ann, q)
    a5 = a5_verdicts(ann, q)
    new, counts = attach(rows, a7, a6, a5)

    if dst.exists():
        lost = annotation_loss(_read_jsonl(dst), new)
        if lost and not allow_loss:
            raise AnnotationLoss(
                f"writing {dst} would drop annotations it currently carries: {lost}. The "
                "immutable records are the system of record, so this means the root is "
                "missing a campaign the target was built with. Pass allow_loss=True only if "
                "that is intended."
            )

    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in new)
    dst.write_text(payload)

    blocks = {
        "a7_campaign": {
            "pairs_annotated": counts.get("a7", 0),
            "verdicts_available": len(a7),
            "quarantined": {b: v for b, v in q.items() if v and not b.startswith("_")},
            "slot_bias": slot_bias,
            "n_retest_duplicates": ann.n_retest_duplicates,
        },
        "a6_campaign": {
            "pairs_annotated": counts.get("a6", 0),
            "preferences_available": len(a6),
            "slot_bias": a6_slot_bias,
        },
        "a5_campaign": {
            "pairs_annotated": counts.get("a5", 0),
            "verdicts_available": len(a5),
        },
        "promote": {
            "annotations_root": Path(annotations).name,
            "quarantine_sha256": hashlib.sha256(Path(quarantine).read_bytes()).hexdigest()[:16],
            "n_unjoinable_items": ann.n_unjoinable_items,
            "q_distinctness": {k[2:]: v for k, v in sorted(counts.items()) if k.startswith("q_")},
            "n_rows": len(new),
        },
    }
    return blocks
