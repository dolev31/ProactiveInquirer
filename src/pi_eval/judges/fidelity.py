"""How far our reimplemented judges sit from the published ones. Measured, then tabled.

THE OBLIGATION THIS DISCHARGES. kpr.py, citation.py and quality.py reimplement a rubric whose
original prompt text we may not redistribute (cxcscmu/deepresearch_benchmarking has NO
LICENCE FILE -- GitHub's licence API returns 404, verified 2026-08-24). A reimplemented judge
is a DIFFERENT INSTRUMENT until someone measures the difference, so every DRGym number in the
paper ships next to the table this module emits. "Prompts adapted from" is not a measurement.

WHAT IS COMPARED, AND WHY IT IS THE PER-UNIT LABELS. Upstream commits its own judge's output
per system at `results/mini/<system>/evaluation_results_kpr_<model>.json`, in the shape

    {"<query_id>": {"labels": {"<point_number>": ["Supported", "<justification>"]}}, ...}

which is one label per (query, key point) -- the same unit our Judgments carry. So the
comparison is a confusion matrix over shared units, plus the aggregate each side computes
FROM THOSE SHARED UNITS ONLY. Comparing our mean over our subset against their published mean
over their subset would confound the instrument with the sample, which is the exact error the
table exists to rule out.

WHAT THIS MODULE REFUSES TO DO
  * Invent reference numbers. Nothing is hardcoded: a reference is loaded from a file, and
    the loader raises unless system, judge model, source URL and commit are all supplied. A
    number without provenance is not a result.
  * Emit a table when the unit sets do not overlap. An empty delta renders as 0.000 and reads
    as perfect agreement, which is the most expensive possible way to be wrong here.
  * Compare against a reference produced by a different judge model without saying so: the
    model id appears in every row, because our-vs-theirs and gpt-4.1-mini-vs-something-else
    are different questions.

WHAT IS AVAILABLE UPSTREAM, checked at commit d4d2433 on 2026-08-24: five reference systems
under `results/mini/` (GPTResearcher, GPTResearcher_custom, gpt-4o-search-preview,
hf_deepresearch_gpt-4o-mini, open_deep_search), each with one
`evaluation_results_kpr_gpt-4.1-mini.json`. open_deep_search covers 10 queries / 514 labelled
key points and its Supported share is 0.337. The REPORTS those labels were computed from are
NOT in that repository, so reproducing the comparison means obtaining the same `.q`/`.a`
files; until then `kpr_delta` refuses rather than comparing across samples.

Running our judge over the same reports is the caller's job: point kpr.grade() at the
system's `.q`/`.a` directory (pinq_adapters.drgym.report.read_system_dir) and pass the
resulting Judgments in.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .kpr import LABELS as KPR_LABELS
from .paired import krippendorff_alpha_nominal
from .types import Judgment


class MissingProvenance(ValueError):
    """A reference number arrived without the four things that make it citable."""


class NoOverlap(ValueError):
    """No shared (task, key point) unit. A delta over nothing is not a delta."""


# ------------------------------------------------------------------ references


@dataclass(frozen=True, slots=True)
class ReferenceKPR:
    """Someone else's per-unit key-point labels, with the provenance to cite them."""

    system: str
    judge_model: str
    source_url: str
    commit: str
    sha256: str
    labels: Mapping[tuple[str, str], str] = field(default_factory=dict)

    @property
    def recall(self) -> float:
        vals = list(self.labels.values())
        return sum(1 for v in vals if v == "Supported") / len(vals) if vals else float("nan")


def _require(**kw: str) -> None:
    missing = sorted(k for k, v in kw.items() if not (isinstance(v, str) and v.strip()))
    if missing:
        raise MissingProvenance(
            f"reference is missing {missing}. Every published comparison needs the system, "
            "the judge model, the source URL and the pinned commit, or the row cannot be "
            "cited and must not be printed."
        )


def normalize_key_point_id(kp_id: str) -> str:
    """`kp7` (our node id) and `7` (upstream's point_number) are the same unit.

    The builder mints node ids as kp<point_number> precisely so this is a prefix strip and
    not a join through a lookup table that could silently mismatch.
    """
    s = str(kp_id).strip()
    return s[2:] if s.lower().startswith("kp") else s


def load_kpr_reference(
    path: Path | str, *, system: str, judge_model: str, source_url: str, commit: str
) -> ReferenceKPR:
    """Read upstream's evaluation_results_kpr_*.json shape. sha256 is of the file as read."""
    _require(system=system, judge_model=judge_model, source_url=source_url, commit=commit)
    p = Path(path)
    raw = p.read_bytes()
    data = json.loads(raw)
    labels: dict[tuple[str, str], str] = {}
    for task_id, rec in data.items():
        if not rec:  # upstream writes null for a query it could not evaluate
            continue
        for kp, value in rec.get("labels", {}).items():
            label = value[0] if isinstance(value, (list, tuple)) and value else value
            if isinstance(label, str) and label in KPR_LABELS:
                labels[(str(task_id), normalize_key_point_id(kp))] = label
    return ReferenceKPR(
        system=system,
        judge_model=judge_model,
        source_url=source_url,
        commit=commit,
        sha256=hashlib.sha256(raw).hexdigest(),
        labels=labels,
    )


# ------------------------------------------------------------------ the delta


@dataclass(frozen=True, slots=True)
class KPRDelta:
    system: str
    ours_model: str
    ref_model: str
    n_units: int
    ours_recall: float
    ref_recall: float
    agreement: float
    alpha: float
    confusion: Mapping[tuple[str, str], int]
    ref_sha256: str
    ref_commit: str
    n_ours_only: int
    n_ref_only: int

    @property
    def delta_recall(self) -> float:
        return self.ours_recall - self.ref_recall

    def as_dict(self) -> dict[str, object]:
        return {
            "system": self.system,
            "ours_model": self.ours_model,
            "ref_model": self.ref_model,
            "n_units": self.n_units,
            "ours_recall": self.ours_recall,
            "ref_recall": self.ref_recall,
            "delta_recall": self.delta_recall,
            "agreement": self.agreement,
            "alpha": self.alpha,
            "confusion": {f"{a}->{b}": n for (a, b), n in sorted(self.confusion.items())},
            "ref_sha256": self.ref_sha256,
            "ref_commit": self.ref_commit,
            "n_ours_only": self.n_ours_only,
            "n_ref_only": self.n_ref_only,
        }


def kpr_delta(ours: Sequence[Judgment], ref: ReferenceKPR, *, ours_model: str = "") -> KPRDelta:
    """Confusion, agreement, alpha and the recall gap over the SHARED units only."""
    mine: dict[tuple[str, str], str] = {}
    model = ours_model
    for j in ours:
        if j.criterion != "keypoint" or j.label is None or j.key_point_id is None:
            continue
        mine[(str(j.task_id), normalize_key_point_id(j.key_point_id))] = j.label
        model = model or j.judge_model

    shared = sorted(set(mine) & set(ref.labels))
    if not shared:
        raise NoOverlap(
            f"our judgments and reference '{ref.system}' share no (task, key point) unit "
            f"(ours {len(mine)}, reference {len(ref.labels)}). Judge the same reports with "
            "the same key-point ids before asking for a fidelity delta."
        )

    confusion: dict[tuple[str, str], int] = {}
    agree = 0
    for unit in shared:
        pair = (mine[unit], ref.labels[unit])
        confusion[pair] = confusion.get(pair, 0) + 1
        agree += pair[0] == pair[1]

    ours_recall = sum(1 for u in shared if mine[u] == "Supported") / len(shared)
    ref_recall = sum(1 for u in shared if ref.labels[u] == "Supported") / len(shared)
    alpha = krippendorff_alpha_nominal(
        {f"{t}/{k}": {"ours": mine[(t, k)], "reference": ref.labels[(t, k)]} for t, k in shared}
    )
    return KPRDelta(
        system=ref.system,
        ours_model=model,
        ref_model=ref.judge_model,
        n_units=len(shared),
        ours_recall=ours_recall,
        ref_recall=ref_recall,
        agreement=agree / len(shared),
        alpha=alpha,
        confusion=confusion,
        ref_sha256=ref.sha256,
        ref_commit=ref.commit,
        n_ours_only=len(set(mine) - set(ref.labels)),
        n_ref_only=len(set(ref.labels) - set(mine)),
    )


# ------------------------------------------------------------------ aggregate references


@dataclass(frozen=True, slots=True)
class ReferenceAggregate:
    """A published per-criterion mean (report quality, citation support), with provenance."""

    system: str
    judge_model: str
    source_url: str
    commit: str
    values: Mapping[str, float]
    n: int = 0


def load_aggregate_reference(path: Path | str) -> ReferenceAggregate:
    """JSON: {system, judge_model, source_url, commit, n, values:{criterion: mean}}.

    The four provenance fields are REQUIRED by the loader, so a hand-transcribed number
    cannot enter a table without saying where it was transcribed from.
    """
    d = json.loads(Path(path).read_text())
    _require(
        system=d.get("system", ""),
        judge_model=d.get("judge_model", ""),
        source_url=d.get("source_url", ""),
        commit=d.get("commit", ""),
    )
    values = {str(k): float(v) for k, v in dict(d.get("values", {})).items()}
    if not values:
        raise MissingProvenance("reference carries no values; an empty table is not a comparison")
    return ReferenceAggregate(
        system=d["system"],
        judge_model=d["judge_model"],
        source_url=d["source_url"],
        commit=d["commit"],
        values=values,
        n=int(d.get("n", 0)),
    )


def aggregate_delta(
    ours: Mapping[str, float], ref: ReferenceAggregate
) -> tuple[list[dict[str, object]], list[str]]:
    """(rows, criteria we could not compare). The second element is never silently dropped."""
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for criterion, ref_value in sorted(ref.values.items()):
        if criterion not in ours:
            missing.append(criterion)
            continue
        rows.append(
            {
                "criterion": criterion,
                "ours": ours[criterion],
                "reference": ref_value,
                "delta": ours[criterion] - ref_value,
            }
        )
    if not rows:
        raise NoOverlap(
            f"no criterion is present in both our scores {sorted(ours)} and reference "
            f"'{ref.system}' {sorted(ref.values)}"
        )
    return rows, missing


# ------------------------------------------------------------------ rendering


def render_kpr_delta(d: KPRDelta) -> str:
    """A markdown table. Provenance is in the caption, not a footnote nobody copies."""
    lines = [
        f"**Judge fidelity vs `{d.system}`** "
        f"(reference judge `{d.ref_model}`, ours `{d.ours_model or 'unpinned'}`; "
        f"reference commit `{d.ref_commit[:12]}`, sha256 `{d.ref_sha256[:12]}`)",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| shared (task, key point) units | {d.n_units} |",
        f"| key-point recall, ours | {d.ours_recall:.3f} |",
        f"| key-point recall, reference | {d.ref_recall:.3f} |",
        f"| delta (ours - reference) | {d.delta_recall:+.3f} |",
        f"| label agreement | {d.agreement:.3f} |",
        f"| Krippendorff alpha (nominal) | {d.alpha:.3f} |",
        f"| units only ours / only reference | {d.n_ours_only} / {d.n_ref_only} |",
        "",
        "| ours \\ reference | " + " | ".join(KPR_LABELS) + " |",
        "|---" * (len(KPR_LABELS) + 1) + "|",
    ]
    for a in KPR_LABELS:
        cells = " | ".join(str(d.confusion.get((a, b), 0)) for b in KPR_LABELS)
        lines.append(f"| {a} | {cells} |")
    return "\n".join(lines)


def render_aggregate_delta(rows: Sequence[Mapping[str, object]], ref: ReferenceAggregate) -> str:
    head = [
        f"**Judge fidelity vs `{ref.system}`** (reference judge `{ref.judge_model}`, "
        f"n={ref.n}, source {ref.source_url} @ `{ref.commit[:12]}`)",
        "",
        "| criterion | ours | reference | delta |",
        "|---|---|---|---|",
    ]
    body = [
        f"| {r['criterion']} | {float(r['ours']):.3f} | {float(r['reference']):.3f} | "
        f"{float(r['delta']):+.3f} |"
        for r in rows
    ]
    return "\n".join(head + body)
