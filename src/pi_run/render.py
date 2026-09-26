"""RENDER: parquet -> tables/*.tex and figures/*.pdf. Nothing else is an input.

WHY RENDER REFUSES INSTEAD OF OVERWRITING. The failure this stage exists to prevent is not a
crash, it is a table in a paper whose numbers came from a person. Two refusals enforce that:

  * PROVENANCE DRIFT. Every table carries a provenance digest over the run_id set, the
    scorer_hash, the graph_version, the code versions and the CONTENT HASH OF EVERY INPUT
    PARQUET FILE. If any of those moved since `pi agg` produced the table, rendering stops.
    The remedy is to re-run `pi agg` and look at what changed, never to render anyway.

  * HAND EDITS. Each .tex carries the sha256 of its own body. If the body no longer matches,
    a human edited a rendered number, and render refuses to silently overwrite the evidence.
    `--force` exists, prints loudly, and is the only way past it.

matplotlib only, no seaborn. The palette is Okabe-Ito (colorblind-safe) and EVERY series is
also distinguished by linestyle, marker and hatch, because the paper prints in grayscale and
a figure whose only encoding is hue becomes one line in a photocopier.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pi_eval import report as rp

HEADER_END = "% pi-header-end"

# Okabe-Ito. Chosen for deuteranopia/protanopia separation AND for spread in luminance, so
# the series stay distinguishable once the printer throws the hue away.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 3, 1)), (0, (1, 1)))
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
HATCHES = ("///", "\\\\\\", "xxx", "...", "ooo", "++", "**")


class RenderRefused(RuntimeError):
    """A refusal, never a warning. Exits non-zero so CI cannot step over it."""


# --------------------------------------------------------------------------- latex


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(s: str) -> str:
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in str(s))


def to_latex(table: rp.Table) -> str:
    """booktabs, one column per declared column, warnings first and impossible to miss."""
    cols = table.columns
    out = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{latex_escape(table.title)}}}",
        f"\\label{{tab:{latex_escape(table.table_id)}}}",
        "\\begin{tabular}{" + "l" * len(cols) + "}",
        "\\toprule",
        " & ".join(f"\\textbf{{{latex_escape(c)}}}" for c in cols) + " \\\\",
        "\\midrule",
    ]
    dropped: list[str] = []
    for r in table.rows:
        out.append(" & ".join(latex_escape(rp.fmt(r.get(c))) for c in cols) + " \\\\")
        # A ROW FIELD OUTSIDE `columns` USED TO VANISH. The primary endpoint's row carries a
        # `notes` explaining a reduced n -- "excluded 6 task(s) split across seeds (no
        # consistent direction; coercing them would have scored each a loss)" -- and the .tex
        # printed the shrunken n with the reason deleted. A number whose denominator changed
        # for a stated reason, published without the reason.
        #
        # Underscore-prefixed fields are internal plumbing (`_run_ids`) and are not data.
        for k, v in sorted(r.items()):
            if k in cols or k.startswith("_") or v in (None, "", ()):
                continue
            line = f"{k}: {v}"
            if line not in dropped:
                dropped.append(line)
    if not table.rows:
        out.append(f"\\multicolumn{{{len(cols)}}}{{c}}{{\\emph{{no eligible rows}}}} \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    for w in table.warnings:
        out.append(f"\\\\[2pt]\\textbf{{WARNING:}} {latex_escape(w)}")
    for n in table.notes:
        out.append(f"\\\\[2pt]\\footnotesize {latex_escape(n)}")
    for d in dropped:
        out.append(f"\\\\[2pt]\\footnotesize {latex_escape(d)}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def _stamp(table: rp.Table, body: str) -> str:
    prov = dict(table.provenance)
    head = [
        f"% pi-render-table: {table.table_id}",
        f"% pi-provenance-digest: {prov.get('provenance_digest', '')}",
        f"% pi-inputs-hash: {prov.get('inputs_hash', '')}",
        f"% pi-scorer-hash: {prov.get('scorer_hash', '')}",
        f"% pi-graph-version: {prov.get('graph_version', '')}",
        f"% pi-n-run-ids: {prov.get('n_run_ids', 0)}",
        # THE SEAL STATUS, WHICH THE ARTIFACT DID NOT CARRY. Every .tex here captions itself
        # "(preregistered, uncorrected)" and footnotes "one preregistered contrast per suite",
        # while `make prereg-verify` exits 1 -- there is no prereg/ directory at all -- and the
        # header carried provenance-digest, inputs-hash, scorer-hash, graph-version,
        # n-run-ids, table-digest and content-sha and said nothing about whether anything had
        # been sealed. The claim was in the prose and the evidence was in neither.
        #
        # "unsealed" is the honest reading of None: no prereg directory exists, which is not
        # the same as a seal that failed to verify.
        f"% pi-prereg-verified: "
        f"{'unsealed' if prov.get('prereg_verified') is None else prov.get('prereg_verified')}",
        f"% pi-prereg-exclusions: {prov.get('prereg_exclusions', '')}",
        f"% pi-table-digest: {table.digest()}",
        f"% pi-content-sha256: {hashlib.sha256(body.encode()).hexdigest()}",
        "% DO NOT EDIT BY HAND. `pi render` compares the body against the hash above and",
        "% refuses to overwrite a table a human has changed. Fix the pipeline, not the file.",
        HEADER_END,
    ]
    return "\n".join(head) + "\n" + body


def read_stamp(path: Path) -> tuple[dict[str, str], str] | None:
    """(header fields, body) for a rendered .tex, or None if it carries no stamp."""
    if not path.exists():
        return None
    text = path.read_text()
    if HEADER_END not in text:
        return None
    head, body = text.split(HEADER_END + "\n", 1)
    fields: dict[str, str] = {}
    for line in head.splitlines():
        if line.startswith("% pi-") and ":" in line:
            k, v = line[2:].split(":", 1)
            fields[k.strip()] = v.strip()
    return fields, body


def check_not_hand_edited(path: Path) -> None:
    st = read_stamp(path)
    if st is None:
        return
    fields, body = st
    want = fields.get("pi-content-sha256", "")
    got = hashlib.sha256(body.encode()).hexdigest()
    if want and want != got:
        raise RenderRefused(
            f"{path} was edited by hand: its body hashes to {got[:12]} but its own stamp "
            f"says {want[:12]}. Refusing to overwrite the evidence. Re-run the pipeline, or "
            "pass --force if you are certain the edit was worthless."
        )


# --------------------------------------------------------------------------- provenance gate


def check_provenance(table: rp.Table, tables_dir: Path) -> None:
    """Refuse when anything the table was scored against has moved."""
    _check_provenance(table.table_id, dict(table.provenance), tables_dir)


def _check_provenance(artifact_id: str, fresh: dict, root: Path) -> None:
    """The gate, for a TABLE OR A FIGURE.

    Figures did not pass through it. `pi render --all` would refuse every table with "input
    hash changed since the table was scored" and then overwrite every figure PDF with numbers
    computed from exactly those changed inputs -- and `_write_figure` saved the PDF BEFORE
    computing provenance, so there was no moment at which a check could have run.

    Reproduced: append one row to scores.parquet without re-running `pi agg`, then
    `pi render --all` -> 7 tables refused, "tables": [], 14 refusal lines, and F1_frontier.pdf
    rewritten. The figure is where the frontier's simultaneous band is published, so the
    artifact that survived the refusal is the one that licenses "that dip is noise".
    """
    stored_path = root / artifact_id / "provenance.json"
    if not stored_path.exists():
        return
    try:
        stored = json.loads(stored_path.read_text())
    except ValueError as exc:
        raise RenderRefused(f"{stored_path} is not readable JSON: {exc}") from exc
    if stored.get("inputs_hash") != fresh.get("inputs_hash"):
        moved = [
            k
            for k, v in (fresh.get("inputs") or {}).items()
            if (stored.get("inputs") or {}).get(k) != v
        ]
        raise RenderRefused(
            f"{artifact_id}: input hash changed since it was scored "
            f"({', '.join(moved) or 'unknown file'}). The rendered artifact would not be the "
            "one that was aggregated and reviewed. Re-run `pi agg` and inspect the diff."
        )
    if stored.get("provenance_digest") != fresh.get("provenance_digest"):
        changed = sorted(
            k
            for k in set(stored) | set(fresh)
            if k not in ("generated_at_utc", "provenance_digest") and stored.get(k) != fresh.get(k)
        )
        raise RenderRefused(
            f"{artifact_id}: provenance no longer matches (changed: {changed}). "
            "Refusing to overwrite an artifact whose provenance moved."
        )


# --------------------------------------------------------------------------- tables


def _inputs_moved(agg: rp.Agg, tables_dir: Path, table_ids: Sequence[str]) -> str:
    """Why the aggregation is stale, or "" when it is not.

    FIGURES BYPASSED THE PROVENANCE GATE ENTIRELY. `pi render --all` would refuse every table
    with "input hash changed since it was scored" and then overwrite every figure PDF with
    numbers computed from exactly those changed inputs. Reproduced: append one row to
    scores.parquet without re-running `pi agg`, then render -> 7 tables refused, "tables": [],
    and F1_frontier.pdf rewritten anyway. F1 is where the frontier's simultaneous band is
    published, so the artifact that survived the refusal is the one that licenses "that dip is
    noise".

    Checked against the TABLES' stored provenance rather than a figure's own, because that is
    the record `pi agg` writes and refreshes -- so the figures are stale exactly when the tables
    are, and one `pi agg` clears both.

    ONLY the tables this render builds. Globbing every directory under tables/ refused all three
    figures on the strength of K_killswitch, which `pi report killswitch` writes and
    `pi render --all` does not touch: 7 of 8 stored provenances matched and the eighth, from a
    different command, poisoned the check.
    """
    from pinq.ids import canon, h

    fresh = h("inputs", canon(sorted(rp.file_hashes(agg.parquet_dir).items())))
    for tid in sorted(table_ids):
        prov_path = Path(tables_dir) / tid / "provenance.json"
        if not prov_path.exists():
            continue
        try:
            stored = json.loads(prov_path.read_text())
        except ValueError:
            continue
        if stored.get("inputs_hash") and stored["inputs_hash"] != fresh:
            moved = [
                k
                for k, v in rp.file_hashes(agg.parquet_dir).items()
                if (stored.get("inputs") or {}).get(k) != v
            ]
            return (
                f"input hash changed since the aggregation ({', '.join(moved) or 'unknown file'})."
                " A figure drawn now would not match the tables. Re-run `pi agg`."
            )
    return ""


def render_table(table: rp.Table, tables_dir: str | Path, *, force: bool = False) -> Path:
    d = Path(tables_dir)
    d.mkdir(parents=True, exist_ok=True)
    tex = d / f"{table.table_id}.tex"
    if not force:
        check_provenance(table, d)
        check_not_hand_edited(tex)
    tex.write_text(_stamp(table, to_latex(table)))
    table.write(d)
    (d / table.table_id / f"{table.table_id}.txt").write_text(table.to_text() + "\n")
    return tex


# --------------------------------------------------------------------------- figures


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
    return plt


@dataclass(frozen=True, slots=True)
class Figure:
    figure_id: str
    pdf: Path
    png: Path
    provenance: Mapping[str, Any]
    summary: Mapping[str, Any]


def _write_figure(fig, agg: rp.Agg, figures_dir, figure_id, run_ids, query, summary) -> Figure:
    d = Path(figures_dir)
    d.mkdir(parents=True, exist_ok=True)
    pdf, png = d / f"{figure_id}.pdf", d / f"{figure_id}.png"
    # The staleness gate for figures lives in `render_all` (see `_inputs_moved`): a figure has
    # no stored provenance but its own last render, which `pi agg` cannot refresh, so gating it
    # against itself would make a stale figure permanently unrenderable.
    prov = agg.prov(figure_id, run_ids, query, f"pi render --figure {figure_id}", **summary)
    fig.savefig(pdf)
    fig.savefig(png)
    side = d / figure_id
    side.mkdir(parents=True, exist_ok=True)
    (side / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return Figure(figure_id, pdf, png, prov, summary)


def f1_frontier(agg: rp.Agg, figures_dir: str | Path, *, n_boot: int = 2000) -> Figure:
    """F1: quality vs CUMULATIVE SPEND, with pointwise AND sup-t simultaneous bands.

    The scalar AUC appears only in the legend of the curve that produced it: `rp.auc_row`
    raises rather than hand back a number with no curve behind it.
    """
    plt = _mpl()
    curves, grid = rp.frontier_curves(agg, n_boot=n_boot)
    if not curves:
        raise RenderRefused("F1: no frontier ladders in scores.parquet; run `pi score` first.")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    summary: dict[str, Any] = {"grid": list(grid), "arms": {}}
    for i, (arm, curve) in enumerate(sorted(curves.items())):
        scalars = rp.auc_row(curve)  # refuses if the curve is not there
        col, ls, mk, ht = (
            PALETTE[i % len(PALETTE)],
            LINESTYLES[i % len(LINESTYLES)],
            MARKERS[i % len(MARKERS)],
            HATCHES[i % len(HATCHES)],
        )
        ax.fill_between(
            curve.grid,
            curve.lo_simultaneous,
            curve.hi_simultaneous,
            color=col,
            alpha=0.10,
            hatch=ht,
            edgecolor=col,
            linewidth=0.0,
            zorder=1,
        )
        ax.fill_between(
            curve.grid,
            curve.lo_pointwise,
            curve.hi_pointwise,
            color=col,
            alpha=0.22,
            linewidth=0.0,
            zorder=2,
        )
        ax.plot(
            curve.grid,
            curve.mean,
            color=col,
            linestyle=ls,
            marker=mk,
            markersize=4,
            linewidth=1.8,
            zorder=3,
            label=(
                f"{arm}  AUC={scalars['auc']:.3f} "
                f"[{scalars['auc_lo']:.3f},{scalars['auc_hi']:.3f}], n={curve.n_tasks}"
            ),
        )
        summary["arms"][arm] = {
            "mean": list(curve.mean),
            "lo_pointwise": list(curve.lo_pointwise),
            "hi_pointwise": list(curve.hi_pointwise),
            "lo_simultaneous": list(curve.lo_simultaneous),
            "hi_simultaneous": list(curve.hi_simultaneous),
            **scalars,
        }
    ax.set_xscale("log", base=2)
    ax.set_xlabel("cumulative spend (retrieval calls, log$_2$)")
    ax.set_ylabel("Q = evidence coverage")
    ax.set_title("F1  Budget frontier: pointwise (solid fill) and sup-t (hatched) bands")
    ax.legend(loc="best", frameon=True, framealpha=0.85, edgecolor="none", fontsize=7)
    return _write_figure(
        fig,
        agg,
        figures_dir,
        "F1_frontier",
        _frontier_run_ids(agg),
        rp.frontier_query(agg, None),
        summary,
    )


def _frontier_run_ids(agg: rp.Agg) -> list[str]:
    out: set[str] = set()
    for key in rp.frontier_ladders(agg):
        out |= set(key[4])
    return sorted(out)


def f2_coverage_at_depth(agg: rp.Agg, figures_dir: str | Path) -> Figure:
    """F2: C@d by arm. Every tick label carries |V_d|, exactly as the table does.

    THE SERIES KEY IS (suite, arm), NOT arm. `cad_query` groups by
    (suite_id, arm_id, depth, fam); this accumulator keyed on (arm, depth) alone, so with more
    than one suite in the parquet the LAST suite in `ORDER BY suite_id` silently overwrote the
    others and the figure showed one suite's numbers under a label naming only the arm. The
    tier-1 grid runs three suites at once, so that is the ordinary case rather than an edge
    one. `|V_d|` was `max`ed across suites for the same reason -- a cardinality that belongs to
    whichever suite happened to have the most nodes at that depth.
    """
    plt = _mpl()
    q = rp.cad_query(agg)
    raw = agg.sql(q)
    acc: dict[tuple[str, str], dict[int, float]] = {}
    card: dict[tuple[str, int], float] = {}
    run_ids: set[str] = set()
    for r in raw:
        d, fam, arm = int(r["depth"]), str(r["fam"]), str(r["arm_id"])
        suite = str(r.get("suite_id", ""))
        run_ids |= {str(x) for x in (r.get("run_ids") or ())}
        if fam == "cad":
            acc.setdefault((suite, arm), {})[d] = float(r["value"])
        else:
            # Keyed by (suite, depth). It used to be `card[d] = max(card.get(d, 0), value)`,
            # which reported whichever suite happened to have the most nodes at that depth as
            # everyone's |V_d|.
            card[(suite, d)] = float(r["value"])
    if not acc:
        raise RenderRefused("F2: no C@d rows in scores.parquet; run `pi score` first.")
    depths = sorted({d for (_s, d) in card})
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    arms = sorted(acc)  # (suite, arm) pairs: one series each, nothing overwritten
    width = 0.8 / max(1, len(arms))
    for i, arm in enumerate(arms):
        xs = [d + (i - (len(arms) - 1) / 2) * width for d in depths]
        # A depth this (suite, arm) has no row for is MISSING, not zero: a zero-height bar in a
        # coverage figure reads as "covered nothing", which is a claim about the policy.
        ys = [acc[arm].get(d, float("nan")) for d in depths]
        ax.bar(
            xs,
            ys,
            width=width * 0.92,
            color=PALETTE[i % len(PALETTE)],
            hatch=HATCHES[i % len(HATCHES)],
            edgecolor="black",
            linewidth=0.6,
            label="/".join(arm),
        )
    ax.set_xticks(depths)
    # |V_d| summed over the suites in the figure, and the suites are named in the legend --
    # rather than max'd, which silently reported one suite's cardinality as everyone's.
    totals = {d: sum(v for (_s, dd), v in card.items() if dd == d) for d in depths}
    ax.set_xticklabels([f"d={d}\n$|V_{d}|$={totals[d]:.0f}" for d in depths])
    ax.set_ylim(0, 1.30)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("C@d (resolve rung)")
    ax.set_title("F2  Coverage by dependency depth, with $|V_d|$")
    # Headroom plus a single legend row: a legend box sitting on top of a bar is how a
    # reader mistakes an occluded arm for a missing one.
    ax.legend(loc="upper center", ncol=max(1, len(arms)), frameon=False, fontsize=7)
    return _write_figure(
        fig,
        agg,
        figures_dir,
        "F2_coverage_at_depth",
        sorted(run_ids),
        q,
        {
            "depths": depths,
            "cardinality": {f"{s_}/{d}": v for (s_, d), v in sorted(card.items())},
            "series": {"/".join(a): acc[a] for a in arms},
        },
    )


def phi_query(agg: rp.Agg) -> str:
    return (
        "SELECT r.arm_id, split_part(s.metric_name, '#', 1) AS fam, s.value, r.run_id\n"
        "FROM scores s JOIN runs r ON r.run_id = s.run_id\n"
        f"WHERE s.scorer_hash = {rp.sql_literal(agg.scorer_hash)} AND ({agg.predicate})\n"
        "  AND split_part(s.metric_name, '#', 1) IN ('phi_loo', 'phi_null')\n"
        "ORDER BY 1, 2"
    )


def f3_phi_vs_null(agg: rp.Agg, figures_dir: str | Path) -> Figure:
    """F3: phi distribution against the matched-volume random-question null.

    The null is what gives phi a meaningful zero. Without it, 'the mean phi is 0.08' is a
    statement about how much any k units help, not about question selection.
    """
    plt = _mpl()
    q = phi_query(agg)
    raw = agg.sql(q)
    phis: dict[str, list[float]] = {}
    nulls: list[float] = []
    run_ids: set[str] = set()
    for r in raw:
        run_ids.add(str(r["run_id"]))
        v = float(r["value"])
        if math.isnan(v):
            continue
        if str(r["fam"]) == "phi_null":
            nulls.append(v)
        else:
            phis.setdefault(str(r["arm_id"]), []).append(v)
    if not phis and not nulls:
        raise RenderRefused("F3: no phi rows in scores.parquet; run `pi score` first.")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    series = sorted(phis.items()) + ([("random-question null", nulls)] if nulls else [])
    lo = min((min(v) for _, v in series if v), default=0.0)
    hi = max((max(v) for _, v in series if v), default=1.0)
    if hi - lo < 1e-9:
        lo, hi = lo - 0.05, hi + 0.05
    bins = [lo + (hi - lo) * i / 24 for i in range(25)]
    for i, (label, vals) in enumerate(series):
        if not vals:
            continue
        ax.hist(
            vals,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            color=PALETTE[i % len(PALETTE)],
            hatch=HATCHES[i % len(HATCHES)],
            label=f"{label} (n={len(vals)}, mean={sum(vals) / len(vals):+.3f})",
            density=True,
        )
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"$\phi_{LOO}(q) = Q(E_T) - Q(E_T \setminus ev(q))$,  Q = evidence coverage")
    ax.set_ylabel("density")
    ax.set_title("F3  Per-question value against the matched-volume null")
    ax.legend(frameon=False, fontsize=7)
    return _write_figure(
        fig,
        agg,
        figures_dir,
        "F3_phi_vs_null",
        sorted(run_ids),
        q,
        {
            "n_null": len(nulls),
            "null_mean": (sum(nulls) / len(nulls)) if nulls else float("nan"),
            "arms": {a: len(v) for a, v in sorted(phis.items())},
        },
    )


FIGURES: dict[str, Any] = {
    "F1_frontier": f1_frontier,
    "F2_coverage_at_depth": f2_coverage_at_depth,
    "F3_phi_vs_null": f3_phi_vs_null,
}


# --------------------------------------------------------------------------- the stage


@dataclass(frozen=True, slots=True)
class RenderResult:
    tables: tuple[str, ...]
    figures: tuple[str, ...]
    refused: tuple[str, ...]
    warnings: tuple[str, ...]
    # Trained-arm rows that reached a table they must not be in. RETURNED, not printed: the
    # wording belongs to `pi_run.cli.contamination_report`, which `pi agg` already prints, and
    # two copies of a refusal are two things to keep in step. Never a warning -- see `ok`.
    contaminated: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "tables": list(self.tables),
            "figures": list(self.figures),
            "refused": list(self.refused),
            "warnings": list(self.warnings),
            "contaminated": [dict(v) for v in self.contaminated],
            "ok": not (self.refused or self.contaminated),
        }


def render_all(
    agg: rp.Agg,
    *,
    tables_dir: str | Path = "tables",
    figures_dir: str | Path = "figures",
    table_ids: Sequence[str] | None = None,
    figure_ids: Sequence[str] | None = None,
    tasks: int | None = None,
    force: bool = False,
    n_boot: int = 2000,
    ids_dir: str | Path | None = None,
) -> RenderResult:
    """Build every artifact from parquet, refusing rather than papering over a mismatch.

    THE CONTAMINATION CHECK RUNS HERE TOO, exactly as `cmd_agg` runs it. `pi agg` called
    `train_split_violations` on every table it built and was its ONLY caller; this function
    builds the same tables and wrote the .tex files the paper includes, and checked nothing.
    So a trained arm scored on tasks it may have been fitted on was one command away from
    reaching the paper unremarked -- and the command it was away from is the one that produces
    a screenful of numbers rather than the artifact.

    `TrainIdSetUnresolved` is deliberately NOT caught: a stamped id set that cannot be found is
    an unanswerable question, and answering it "clean" is the failure the whole layer exists to
    remove.
    """
    wanted_t = list(table_ids) if table_ids is not None else list(rp.ALL_TABLES)
    wanted_f = list(figure_ids) if figure_ids is not None else list(FIGURES)
    done_t: list[str] = []
    done_f: list[str] = []
    refused: list[str] = []
    warns: list[str] = []
    built: dict[str, rp.Table] = {}
    contaminated: list[Mapping[str, Any]] = []

    for tid in wanted_t:
        builder = rp.ALL_TABLES.get(tid)
        if builder is None:
            refused.append(f"{tid}: unknown table")
            continue
        try:
            table = builder(agg, tasks=tasks) if _takes_tasks(builder) else builder(agg)
            built[tid] = table
            render_table(table, tables_dir, force=force)
            done_t.append(tid)
            warns += [f"{tid}: {w}" for w in table.warnings]
            # ALWAYS, not behind a flag, and on the table that was actually built -- same call
            # `cmd_agg` makes, on the same `provenance["run_ids"]`.
            contaminated += [
                {**v, "table": tid}
                for v in rp.train_split_violations(
                    agg, table.provenance.get("run_ids", []), ids_dir=ids_dir
                )
            ]
        except RenderRefused as exc:
            refused.append(str(exc))

    stale = "" if force else _inputs_moved(agg, Path(tables_dir), wanted_t)
    for fid in wanted_f:
        builder = FIGURES.get(fid)
        if builder is None:
            refused.append(f"{fid}: unknown figure")
            continue
        if stale:
            refused.append(f"{fid}: {stale}")
            continue
        try:
            if fid == "F1_frontier":
                builder(agg, figures_dir, n_boot=n_boot)
            else:
                builder(agg, figures_dir)
            done_f.append(fid)
        except RenderRefused as exc:
            refused.append(str(exc))

    # The scalar never appears without the curve, enforced at the ARTIFACT level: a
    # Delta-AUC row in the secondary table is only legible next to F1.
    sec = built.get("T1b_secondary")
    if sec is not None and any(
        r.get("metric") == "frontier_auc" and int(r.get("n") or 0) > 0 for r in sec.rows
    ):
        if "F1_frontier" not in done_f:
            raise RenderRefused(
                "T1b_secondary carries a frontier Delta-AUC row but F1_frontier was not "
                "produced. The scalar hides curve crossings and must never be emitted "
                "alone; render the figure or drop the row."
            )
    return RenderResult(
        tuple(done_t), tuple(done_f), tuple(refused), tuple(warns), tuple(contaminated)
    )


def _takes_tasks(fn: Any) -> bool:
    import inspect

    return "tasks" in inspect.signature(fn).parameters
