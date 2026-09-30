#!/usr/bin/env python3
"""ICLR 2027 figures and tables, generated from the artifact records and never by hand.

    .venv/bin/python scripts/paper_figures_iclr.py --out "paper/iclr2027 2/figures" [--only fig2]

Four assets: `fig2_frontier` (the cost frontier on three suites), `fig3_stopping` (the two
per-decision stopping cells), `table1_heldout.tex` (both axes at equal spend on the held-out
splits) and `table2_mechanism.tex` (which supervision moves what, on development).

The style is copied from `src/pi_run/render.py` and `scripts/figures_programme.py` so the three
sets sit together without a seam: the same rcParams (these PDFs embed TrueType fonts, not Type 3),
the same Okabe-Ito palette, a PDF and a PNG
per figure, and a provenance record whose `inputs` map every source file to its sha256.

WHAT THIS SCRIPT REFUSES TO DO.

  * It does not draw a number it has not reproduced. The frontier panels are parsed out of the
    promoted `.md` records, which are the citable post-fix intervals, and then every level cell
    is RE-DERIVED from the scored parquet store beside the record, selected by that cell's own
    run-id list. A cell whose recomputation disagrees with the record past `TOL` raises
    `Mismatch`, naming both numbers, and NOTHING is written. An axis with plausible numbers on
    it is the easiest way for an invented number to reach a paper.

  * It records WHICH ESTIMAND reproduced. `FRONTIER.md`'s coverage column is the unweighted mean
    over musique's 186 template clusters and NOT the pooled run mean (the two differ by up to
    ~0.006 there, because 14 templates hold two tasks each; see `FRONTIER.md`'s "Estimator"
    section and `paper/sections/mechanism.tex`). Its `mean retrieval_calls` column, on the same
    rows, is the POOLED run mean. Both readings are computed for every cell, the one that
    reproduces is recorded per cell in the provenance, and a cell that reproduces under neither
    is a failure rather than a choice.

  * It puts no arm id, field name, file name or hash into a rendered label or legend. Those
    belong in the provenance record and in the `%` comments of the emitted tabulars, which is
    where a reader who wants them can find them. The reader words are the paper's own.

  * The two dev-population assets are cross-checked against the PRINTED appendix. The gate
    verdicts on disk carry PRE-FIX bootstrap endpoints: `artifacts/bca_provenance_20260918/`
    records, per printed tex cell, the pre-fix pair, the post-fix pair and the point estimate,
    and the appendix prints the post-fix pair. So the point estimate, `n` and the two stopping
    cells are read from the verdict JSON, the interval is read from the post-fix record, and
    every one of them is asserted equal to the string `paper/appendix_training.tex` actually
    prints before anything is written.

Dependencies are thin on purpose: stdlib, matplotlib and duckdb. Nothing first-party is
imported, so this script cannot be the thing that breaks when the run stage changes, and it
cannot reach gold.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import duckdb

REPO = Path(__file__).resolve().parents[1]
TOOL = "scripts/paper_figures_iclr.py"
# SEED 0 IS NOT DRAWN in the main-text assets (author decision 2026-09-23): the resumed run is
# disclosed once where the questioner is defined and read in one appendix section. Every lock that
# reproduces a published seed-0 value still runs; only the drawing and the printing stop.
SHOW_SEED0 = False

# Equality tolerance for "the recomputation reproduces the record". The records print four
# decimals, so the worst honest rounding error is 5e-5; 1e-4 admits that and nothing larger.
TOL = 1e-4


class Mismatch(Exception):
    """A drawn value does not reproduce from its own store. Nothing is written."""


class SourceMissing(Exception):
    """A named input does not exist on this checkout."""


# --------------------------------------------------------------------------- sources

FRONTIER_MUSIQUE = REPO / "artifacts/frontier/FRONTIER.md"
FRONTIER_STRATEGYQA = REPO / "artifacts/frontier_strategyqa/FRONTIER_STRATEGYQA.md"
FRONTIER_FRAMES = REPO / "artifacts/frames_frontier/FRAMES_FRONTIER.md"

TESTSPLIT_CONTRASTS = REPO / "artifacts/testsplit_plan_metrics_20260918/contrasts.json"
TESTSPLIT_RESULT = REPO / "artifacts/testsplit_plan_metrics_20260918/RESULT.md"
SYMMETRIC_RECORD = REPO / "artifacts/symmetric_matched_cost_20260919/contrasts.symmetric.json"
TESTSPLIT_VERDICTS = REPO / "artifacts/testsplit_qa/verdicts"
TESTSPLIT_STORE = REPO / "artifacts/testsplit_qa/scores_parquet"

BCA_CELLS = REPO / "artifacts/bca_provenance_20260918/cells.json"
APPENDIX_TRAINING = REPO / "paper/appendix_training.tex"

CAPS = (4, 8, 12, 16, 24)


# --------------------------------------------------------------------------- provenance


def sha256_file(p: Path) -> str:
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def require(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            raise SourceMissing(rel(p))


def rel(p: Path) -> str:
    """A path as the repository names it. An absolute home path in a generated artifact is how
    a research repo stops being reproducible for anyone but its author, and
    `scripts/check_no_home_paths.sh` fails the build over exactly that."""
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def provenance(
    *,
    asset_id: str,
    sources: Sequence[Path],
    query: str,
    values: Any,
    scorer_hash: Iterable[str] = (),
    graph_version: Iterable[str] = (),
    run_ids_from: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """`inputs` maps every source file to its sha256, `values` carries every number drawn, and
    `scorer_hash` / `graph_version` are the identities OF THE ROWS THIS ASSET READ (rule 1),
    quoted from the record rather than from the pipeline in general."""
    files = {rel(p): sha256_file(p) for p in sorted(set(sources))}
    body: dict[str, Any] = {
        "asset_id": asset_id,
        "tool": f"{TOOL} --only {asset_id.split('_')[0]}",
        "inputs": files,
        "inputs_hash": hashlib.sha256(
            json.dumps(sorted(files.items()), sort_keys=True).encode()
        ).hexdigest(),
        "query": query,
        "scorer_hash": sorted({s for s in scorer_hash if s}),
        "graph_version": sorted({g for g in graph_version if g}),
        "values": values,
    }
    if run_ids_from:
        body["run_ids_from"] = dict(run_ids_from)
    if extra:
        body["extra"] = dict(extra)
    body["provenance_digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str).encode()
    ).hexdigest()
    body["generated_at_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    return body


# --------------------------------------------------------------------------- style

# Copied from src/pi_run/render.py via scripts/figures_programme.py. Okabe-Ito, chosen for
# deuteranopia/protanopia separation and for spread in luminance so the series survive a
# grayscale printer. Marker shapes differ per series as well as colors, so the figures are
# readable with no color at all.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
GREY = "#8C8C8C"
INK = "#000000"

# The reader words for the arms (the main-text owner's G3 terms; the main text writes the 120B model
# as GPT-OSS-120B, and keeps lowercase gpt-oss-120b for code pins).
ARM_TERM = {
    "trained": "the trained questioner",
    "same": "the same model, prompted",
    "teacher": "GPT-OSS-120B, prompted",
    "opus": "Claude Opus 5, prompted",
}

LINEWIDTH_IN = 5.5  # \linewidth at ICLR 2027 size; figures are placed 1:1, so 8pt stays 8pt


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            # TrueType, not matplotlib's default Type 3, which Google Scholar asks PDFs to avoid
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


# The page box (MediaBox, points) of each main-text figure as the main text was typeset with it at
# 1b1dd08. The paper places these at a fixed fraction of \linewidth, so a page that moved by a
# fraction of a point would rescale the figure and move the text around it, and a label edit must
# not do that (the user, 2026-09-24). A tight box cannot hold the size by itself: it is the extent
# of the drawn text as the PDF renderer measures it, which moves with every label. So the content
# is laid out and centred in a pinned box (MAIN_PAGE_PT), and a figure whose content does not fit is
# refused. The 1b1dd08 boxes are kept here as the record the round-2 boxes below are measured from.
MAIN_PAGE_PT_1B1DD08: dict[str, tuple[float, float]] = {
    "fig2_frontier_recipe": (392.134375, 191.230125),
    "fig12_comparators": (403.2925, 171.418125),
    "fig13_tau2": (403.1155177239, 162.742125),
}
# The user's round 2 (INBOX_FOR_PAPER_LANE section 14, and the redirect after the user read the
# interim PDF): Figure 2's two parts, fig2_frontier_recipe (a) and fig12_comparators (b), sit in one
# float, each at width=\linewidth, "the same width, with matching font sizes"; Figure 4 is at
# \linewidth too. \textwidth is 5.5in = 397.5pt = 396.0bp, so every box is now 396.0bp wide and prints
# 1:1 (8pt type prints at 8pt; the 1b1dd08 boxes printed at 1.0099, 0.9819 and 0.9823). The content is
# laid out at that width (figsize 5.5in), not scaled into it. Heights: the 9-page trim (main-text
# owner, 2026-09-24) cuts each part of Figure 2 by about 15% of what it printed at \linewidth under
# its 1b1dd08 box (193.1 -> 164bp, 168.3 -> 143bp), fonts unchanged; Figure 4 is at most 110bp (the
# redirect; it printed 159.9bp).
LINEWIDTH_BP = 396.0
TAU2_PAGE_H_BP = 110.0
# The user, 2026-09-25: Figure 2 at width=0.88\linewidth, its type at 8pt and its part letters at 9pt
# AS PRINTED. On the round-2 pages (396.0 x 164 and 396.0 x 143bp) that placement printed 8pt type at
# 7.04pt and the letters at 7.92pt. So both parts are laid out at the placed width, 0.88 x 396.0 =
# 348.48bp (4.84in), not scaled into it, and print 1:1; each keeps the height it printed at before
# (164 x 0.88 = 144.32bp, 143 x 0.88 = 125.84bp). Figure 4 stays at \linewidth.
FIG2_PARTS_W_BP = 348.48
FIG2_PARTS_W_IN = FIG2_PARTS_W_BP / 72
MAIN_PAGE_PT: dict[str, tuple[float, float]] = {
    "fig2_frontier_recipe": (FIG2_PARTS_W_BP, 144.32),
    "fig12_comparators": (FIG2_PARTS_W_BP, 125.84),
    "fig13_tau2": (LINEWIDTH_BP, TAU2_PAGE_H_BP),
}


@contextmanager
def _pdf_renderer(fig):
    """The figure laid out and measured as the PDF writer sets it (a disabled draw, as
    print_figure does), in points; the figure's own canvas and dpi are given back on exit. Text is
    measured narrower here than on the Agg canvas (tick labels by up to a point), so anything placed
    against the PDF page's edges is placed on this renderer's numbers."""
    from matplotlib.backend_bases import _get_renderer
    from matplotlib.backends.backend_pdf import FigureCanvasPdf

    canvas, dpi = fig.canvas, fig.dpi
    try:
        renderer = _get_renderer(fig, FigureCanvasPdf(fig).print_pdf)
        with getattr(renderer, "_draw_disabled", nullcontext)():
            fig.draw(renderer)
        yield renderer
    finally:
        fig.set_canvas(canvas)
        fig.dpi = dpi


PANEL_LABEL_PT = 9  # the bold (a) and (b) of Figure 2's two stacked parts


def _panel_label(fig, axes, letter: str, beside=None) -> Any:
    """Set a part's letter in bold at the figure's top left (reviewer 2026-09-24: Figure 2 is two
    PDFs stacked, and its caption's (a) and (b) had nothing on the page to point at).

    Left-aligned on the leftmost drawn edge of the panels (the outer axis label or tick labels), its
    top on the top of `beside`, a text already set in the figure's top line (a legend entry), or of
    the panels' titles when there is none, both as the PDF sets them, so the letter adds nothing to
    the drawn extent and a pinned page holds. Called after the layout, whose positions it reads."""
    with _pdf_renderer(fig) as r:
        left = min(ax.get_tightbbox(r).x0 for ax in axes)
        top = (
            beside.get_window_extent(r).y1
            if beside is not None
            else max(ax.get_tightbbox(r).y1 for ax in axes)
        )
        x, y = fig.transFigure.inverted().transform((left, top))
    return fig.text(x, y, letter, ha="left", va="top", fontsize=PANEL_LABEL_PT, fontweight="bold")


def _pinned_page(fig, asset_id: str) -> tuple[Any, tuple[float, float]]:
    """The box (inches) savefig writes as the pinned page, and the content's own size (points).

    The content box is the one savefig itself would cut with bbox_inches="tight": measured with the
    PDF renderer after a disabled draw, padded by savefig.pad_inches. Refuses content that does not
    fit the page."""
    from matplotlib import rcParams
    from matplotlib.transforms import Bbox

    with _pdf_renderer(fig) as renderer:
        pad = rcParams["savefig.pad_inches"]
        content = fig.get_tightbbox(renderer).padded(pad, pad)
    w, h = (v / 72 for v in MAIN_PAGE_PT[asset_id])
    got = (content.width * 72, content.height * 72)
    if content.width > w + 1e-9 or content.height > h + 1e-9:
        raise Mismatch(
            f"{asset_id}: the content is {got[0]:.3f} x {got[1]:.3f}pt, larger than the "
            f"{MAIN_PAGE_PT[asset_id][0]} x {MAIN_PAGE_PT[asset_id][1]}pt page the main text was "
            "typeset with. NOT DRAWING."
        )
    x0 = content.x0 - (w - content.width) / 2
    y0 = content.y0 - (h - content.height) / 2
    return Bbox.from_bounds(x0, y0, w, h), got


def _emit_figure(fig, out: Path, asset_id: str, prov: Mapping[str, Any]) -> list[Path]:
    pinned = _pinned_page(fig, asset_id) if asset_id in MAIN_PAGE_PT else None
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        p = out / f"{asset_id}.{ext}"
        fig.savefig(p, **({"bbox_inches": pinned[0]} if pinned else {}))
        written.append(p)
    # The bounding box is trimmed on save, so the figure's final width is not the figsize it
    # was built at. It is read back off the PDF and recorded, because "fonts are at least 8pt
    # at final size" is a claim about the width the document PLACES it at, and a reader of this
    # record needs the number to check it rather than a promise.
    body = dict(prov)
    # Every string the figure renders (titles, axis labels, tick labels, legend entries, notes),
    # read off the drawn figure, so a test can check the words a reader sees rather than the
    # constants that were meant to produce them.
    import matplotlib.text as mtext

    body["rendered_text"] = sorted(
        {t.get_text() for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()}
    )
    natural_in = _pdf_width_in(out / f"{asset_id}.pdf")
    body["placement"] = {
        "natural_width_in": round(natural_in, 4),
        "base_font_pt": 8,
        "note": (
            f"Type is 8pt at {natural_in:.3f}in. Place at this width or wider and no label "
            f"falls below 8pt; at \\linewidth (5.5in) the scale is {5.5 / natural_in:.3f}."
        ),
    }
    if pinned:
        body["placement"]["page_pt"] = list(MAIN_PAGE_PT[asset_id])
        body["placement"]["content_pt"] = [round(v, 4) for v in pinned[1]]
        body["placement"]["page_note"] = (
            "page box pinned to the size the main text was typeset with (MAIN_PAGE_PT); the "
            "content, padded as a tight save pads it, is centred inside it"
        )
    p = out / f"{asset_id}.provenance.json"
    p.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    written.append(p)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return written


def _pdf_width_in(p: Path) -> float:
    box = re.search(
        rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", p.read_bytes()
    )
    if not box:
        raise Mismatch(f"{rel(p)}: no /MediaBox, so the figure's final width is unknown")
    return (float(box.group(3)) - float(box.group(1))) / 72.0


def _emit_tex(out: Path, asset_id: str, body: str, prov: Mapping[str, Any]) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{asset_id}.tex"
    p.write_text(body if body.endswith("\n") else body + "\n")
    q = out / f"{asset_id}.provenance.json"
    q.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")
    return [p, q]


# --------------------------------------------------------------------------- markdown tables


def md_tables(text: str) -> list[list[list[str]]]:
    """Every GitHub pipe table in a document, as a list of rows of trimmed cells.

    Deliberately not a Markdown parser: a table is a run of lines that start and end with `|`,
    whose second line is the `---` separator. That is the only table shape these records use,
    and a parser that accepted more would accept things that are not tables.
    """
    tables: list[list[list[str]]] = []
    block: list[list[str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("|") and line.endswith("|") and len(line) > 1:
            block.append([c.strip() for c in line[1:-1].split("|")])
            continue
        if len(block) >= 3 and _is_sep(block[1]):
            tables.append(block)
        block = []
    if len(block) >= 3 and _is_sep(block[1]):
        tables.append(block)
    return tables


def _is_sep(cells: Sequence[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


INTERVAL_RE = re.compile(
    r"^\s*([+-]?\d+\.\d+)\s*\[\s*([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)\s*\]\s*$"
)


def parse_interval(cell: str) -> tuple[float, float, float]:
    m = INTERVAL_RE.match(cell)
    if not m:
        raise Mismatch(f"not a `value [lo, hi]` cell: {cell!r}")
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


@dataclass(frozen=True, slots=True)
class LevelCell:
    arm: str
    cap: int
    n: int
    mean_asks: float
    mean_calls: float
    value: float
    lo: float
    hi: float


def parse_level_table(text: str, metric: str) -> dict[tuple[str, int], LevelCell]:
    """The one table in a frontier record whose header is `arm | cap | ... | <metric> [BCa 95%]`.

    Selected by HEADER NAMES and not by position or by order of appearance: these records carry
    five to eight other pipe tables each (contrasts, the pre/post-fix comparison, the three-suite
    question table), and two of them also start with `arm`.
    """
    want = f"{metric} [bca 95%]"
    hit = None
    for tbl in md_tables(text):
        head = [c.lower() for c in tbl[0]]
        if "arm" in head and "cap" in head and "mean retrieval_calls" in head and want in head:
            if hit is not None:
                raise Mismatch(f"two level tables for {metric!r}; the record is ambiguous")
            hit = tbl
    if hit is None:
        raise Mismatch(f"no level table for {metric!r} in this record")
    head = [c.lower() for c in hit[0]]
    ix = {name: head.index(name) for name in ("arm", "cap", "n", "mean retrieval_calls")}
    ix["asks"] = head.index("mean n_asks")
    ix["metric"] = head.index(want)
    out: dict[tuple[str, int], LevelCell] = {}
    for row in hit[2:]:
        arm = row[ix["arm"]].strip("`")
        cap = int(row[ix["cap"]])
        v, lo, hi = parse_interval(row[ix["metric"]])
        cell = LevelCell(
            arm=arm,
            cap=cap,
            n=int(row[ix["n"]].replace(",", "")),
            mean_asks=float(row[ix["asks"]]),
            mean_calls=float(row[ix["mean retrieval_calls"]]),
            value=v,
            lo=lo,
            hi=hi,
        )
        if (arm, cap) in out:
            raise Mismatch(f"duplicate level cell {arm} cap {cap}")
        out[(arm, cap)] = cell
    return out


# --------------------------------------------------------------------------- reproduction


def read_run_ids(path: Path) -> list[str]:
    """The run ids of one cell. Two layouts are in use and both are declared in the file's own
    `# columns:` / header line, so the column is found rather than assumed: `FRONTIER.md`'s
    lists are `run_id, task_id, seed` and the two later campaigns' are `task_id, seed, run_id`.
    A run id is 32 hex characters, which is what settles it when a header is absent."""
    ids: list[str] = []
    col: int | None = None
    for raw in path.read_text().splitlines():
        if raw.startswith("#"):
            cells = [c.strip() for c in raw.lstrip("#").strip().split("\t")]
            if "run_id" in cells:
                col = cells.index("run_id")
            elif "columns:" in raw and "run_id" in raw:
                names = raw.split("columns:")[1].split("(")[0].replace("\\t", "\t")
                col = [c.strip() for c in names.split("\t")].index("run_id")
            continue
        if not raw.strip():
            continue
        cells = raw.split("\t")
        if col is None:
            col = next(
                (i for i, c in enumerate(cells) if re.fullmatch(r"[0-9a-f]{32}", c.strip())), 0
            )
        ids.append(cells[col].strip())
    if not ids:
        raise Mismatch(f"{rel(path)}: no run ids")
    if len(set(ids)) != len(ids):
        raise Mismatch(f"{rel(path)}: {len(ids) - len(set(ids))} duplicate run ids")
    return ids


def recompute_cell(store: Path, run_ids: Sequence[str], metric: str) -> dict[str, float]:
    """The two readings of one cell's mean, from the scored store beside the record.

    Clusters are `COALESCE(NULLIF(template_id, ''), task_id)`, the rule the records state. On
    musique that is 186 template clusters over 200 tasks and the two readings differ; on
    strategyqa and frames `template_id` is NULL throughout, so the cluster is the task, every
    cluster holds the same number of units and the two readings coincide exactly.
    """
    require(store / "runs.parquet", store / "scores.parquet")
    lst = ", ".join("'" + i.replace("'", "''") + "'" for i in run_ids)
    sql = f"""
    WITH r AS (
        SELECT run_id, COALESCE(NULLIF(template_id, ''), task_id) AS cl, retrieval_calls AS rc
        FROM read_parquet('{store / "runs.parquet"}') WHERE run_id IN ({lst})),
         s AS (
        SELECT run_id, value AS y, scorer_hash, graph_version
        FROM read_parquet('{store / "scores.parquet"}')
        WHERE metric_name = '{metric}' AND run_id IN ({lst})),
         j AS (SELECT r.cl, s.y, r.rc FROM r JOIN s USING (run_id)),
         g AS (SELECT cl, avg(y) AS y, avg(rc) AS rc FROM j GROUP BY cl)
    SELECT (SELECT count(*) FROM j), (SELECT count(*) FROM g),
           (SELECT avg(y) FROM j), (SELECT avg(rc) FROM j),
           (SELECT avg(y) FROM g), (SELECT avg(rc) FROM g)
    """
    con = duckdb.connect()
    n, n_cl, pooled_y, pooled_rc, cluster_y, cluster_rc = con.execute(sql).fetchone()
    hashes = con.execute(
        f"SELECT DISTINCT scorer_hash, graph_version FROM "
        f"read_parquet('{store / 'scores.parquet'}') "
        f"WHERE metric_name = '{metric}' AND run_id IN ({lst})"
    ).fetchall()
    con.close()
    return {
        "n": int(n),
        "n_clusters": int(n_cl),
        "pooled_y": float(pooled_y),
        "pooled_rc": float(pooled_rc),
        "cluster_y": float(cluster_y),
        "cluster_rc": float(cluster_rc),
        "scorer_hash": sorted({h[0] for h in hashes}),
        "graph_version": sorted({h[1] for h in hashes}),
    }


def reconcile(label: str, recorded: float, pooled: float, cluster: float, tol: float = TOL) -> str:
    """Which estimand reproduces the record, or `Mismatch` naming both numbers.

    Returning the NAME of the estimand rather than a boolean is the point: the frontier records
    report coverage as a cluster mean and retrieval calls as a pooled mean, in adjacent columns
    of one table, and an asset that silently accepted either would hide that.
    """
    ok_cluster = abs(cluster - recorded) <= tol
    ok_pooled = abs(pooled - recorded) <= tol
    if ok_cluster and ok_pooled:
        return "both (the two readings coincide on this suite)"
    if ok_cluster:
        return "unweighted mean over clusters"
    if ok_pooled:
        return "pooled run mean"
    raise Mismatch(
        f"{label}: the record says {recorded!r}; recomputed pooled {pooled!r} "
        f"(delta {pooled - recorded:+.6f}) and clustered {cluster!r} "
        f"(delta {cluster - recorded:+.6f}); tolerance {tol}. NOT DRAWING."
    )


# --------------------------------------------------------------------------- fig 2


ARM_LABEL = {
    "trained": "the trained questioner",
    "base8b": "the same weights prompted (8B)",
    "teacher": "a prompted model 15x larger (120B)",
    "drafter_only": "no-question baseline",
}
ARM_STYLE = {
    "trained": dict(color=PALETTE[0], marker=MARKERS[0], zorder=5, lw=1.6, ms=4.5),
    "base8b": dict(color=PALETTE[1], marker=MARKERS[1], zorder=4, lw=1.2, ms=4.0),
    "teacher": dict(color=PALETTE[2], marker=MARKERS[2], zorder=3, lw=1.2, ms=4.0),
}


@dataclass(frozen=True, slots=True)
class PanelSpec:
    key: str
    title: str
    md: Path
    metric: str
    ylabel: str
    arms: tuple[str, ...]
    units_note: str
    store: Callable[[str], Path]
    ids: Callable[[str, int], Path]
    flat_arm: str | None = None  # an arm drawn as a reference line rather than a curve


def _panels() -> tuple[PanelSpec, ...]:
    d1, d2, d3 = (
        REPO / "artifacts/frontier",
        REPO / "artifacts/frontier_strategyqa",
        REPO / "artifacts/frames_frontier",
    )
    return (
        PanelSpec(
            key="musique",
            title="MuSiQue",
            md=FRONTIER_MUSIQUE,
            metric="evidence_coverage",
            ylabel="required-evidence coverage",
            arms=("trained", "base8b", "teacher"),
            units_note="test split, 200 tasks x 2 seeds = 400 units per cell",
            store=lambda a, d=d1: d / f"scores_parquet.{a}",
            ids=lambda a, c, d=d1: d / f"run_ids.{a}.cap{c}.txt",
        ),
        PanelSpec(
            key="strategyqa",
            title="StrategyQA",
            md=FRONTIER_STRATEGYQA,
            metric="evidence_coverage",
            ylabel="required-evidence coverage",
            arms=("trained", "base8b", "teacher"),
            units_note="test split, 200 tasks x 2 seeds = 400 units per cell",
            store=lambda a, d=d2: d / "scores_parquet.strategyqa",
            ids=lambda a, c, d=d2: d / f"run_ids.{a}.cap{c}.tsv",
        ),
        PanelSpec(
            key="frames",
            title="FRAMES",
            md=FRONTIER_FRAMES,
            metric="answer_correct",
            ylabel="answer correctness",
            arms=("trained", "base8b", "teacher", "drafter_only"),
            units_note="test split, 824 tasks x 1 seed = 824 units per cell",
            store=lambda a, d=d3: d / "scores_parquet.frames",
            ids=lambda a, c, d=d3: d / f"run_ids.{a}.cap{c}.tsv",
            flat_arm="drafter_only",
        ),
    )


def fig2_frontier(out: Path) -> list[Path]:
    panels = _panels()
    require(*[p.md for p in panels])
    drawn: dict[str, Any] = {}
    sources: list[Path] = []
    scorer, graph = set(), set()

    for p in panels:
        text = p.md.read_text()
        sources.append(p.md)
        cells = parse_level_table(text, p.metric)
        rows = {}
        for arm in p.arms:
            for cap in CAPS:
                rec = cells.get((arm, cap))
                if rec is None:
                    raise Mismatch(f"{rel(p.md)}: no level row for {arm} cap {cap}")
                idp = p.ids(arm, cap)
                require(idp)
                sources.append(idp)
                got = recompute_cell(p.store(arm), read_run_ids(idp), p.metric)
                if got["n"] != rec.n:
                    raise Mismatch(
                        f"{p.key} {arm} cap {cap}: the record says n={rec.n}, "
                        f"the store joins {got['n']} rows. NOT DRAWING."
                    )
                est_y = reconcile(
                    f"{p.key} {arm} cap {cap} {p.metric}",
                    rec.value,
                    got["pooled_y"],
                    got["cluster_y"],
                )
                est_rc = reconcile(
                    f"{p.key} {arm} cap {cap} retrieval_calls",
                    rec.mean_calls,
                    got["pooled_rc"],
                    got["cluster_rc"],
                )
                scorer.update(got["scorer_hash"])
                graph.update(got["graph_version"])
                rows[f"{arm}.cap{cap}"] = {
                    "arm_reader_name": ARM_LABEL[arm],
                    "cap": cap,
                    "n": rec.n,
                    "n_clusters": got["n_clusters"],
                    "x_mean_retrieval_calls": rec.mean_calls,
                    "y": rec.value,
                    "ci_lo": rec.lo,
                    "ci_hi": rec.hi,
                    "estimand_reproduced_y": est_y,
                    "estimand_reproduced_x": est_rc,
                    "recomputed": {
                        k: got[k] for k in ("pooled_y", "cluster_y", "pooled_rc", "cluster_rc")
                    },
                    "run_id_list": rel(p.ids(arm, cap)),
                }
        drawn[p.key] = {
            "metric": p.metric,
            "y_axis": p.ylabel,
            "units": p.units_note,
            "cells": rows,
        }

    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(LINEWIDTH_IN, 2.1))
    handles: dict[str, Any] = {}
    for ax, p in zip(axes, panels):
        cells = drawn[p.key]["cells"]
        for arm in p.arms:
            xs = [cells[f"{arm}.cap{c}"]["x_mean_retrieval_calls"] for c in CAPS]
            ys = [cells[f"{arm}.cap{c}"]["y"] for c in CAPS]
            lo = [ys[i] - cells[f"{arm}.cap{c}"]["ci_lo"] for i, c in enumerate(CAPS)]
            hi = [cells[f"{arm}.cap{c}"]["ci_hi"] - ys[i] for i, c in enumerate(CAPS)]
            if arm == p.flat_arm:
                ax.axhline(ys[0], color=GREY, lw=1.0, ls="--", zorder=1)
                ax.axhspan(
                    cells[f"{arm}.cap4"]["ci_lo"],
                    cells[f"{arm}.cap4"]["ci_hi"],
                    color=GREY,
                    alpha=0.13,
                    lw=0,
                    zorder=0,
                )
                handles.setdefault(
                    ARM_LABEL[arm],
                    plt.Line2D([], [], color=GREY, lw=1.0, ls="--"),
                )
                continue
            st = ARM_STYLE[arm]
            ax.errorbar(
                xs,
                ys,
                yerr=[lo, hi],
                color=st["color"],
                marker=st["marker"],
                markersize=st["ms"],
                lw=st["lw"],
                elinewidth=0.8,
                capsize=1.8,
                markeredgecolor=INK if arm == "trained" else st["color"],
                markeredgewidth=0.6 if arm == "trained" else 0.0,
                zorder=st["zorder"],
            )
            handles.setdefault(
                ARM_LABEL[arm],
                plt.Line2D(
                    [],
                    [],
                    color=st["color"],
                    marker=st["marker"],
                    markersize=st["ms"],
                    lw=st["lw"],
                ),
            )
        ax.set_title(p.title)
        ax.set_xlabel("mean retrieval calls")
        # The middle panel shares the left panel's quantity, so its axis label is dropped
        # rather than repeated: at this width the repeat costs more room than it earns.
        if p.key != "strategyqa":
            ax.set_ylabel(p.ylabel)
        ax.set_xlim(left=0)
        ax.margins(x=0.08, y=0.10)
    fig.tight_layout(w_pad=0.6)
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    prov = provenance(
        asset_id="fig2_frontier",
        sources=sources,
        query=(
            "per (suite, arm, cap): the level row of the promoted record's frontier table, "
            "re-derived from the scored store beside it over that cell's own run-id list "
            "(clusters = COALESCE(NULLIF(template_id,''), task_id))"
        ),
        values=drawn,
        scorer_hash=scorer,
        graph_version=graph,
        extra={
            "tolerance": TOL,
            "cells_checked": sum(len(v["cells"]) for v in drawn.values()),
            "estimand_note": (
                "The coverage column of artifacts/frontier/FRONTIER.md is the unweighted mean "
                "over 186 template clusters and the retrieval-calls column beside it is the "
                "pooled run mean. On strategyqa and frames template_id is NULL throughout, so "
                "the two readings coincide. Each cell records which one reproduced."
            ),
            "frames_note": (
                "FRAMES ships no need graph and is gold-free, so the panel reports answer "
                "correctness and no coverage or depth statement is made on it."
            ),
        },
    )
    return _emit_figure(fig, out, "fig2_frontier", prov)


FRONTIER_RECIPE = REPO / "artifacts" / "frontier_recipe_20260923" / "frontier_recipe.json"
FRONTIER_RECIPE_LABEL = {
    "recipe": ARM_TERM["trained"],
    "base": ARM_TERM["same"],
    "teacher": ARM_TERM["teacher"],
}
FRONTIER_RECIPE_STYLE = {"recipe": "trained", "base": "base8b", "teacher": "teacher"}
# The coverage axis in percent, on two lines: on one ("required-evidence coverage (%)", 130pt at
# 8pt) it is longer than the panel is tall plus the title's height, so it stood above the titles
# and grew the page by 1.5pt; broken, it clears the titles and the page keeps its size.
FIG2_COVERAGE_YLABEL = "required-evidence\ncoverage (%)"


def _frontier_recipe_record() -> dict:
    """The recipe's budget sweep, refused unless its reader locked and its population passed.

    scripts/seed_identity/frontier_recipe.py writes the record only after its level function has
    reproduced all 30 level rows of the two published frontier records, and it writes
    `population_failures` instead of levels when any cell is short, spans two code versions or two
    pins, or when any manifest is dirty or carries a second base URL. This re-asserts all of it
    from the record rather than trusting that the reader ran to completion.
    """
    require(FRONTIER_RECIPE)
    rec = json.loads(FRONTIER_RECIPE.read_text())
    lock = rec.get("lock") or {}
    bad = [k for k, v in lock.items() if not v.get("ok")]
    if len(lock) != 30 or bad:
        raise Mismatch(
            f"{rel(FRONTIER_RECIPE)}: lock {len(lock) - len(bad)}/30, failing {bad}. NOT DRAWING."
        )
    if rec.get("population_failures") or not (rec.get("manifests") or {}).get("ok"):
        raise Mismatch(
            f"{rel(FRONTIER_RECIPE)}: population refused ({rec.get('population_failures')}, "
            f"manifests {rec.get('manifests')}). NOT DRAWING."
        )
    for suite in ("musique", "strategyqa"):
        for arm in FRONTIER_RECIPE_LABEL:
            for cap in CAPS:
                cell = rec["levels"][suite][arm][str(cap)]
                if cell["n"] != 400:
                    raise Mismatch(f"{suite} {arm} cap {cap}: n={cell['n']}, not 400. NOT DRAWING.")
    return rec


FRAMES_RECIPE_JSON = REPO / "artifacts" / "frames_recipe_20260923" / "frames_recipe.json"


def _frames_recipe_record() -> dict:
    """The FRAMES reading for the recipe, refused unless every published row reproduces and the
    population check found nothing (scripts/seed_identity/frames_recipe.py)."""
    rec = json.loads(FRAMES_RECIPE_JSON.read_text())
    bad = [
        f"{sec}/{k}"
        for sec in ("levels", "contrasts", "budget_response")
        for k, v in rec["lock"][sec].items()
        if not v.get("ok")
    ]
    if bad or rec.get("population_failures"):
        raise Mismatch(
            f"FRAMES record: lock rows failing {bad[:5]}, population failures "
            f"{rec.get('population_failures')}. NOT DRAWING."
        )
    return rec


CEILING_MARK_PT = 6.5  # the ceiling numbers beside the trained questioner's points
CEILING_MARK_CLEAR_PT = 1.0  # least white space between a leader and any other number
CEILING_MARK_DATA_PT = 3.5  # least white space between a number and any drawn mark (3pt tested; a reviewer read 1.5pt as touching)


def _drawn_marks(ax) -> tuple[list, list]:
    """Every drawn data mark in a panel, in display pixels: line segments (curves, error bars,
    reference lines) and marker boxes (points and error-bar caps). Annotations are not included."""
    from matplotlib.collections import LineCollection
    from matplotlib.transforms import Bbox

    segments, boxes = [], []
    px = ax.figure.dpi / 72
    for ln in ax.lines:
        xy = ln.get_transform().transform(ln.get_xydata())
        if ln.get_linestyle() not in ("None", "none", "", " ") and len(xy) > 1:
            segments += list(zip(xy[:-1], xy[1:]))
        if ln.get_marker() not in (None, "None", "none", "", " "):
            half = ln.get_markersize() * px / 2
            boxes += [Bbox.from_extents(x - half, y - half, x + half, y + half) for x, y in xy]
    for coll in ax.collections:
        if isinstance(coll, LineCollection):
            for seg in coll.get_segments():
                xy = coll.get_transform().transform(seg)
                segments += list(zip(xy[:-1], xy[1:]))
    return segments, boxes


def _mark_ceilings(ax, fig, points) -> list[dict]:
    """Tie each of the trained questioner's points to its ceiling (reviewer 2026-09-24).

    `points` is the trained questioner's five (cap, x, y) in ceiling order. They sit within a few
    typographic points of each other on StrategyQA and FRAMES, so a number beside each would
    collide. The numbers are stacked above the cluster in three short lines, 12 on top, 8 and 16
    below it on either side of 12's leader, 4 and 24 lowest and outermost, each joined to its own
    point by a grey hairline:

                    12
                 8      16
              4             24

    Left to right the numbers are in ceiling order, so no two leaders cross, and each number is
    set outside the leaders that pass its line (by CEILING_MARK_CLEAR_PT), so no leader passes
    through another number. The stack goes where no number comes within CEILING_MARK_DATA_PT of
    any drawn mark (a number on another arm's point or bar would read as that arm's), as low and
    as close over point 12 as that allows. Everything stays inside the panel (the y-limit is
    raised if it must be, and the placement redone on the new scale), so the figure keeps its
    size; if no place fits, it refuses. Called after the layout, whose transforms it reads.
    Returns what it drew, for the provenance record.
    """
    import matplotlib.text as mtext

    assert [c for c, _x, _y in points] == list(CAPS)
    r = fig.canvas.get_renderer()
    anns = [
        ax.annotate(
            str(cap),
            (x, y),
            xytext=(x, y),
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=CEILING_MARK_PT,
            color=PALETTE[0],
            # grey hairlines, so a leader is not read as one more blue error bar
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.45, shrinkA=0.5, shrinkB=2.6),
            zorder=7,
        )
        for cap, x, y in points
    ]
    w = [mtext.Text.get_window_extent(t, r).width for t in anns]
    h = max(mtext.Text.get_window_extent(t, r).height for t in anns)
    pt = fig.dpi / 72
    clear, data_clear = CEILING_MARK_CLEAR_PT * pt, CEILING_MARK_DATA_PT * pt

    def stack(c12: float, y0: float, anchors) -> list[tuple[float, float, float, float]]:
        """The five boxes (x0, y0, x1, y1), 12 centred at c12, the lowest line at y0 (pixels)."""
        y1, y2 = y0 + h + pt, y0 + 2 * (h + pt)

        def leader(x0: float, yb: float, i: int):  # x of number i's leader at height y
            cx, cy, (px, py) = x0 + w[i] / 2, yb + h / 2, anchors[i]
            return lambda y: cx + (px - cx) * (cy - y) / (cy - py)

        x12 = c12 - w[2] / 2
        l12 = leader(x12, y2, 2)
        x8 = min(min(l12(y1), l12(y1 + h)) - clear, anchors[1][0] + w[1] / 2) - w[1]
        x16 = max(max(l12(y1), l12(y1 + h)) + clear, anchors[3][0] - w[3] / 2)
        low = [f(y) for f in (leader(x8, y1, 1), l12, leader(x16, y1, 3)) for y in (y0, y0 + h)]
        x4 = min(min(low) - clear, anchors[0][0] + w[0] / 2) - w[0]
        x24 = max(max(low) + clear, anchors[4][0] - w[4] / 2)
        return [
            (x, yb, x + wi, yb + h)
            for x, yb, wi in zip((x4, x8, x12, x16, x24), (y0, y1, y2, y1, y0), w)
        ]

    def place() -> list[tuple[float, float, float, float]]:
        import numpy as np

        to_px = ax.transData.transform
        box = ax.get_window_extent(r)
        anchors = [to_px((x, y)) for _c, x, y in points]
        segments, marks = _drawn_marks(ax)
        # every drawn segment sampled every half pixel, so a box test is one vectorised compare
        pts = np.concatenate(
            [
                np.linspace(p, q, max(2, int(np.hypot(*(np.subtract(q, p))) * 2) + 1))
                for p, q in segments
            ]
            or [np.empty((0, 2))]
        )
        mk = np.array([m.extents for m in marks]).reshape(-1, 4)

        def on_data(boxes) -> bool:
            e = (np.array(boxes) + (-data_clear, -data_clear, data_clear, data_clear))[:, None, :]
            hit_pt = (
                (pts[None, :, 0] >= e[..., 0])
                & (pts[None, :, 0] <= e[..., 2])
                & (pts[None, :, 1] >= e[..., 1])
                & (pts[None, :, 1] <= e[..., 3])
            )
            hit_mk = (
                (mk[None, :, 0] <= e[..., 2])
                & (mk[None, :, 2] >= e[..., 0])
                & (mk[None, :, 1] <= e[..., 3])
                & (mk[None, :, 3] >= e[..., 1])
            )
            return bool(hit_pt.any() or hit_mk.any())

        floor = max(a[1] for a in anchors)
        shifts = sorted(range(-int(box.width), int(box.width) + 1), key=abs)
        best = None
        for lift in range(int(box.height)):
            if best is not None and lift >= best[0]:  # lift alone already costs more
                break
            for shift in shifts:
                cost = lift + 2 * abs(shift)  # sideways costs more than up: leaders stay short
                if best is not None and cost >= best[0]:
                    break
                boxes = stack(anchors[2][0] + shift, floor + lift, anchors)
                if boxes[0][0] < box.x0 + pt or boxes[4][2] > box.x1 - pt or on_data(boxes):
                    continue
                # a stack that runs over the top of the panel costs the space it takes from it
                cost += 3 * max(0.0, boxes[2][3] + pt - box.y1)
                if best is None or cost < best[0]:
                    best = (cost, boxes)
        if best is None:
            raise Mismatch("no place in the panel sets the ceiling numbers clear. NOT DRAWING.")
        return best[1]

    for _ in range(30):
        boxes = place()
        top = max(b[3] for b in boxes) + pt
        if top <= ax.get_window_extent(r).y1 + 0.25:  # a quarter pixel: inside the panel
            break
        ax.set_ylim(top=ax.transData.inverted().transform((0.0, top))[1])
    else:
        raise Mismatch("the ceiling numbers do not fit inside the panel. NOT DRAWING.")
    to_data = ax.transData.inverted().transform
    out = []
    for t, (cap, x, y), (x0, y0, x1, _y1) in zip(anns, points, boxes):
        t.xyann = tuple(float(v) for v in to_data(((x0 + x1) / 2, y0)))
        out.append({"cap": cap, "xy": [x, y], "label_xy": list(t.xyann)})
    return out


# Figure 2(a)'s layout for the 9-page trim (main-text owner, 2026-09-24): the figure's height
# (inches; it was 2.2), the y-axis padding around the data (it was 0.10 of the range) and the legend's
# distance above the panels' titles (points). The content fills the 144.32bp page (MAIN_PAGE_PT) to
# within a point: the height was 1.945 on the 396.0 x 164bp page, and at 0.88\linewidth (the user,
# 2026-09-25) it is 1.575, laid out at FIG2_PARTS_W_IN. At that height the FRAMES panel's rotated label
# is longer than the panel and its title, so it sets the top the legend clears, and the page grows by
# only half of any height added to the figure.
FIG2_H_IN = 1.575
FIG2_Y_MARGIN = 0.04
FIG2_LEGEND_GAP_PT = 2.0
# Both parts of Figure 2 at 0.88\linewidth (the user, 2026-09-25): 8pt type on pages 0.88 as tall as
# before leaves the panels less height, and it goes back to them from the white around the text: the
# panel titles' pad and the x-axis labels' pad, points (they were the rcParams' 6 and 4). One value
# in both parts, so the stacked parts set their titles and axis labels alike.
FIG2_PARTS_TITLE_PAD_PT = 3.0
FIG2_PARTS_LABELPAD_PT = 2.0


def fig2_frontier_recipe(out: Path) -> list[Path]:
    """What a budget buys each questioner, for the reported questioner, at one commit.

    Required-evidence coverage against mean retrieval calls at five budget ceilings, the recipe
    (seeds 1 and 2 averaged within each unit) against the same weights prompted and a prompted model
    fifteen times larger, every arm at code a61c4f4be3e9 through one base URL. Seed 0, the resumed
    run the published sweep used, is marked at the one ceiling it was run at here.
    """
    rec = _frontier_recipe_record()
    frames = _frames_recipe_record() if FRAMES_RECIPE_JSON.exists() else None
    plt = _mpl()
    fig, axes = plt.subplots(
        1, 3 if frames else 2, figsize=(FIG2_PARTS_W_IN, FIG2_H_IN), sharey=False
    )
    handles: dict[str, Any] = {}
    drawn: dict[str, Any] = {}
    for ax, (suite, title) in zip(axes, (("musique", "MuSiQue"), ("strategyqa", "StrategyQA"))):
        lv = rec["levels"][suite]
        drawn[suite] = {}
        for arm in ("teacher", "base", "recipe"):
            cells = [lv[arm][str(c)] for c in CAPS]
            xs = [c["x_mean_retrieval_calls"] for c in cells]
            ys = [c["y"] for c in cells]
            st = ARM_STYLE[FRONTIER_RECIPE_STYLE[arm]]
            ax.errorbar(
                xs,
                ys,
                yerr=[
                    [y - c["ci_lo"] for y, c in zip(ys, cells)],
                    [c["ci_hi"] - y for y, c in zip(ys, cells)],
                ],
                color=st["color"],
                marker=st["marker"],
                markersize=st["ms"],
                lw=st["lw"],
                elinewidth=0.8,
                capsize=1.8,
                markeredgecolor=INK if arm == "recipe" else st["color"],
                markeredgewidth=0.6 if arm == "recipe" else 0.0,
                zorder=st["zorder"],
            )
            handles.setdefault(
                FRONTIER_RECIPE_LABEL[arm],
                plt.Line2D(
                    [], [], color=st["color"], marker=st["marker"], markersize=st["ms"], lw=st["lw"]
                ),
            )
            drawn[suite][arm] = {
                str(c): {
                    k: lv[arm][str(c)][k]
                    for k in ("n", "y", "ci_lo", "ci_hi", "x_mean_retrieval_calls", "ceiling_hit")
                }
                for c in CAPS
            }
        s0 = lv["s0"]["8"]
        drawn[suite]["s0_cap8"] = {
            k: s0[k] for k in ("n", "y", "ci_lo", "ci_hi", "x_mean_retrieval_calls", "ceiling_hit")
        }
        if not SHOW_SEED0:
            ax.set_title(title, pad=FIG2_PARTS_TITLE_PAD_PT)
            ax.set_xlabel("mean retrieval calls", labelpad=FIG2_PARTS_LABELPAD_PT)
            if suite == "musique":
                ax.set_ylabel(FIG2_COVERAGE_YLABEL)
            ax.set_xlim(left=0)
            ax.margins(x=0.08, y=FIG2_Y_MARGIN)
            continue
        ax.errorbar(
            [s0["x_mean_retrieval_calls"]],
            [s0["y"]],
            yerr=[[s0["y"] - s0["ci_lo"]], [s0["ci_hi"] - s0["y"]]],
            fmt="o",
            color=PALETTE[0],
            markerfacecolor="white",
            markeredgewidth=1.0,
            markersize=4.5,
            elinewidth=0.8,
            capsize=1.8,
            zorder=6,
        )
        handles.setdefault(
            "seed 0, the resumed run (ceiling of 8 only)",
            plt.Line2D(
                [], [], color=PALETTE[0], marker="o", markerfacecolor="white", lw=0, markersize=4.5
            ),
        )
        drawn[suite]["s0_cap8"] = {
            k: s0[k] for k in ("n", "y", "ci_lo", "ci_hi", "x_mean_retrieval_calls", "ceiling_hit")
        }
        ax.set_title(title, pad=FIG2_PARTS_TITLE_PAD_PT)
        ax.set_xlabel("mean retrieval calls", labelpad=FIG2_PARTS_LABELPAD_PT)
        if suite == "musique":
            ax.set_ylabel(FIG2_COVERAGE_YLABEL)
        ax.set_xlim(left=0)
        ax.margins(x=0.08, y=FIG2_Y_MARGIN)
    if frames:
        # FRAMES has no need graph: its panel is answer correctness, a different quantity from the
        # other two, drawn at the same five ceilings, arms and commit.
        ax = axes[2]
        lv = frames["reading"]["levels"]
        drawn["frames"] = {}
        for arm in ("teacher", "base", "recipe"):
            cells = [lv[arm][str(c)] for c in CAPS]
            xs = [c["mean_retrieval_calls"] for c in cells]
            ys = [c["y"] for c in cells]
            st = ARM_STYLE[FRONTIER_RECIPE_STYLE[arm]]
            ax.errorbar(
                xs,
                ys,
                yerr=[
                    [y - c["ci_lo"] for y, c in zip(ys, cells)],
                    [c["ci_hi"] - y for y, c in zip(ys, cells)],
                ],
                color=st["color"],
                marker=st["marker"],
                markersize=st["ms"],
                lw=st["lw"],
                elinewidth=0.8,
                capsize=1.8,
                markeredgecolor=INK if arm == "recipe" else st["color"],
                markeredgewidth=0.6 if arm == "recipe" else 0.0,
                zorder=st["zorder"],
            )
            drawn["frames"][arm] = {
                str(c): {
                    k: lv[arm][str(c)][k]
                    for k in ("n", "y", "ci_lo", "ci_hi", "mean_retrieval_calls", "ceiling_hit")
                }
                for c in CAPS
            }
        floor = lv["drafter_only"]["8"]
        ax.axhline(floor["y"], color=GREY, lw=0.9, ls="--", zorder=1)
        handles.setdefault(
            "no questions asked (FRAMES)",
            plt.Line2D([], [], color=GREY, lw=0.9, ls="--"),
        )
        drawn["frames"]["drafter_only"] = {"8": floor}
        ax.set_title("FRAMES", pad=FIG2_PARTS_TITLE_PAD_PT)
        ax.set_xlabel("mean retrieval calls", labelpad=FIG2_PARTS_LABELPAD_PT)
        # A different quantity from the two panels on its left, so its axis and label sit on its
        # own outer (right) side rather than in the gap it shares with StrategyQA (G3).
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(False)
        ax.set_ylabel("answer correctness (%)")
        ax.set_xlim(left=0)
        ax.margins(x=0.08, y=FIG2_Y_MARGIN)
    # Coverage and answer correctness in percent (the user's comment, 2026-09-24). The drawn values
    # stay the records' fractions, so `values` and the ceiling numbers' data anchors are unchanged.
    for ax in axes:
        _axis_in_points(ax.yaxis)

    def overhangs() -> list[tuple[float, ...]]:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        got = []
        for ax in axes:
            p, t = ax.get_window_extent(r), ax.get_tightbbox(r)
            got.append(
                tuple(round(v, 3) for v in (p.x0 - t.x0, t.x1 - p.x1, p.y0 - t.y0, t.y1 - p.y1))
            )
        return got

    # The ceiling numbers can raise a panel's y-limit after the layout, and in points a raised limit
    # can add a tick label wider than any the layout was solved with ("100" over "95"; the fraction
    # labels were all four characters), which pushes the axis label past the edge the layout set and
    # grows the page. So the layout is solved again on the final ticks and the numbers set again on
    # it, until the panels' decorations stop moving (one pass when nothing changes, as before).
    for _ in range(4):
        fig.tight_layout(w_pad=0.8)
        before = overhangs()
        kept = {ax: list(ax.texts) for ax in axes}
        ceiling_marks: dict[str, list[dict]] = {}
        for ax, suite in zip(axes, ("musique", "strategyqa", "frames")):
            if suite not in drawn:
                continue
            xkey = "mean_retrieval_calls" if suite == "frames" else "x_mean_retrieval_calls"
            trained = [drawn[suite]["recipe"][str(c)] for c in CAPS]
            ceiling_marks[suite] = _mark_ceilings(
                ax, fig, [(c, cell[xkey], cell["y"]) for c, cell in zip(CAPS, trained)]
            )
        if overhangs() == before:
            break
        for ax in axes:
            for t in [t for t in ax.texts if t not in kept[ax]]:
                t.remove()
    else:
        raise Mismatch("the panels' ticks do not settle under the ceiling numbers. NOT DRAWING.")
    geometry = None
    if frames:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        geometry = {
            "text": axes[2].yaxis.label.get_text(),
            "label_x0": axes[2].yaxis.label.get_window_extent(r).x0,
            "frames_axes_x1": axes[2].get_window_extent(r).x1,
            "strategyqa_axes_x1": axes[1].get_window_extent(r).x1,
            "units": "display pixels at the figure's dpi",
        }
        if geometry["label_x0"] < geometry["frames_axes_x1"]:
            raise Mismatch("the FRAMES y-label is not beside its own panel. NOT DRAWING.")
    order = [
        FRONTIER_RECIPE_LABEL["recipe"],
        *(["seed 0, the resumed run (ceiling of 8 only)"] if SHOW_SEED0 else []),
        FRONTIER_RECIPE_LABEL["base"],
        FRONTIER_RECIPE_LABEL["teacher"],
        *(["no questions asked (FRAMES)"] if frames else []),
    ]
    # Above the panels, not below them: in the main text this figure is stacked over Figure 2(b),
    # and a legend set below (a) sat between the two parts and read as keying both (reviewer
    # 2026-09-24). Set FIG2_LEGEND_GAP_PT above the panels' titles, its rows closer together, for the
    # 9-page trim (2026-09-24; it was anchored at 98% of the figure's height, 5pt higher).
    with _pdf_renderer(fig) as r:
        panels_top = max(ax.get_tightbbox(r).y1 for ax in axes)
        fig_h = fig.bbox.height
    legend = fig.legend(
        [handles[k] for k in order],
        order,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, (panels_top + FIG2_LEGEND_GAP_PT) / fig_h),
        borderaxespad=0.0,
        labelspacing=0.3,
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    _panel_label(fig, axes, "(a)", beside=legend.get_texts()[0])
    prov = provenance(
        asset_id="fig2_frontier_recipe",
        sources=[FRONTIER_RECIPE, REPO / "scripts" / "seed_identity" / "frontier_recipe.py"]
        + ([FRAMES_RECIPE_JSON] if frames else []),
        query=(
            "levels[suite][arm][cap] of the record written by scripts/seed_identity/frontier_recipe.py: "
            "cluster_bootstrap over template clusters at 10,000 resamples, x the pooled mean of "
            "retrieval_calls; the recipe's unit is (task, seed) with seeds 1 and 2 averaged within it"
        ),
        values=drawn,
        scorer_hash=[FRONTIER_RECIPE_SCORER],
        graph_version=["v1"],
        extra={
            "one_code_version": "a61c4f4be3e9",
            "one_base_url_sha": rec["manifests"]["base_url_sha"],
            "lock": "all 30 level rows of artifacts/frontier/FRONTIER.md and "
            "artifacts/frontier_strategyqa/FRONTIER_STRATEGYQA.md reproduced by the same level "
            "function to 1e-4 before the record was written",
            "caps": list(CAPS),
            "build": "scripts/paper_figures_iclr.py --only fig2_recipe",
            "cap8_source": "artifacts/structured_baselines_clean_20260922 (same code, base URL, roles)",
            "frames_ylabel_geometry": geometry,
            "ceiling_marks": ceiling_marks,
            "units": "coverage and answer correctness in percent: the tick labels are the tick "
            "values moved two places (_pts_tick); `values` are the records' fractions, unchanged",
        },
    )
    return _emit_figure(fig, out, "fig2_frontier_recipe", prov)


# --------------------------------------------------------------------------- dev-population


# Row label in the appendix's own tables -> the reader words the paper uses. The right-hand side
# is what gets rendered; the left-hand side never leaves this file or the provenance record.
DEV_ARMS: tuple[tuple[str, str, str], ...] = (
    ("Reference", "the imitation reference", "reference"),
    ("Stop-weight", "stop-weighted imitation", "stopweight"),
    ("Question-preference", "the question-pair arm", "questionpair"),
    ("Persistence-only", "the stop-contrast-only arm", "stoponly"),
    ("Mixed", "both kinds, from the reference", "mixed"),
    ("Stacked", "both kinds, from the question-pair arm", "stacked"),
)
BEHAVIOUR_LABEL = {"Question-pref.": "Question-preference"}
# The row label as Table 2 prints it: the same words as the reader name, short enough that the
# label column leaves room for five number columns inside 397pt. The full reader name is in the
# provenance record beside every value.
# What each row was fitted from and on, so a reader does not have to hold the recipe table in
# mind. Taken from paper/appendix_training.tex tab:train-recipes and asserted against it below.
TABLE2_TRAINED_ON = {
    "reference": "filtered demonstrations",
    "stopweight": "the same, stop rows down-weighted",
    "questionpair": "question pairs, from the pooled imitation checkpoint",
    "stoponly": "stop contrasts, from the reference",
    "mixed": "both pair kinds, from the reference",
    "stacked": "both pair kinds, from the question-pair arm",
}

TABLE2_LABEL = {
    "reference": "imitation reference",
    "stopweight": "stop-weighted imitation",
    "questionpair": "question pairs",
    "stoponly": "stop contrasts only",
    "mixed": "both kinds, from reference",
    "stacked": r"both kinds, from pairs, seed 0$^{\dagger}$",
}
SUITES = ("musique", "strategyqa")
SUITE_TITLE = {"musique": "MuSiQue", "strategyqa": "StrategyQA"}

# Color per arm. The shape DRAWN is FIG3_GROUP_MARKER's, one per group of arms (reviewer
# 2026-09-24: seven shapes times filled/open did not read); the second element here is kept
# only as the arm's historical marker and is not drawn.
FIG3_STYLE: dict[str, tuple[str, str]] = {
    "stacked": (PALETTE[0], MARKERS[0]),
    "prompted": (PALETTE[1], MARKERS[1]),
    "questionpair": (PALETTE[2], MARKERS[2]),
    "stoponly": (PALETTE[3], MARKERS[3]),
    "mixed": (PALETTE[4], MARKERS[4]),
    "stopweight": (PALETTE[5], MARKERS[5]),
    "reference": (PALETTE[6], MARKERS[6]),
}
# One marker shape per group of arms, so a reader matches shape to group and color to arm: the
# two imitation arms, the two arms trained on one kind of contrast, the two trained on both kinds
# (the trained questioner among them, drawn larger), and the prompted model. The two arms that
# also appear in Figure 2(a) keep the shape they have there (circle, square). Every arm's
# (shape, color) pair is still its own.
FIG3_GROUP_MARKER: dict[str, str] = {
    "reference": "D",
    "stopweight": "D",
    "questionpair": "^",
    "stoponly": "^",
    "mixed": "o",
    "stacked": "o",
    "prompted": "s",
}
# The legend's words where they differ from the record's reader name, which stays in the
# provenance values unchanged.
FIG3_LEGEND_NAME = {"prompted": ARM_TERM["same"]}


def _tex_tables(text: str) -> list[list[str]]:
    """The body rows of every `tabular` in a tex file, `%`-comment lines dropped."""
    out = []
    body: list[str] | None = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(r"\begin{tabular}"):
            body = []
            continue
        if s.startswith(r"\end{tabular}"):
            if body is not None:
                out.append(body)
            body = None
            continue
        if body is not None and "&" in s and not s.startswith("%"):
            body.append(s)
    return out


def _num(tok: str) -> float:
    return float(tok.replace("$", "").replace("{", "").replace("}", "").replace(",", "").strip())


def read_tex_with_inputs(path: Path) -> tuple[str, list[Path]]:
    """Read a tex file with every `\\input{tables/<name>}` line replaced by that rendered file.

    `paper/appendix_training.tex` no longer prints its headline table inline: the table is a
    machine-rendered, hash-locked file under `paper/tables/` pulled in by `\\input`. Parsing the
    appendix literally therefore finds no coverage rows at all, which reads as every arm missing
    rather than as a moved table. The inlined files are returned so provenance can cite them.
    """
    text = path.read_text()
    inlined: list[Path] = []

    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        target = REPO / "paper" / (name if name.endswith(".tex") else f"{name}.tex")
        require(target)
        inlined.append(target)
        return target.read_text()

    text = re.sub(r"^\\input\{(tables/[^}]+)\}\s*$", _sub, text, flags=re.M)
    return text, inlined


def parse_printed_coverage(text: str) -> dict[str, dict[str, Any]]:
    """tab:train-coverage as the appendix prints it: arm -> suite -> (delta, [lo, hi], cap8)."""
    out: dict[str, dict[str, Any]] = {}
    for body in _tex_tables(text):
        for line in body:
            cells = [c.strip() for c in line.rstrip("\\").split("&")]
            if len(cells) != 7:
                continue
            arm = cells[0].strip()
            if arm not in {a for a, _, _ in DEV_ARMS}:
                continue
            # tab:train-behaviour also has seven cells and also starts with an arm name, so the
            # shape of the second and third cells is what distinguishes the two tables: a signed
            # delta followed by an interval pair.
            if not re.fullmatch(r"\$[+-]\d+\.\d+\$", cells[1]):
                continue
            if len(re.findall(r"[+-]?\d+\.\d+", cells[2])) != 2:
                continue
            row = {}
            for i, suite in enumerate(SUITES):
                lo, hi = re.findall(r"[+-]?\d+\.\d+", cells[2 + 3 * i])
                row[suite] = {
                    "delta": _num(cells[1 + 3 * i]),
                    "ci": [float(lo), float(hi)],
                    "cap8": _num(cells[3 + 3 * i]),
                }
            out[arm] = row
    if len(out) != len(DEV_ARMS):
        raise Mismatch(
            f"tab:train-coverage: parsed {sorted(out)}, expected {[a for a, _, _ in DEV_ARMS]}"
        )
    return out


def parse_printed_behaviour(text: str) -> dict[tuple[str, str], dict[str, float]]:
    """tab:train-behaviour as the appendix prints it: (arm, suite) -> asks, stop, ask cells."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    known = {a for a, _, _ in DEV_ARMS} | {"Prompted"}
    for body in _tex_tables(text):
        for line in body:
            cells = [c.strip() for c in line.rstrip("\\").split("&")]
            if len(cells) != 7:
                continue
            arm = BEHAVIOUR_LABEL.get(cells[0].strip(), cells[0].strip())
            suite = {"MuSiQue": "musique", "StrategyQA": "strategyqa"}.get(cells[1].strip())
            if arm not in known or suite is None:
                continue
            out[(arm, suite)] = {
                "asks": _num(cells[2]),
                "distinct3": _num(cells[3]),
                "question_words": _num(cells[4]),
                "p_stop_given_done": _num(cells[5]),
                "p_ask_given_not_done": _num(cells[6]),
            }
    if len(out) != 2 * (len(DEV_ARMS) + 1):
        raise Mismatch(f"tab:train-behaviour: parsed {len(out)} rows, expected {2 * 7}")
    return out


def _bca_post_fix(cells: Sequence[Mapping[str, Any]], arm: str) -> dict[str, dict[str, Any]]:
    """The post-fix interval record for one arm's tab:train-coverage row.

    The gate verdicts on disk carry PRE-FIX bootstrap endpoints. This record maps each printed
    tex cell to its pre-fix pair, its post-fix pair, its point estimate and the verdict files it
    came from, so the appendix's interval is citable without recomputing a bootstrap here.
    """
    out: dict[str, dict[str, Any]] = {}
    for c in cells:
        if c.get("file") != "paper/appendix_training.tex" or c.get("is_comment"):
            continue
        if c.get("line_text", "").split("&")[0].strip() != arm:
            continue
        idx = c.get("cell_index")
        if idx not in (0, 1):
            continue
        out[SUITES[idx]] = c
    if set(out) != set(SUITES):
        raise Mismatch(f"{rel(BCA_CELLS)}: no post-fix interval pair for {arm!r}")
    return out


def _round_eq(a: float, b: float, places: int) -> bool:
    return abs(round(a, places) - round(b, places)) < 10 ** (-places - 3)


def _scorer_hash_split(rows: Mapping[str, Any]) -> dict[str, list[str]]:
    """Which rows were scored under which hash.

    These six arms do NOT share one scoring hash: the imitation reference and the question-pair
    arm were gated out of one root and the other four out of another, and `scorer_hash` is
    global over the gold set, so a re-score of one root re-stamps every run in it. The appendix
    already prints the rows in one table, so this record inherits the mix rather than creating
    it, and naming it is the difference between an inherited caveat and a hidden one.
    """
    out: dict[str, list[str]] = {}
    for row in rows.values():
        for cell in row["suites"].values():
            h = cell.get("scorer_hash")
            if h:
                out.setdefault(h, []).append(row["reader_name"])
    return {h: sorted(set(v)) for h, v in out.items()}


def collect_dev_rows() -> dict[str, Any]:
    """Every dev-population number Fig 3 and Table 2 draw, each checked three ways.

    The point estimate, `n`, the mean question count and the two per-decision cells come from
    the gate verdict JSON; the interval comes from the post-fix interval record; and every one
    of them is asserted equal to what `paper/appendix_training.tex` prints, to four decimals,
    before anything is written.
    """
    require(BCA_CELLS, APPENDIX_TRAINING)
    cells = json.loads(BCA_CELLS.read_text())
    tex, inlined = read_tex_with_inputs(APPENDIX_TRAINING)
    printed_cov = parse_printed_coverage(tex)
    printed_beh = parse_printed_behaviour(tex)

    rows: dict[str, Any] = {}
    sources = [BCA_CELLS, APPENDIX_TRAINING, *inlined]
    scorer, graph = set(), set()
    failures: list[str] = []

    for arm, reader, slug in DEV_ARMS:
        post = _bca_post_fix(cells, arm)
        row: dict[str, Any] = {"reader_name": reader, "appendix_row_label": arm, "suites": {}}
        for suite in SUITES:
            rec = post[suite]
            vpath = REPO / rec["sources"][0].split("#")[0]
            require(vpath)
            verdict = json.loads(vpath.read_text())
            crit = verdict["criteria"]
            ec = crit["evidence_coverage"]
            mc = verdict["matched_cost"]["by_suite"][suite]
            s2 = crit["stop_2x2"]
            want = printed_cov[arm][suite]
            beh = printed_beh[(arm, suite)]

            # Each check is compared at the precision the appendix actually prints, not at a
            # single global one: the coverage columns carry four decimals and the question
            # count carries three, so comparing the latter at four would fail on every row and
            # teach a reader to loosen the check rather than to read it.
            checks = [
                ("point vs record", ec["value"], rec["point"], 4),
                ("point vs appendix", ec["value"], want["delta"], 4),
                ("n vs record", float(ec["n"]), float(rec["n_units"]), 0),
                ("ci_lo vs appendix", rec["recomputed_post"][0], want["ci"][0], 4),
                ("ci_hi vs appendix", rec["recomputed_post"][1], want["ci"][1], 4),
                ("cap8 vs appendix", ec["cap8_value"], want["cap8"], 4),
                ("asks vs appendix", mc["trained_mean_n_asks"], beh["asks"], 3),
                (
                    "stop-when-done vs appendix",
                    s2["value"]["p_stop_given_done"],
                    beh["p_stop_given_done"],
                    4,
                ),
                (
                    "ask-when-not-done vs appendix",
                    s2["value"]["p_ask_given_not_done"],
                    beh["p_ask_given_not_done"],
                    4,
                ),
            ]
            bad = [(w, a, b) for w, a, b, d in checks if not _round_eq(a, b, d)]
            if bad:
                for w, a, b in bad:
                    failures.append(f"{reader} / {SUITE_TITLE[suite]}: {w}: {a!r} vs {b!r}")
                continue

            scorer.add(verdict["selection"]["scorer_hash"])
            graph.update(rec.get("graph_version") or [])
            sources.append(vpath)
            row["suites"][suite] = {
                "verdict": rel(vpath),
                "coverage_delta_matched": ec["value"],
                "coverage_ci_post_fix": list(rec["recomputed_post"]),
                "coverage_ci_pre_fix_on_disk": [ec["ci_lo"], ec["ci_hi"]],
                "coverage_delta_cap8": ec["cap8_value"],
                "n_tasks": ec["n"],
                "mean_questions_asked": mc["trained_mean_n_asks"],
                "p_stop_given_done": s2["value"]["p_stop_given_done"],
                "p_ask_given_not_done": s2["value"]["p_ask_given_not_done"],
                "appendix_printed": {
                    "delta": want["delta"],
                    "ci": want["ci"],
                    "cap8": want["cap8"],
                    **{k: beh[k] for k in ("asks", "p_stop_given_done", "p_ask_given_not_done")},
                },
                "comparator_prompted": {
                    "mean_questions_asked": mc["baseline_mean_n_asks"],
                    "p_stop_given_done": s2["baseline"]["p_stop_given_done"],
                    "p_ask_given_not_done": s2["baseline"]["p_ask_given_not_done"],
                },
                "scorer_hash": verdict["selection"]["scorer_hash"],
                "bootstrap": verdict["bootstrap"],
            }
        if row["suites"]:
            rows[slug] = row

    if failures:
        raise Mismatch(
            "the record and the printed appendix disagree; the arms below are NOT DRAWN:\n  "
            + "\n  ".join(failures)
        )

    prompted: dict[str, Any] = {"reader_name": "the same weights prompted", "suites": {}}
    for suite in SUITES:
        anchor = rows["stacked"]["suites"][suite]
        beh = printed_beh[("Prompted", suite)]
        for what, a, b, d in (
            ("asks", anchor["comparator_prompted"]["mean_questions_asked"], beh["asks"], 3),
            (
                "stop-when-done",
                anchor["comparator_prompted"]["p_stop_given_done"],
                beh["p_stop_given_done"],
                4,
            ),
            (
                "ask-when-not-done",
                anchor["comparator_prompted"]["p_ask_given_not_done"],
                beh["p_ask_given_not_done"],
                4,
            ),
        ):
            if not _round_eq(a, b, d):
                raise Mismatch(
                    f"the comparator / {SUITE_TITLE[suite]}: {what}: {a!r} vs the appendix's {b!r}"
                )
        prompted["suites"][suite] = {
            "mean_questions_asked": anchor["comparator_prompted"]["mean_questions_asked"],
            "p_stop_given_done": anchor["comparator_prompted"]["p_stop_given_done"],
            "p_ask_given_not_done": anchor["comparator_prompted"]["p_ask_given_not_done"],
            "appendix_printed": beh,
        }
    rows["prompted"] = prompted
    return {
        "rows": rows,
        "sources": sources,
        "scorer_hash": sorted(scorer),
        "graph_version": sorted(graph),
    }


# --------------------------------------------------------------------------- fig 3

# The selected recipe's marker is the recipe itself, training seeds 1 and 2 pooled on the
# development split (seed-0 audit C6.2), read from the audit's re-gating of every seed in one store
# (scripts/seed0_audit/gate_by_seed.py). The published marker was the set-aside run's verdict; the
# record's own reading of that run must reproduce it before the recipe's cell is drawn.
FIG3_RECIPE_RECORD = REPO / "artifacts/seed0_audit_20260924/dev_gate_by_seed.json"
FIG3_RECIPE_SEEDS = ("s1", "s2")
# The recipe's marker as the caption prints it (artifacts/seed0_audit_20260924/c6_figures.tex, C6.2).
FIG3_RECIPE_PRINTED = {
    "musique": {
        "p_ask_given_not_done": "0.9436",
        "p_stop_given_done": "0.4272",
        "coverage_delta_matched": "+0.1556",
    },
    "strategyqa": {
        "p_ask_given_not_done": "0.9091",
        "p_stop_given_done": "0.5454",
        "coverage_delta_matched": "+0.0799",
    },
}
_FIG3_RECORD_KEY = {
    "p_ask_given_not_done": "stop.p_ask_given_not_done",
    "p_stop_given_done": "stop.p_stop_given_done",
    "coverage_delta_matched": "mc.delta",
}


def _fig3_recipe_marker(
    seed0_published: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], str, list[Path]]:
    """The recipe's pooled development cell per suite, refused unless (1) the record's own locks are
    clean, (2) its reading of the set-aside run equals the marker the published figure drew,
    (3) each seed's cell equals that seed's gate verdict (the files Table 2 reads), (4) the pooled
    cell is the decision-weighted pool of the two seeds and the within-task average of their
    coverage, and (5) it reproduces the values the caption prints."""
    require(FIG3_RECIPE_RECORD)
    rec = json.loads(FIG3_RECIPE_RECORD.read_text())
    cells = rec["cells"]
    bad: list[str] = []
    for k, v in (rec.get("locks") or {}).items():
        if v.get("mismatch"):
            bad.append(f"the record's lock {k} lists mismatches {v['mismatch']}")
        if k.startswith("pool.") and not (v.get("same_keys") and v.get("one_run_per_key")):
            bad.append(f"the record's pool lock {k} is not clean: {v}")
    models = dict(zip(FIG3_RECIPE_SEEDS, (m for _l, m in DEV_SEED_MODELS)))
    verdicts: list[Path] = []
    out: dict[str, dict[str, Any]] = {}
    for suite in SUITES:
        s0, pub = cells["s0"][suite], seed0_published[suite]
        for k, rk in _FIG3_RECORD_KEY.items():
            if abs(s0[rk] - pub[k]) > 1e-9:
                bad.append(
                    f"seed-0 {suite} {k}: the record reads {s0[rk]!r}, the published marker "
                    f"{pub[k]!r}"
                )
        for s in FIG3_RECIPE_SEEDS:
            path = DEV_SEED_GATE / f"{models[s]}.{suite}.json"
            require(path)
            verdicts.append(path)
            v = json.loads(path.read_text())
            want = {
                "p_ask_given_not_done": v["criteria"]["stop_2x2"]["value"]["p_ask_given_not_done"],
                "p_stop_given_done": v["criteria"]["stop_2x2"]["value"]["p_stop_given_done"],
                "coverage_delta_matched": v["matched_cost"]["by_suite"][suite]["delta"],
            }
            for k, rk in _FIG3_RECORD_KEY.items():
                if abs(cells[s][suite][rk] - want[k]) > 1e-9:
                    bad.append(
                        f"{s} {suite} {k}: the record's {cells[s][suite][rk]!r}, the verdict's {want[k]!r}"
                    )
        a, b, p = (cells[s][suite] for s in (*FIG3_RECIPE_SEEDS, "s1s2"))
        nd = a["stop.n_not_done"] + b["stop.n_not_done"]
        dn = a["stop.n_done"] + b["stop.n_done"]
        pool = {
            "stop.n_not_done": nd,
            "stop.n_done": dn,
            "stop.p_ask_given_not_done": (
                a["stop.p_ask_given_not_done"] * a["stop.n_not_done"]
                + b["stop.p_ask_given_not_done"] * b["stop.n_not_done"]
            )
            / nd,
            "stop.p_stop_given_done": (
                a["stop.p_stop_given_done"] * a["stop.n_done"]
                + b["stop.p_stop_given_done"] * b["stop.n_done"]
            )
            / dn,
            "mc.delta": (a["mc.delta"] + b["mc.delta"]) / 2,
        }
        for k, want in pool.items():
            if abs(p[k] - want) > 1e-9:
                bad.append(
                    f"{suite} {k}: the pooled cell's {p[k]!r} is not the pool of its seeds {want!r}"
                )
        for k, printed in FIG3_RECIPE_PRINTED[suite].items():
            got = p[_FIG3_RECORD_KEY[k]]
            fmt = f"{got:+.4f}" if printed[0] in "+-" else f"{got:.4f}"
            if fmt != printed:
                bad.append(f"{suite} {k}: {got!r} is not the printed {printed}")
        out[suite] = {
            "p_ask_given_not_done": p["stop.p_ask_given_not_done"],
            "p_stop_given_done": p["stop.p_stop_given_done"],
            "coverage_delta_matched": p["mc.delta"],
            "n_not_done": p["stop.n_not_done"],
            "n_done": p["stop.n_done"],
            "n_tasks": p["mc.n_tasks"],
            "scorer_hash": rec["scorer_hash"],
            "verdict": f"{rel(FIG3_RECIPE_RECORD)} cells.s1s2.{suite}",
        }
    if bad:
        raise Mismatch("fig3, the recipe's marker: " + "; ".join(bad) + ". NOT WRITING.")
    return out, rec["scorer_hash"], verdicts


def fig3_stopping(out: Path) -> list[Path]:
    got = collect_dev_rows()
    rows = got["rows"]
    published = rows["stacked"]["suites"]
    recipe, recipe_scorer, recipe_verdicts = _fig3_recipe_marker(published)
    seed0_lock = {
        suite: {
            "x_ask_when_not_done": published[suite]["p_ask_given_not_done"],
            "y_stop_when_done": published[suite]["p_stop_given_done"],
            "coverage_delta_matched": published[suite]["coverage_delta_matched"],
            "verdict": published[suite]["verdict"],
            "reproduced_by": f"{rel(FIG3_RECIPE_RECORD)} cells.s0.{suite}",
        }
        for suite in SUITES
    }
    if not SHOW_SEED0:
        rows["stacked"] = {"reader_name": rows["stacked"]["reader_name"], "suites": recipe}
    order = [slug for _, _, slug in DEV_ARMS] + ["prompted"]
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    drawn: dict[str, Any] = {}
    for slug in order:
        row = rows[slug]
        color, marker = FIG3_STYLE[slug][0], FIG3_GROUP_MARKER[slug]
        big = slug == "stacked"
        for suite in SUITES:
            cell = row["suites"][suite]
            x, y = cell["p_ask_given_not_done"], cell["p_stop_given_done"]
            ax.plot(
                [x],
                [y],
                marker=marker,
                markersize=7.0 if big else 5.0,
                color=color,
                markerfacecolor=color if suite == "musique" else "none",
                # the selected arm's filled marker is ringed in black; its open one keeps the
                # arm's color, or it would read as the black imitation reference
                markeredgecolor=INK if big and suite == "musique" else color,
                markeredgewidth=1.3 if big else 0.9,
                linestyle="none",
                zorder=6 if big else 3,
            )
            drawn.setdefault(slug, {"reader_name": row["reader_name"]})[suite] = {
                "x_ask_when_not_done": x,
                "y_stop_when_done": y,
                "coverage_delta_matched": cell.get("coverage_delta_matched"),
                "scorer_hash": cell.get("scorer_hash"),
                "verdict": cell.get("verdict"),
            }
    ax.set_xlabel("ask-when-not-done rate (higher is better)")
    ax.set_ylabel("stop-when-done rate (higher is better)")
    ax.set_ylim(-0.04, 1.08)
    ax.margins(x=0.10)
    ax.legend(
        [
            plt.Line2D([], [], marker="o", color=GREY, ls="none", markerfacecolor=GREY),
            plt.Line2D([], [], marker="o", color=GREY, ls="none", markerfacecolor="none"),
        ],
        ["MuSiQue (filled)", "StrategyQA (open)"],
        # Lower left is the one empty corner: the recipe's pooled marker sits mid-right.
        loc="lower left",
        frameon=False,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    fig.tight_layout()
    # An arm whose two suites land within one marker width of each other (the prompted model:
    # 0.964/0.153 and 0.963/0.146) shows one marker; say that both are there, beside it.
    coincident = []
    for slug in order:
        m, q = (drawn[slug][s] for s in SUITES)
        a, b = (
            ax.transData.transform((c["x_ask_when_not_done"], c["y_stop_when_done"]))
            for c in (m, q)
        )
        if ((a - b) ** 2).sum() ** 0.5 < (7.0 if slug == "stacked" else 5.0) * fig.dpi / 72:
            coincident.append(slug)
            ax.annotate(
                "both suites",
                (m["x_ask_when_not_done"], m["y_stop_when_done"]),
                xytext=(-6, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=6.5,
                color=FIG3_STYLE[slug][0],
            )
    # The label stays short: the legend sets this figure's width, and the caption says the selected
    # marker pools the two training seeds.
    selected = " (selected)"
    fig.legend(
        [
            plt.Line2D(
                [],
                [],
                marker=FIG3_GROUP_MARKER[s],
                color=FIG3_STYLE[s][0],
                markerfacecolor=FIG3_STYLE[s][0],
                markeredgecolor=INK if s == "stacked" else FIG3_STYLE[s][0],
                ls="none",
                markersize=6.0 if s == "stacked" else 5.0,
            )
            for s in order
        ],
        [
            FIG3_LEGEND_NAME.get(s, rows[s]["reader_name"]) + (selected if s == "stacked" else "")
            for s in order
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 0.02),
        frameon=False,
        ncol=1,
        handletextpad=0.4,
        labelspacing=0.28,
    )

    prov = provenance(
        asset_id="fig3_stopping",
        sources=[*got["sources"], FIG3_RECIPE_RECORD, *recipe_verdicts],
        query=(
            "per (arm, suite) on the development population: "
            "criteria.stop_2x2.value.{p_ask_given_not_done, p_stop_given_done} of the gate "
            "verdict named by artifacts/bca_provenance_20260918/cells.json, each asserted equal "
            "to the value printed in paper/appendix_training.tex tab:train-behaviour; the selected "
            "recipe's marker is cells.s1s2 of artifacts/seed0_audit_20260924/dev_gate_by_seed.json "
            "(training seeds 1 and 2 pooled over decision points, coverage averaged within task)"
        ),
        values=drawn,
        scorer_hash=[*got["scorer_hash"], recipe_scorer],
        graph_version=got["graph_version"],
        extra={
            "population": (
                "development-split rollouts carrying the exploratory flag, 132 MuSiQue tasks "
                "and 333 StrategyQA tasks; one decision point per (run, turn), cap-forced final "
                "states excluded; the recipe's marker pools both training seeds' decision points "
                f"({recipe['musique']['n_not_done'] + recipe['musique']['n_done']} MuSiQue, "
                f"{recipe['strategyqa']['n_not_done'] + recipe['strategyqa']['n_done']} StrategyQA)"
            ),
            "assertion": (
                "every drawn value of the other arms equals the printed appendix value to 4 "
                "decimals; the recipe's marker equals the pool of its two seeds' verdicts and the "
                "values FIG3_RECIPE_PRINTED gives (the caption's), after the record's reading of "
                "the set-aside run reproduces its published marker (seed0_lock)"
            ),
            "seed0_lock": seed0_lock,
            "marker_size": (
                "Marker size is NOT a quantity: the selected recipe is drawn larger, every other "
                "marker at one size. The coverage gain of each arm is recorded here per value, "
                "not drawn."
            ),
            "markers": (
                "shape per group of arms (FIG3_GROUP_MARKER), color per arm, filled MuSiQue and "
                "open StrategyQA; arms whose two suites coincide within one marker width are "
                "labeled 'both suites' beside the marker"
            ),
            "coincident_suites": coincident,
            "scorer_hash_split": _scorer_hash_split(
                {k: v for k, v in rows.items() if k != "prompted"}
            ),
        },
    )
    return _emit_figure(fig, out, "fig3_stopping", prov)


# --------------------------------------------------------------------------- table 1


TABLE1_SUITES = ("musique", "strategyqa", "wiki2")
TABLE1_TITLE = {
    "musique": "MuSiQue",
    "strategyqa": "StrategyQA",
    "wiki2": "2WikiMultiHopQA",
}
# The coverage row is asserted against these three literals before anything is written. They are
# the published-number lock of artifacts/testsplit_plan_metrics_20260918/RESULT.md, and they are
# what paper/results.tex quotes as +0.1238 / +0.0628 / +0.0776.
TABLE1_COVERAGE_LOCK = {
    "musique": 0.12381628787878789,
    "strategyqa": 0.06280952380952382,
    "wiki2": 0.07756024096385543,
}
TABLE1_ROWS: tuple[tuple[str, str, bool], ...] = (
    ("evidence_coverage", "required-evidence coverage", False),
    ("facet_breadth", "breadth", False),
    ("dwr", "depth-weighted recall", False),
    ("max_depth_reached", "deepest need resolved", False),
    ("precedence_violation_rate", r"out-of-order rate$^{\dagger}$", True),
)

# Every emitted tabular is measured against \linewidth at ICLR size, which is 5.5in = 397.5pt.
# The column separators are written INTO the column specification rather than left to
# \tabcolsep, because the asset is a bare tabular: a \setlength outside it would be an
# instruction to the document that this file cannot give, and the default 6pt on six columns is
# 72pt of the 397 that these tables do not have to spare.
COLSEP = r"@{\hspace{5pt}}"

# Measured with tectonic on 2026-09-18 by typesetting each emitted block into a \savebox at
# textwidth 5.5in and printing \wd: the sizes below are the block's own width, not an estimate.
# The note travels with the asset because the size the block needs is a property of the block
# and the document that pastes it cannot be expected to rediscover it.
# Measured with tectonic on 2026-09-23 against the paper's own preamble (T1, times, the ICLR
# style, in which \footnotesize and \small are both 9pt): the widest row is the one-line cell
# "point [lo, hi]" beside the longest label. It fits \linewidth only without row indentation.
# Re-measured the same way on 2026-09-24 night after the restructure into two suite groups of four
# columns under a two-row header (tectonic, the paper's preamble, each size's \wd and \ht+\dp of the
# \input block): was 339.2pt at \small and 373.7pt at \normalsize with the "/" pairs. Re-measured
# the same way on 2026-09-25 after the bold-best cells (bold digits are wider): was 380.4pt at \small
# and 415.8pt at \normalsize without bold; the height is unchanged.
MECHANISM_MAIN_SIZE_NOTE = (
    r"% note: measured against \linewidth at ICLR size (5.5in = 397.5pt): this block is 386.9pt "
    r"at \footnotesize and at \small (both 9pt in the ICLR style) and 422.7pt at \normalsize, so "
    r"set it at \small or smaller. Its height at \small is 106.0pt."
)
# Table 2's difference layout (the appendix copy since 2026-09-25): re-measured the same way on
# 2026-09-24 after the proportions moved to percentage points: 387.6pt (was 396.6pt on the 0.X scale)
# and 407.1pt at \normalsize (was 416.7pt). As Table 2 its height at \small was 157.2pt (14 rows, one
# level line and the question line included; 197.2pt and 18 rows with the four level lines the 9-page
# trim dropped). Re-measured 2026-09-25 without those two lines.
HEADLINE_DIFFS_SIZE_NOTE = (
    r"% note: measured against \linewidth at ICLR size (5.5in = 397.5pt): this block is 387.6pt "
    r"at \footnotesize and at \small (both 9pt in the ICLR style) and 407.1pt at \normalsize, "
    r"so set it at \footnotesize or \small. Its height at \small is 137.2pt (12 rows; 157.2pt as "
    r"Table 2 with its level line and question line, 2026-09-24)."
)
# Table 2 with levels side by side (HEADLINE_LAYOUT), measured 2026-09-25 the same way (tectonic, the
# paper's preamble, \wd and \ht+\dp of the \input block). The four-block layout offered beside it, not
# chosen, measured 374.8pt by 201.0pt at \small.
HEADLINE_SIZE_NOTE = (
    r"% note: measured against \linewidth at ICLR size (5.5in = 397.5pt): this block is 375.3pt "
    r"at \footnotesize and at \small (both 9pt in the ICLR style) and 396.5pt at \normalsize. Its "
    r"height at \small is 156.0pt (15 rows at \arraystretch 0.9: two header rows, two block "
    r"labels, 11 data rows; 171.0pt at \arraystretch 1)."
)
SIZE_NOTE = (
    r"% note: measured against \linewidth at ICLR size (5.5in = 397.5pt): this block is "
    "{small}pt at \\small and {normal}pt at \\normalsize, so set it at \\small or smaller."
)


def _fmt(x: float, places: int, sign: bool = False) -> str:
    """THE one formatter for a printed table value: rounded HALF UP on the exact decimal the value
    reads as (its repr), never on its binary expansion. 0.2175 is exactly 87/400, and the nearest
    double is 0.21749999999999999, so f"{x:.3f}" prints 0.217 where the paper's rule prints 0.218
    (main-text owner, 2026-09-23)."""
    d = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return format(d, "+f" if sign else "f")


def _fmt3(x: float) -> str:
    return _fmt(x, 3, sign=True)


def _cell3(delta: float, lo: float, hi: float) -> str:
    """A delta over its interval, the interval one size down.

    Both belong to the first line of the cell, as the table's brief asks. Setting the interval
    smaller is what makes three of these fit across 397pt beside a row label: at one size the
    row is 616pt wide and overfull by more than half a column.
    """
    return f"${_fmt3(delta)}$ " + _ci3((lo, hi))


def _ci3(ci: Sequence[float]) -> str:
    return r"{\scriptsize $[" + f"{_fmt3(ci[0])}, {_fmt3(ci[1])}" + r"]$}"


def _pts(x: float, sign: bool = False, places: int = 3) -> str:
    """A proportion in percentage points, or a rate in percent: the string `_fmt` prints at
    `places` decimals with its decimal point moved two places right, NEVER a fresh rounding of
    x*100 ("+0.112" -> "+11.2", "-0.035" -> "-3.5", "+0.009" -> "+0.9", "0.13" at two places ->
    "13"). So every value stays digit-identical to the fraction the paper printed before, and a
    value whose 4-dp display is a tie (0.15246... shows 0.1525, prints 0.152) reads 15.2, not the
    15.3 a second rounding of the display would give (the user's comment, 2026-09-24)."""
    return format(Decimal(_fmt(x, places, sign=sign)).scaleb(2), "+f" if sign else "f")


def _pts_tick(v: float, _pos: Any = None) -> str:
    """A figure tick in points: `_pts` of the tick value, trailing zeros dropped (0.35 -> 35), a
    float-noise zero printed 0 and the minus sign matplotlib's own (U+2212)."""
    s = _pts(v)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if Decimal(s) == 0 else s.replace("-", "−")


def _axis_in_points(axis) -> None:
    """Label an axis that draws fractions in points. The drawn values stay the records' own
    fractions, so the provenance `values` and every placement computed in data units are those of
    the fraction figure; only the tick labels move their decimal point."""
    from matplotlib.ticker import FuncFormatter

    axis.set_major_formatter(FuncFormatter(_pts_tick))


def _cell_pts(delta: float, lo: float, hi: float) -> str:
    """`_cell3` in percentage points: "$+11.2$ {\\scriptsize $[+8.7, +14.3]$}"."""
    return (
        f"${_pts(delta, sign=True)}$ "
        + r"{\scriptsize $["
        + f"{_pts(lo, sign=True)}, {_pts(hi, sign=True)}"
        + r"]$}"
    )


def collect_table1() -> dict[str, Any]:
    require(TESTSPLIT_CONTRASTS, TESTSPLIT_RESULT, SYMMETRIC_RECORD)
    contrasts = json.loads(TESTSPLIT_CONTRASTS.read_text())
    index = {(r["metric"], r["suite"], r["basis"]): r for r in contrasts}
    # The symmetric record (2026-09-19) reads BOTH arms at the lower question count. The gate
    # helper read the comparator at min(k, its own asks) and the treatment at k, which flatters
    # the treatment on the pairs where the comparator was the shorter side. Only the coverage
    # cells are recomputed there so far; every other cell carries a null with a reason string.
    symmetric = json.loads(SYMMETRIC_RECORD.read_text())
    sources = [TESTSPLIT_CONTRASTS, TESTSPLIT_RESULT, SYMMETRIC_RECORD]
    scorer, graph = set(), set()
    values: dict[str, Any] = {}

    for suite in TABLE1_SUITES:
        vpath = TESTSPLIT_VERDICTS / f"stacked-notdone.{suite}.json"
        require(vpath)
        sources.append(vpath)
        verdict = json.loads(vpath.read_text())
        ec = verdict["criteria"]["evidence_coverage"]
        if verdict.get("coverage_rule") != "matched_cost":
            raise Mismatch(
                f"{rel(vpath)}: coverage_rule is {verdict.get('coverage_rule')!r}, not "
                "'matched_cost'. `value` means a different metric under the other rule."
            )
        lock = TABLE1_COVERAGE_LOCK[suite]
        if ec["value"] != lock:
            raise Mismatch(
                f"{rel(vpath)}: required-evidence coverage is {ec['value']!r}, the published "
                f"lock is {lock!r}. NOT WRITING."
            )
        cross = index[("evidence_coverage", suite, "matched")]["task"]["delta"]
        if abs(cross - lock) > 1e-12:
            raise Mismatch(
                f"{suite}: the verdict says {lock!r} and {rel(TESTSPLIT_CONTRASTS)} says "
                f"{cross!r} for the same cell. NOT WRITING."
            )
        cap8_cross = index[("evidence_coverage", suite, "unmatched")]["task"]["delta"]
        if abs(cap8_cross - ec["cap8_value"]) > 1e-12:
            raise Mismatch(
                f"{suite}: cap-8 coverage is {ec['cap8_value']!r} on the verdict and "
                f"{cap8_cross!r} in the contrast record. NOT WRITING."
            )
        sym = symmetric[f"evidence_coverage::{suite}::matched"]
        if sym.get("symmetric_delta") is None:
            raise Mismatch(
                f"{suite}: the symmetric record carries no coverage value "
                f"({sym.get('reason')!r}). NOT WRITING."
            )
        if (
            sym["published_value"] != lock
            or sym["scorer_hash"] != verdict["selection"]["scorer_hash"]
        ):
            raise Mismatch(
                f"{suite}: the symmetric record supersedes {sym['published_value']!r} under "
                f"{sym['scorer_hash'][:16]}, the verdict carries {lock!r} under "
                f"{verdict['selection']['scorer_hash'][:16]}. Not the same cell. NOT WRITING."
            )
        if sym["n"] != ec["n"] or sym["sign_changed"] or sym["significance_changed"]:
            raise Mismatch(
                f"{suite}: symmetric record disagrees with the verdict on n, sign or significance."
            )
        scorer.add(verdict["selection"]["scorer_hash"])
        graph.add("v1")
        col: dict[str, Any] = {
            "n_tasks": ec["n"],
            "evidence_coverage": {
                "source": rel(SYMMETRIC_RECORD),
                "matched": {
                    "delta": sym["symmetric_delta"],
                    "ci": [sym["symmetric_ci_lo"], sym["symmetric_ci_hi"]],
                    "n": sym["n"],
                },
                "superseded_helper_reading": {
                    "delta": ec["value"],
                    "ci": [ec["ci_lo"], ec["ci_hi"]],
                    "n_unsafe_pairs": sym["n_unsafe_pairs"],
                },
                "cap8_delta": ec["cap8_value"],
                "cap8_n": ec["n"],
                "ci_method": ec["ci_method"],
                "cross_check_contrasts_json": {"matched": cross, "cap8": cap8_cross},
            },
        }
        for metric, _, _ in TABLE1_ROWS[1:]:
            m = index[(metric, suite, "matched")]
            u = index[(metric, suite, "unmatched")]
            helper = {
                "delta": m["task"]["delta"],
                "ci": [m["task"]["lo"], m["task"]["hi"]],
                "n": m["task"]["n"],
            }
            # Prefer the symmetric record once the cell carries a value. A null with a reason
            # string means "not recomputed", and the helper reading is printed FLAGGED; a value
            # is used only if the record agrees with the helper on what it supersedes.
            symcell = symmetric.get(f"{metric}::{suite}::matched")
            if symcell is not None and symcell.get("symmetric_delta") is not None:
                if abs(symcell["published_value"] - helper["delta"]) > 1e-12:
                    raise Mismatch(
                        f"{metric}/{suite}: the symmetric record supersedes "
                        f"{symcell['published_value']!r} but the helper record reads "
                        f"{helper['delta']!r}. Not the same cell. NOT WRITING."
                    )
                matched = {
                    "delta": symcell["symmetric_delta"],
                    "ci": [symcell["symmetric_ci_lo"], symcell["symmetric_ci_hi"]],
                    "n": symcell["n"],
                }
                rule = "symmetric (both arms at the lower question count)"
                source = rel(SYMMETRIC_RECORD)
            else:
                matched = helper
                rule = "helper (comparator at min(k, own asks)); symmetric reading pending"
                source = rel(TESTSPLIT_CONTRASTS)
            col[metric] = {
                "source": source,
                "matched_rule": rule,
                "matched": matched,
                "superseded_helper_reading": helper if source == rel(SYMMETRIC_RECORD) else None,
                "cap8_delta": u["task"]["delta"],
                "cap8_n": u["task"]["n"],
                "declared": m["declared"],
                "cluster_by": "task",
            }
        values[suite] = col
    return {
        "values": values,
        "sources": sources,
        "scorer_hash": sorted(scorer),
        "graph_version": sorted(graph),
    }


def table1_heldout(out: Path) -> list[Path]:
    got = collect_table1()
    v = got["values"]
    lines = [
        "% table1_heldout.tex -- generated by " + TOOL + ", do not hand-edit.",
        "% prov: see table1_heldout.provenance.json beside this file "
        "(inputs with sha256, every value drawn, scorer_hash, graph_version).",
        "% note: the first line of a cell is the paired difference at equal retrieval spend and "
        "the second is the same contrast at a fixed budget of eight retrieval calls.",
        "% note: the out-of-order row is emitted only where both arms' trajectories define it, "
        "so its n is smaller than the column n. Both counts are on the dagger line.",
        "% note: every contrast here is exploratory as a class: no declared endpoint names the "
        "trained arm.",
        SIZE_NOTE.format(small="391.7", normal="408.2"),
        r"\begin{tabular}{@{}l" + COLSEP + "c" + COLSEP + "c" + COLSEP + r"c@{}}",
        r"\toprule",
        " & " + " & ".join(TABLE1_TITLE[s] for s in TABLE1_SUITES) + r" \\",
        " & " + " & ".join(f"($n={v[s]['n_tasks']}$)" for s in TABLE1_SUITES) + r" \\",
        r"\midrule",
    ]
    odd = []
    for metric, label, lower_is_better in TABLE1_ROWS:
        top, bot = [], []
        for suite in TABLE1_SUITES:
            cell = v[suite][metric]
            m = cell["matched"]
            top.append(_cell3(m["delta"], m["ci"][0], m["ci"][1]))
            bot.append(r"{\scriptsize $" + _fmt3(cell["cap8_delta"]) + "$}")
            if m["n"] != v[suite]["n_tasks"]:
                odd.append((m["n"], cell["cap8_n"]))
        lines.append(label + " & " + " & ".join(top) + r" \\")
        lines.append(" & " + " & ".join(bot) + r" \\[2pt]")
    lines += [r"\bottomrule"]
    # The dagger line carries what will not fit in a cell: the direction of the quantity, and
    # the population it is defined on, which is smaller than the column's AND different between
    # the two cost bases. Naming one of those counts and not the other is how a reader ends up
    # attributing a count to the wrong reading.
    # Two rows and not one: a \scriptsize \multicolumn does not shrink with the surrounding
    # font size, so a single long line becomes the widest object in the tabular and the whole
    # block is overfull no matter what size the document sets it at. Measured: one line put
    # this table at 590pt against a 397pt line, at every size.
    if odd:
        lines.append(
            r"\multicolumn{4}{@{}l@{}}{\scriptsize $\dagger$ lower is better, and defined only "
            r"where both policies' trajectories define it:} \\"
        )
        lines.append(
            r"\multicolumn{4}{@{}l@{}}{\scriptsize \phantom{$\dagger$} "
            + " / ".join(str(a) for a, _ in odd)
            + " tasks at equal spend and "
            + " / ".join(str(b) for _, b in odd)
            + r" at the fixed budget.} \\"
        )
    lines += [r"\end{tabular}"]

    prov = provenance(
        asset_id="table1_heldout",
        sources=got["sources"],
        query=(
            "required-evidence coverage from artifacts/testsplit_qa/verdicts/"
            "stacked-notdone.<suite>.json (criteria.evidence_coverage value/ci_lo/ci_hi/n and "
            "cap8_value); the other four rows from artifacts/testsplit_plan_metrics_20260918/"
            "contrasts.json, task-clustered rows, basis matched and unmatched"
        ),
        values=got["values"],
        scorer_hash=got["scorer_hash"],
        graph_version=got["graph_version"],
        extra={
            "coverage_lock": TABLE1_COVERAGE_LOCK,
            "population": (
                "split = 'test', grid tier1_trained_qa_base, budget_cap 8, max_turns 16; the "
                "trained questioner against the same weights prompted, paired per (suite, task, "
                "seed), seeds averaged into the task, task-clustered BCa, 1,000 resamples, seed 0"
            ),
            "two_interval_estimators": (
                "The coverage row's interval is the verdict's own (pinq_train.gate, "
                "task-clustered BCa) and the other four rows' intervals are "
                "pi_eval.stats.inference.paired_difference as contrasts.json records them. The "
                "POINT estimates of the coverage row agree between the two to 1e-12, which is "
                "asserted before writing; the intervals differ slightly and the table prints the "
                "verdict's, which is what paper/results.tex quotes."
            ),
            "wiki2_depth_limit": (
                "The 200 wiki2 test tasks carry required gold nodes at depths 0 and 1 only, so "
                "the deepest-need row cannot exceed 1 there and no depth-two statement is made."
            ),
            "wiki2_breadth_dilution": (
                "47 of the 166 paired wiki2 tasks carry no gold facet and the scorer emits a "
                "breadth of 0.0 there. Restricted to the 119 tasks that have at least one facet "
                "the matched delta is +0.15966 [+0.08024, +0.23950]; the table prints the "
                "unrestricted +0.11446 over all 166."
            ),
        },
    )
    return _emit_tex(out, "table1_heldout", "\n".join(lines) + "\n", prov)


# --------------------------------------------------------------------------- table 2


DEV_SEED_GATE = REPO / "artifacts/seed_identity_20260923/dev_gate"
DEV_SEED_MODELS = (
    ("both kinds, from pairs, seed 1", "qwen3-8b-dpo-stacked-notdone-both-s1"),
    ("both kinds, from pairs, seed 2", "qwen3-8b-dpo-stacked-notdone-both-s2"),
)


def _dev_seed_rows(seed0_published: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Dev rows for the recipe's fresh seeds, locked on seed 0 re-gated in the same store."""

    def read(model: str, suite: str) -> dict[str, Any]:
        path = DEV_SEED_GATE / f"{model}.{suite}.json"
        require(path)
        v = json.loads(path.read_text())
        mc = v["matched_cost"]["by_suite"][suite]
        st = v["criteria"]["stop_2x2"]["value"]
        return {
            "verdict": rel(path),
            "coverage_delta_matched": mc["delta"],
            "coverage_ci": [mc["ci_lo"], mc["ci_hi"]],
            "n_tasks": mc["n_tasks"],
            "mean_questions_asked": mc["trained_mean_n_asks"],
            "p_stop_given_done": st["p_stop_given_done"],
            "p_ask_given_not_done": st["p_ask_given_not_done"],
            "gate_passed": v["passed"],
        }

    for suite in SUITES:
        mine = read("qwen3-8b-dpo-stacked-notdone-both", suite)
        pub = seed0_published[suite]
        for k in (
            "coverage_delta_matched",
            "p_stop_given_done",
            "p_ask_given_not_done",
            "mean_questions_asked",
        ):
            if abs(mine[k] - pub[k]) > 1e-9:
                raise Mismatch(
                    f"seed 0 re-gated in the seed store reads {k}={mine[k]!r} on {suite}, the "
                    f"published row {pub[k]!r}. NOT WRITING."
                )
    return [(label, {s: read(model, s) for s in SUITES}) for label, model in DEV_SEED_MODELS]


def _mechanism_behavior(
    cells: Mapping[str, Any], rate: Callable[[float], str] = lambda x: _fmt(x, 3)
) -> str:
    """The three per-suite columns. The slash carries no space around it because the three
    columns together have 130pt of a 397pt line and a spaced slash costs 20 of them. `rate`
    prints the two per-decision rates: fractions in the appendix, `_pts` (percent) in the main
    text; the question counts are counts either way."""
    m, s = cells["musique"], cells["strategyqa"]
    return (
        f"{rate(m['p_stop_given_done'])}/{rate(s['p_stop_given_done'])} & "
        f"{rate(m['p_ask_given_not_done'])}/{rate(s['p_ask_given_not_done'])} & "
        f"{_fmt(m['mean_questions_asked'], 2)}/{_fmt(s['mean_questions_asked'], 2)}"
    )


def table2_mechanism(out: Path) -> list[Path]:
    got = collect_dev_rows()
    rows = got["rows"]
    lines = [
        "% table2_mechanism.tex -- generated by " + TOOL + ", do not hand-edit.",
        "% prov: see table2_mechanism.provenance.json beside this file "
        "(inputs with sha256, every value drawn, scorer_hash, graph_version).",
        "% note: development population, 132 MuSiQue tasks and 333 StrategyQA tasks, one "
        "training checkpoint at one training seed per row. This is a selection record and no "
        "row carries a headline claim on its own.",
        "% note: the imitation reference's MuSiQue lower bound is COUNT-UNSTABLE. Re-derived "
        "from its own per-task deltas it is positive on 8 of 20 bootstrap seeds at 1,000 "
        "resamples and on 0 of 20 at both 10,000 and 50,000, where the largest bound across "
        "twenty seeds is -0.0006. The point estimate is +0.0492 at every count. The value "
        "printed is the verdict's; the pass it implies is reproducible only at the count it was "
        "taken at (paper/appendix_training.tex:132-137).",
        "% note: the comparator row carries no coverage delta because it is the comparator every "
        "delta in this table is taken against.",
        "% note: these rows do not share one scoring hash. The imitation reference and the "
        "question-pair arm were gated out of one root and the other four out of another, which "
        "the appendix's own table also does; the two hashes and the rows under each are named in "
        "table2_mechanism.provenance.json under extra.scorer_hash_split.",
        SIZE_NOTE.format(small="389.0", normal="409.1"),
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.0cm}" + (COLSEP + "c") * 5 + r"@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{gate-matched gain} "
        r"& \multicolumn{3}{c}{per suite, MuSiQue / StrategyQA} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        r"supervision & MuSiQue & StrategyQA & stop when & ask when & questions \\",
        r" & & & done & not done & asked \\",
        r"\midrule",
    ]
    drawn: dict[str, Any] = {}
    behavior = _mechanism_behavior

    lines.append(
        "same weights, prompted & --- & --- & " + behavior(rows["prompted"]["suites"]) + r" \\"
    )
    lines.append(r"\midrule")
    drawn["the same weights prompted"] = rows["prompted"]["suites"]
    # The interval goes UNDER its delta rather than beside it. Beside it, the two coverage
    # columns are 88pt each and the tabular is 452pt against a 397pt line; under it, they are as
    # wide as their own header and the block fits. Measured both ways.
    # The largest coverage gain per suite, so the table can be read down a column. Computed from
    # the values drawn rather than asserted as a constant, so it cannot go stale.
    printed = [arm for arm in DEV_ARMS if SHOW_SEED0 or arm[2] != "stacked"]
    best = {
        suite: max(
            (rows[slug]["suites"][suite]["coverage_delta_matched"] for _, _, slug in printed)
        )
        for suite in ("musique", "strategyqa")
    }

    def cov(cell: Mapping[str, Any], suite: str) -> str:
        v = cell["coverage_delta_matched"]
        # \textbf around math does not bold digits, so the bold has to be inside the math.
        return rf"$\mathbf{{{_fmt3(v)}}}$" if v == best[suite] else f"${_fmt3(v)}$"

    for _, reader, slug in printed:
        row = rows[slug]
        m, s = row["suites"]["musique"], row["suites"]["strategyqa"]
        lines.append(
            TABLE2_LABEL[slug]
            + r" \newline {\scriptsize\color{black!60} "
            + TABLE2_TRAINED_ON[slug]
            + "}"
            + " & "
            + cov(m, "musique")
            + " & "
            + cov(s, "strategyqa")
            + " & "
            + behavior(row["suites"])
            + r" \\"
        )
        lines.append(
            " & "
            + _ci3(m["coverage_ci_post_fix"])
            + " & "
            + _ci3(s["coverage_ci_post_fix"])
            + r" & & & \\[2pt]"
        )
        drawn[reader] = row["suites"]
    # The recipe's two fresh seeds (seed 0 above resumed its final stage and kept about a tenth of
    # it). Read by `pi train gate` in one store with seed 0's own dev runs, whose verdict there must
    # equal the row printed above before either seed row is written.
    seed_rows = _dev_seed_rows(rows["stacked"]["suites"])
    lines.append(r"\midrule")
    for label, cells in seed_rows:
        m, s = cells["musique"], cells["strategyqa"]
        lines.append(
            label
            + r" \newline {\scriptsize\color{black!60} "
            + TABLE2_TRAINED_ON["stacked"]
            + "}"
            + " & $"
            + _fmt3(m["coverage_delta_matched"])
            + "$ & $"
            + _fmt3(s["coverage_delta_matched"])
            + "$ & "
            + behavior(cells)
            + r" \\"
        )
        lines.append(
            " & " + _ci3(m["coverage_ci"]) + " & " + _ci3(s["coverage_ci"]) + r" & & & \\[2pt]"
        )
        drawn[label] = cells
    lines += [
        r"\bottomrule",
        r"\multicolumn{6}{@{}p{13.6cm}@{}}{\scriptsize Every trained row fails the gated "
        r"length criterion on at least one suite (Appendix~\ref{app:matchedcost}). Every "
        r"gain is measured against the comparator row. \textbf{Bold} marks the largest value in a "
        r"coverage column, which is a comparison of printed values and not a tested difference: "
        r"only one pair of these rows was contrasted directly. The two per-decision columns are "
        r"deliberately not ranked, because the recipe fitted on stop contrasts alone holds the "
        r"highest ask-when-not-done rate in the table together with the lowest stop-when-done "
        r"rate, so they are read as a pair rather than as two scores.} \\",
        r"\end{tabular}",
    ]

    prov = provenance(
        asset_id="table2_mechanism",
        sources=got["sources"],
        query=(
            "per (arm, suite) on development: criteria.evidence_coverage.value / .n and "
            "criteria.stop_2x2.value.* and matched_cost.by_suite.<suite>.trained_mean_n_asks of "
            "the gate verdict; the interval from the post-fix pair recorded in "
            "artifacts/bca_provenance_20260918/cells.json; every one asserted equal to the value "
            "printed in paper/appendix_training.tex to 4 decimals"
        ),
        values=drawn,
        scorer_hash=got["scorer_hash"],
        graph_version=got["graph_version"],
        extra={
            "dagger": "the selected trained questioner",
            "interval_provenance": (
                "The gate verdicts on disk carry PRE-FIX bootstrap endpoints, from before the "
                "BCa order fix. Both pairs are recorded per cell in the provenance under "
                "coverage_ci_pre_fix_on_disk and coverage_ci_post_fix; the table prints the "
                "post-fix pair, which is what the appendix prints. Point estimates, n and the "
                "two decision cells are identical between the two."
            ),
            "scorer_hash_split": _scorer_hash_split(
                {k: v for k, v in rows.items() if k != "prompted"}
            ),
            "reference_row_count_instability": (
                "paper/appendix_training.tex:132-137: the imitation reference's MuSiQue lower "
                "bound of +0.0021 is positive on 8 of 20 bootstrap seeds at 1,000 resamples and "
                "0 of 20 at 10,000 and 50,000."
            ),
        },
    )
    return _emit_tex(out, "table2_mechanism", "\n".join(lines) + "\n", prov)


# The main-text version of tab:mechanism: printed label -> the appendix version's row (its
# provenance key). Six rows, points only; the intervals and the other rows stay in the appendix.
# The labels since the user's round 2 (INBOX_FOR_PAPER_LANE section 14, item 3, and the redirect
# after the interim PDF): sec 5.5 reads the table as what each stage of the training teaches. Until
# then "imitation reference", "stop contrasts only" and "final stage (both kinds), seed 1/2".
MECHANISM_MAIN_SOURCE = {
    "same model, prompted": "the same weights prompted",
    "imitation (first stage)": "the imitation reference",
    "question pairs": "the question-pair arm",  # "..., pooled export" until 2026-09-24
    "stop contrasts alone": "the stop-contrast-only arm",
    "final stage, seed 1": "both kinds, from pairs, seed 1",
    "final stage, seed 2": "both kinds, from pairs, seed 2",
}
# One column group per suite, MuSiQue then StrategyQA, each of four columns: the coverage gain in
# percentage points, stop when done and ask when not done in percent, and questions asked (the
# redirect: no "/" pairs). Exactly two header rows (the 9-page trim): the group names, then short
# column names whose meaning and units the caption gives. "asks" rather than "questions": with it
# the block was 380.4pt at \small before the bold cells (386.9pt with them, MECHANISM_MAIN_SIZE_NOTE);
# "questions" in both groups adds about 33pt, past \linewidth at these column gaps.
MECHANISM_MAIN_HEADER = (
    r"\toprule",
    r" & \multicolumn{4}{c}{MuSiQue} & \multicolumn{4}{c}{StrategyQA} \\",
    r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
    r"supervision & gain & stop\,$|$\,done & ask\,$|$\,not done & asks & gain & stop\,$|$\,done & "
    r"ask\,$|$\,not done & asks \\",
    r"\midrule",
)
# The eight cells of a row as "<suite>::<column>" for the three columns that take bold, None for
# asks, which never does.
MECHANISM_MAIN_BOLD_COLUMNS = (
    "musique::gain",
    "musique::stop",
    "musique::ask",
    None,
    "strategyqa::gain",
    "strategyqa::stop",
    "strategyqa::ask",
    None,
)


def table_mechanism_main(out: Path) -> list[Path]:
    """tab:mechanism for the main text: six rows, one line each, points only; the largest printed
    value of each gain, stop | done and ask | not done column in bold, per suite.

    Every value is read through the appendix version's own loaders and locks (collect_dev_rows
    asserts each cell equal to paper/appendix_training.tex; _dev_seed_rows re-gates seed 0 in the
    seed store and requires it to equal its published row before either fresh seed is read), so the
    two versions cannot disagree on a row they share.
    """
    got = collect_dev_rows()
    rows = got["rows"]
    source: dict[str, Any] = {
        "the same weights prompted": rows["prompted"]["suites"],
        **{reader: rows[slug]["suites"] for _, reader, slug in DEV_ARMS},
        **dict(_dev_seed_rows(rows["stacked"]["suites"])),
    }
    lines = [
        "% table_mechanism_main.tex -- generated by " + TOOL + ", do not hand-edit.",
        "% prov: see table_mechanism_main.provenance.json beside this file; the full version with "
        "intervals and every row is table2_mechanism.tex.",
        "% note: development population, 132 MuSiQue tasks and 333 StrategyQA tasks, one training "
        "checkpoint per row. The imitation row's MuSiQue lower bound (not printed here) is "
        "count-unstable; see table2_mechanism.tex.",
        MECHANISM_MAIN_SIZE_NOTE,
        r"\begin{tabular}{@{}l@{\hspace{8pt}}c@{\hspace{5pt}}c@{\hspace{5pt}}c@{\hspace{5pt}}c"
        r"@{\hspace{10pt}}c@{\hspace{5pt}}c@{\hspace{5pt}}c@{\hspace{5pt}}c@{}}",
        *MECHANISM_MAIN_HEADER,
    ]
    drawn: dict[str, Any] = {}
    printed: dict[str, list[str]] = {}
    # The gains in percentage points and the two rates in percent (the user's comment,
    # 2026-09-24), each the appendix's 3-decimal print with its point moved; questions are counts.
    # One group of four cells per suite; each cell is the string that was one half of its
    # "MuSiQue/StrategyQA" pair before the restructure.
    for label, key in MECHANISM_MAIN_SOURCE.items():
        cells = source[key]
        groups = []
        for suite in ("musique", "strategyqa"):
            c = cells[suite]
            gain = (
                "---"
                if key == "the same weights prompted"
                else f"${_pts(c['coverage_delta_matched'], sign=True)}$"
            )
            groups += [
                gain,
                _pts(c["p_stop_given_done"]),
                _pts(c["p_ask_given_not_done"]),
                _fmt(c["mean_questions_asked"], 2),
            ]
        printed[label] = groups
        drawn[label] = cells
    # Bold (INBOX_FOR_PAPER_LANE section 15, the user's request): per suite, in the gain, stop | done
    # and ask | not done columns, the largest PRINTED value across the six rows, since the printed
    # string is what the reader compares; every tied maximum is bold. The same model's "---" gain
    # takes no part, and asks is never bold. It marks a leader, not a tested difference.
    bold_cells: dict[str, list[str]] = {}
    for j, where in enumerate(MECHANISM_MAIN_BOLD_COLUMNS):
        if where is None:
            continue
        vals = {lb: Decimal(g[j].strip("$")) for lb, g in printed.items() if g[j] != "---"}
        top = max(vals.values())
        bold_cells[where] = [lb for lb, v in vals.items() if v == top]
        for lb in bold_cells[where]:
            s = printed[lb][j]
            printed[lb][j] = (
                r"$\boldsymbol{" + s[1:-1] + "}$" if s.startswith("$") else r"\textbf{" + s + "}"
            )
    for label, groups in printed.items():
        if label in ("imitation (first stage)", "final stage, seed 1"):
            lines.append(r"\midrule")
        lines.append(label + " & " + " & ".join(groups) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    prov = provenance(
        asset_id="table_mechanism_main",
        sources=got["sources"],
        query=(
            "the appendix version's rows (table2_mechanism): collect_dev_rows and _dev_seed_rows, "
            "points only; every value asserted equal to paper/appendix_training.tex by those loaders"
        ),
        values=drawn,
        scorer_hash=got["scorer_hash"],
        graph_version=got["graph_version"],
        extra={
            "build": TOOL + " --only table_mechanism_main",
            "printed_units": "gate-matched gains in percentage points, stop-when-done and "
            "ask-when-not-done in percent, one decimal: each the 3-decimal half-up print of the "
            "value in `values` with its decimal point moved two places (_pts); questions asked "
            "are counts, two decimals. `values` are the records' fractions, unchanged",
            "row_source": MECHANISM_MAIN_SOURCE,
            "scorer_hash_split": _scorer_hash_split(
                {k: v for k, v in rows.items() if k != "prompted"}
            ),
            "rules": {
                "bold": "bold = largest printed value in the column, not a tested difference: "
                "per suite, in the gain, stop | done and ask | not done columns, across the six "
                "rows, compared on the printed strings (every tied maximum bold); the same "
                "model's --- gain takes no part; asks is never bold",
            },
            "bold_cells": bold_cells,
        },
    )
    return _emit_tex(out, "table_mechanism_main", "\n".join(lines) + "\n", prov)


# --------------------------------------------------------------------------- cli


# --------------------------------------------------------------------------- qualitative example

MUSIQUE_TASKS = REPO / "data/corpora/musique/38f5afb69fb7ea18/tasks.jsonl"
MUSIQUE_GOLD = REPO / "data/gold/graphs/musique/v1.jsonl"

# A canary nonce rides inside gold strings so that a gold string reaching a served request can be
# caught (CONTRIBUTING.md, firewall layer 4). A generated paper asset is a published artifact, so a
# canary must never reach one either: every gold string drawn here is stripped and the output is
# then asserted canary-free.
CANARY = re.compile(r"\s*PINQCANARY_[0-9A-Fa-f]+")

# THE SELECTION RULE, fixed before any transcript was read, and asserted in code so that the
# figure cannot quietly become a different one. Among held-out MuSiQue tasks where every drawn
# policy has an ok run at rollout seed 0, take those where the trained questioner resolves a
# required need at depth two or more that the prompted arm does not, and select the
# lexicographically first task id. The complementary count (the same condition with the arms
# exchanged) is recorded in the provenance and printed in the caption, because a rule that selects
# for a phenomenon must report how often the phenomenon runs the other way.
#
# The trained questioner is the recipe, at its two training seeds. The rule names ONE trained run
# per task, so it is read with EACH training seed required to satisfy it (and, for the reverse
# count, the prompted arm required to beat each). The published figure drew the set-aside run
# instead; the same rule keyed to that run's model must reproduce its selection exactly before the
# recipe's reading counts (EXAMPLE_PUBLISHED_SEED0). That run is read, never drawn.
EXAMPLE_SUITE = "musique"
# The published no-question and prompted runs plus every training seed of the trained questioner,
# scored together. The store holds three trained models, so a trained run is keyed by the inquirer
# model its calls name, as every gate does, and the prompted arm by the 8B base.
EXAMPLE_STORE = REPO / "artifacts/seedrep_gate_20260919/scores_parquet"
# The seed-0 audit's own reading of the same rule (scripts/seed0_audit/example_by_seed.py). The
# selection and the four transcripts are read twice and must agree.
EXAMPLE_BY_SEED = REPO / "artifacts/seed0_audit_20260924/example_by_seed.json"
EXAMPLE_PROMPTED_MODEL = "qwen3-8b-base"
EXAMPLE_TRAINED_MODEL = {
    "s0": "qwen3-8b-dpo-stacked-notdone-both",
    "s1": "qwen3-8b-dpo-stacked-notdone-both-s1",
    "s2": "qwen3-8b-dpo-stacked-notdone-both-s2",
}
EXAMPLE_RECIPE_SEEDS = ("s1", "s2")
# The drawn panels, in order.
EXAMPLE_ARMS = ("drafter_only", "inquirer_prompted", "trained_s1", "trained_s2")
EXAMPLE_READER_NAME = {
    "drafter_only": "No questions asked",
    "inquirer_prompted": "The same model, prompted",
    "trained_s1": "The trained questioner, training seed 1",
    "trained_s2": "The trained questioner, training seed 2",
}
# THE LOCK: the published figure's selection (paper/iclr2027 2/figures/examples_transcripts
# .provenance.json at b0df080; the tex prints "selection counts 16 forward, 7 reverse, 175 paired
# tasks" in its prov line).
EXAMPLE_PUBLISHED_SEED0 = {
    "task_id": "3hop1__738245_144587_56749",
    "run_ids": {
        "drafter_only": "8dac459d4191b93dac2ddfc81055eb84",
        "inquirer_prompted": "6e0d6f8f1b47c53f8cf6de8ce7acac80",
        "trained": "8cb53e1942a514274cd30d2a9246f006",
    },
    "n_paired_tasks": 175,
    "n_forward": 16,
    "n_reverse": 7,
}
# The recipe's counts as the caption prints them (artifacts/seed0_audit_20260924/
# c5_example_figure.tex, option (a)): both seeds jointly, and each seed alone.
EXAMPLE_RECIPE_PRINTED = {
    "joint": {"n_paired_tasks": 175, "n_forward": 16, "n_reverse": 3},
    "s1": {"n_forward": 18, "n_reverse": 4},
    "s2": {"n_forward": 17, "n_reverse": 5},
}
# How the float's `% prov:` line names each panel's run.
EXAMPLE_PROV_TAG = {
    "drafter_only": "no-question",
    "inquirer_prompted": "prompted",
    "trained_s1": "recipe seed 1",
    "trained_s2": "recipe seed 2",
}
# The audit record names its transcripts by these keys.
EXAMPLE_AUDIT_KEY = {
    "drafter_only": "no_question",
    "inquirer_prompted": "prompted",
    "trained_s1": "s1",
    "trained_s2": "s2",
}


# The gold node labels are the benchmark's own decomposition forms ("Gracie >> director",
# "What is the birthplace of #1 ?"), precise but unreadable in a figure. The chain is therefore
# printed as a gloss, and the raw form each gloss stands for is ASSERTED before writing, so the
# figure cannot drift from the graph it claims to show. The raw forms are in the provenance.
EXAMPLE_CHAIN_GLOSS = {
    "s1": ("Gracie >> director", r"who directed the film \emph{Gracie}"),
    "s2": ("What is the birthplace of #1 ?", r"that director's birthplace"),
    "s3": (
        "when does meet me in #2 take place",
        r"when the film set in that city takes place",
    ),
}


def _gold_depths_and_chain(
    task_id: str,
) -> tuple[dict[str, int], list[dict[str, Any]], list[tuple[str, str]], str]:
    """Per-node depth, the node records, the prerequisite edges and the answer, canary stripped."""
    require(MUSIQUE_GOLD)
    for line in MUSIQUE_GOLD.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("gold_task_key") != task_id:
            continue
        nodes = sorted(d["gold_nodes"], key=lambda n: (n["gold_depth"], n["gold_node_id"]))
        depths = {n["gold_node_id"]: n["gold_depth"] for n in nodes}
        edges = [
            (e["gold_src_node_id"], e["gold_dst_node_id"])
            for e in d["gold_edges"]
            if e["gold_edge_kind"] == "prerequisite"
        ]
        return depths, nodes, edges, CANARY.sub("", d["gold_answer"])
    raise Mismatch(f"{rel(MUSIQUE_GOLD)}: no gold graph for {task_id}. NOT WRITING.")


def _all_musique_depths() -> dict[tuple[str, str], int]:
    require(MUSIQUE_GOLD)
    out: dict[tuple[str, str], int] = {}
    for line in MUSIQUE_GOLD.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        for n in d["gold_nodes"]:
            out[(n["gold_task_key"], n["gold_node_id"])] = n["gold_depth"]
    return out


def _example_runs(con: Any) -> dict[str, dict[str, str]]:
    """task -> panel key -> run id, over held-out MuSiQue at rollout seed 0, ok runs only. The keys
    are `drafter_only`, `inquirer_prompted` (the 8B base) and one per training seed."""
    by_model = {m: s for s, m in EXAMPLE_TRAINED_MODEL.items()}
    rows = con.execute(
        f"""with m as (select distinct run_id, model from '{EXAMPLE_STORE}/calls.parquet'
                       where actor = 'inquirer')
            select r.arm_id, r.task_id, r.run_id, m.model
            from '{EXAMPLE_STORE}/runs.parquet' r left join m using (run_id)
            where r.suite_id = ? and r.split = 'test' and r.seed = 0 and r.status = 'ok'
              and r.arm_id in ('drafter_only', 'inquirer_prompted', 'inquirer_trained')""",
        [EXAMPLE_SUITE],
    ).fetchall()
    have: dict[str, dict[str, str]] = {}
    for arm, task, run, model in rows:
        if arm == "drafter_only":
            key = arm
        elif arm == "inquirer_prompted":
            if model != EXAMPLE_PROMPTED_MODEL:
                continue
            key = arm
        else:
            key = by_model.get(model)
            if key is None:
                continue
        if key in have.get(task, {}):
            raise Mismatch(
                f"{task}: two {key} runs at rollout seed 0, and the rule names one. NOT WRITING."
            )
        have.setdefault(task, {})[key] = run
    return have


def _apply_example_rule(
    have: Mapping[str, Mapping[str, str]],
    resolved: Mapping[str, set[str]],
    depths: Mapping[tuple[str, str], int],
    trained: Sequence[str],
) -> dict[str, list[str]]:
    """The rule, with EVERY key in `trained` required to resolve a deep need the prompted arm does
    not (forward), or the prompted arm required to resolve one that every key does not (reverse)."""
    need = ("drafter_only", "inquirer_prompted", *trained)
    paired = sorted(t for t, d in have.items() if set(need) <= set(d))

    def deep(task: str, key: str) -> set[str]:
        return {n for n in resolved.get(have[task][key], set()) if depths.get((task, n), 0) >= 2}

    forward = [t for t in paired if all(deep(t, k) - deep(t, "inquirer_prompted") for k in trained)]
    reverse = [t for t in paired if all(deep(t, "inquirer_prompted") - deep(t, k) for k in trained)]
    return {"paired": paired, "forward": forward, "reverse": reverse}


def _audit_disagreements(
    audit: Mapping[str, Any],
    seed0: Mapping[str, Any],
    joint: Mapping[str, list[str]],
    per_seed: Mapping[str, Mapping[str, list[str]]],
) -> list[str]:
    bad = []
    if not all((audit.get("lock") or {"missing": False}).values()):
        bad.append(f"its own seed-0 lock is not clean: {audit.get('lock')}")
    a0 = audit["by_seed"]["s0"]
    for k in ("n_paired", "n_forward", "n_reverse"):
        mine = seed0["n_paired_tasks" if k == "n_paired" else k]
        if a0[k] != mine:
            bad.append(f"seed-0 {k} {a0[k]} against {mine}")
    if a0["run_ids"] != {
        "drafter_only": seed0["run_ids"]["drafter_only"],
        "inquirer_prompted": seed0["run_ids"]["inquirer_prompted"],
        "inquirer_trained": seed0["run_ids"]["trained"],
    }:
        bad.append(f"seed-0 run ids {a0['run_ids']}")
    for s, r in per_seed.items():
        a = audit["by_seed"][s]
        for k, mine in (
            ("n_paired", len(r["paired"])),
            ("n_forward", len(r["forward"])),
            ("n_reverse", len(r["reverse"])),
            ("forward", r["forward"]),
            ("reverse", r["reverse"]),
        ):
            if a[k] != mine:
                bad.append(f"{s} {k}: {a[k]} against {mine}")
    if audit["both_seeds_forward"] != joint["forward"]:
        bad.append("the joint forward list")
    if audit["both_seeds_reverse"] != joint["reverse"]:
        bad.append("the joint reverse list")
    return bad


def select_example_task() -> dict[str, Any]:
    """Apply the selection rule and return the chosen task with the counts that calibrate it.

    The rule keyed to the set-aside run's model is applied FIRST and must reproduce the published
    figure's selection exactly; then the recipe's joint reading, and each training seed alone, are
    taken from the same store and asserted equal to the audit's independent reading and to the
    counts the caption prints.
    """
    require(
        EXAMPLE_STORE / "runs.parquet",
        EXAMPLE_STORE / "calls.parquet",
        EXAMPLE_STORE / "matches.parquet",
        EXAMPLE_BY_SEED,
    )
    con = duckdb.connect()
    have = _example_runs(con)
    resolved: dict[str, set[str]] = {}
    for run, node in con.execute(
        f"select run_id, node_id from '{EXAMPLE_STORE}/matches.parquet' where match_kind = 'resolve'"
    ).fetchall():
        resolved.setdefault(run, set()).add(node)
    depths = _all_musique_depths()

    s0 = _apply_example_rule(have, resolved, depths, ("s0",))
    if not s0["forward"]:
        raise Mismatch("the seed-0 selection finds no task. NOT WRITING.")
    t0 = s0["forward"][0]
    seed0 = {
        "task_id": t0,
        "run_ids": {
            "drafter_only": have[t0]["drafter_only"],
            "inquirer_prompted": have[t0]["inquirer_prompted"],
            "trained": have[t0]["s0"],
        },
        "n_paired_tasks": len(s0["paired"]),
        "n_forward": len(s0["forward"]),
        "n_reverse": len(s0["reverse"]),
    }
    if seed0 != EXAMPLE_PUBLISHED_SEED0:
        raise Mismatch(
            f"the rule no longer reproduces the published seed-0 selection: {seed0} against "
            f"{EXAMPLE_PUBLISHED_SEED0}. NOT WRITING."
        )

    joint = _apply_example_rule(have, resolved, depths, EXAMPLE_RECIPE_SEEDS)
    per_seed = {s: _apply_example_rule(have, resolved, depths, (s,)) for s in EXAMPLE_RECIPE_SEEDS}
    if not joint["forward"]:
        raise Mismatch("no task satisfies the selection rule for the recipe. NOT WRITING.")
    task_id = joint["forward"][0]
    counts = {
        "joint": {
            "n_paired_tasks": len(joint["paired"]),
            "n_forward": len(joint["forward"]),
            "n_reverse": len(joint["reverse"]),
        },
        **{
            s: {"n_forward": len(r["forward"]), "n_reverse": len(r["reverse"])}
            for s, r in per_seed.items()
        },
    }
    if counts != EXAMPLE_RECIPE_PRINTED:
        raise Mismatch(
            f"the recipe's selection counts {counts} are not the printed {EXAMPLE_RECIPE_PRINTED}. "
            "NOT WRITING."
        )
    audit = json.loads(EXAMPLE_BY_SEED.read_text())
    bad = _audit_disagreements(audit, seed0, joint, per_seed)
    if audit.get("recipe_example", {}).get("task_id") != task_id:
        bad.append(f"the recipe's task {audit.get('recipe_example', {}).get('task_id')}")
    if bad:
        raise Mismatch(
            f"the audit record {rel(EXAMPLE_BY_SEED)} disagrees with this reading: {bad}. NOT WRITING."
        )
    # The rule is only meaningful if the comparator RAN: a task where the prompted arm is absent
    # would satisfy "the prompted arm resolved no deep need" by absence rather than by failure.
    keys = {"drafter_only": "drafter_only", "inquirer_prompted": "inquirer_prompted"}
    keys.update({f"trained_{s}": s for s in EXAMPLE_RECIPE_SEEDS})
    if set(keys.values()) - set(have[task_id]):
        raise Mismatch(f"{task_id}: not every drawn policy has an ok run. NOT WRITING.")
    paired = joint["paired"]
    return {
        "task_id": task_id,
        "run_ids": {panel: have[task_id][key] for panel, key in keys.items()},
        "n_paired_tasks": len(paired),
        "n_forward": len(joint["forward"]),
        "n_reverse": len(joint["reverse"]),
        "per_seed": {
            s: {
                "n_paired_tasks": len(r["paired"]),
                "n_forward": len(r["forward"]),
                "n_reverse": len(r["reverse"]),
                "first_task": r["forward"][0] if r["forward"] else None,
            }
            for s, r in per_seed.items()
        },
        "seed0_lock": seed0,
        "n_tasks_with_a_deep_need": len(
            {t for t in paired if any(depths.get((t, n), 0) >= 2 for (tt, n) in depths if tt == t)}
        ),
    }


def _task_question(task_id: str) -> str:
    require(MUSIQUE_TASKS)
    for line in MUSIQUE_TASKS.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("id") == task_id:
            return CANARY.sub("", d["question"]).strip()
    raise Mismatch(f"{rel(MUSIQUE_TASKS)}: no question for {task_id}. NOT WRITING.")


# Typographic characters a recorded model string carries, and the TeX that prints them. The paper
# sets T1 Times, which drops U+201C, U+201D and U+2019 WITHOUT an error (request 1a from the
# main-text owner), so every such character is written as TeX's own ligature instead.
TEX_TYPOGRAPHY = (
    ("“", "``"),
    ("”", "''"),
    ("‘", "`"),
    ("’", "'"),
    ("—", "---"),
    ("–", "--"),
    ("…", r"\ldots{}"),
    (" ", "~"),
)


def _tex_escape(text: str) -> str:
    out = text
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        out = out.replace(a, b)
    for a, b in TEX_TYPOGRAPHY:
        out = out.replace(a, b)
    # A straight double-quoted span inside a recorded question becomes a matched SINGLE pair,
    # because the template already wraps the whole question in double quotes, so a title reads
    # `Gracie' inside ``...'' rather than doubling the quote level.
    parts = out.split('"')
    if len(parts) % 2 == 1 and len(parts) > 1:
        rebuilt = parts[0]
        for i in range(1, len(parts), 2):
            rebuilt += "`" + parts[i] + "'"
            if i + 1 < len(parts):
                rebuilt += parts[i + 1]
        return rebuilt
    return out.replace('"', "''")


def _example_transcript(con: Any, run: str, depths: Mapping[str, int]) -> dict[str, Any]:
    n_asks, calls, stop, cap = con.execute(
        f"""select n_asks, retrieval_calls, stop_reason, budget_cap
            from '{EXAMPLE_STORE}/runs.parquet' where run_id = ?""",
        [run],
    ).fetchone()
    resolved_at: dict[int, list[str]] = {}
    for node, turn in con.execute(
        f"""select node_id, matched_turn_idx from '{EXAMPLE_STORE}/matches.parquet'
            where run_id = ? and match_kind = 'resolve'""",
        [run],
    ).fetchall():
        resolved_at.setdefault(int(turn), []).append(node)
    turns = []
    for turn, kind, question in con.execute(
        f"""select turn_idx, action_kind, question from '{EXAMPLE_STORE}/turns.parquet'
            where run_id = ? order by turn_idx""",
        [run],
    ).fetchall():
        if not question:
            continue
        got = sorted(resolved_at.get(int(turn), []))
        turns.append(
            {
                "turn_idx": int(turn),
                "action_kind": kind,
                "question": CANARY.sub("", question).strip(),
                "resolved": got,
                "resolved_depths": [depths.get(n) for n in got],
            }
        )
    outcome = json.loads((REPO / "runs" / run / "outcome.json").read_text())
    row: dict[str, Any] = {
        "run_id": run,
        "n_asks": int(n_asks),
        "retrieval_calls": float(calls),
        "stop_reason": stop,
        "budget_cap": None if cap is None else int(cap),
        "turns": turns,
        "answer_text": CANARY.sub("", outcome["answer"]["text"]).strip(),
        "n_resolved": len({n for t in turns for n in t["resolved"]}),
    }
    for metric in ("task_success", "answer_correct", "answer_token_f1"):
        got_m = con.execute(
            f"select value from '{EXAMPLE_STORE}/scores.parquet' where run_id = ? and metric_name = ?",
            [run, metric],
        ).fetchone()
        row[metric] = None if got_m is None else float(got_m[0])
    return row


def _transcript_disagreements(
    panel: str, mine: Mapping[str, Any], theirs: Mapping[str, Any]
) -> list[str]:
    bad = [
        f"{panel} {k}: {theirs.get(k)!r} against {mine[k]!r}"
        for k in (
            "run_id",
            "n_asks",
            "stop_reason",
            "answer_text",
            "task_success",
            "answer_correct",
        )
        if theirs.get(k) != mine[k]
    ]
    a = [(t["turn_idx"], t["question"], t["resolved"]) for t in theirs.get("turns", [])]
    b = [(t["turn_idx"], t["question"], t["resolved"]) for t in mine["turns"]]
    if a != b:
        bad.append(f"{panel}: the questions or the needs they resolved")
    return bad


def collect_example() -> dict[str, Any]:
    sel = select_example_task()
    task_id = sel["task_id"]
    con = duckdb.connect()
    depths, nodes, edges, answer = _gold_depths_and_chain(task_id)
    arms = {panel: _example_transcript(con, run, depths) for panel, run in sel["run_ids"].items()}
    audit = json.loads(EXAMPLE_BY_SEED.read_text())["recipe_example"]["transcripts"]
    bad = [
        msg
        for panel, row in arms.items()
        for msg in _transcript_disagreements(panel, row, audit.get(EXAMPLE_AUDIT_KEY[panel], {}))
    ]
    if bad:
        raise Mismatch(
            f"the audit record {rel(EXAMPLE_BY_SEED)} disagrees with this reading: {bad}. NOT WRITING."
        )
    scorer = {
        r[0]
        for r in con.execute(
            f"""select distinct scorer_hash from '{EXAMPLE_STORE}/scores.parquet'
                where run_id in (select unnest(?))""",
            [list(sel["run_ids"].values())],
        ).fetchall()
        if r[0]
    }
    return {
        "selection": sel,
        "task_id": task_id,
        "question": _task_question(task_id),
        "gold_answer": answer,
        "gold_nodes": [
            {
                "node_id": n["gold_node_id"],
                "depth": n["gold_depth"],
                "text": CANARY.sub("", n["gold_text"]).strip(),
                "aliases": [CANARY.sub("", a).strip() for a in n.get("gold_aliases") or []],
            }
            for n in nodes
        ],
        "gold_edges": edges,
        "arms": arms,
        "scorer_hash": sorted(scorer),
        "graph_version": ["v1"],
    }


# The chain drawn in part (a): the entity that resolving each need reveals, which is what the next
# question has to name. Each label is asserted equal to the gold answer of the need before it.
EXAMPLE_CHAIN_BOXES = (
    ("s1", 0, r"who directed \emph{Gracie}"),
    ("s2", 1, "that director's\\\\birthplace"),
    ("s3", 2, r"when the film set\\there takes place"),
)
EXAMPLE_CHAIN_REVEALS = ("Davis Guggenheim", r"St.\,Louis")


def _collapse_repeats(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Consecutive identical questions become one row carrying the count. The strings must be
    identical, so the collapse cannot hide a difference between two questions."""
    out: list[dict[str, Any]] = []
    for t in turns:
        if (
            out
            and out[-1]["question"] == t["question"]
            and not t["resolved"]
            and not out[-1]["resolved"]
        ):
            out[-1]["repeats"] += 1
            out[-1]["last_index"] = t["turn_idx"] + 1
            continue
        row = dict(t)
        row["repeats"] = 1
        row["first_index"] = t["turn_idx"] + 1
        row["last_index"] = t["turn_idx"] + 1
        out.append(row)
    return out


# A question that resolves nothing and opens with the same words as the question drawn just above
# it is a rewording of that question. It is counted on one note row under it rather than drawn
# again, and its full text stays in the provenance. The opening must be long enough that a shared
# stem is a rewording and not a common question form ("What is the birthplace of ...").
EXAMPLE_REWORDING_WORDS = 10
COUNT_WORD = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def _opening(text: str) -> tuple[str, ...] | None:
    words = text.split()
    return tuple(words[:EXAMPLE_REWORDING_WORDS]) if len(words) > EXAMPLE_REWORDING_WORDS else None


def _fold_rewordings(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    anchor: Mapping[str, Any] | None = None
    for r in rows:
        opening = _opening(r["question"])
        if (
            anchor is not None
            and not r["resolved"]
            and r["repeats"] == 1
            and opening is not None
            and opening == _opening(anchor["question"])
        ):
            if not out[-1].get("note"):
                out.append({"note": True, "resolved": [], "folded": [], "turns": []})
            out[-1]["folded"].append(r["question"])
            out[-1]["turns"].append(r["turn_idx"])
            continue
        out.append(dict(r))
        anchor = r
    return out


def _clip(text: str, n: int) -> str:
    """One line of whitespace-normalized text, ellipsized if it would not fit the panel."""
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _example_prov_line(got: Mapping[str, Any]) -> str:
    """The one-line `% prov:` the float carries, generated so it is byte-identical to this record."""
    sel, arms = got["selection"], got["arms"]
    runs = ", ".join(f"{EXAMPLE_PROV_TAG[p]} {arms[p]['run_id'][:16]}" for p in EXAMPLE_ARMS)
    return (
        f"% prov: one task, {len(EXAMPLE_ARMS)} run ids, every question and resolved need as the "
        f"store records them: {runs}; selection counts for the recipe, both seeds jointly, "
        f"{sel['n_forward']} forward, {sel['n_reverse']} reverse, {sel['n_paired_tasks']} paired "
        f"tasks | paper/iclr2027 2/figures/examples_transcripts.provenance.json "
        f"({rel(EXAMPLE_STORE)}, {rel(MUSIQUE_GOLD)}, {rel(MUSIQUE_TASKS)}) | "
        f"scorer_hash={','.join(h[:16] for h in got['scorer_hash'])} | "
        f"graph_version={','.join(got['graph_version'])} | n=1 task, {len(EXAMPLE_ARMS)} runs | "
        "generated by scripts/paper_figures_iclr.py --only examples; the rule's seed-0 reading "
        f"({sel['seed0_lock']['n_paired_tasks']}/{sel['seed0_lock']['n_forward']}/"
        f"{sel['seed0_lock']['n_reverse']}, the published figure) reproduced first and not drawn; "
        "gold strings canary-stripped and the body asserted canary-free"
    )


def examples_transcripts(out: Path) -> list[Path]:
    """One task, three policies, drawn: the qualitative counterpart of Table~\\ref{tab:heldout}.

    Part (a) is laid out at fixed coordinates. The panels of part (b) are CHAINED: every row is
    placed below the one before it and every panel below the last, so TeX, not an estimate of how
    many characters fit a line, decides where a wrapped question ends.
    """
    got = collect_example()
    sel = got["selection"]
    gold = {n["node_id"]: n for n in got["gold_nodes"]}
    for n in got["gold_nodes"]:
        raw, _ = EXAMPLE_CHAIN_GLOSS.get(n["node_id"], ("", ""))
        if n["text"] != raw:
            raise Mismatch(
                f"{n['node_id']}: the gold label is {n['text']!r} but the figure glosses {raw!r}. "
                "The gloss no longer stands for the graph. NOT WRITING."
            )
    # Each arrow label names what the need at its tail reveals, which is that need's gold answer.
    for (nid, _d, _g), reveal in zip(EXAMPLE_CHAIN_BOXES, EXAMPLE_CHAIN_REVEALS):
        plain = reveal.replace(r"\,", " ")
        if plain not in gold[nid]["aliases"]:
            raise Mismatch(
                f"the chain labels {plain!r} as what {nid} reveals, but {nid}'s gold answer is "
                f"{gold[nid]['aliases']}. NOT WRITING."
            )

    W = 13.6
    x_text = 0.75
    x_badge = 0.36
    text_w = W - x_text - 0.4
    L: list[str] = [
        "% examples_transcripts.tex -- generated by scripts/paper_figures_iclr.py, do not hand-edit.",
        "% prov: see examples_transcripts.provenance.json beside this file (run ids, scorer hash,",
        "% graph version, the selection rule and the counts that calibrate it).",
        "% Every gold string is canary-stripped and the body is asserted canary-free.",
        "% Needs the TikZ libraries arrows.meta and fit, which the paper's preamble loads.",
        r"\definecolor{pxhit}{HTML}{0B7A5D}",
        r"\definecolor{pxmiss}{HTML}{9A9A9A}",
        r"\definecolor{pxrule}{HTML}{C8C8C8}",
        r"\definecolor{pxband}{HTML}{F2F4F5}",
        r"\begin{tikzpicture}[",
        r"  font=\small,",
        r"  hit/.style={circle, fill=pxhit, text=white, inner sep=0pt, minimum size=3.6mm,"
        r" font=\scriptsize\bfseries},",
        r"  miss/.style={circle, draw=pxmiss, text=pxmiss, inner sep=0pt, minimum size=3.6mm,"
        r" font=\scriptsize},",
        r"  need/.style={draw=pxrule, fill=pxband, rounded corners=2pt, align=center,"
        r" inner sep=3pt, font=\scriptsize, text width=2.7cm},",
        r"  panel/.style={draw=pxrule, rounded corners=2pt, line width=0.4pt},",
        r"  lbl/.style={font=\scriptsize, inner sep=1pt},",
        r"  arr/.style={-{Latex[length=1.5mm]}, draw=black!55},",
        rf"  row/.style={{anchor=north west, text width={text_w:.2f}cm, align=left, inner ysep=1pt}},",
        r"]",
        r"\coordinate (pxorigin) at (0,0);",
        rf"\coordinate (pxright) at ({W:.2f},0);",
        rf"\coordinate (pxrulend) at ({W - 0.22:.2f},0);",
    ]
    y = 0.0

    # ---- (a) the chain ---------------------------------------------------------------
    L.append(
        rf"\node[anchor=north west, font=\small\bfseries] at (0,{y:.2f}) "
        r"{(a) What the task requires};"
    )
    y -= 0.50
    L.append(
        rf"\node[anchor=north west, text width={W - 0.2:.2f}cm, align=left, font=\scriptsize] "
        rf"at (0,{y:.2f}) {{``{_tex_escape(got['question'])}'' The film is named only by the "
        r"start of its title, and where it is set cannot be known until the first two needs are "
        r"resolved.};"
    )
    y -= 0.86
    box_w, box_h, gap = 2.90, 0.92, 2.45
    for i, (nid, depth, gloss) in enumerate(EXAMPLE_CHAIN_BOXES):
        x = i * (box_w + gap)
        L.append(
            rf"\node[need, anchor=north west, text width={box_w - 0.3:.2f}cm, "
            rf"minimum height={box_h:.2f}cm] (n{i}) at ({x:.2f},{y:.2f}) {{{gloss}}};"
        )
        L.append(rf"\node[hit, anchor=center] at ({x:.2f},{y:.2f}) {{{depth}}};")
        if i:
            ay = y - box_h / 2
            L.append(rf"\draw[arr] ({x - gap:.2f},{ay:.2f}) -- ({x - 0.06:.2f},{ay:.2f});")
            # A long entity name is set on two lines: the label sits in the narrow gap between two
            # boxes, and one line of "Davis Guggenheim" ran over the next box.
            reveal = EXAMPLE_CHAIN_REVEALS[i - 1]
            if len(reveal) > 10 and " " in reveal:
                reveal = r"\\".join(reveal.rsplit(" ", 1))
            L.append(
                rf"\node[lbl, anchor=south, align=center, text=black!65, fill=white, inner sep=1.5pt] "
                rf"at ({x - gap / 2:.2f},{ay + 0.04:.2f}) {{{reveal}}};"
            )
    L.append(
        rf"\node[anchor=north west, font=\scriptsize, text=black!60] at "
        rf"(0,{y - box_h - 0.04:.2f}) {{badge: the depth of the need}};"
    )
    y -= box_h + 0.58

    # ---- (b) the policies ------------------------------------------------------------
    L.append(
        rf"\node[anchor=north west, font=\small\bfseries] at (0,{y:.2f}) "
        r"{(b) What each policy asked, and what it answered};"
    )
    y -= 0.62

    n_needed = len(got["gold_nodes"])
    drawn: dict[str, Any] = {}
    for i, arm in enumerate(EXAMPLE_ARMS):
        a = got["arms"][arm]
        rows = _fold_rewordings(_collapse_repeats(a["turns"]))
        accounted = sum(len(r["folded"]) if r.get("note") else r["repeats"] for r in rows)
        if accounted != a["n_asks"]:
            raise Mismatch(
                f"{arm}: {accounted} questions drawn, repeated or folded, but the store records "
                f"{a['n_asks']}. A question is missing from the drawing. NOT WRITING."
            )
        a["n_questions_accounted"] = accounted
        drawn[arm] = [
            {"folded_rewordings": r["folded"], "turn_idx": r["turns"]}
            if r.get("note")
            else {"question": r["question"], "repeats": r["repeats"], "resolved": r["resolved"]}
            for r in rows
        ]
        p = f"px{i}"
        if i == 0:
            L.append(rf"\coordinate ({p}l) at (0,{y:.2f});")
        else:
            L.append(rf"\coordinate ({p}t) at (px{i - 1}b -| pxorigin);")
            L.append(rf"\coordinate ({p}l) at ([yshift=-0.26cm]{p}t);")
        L.append(rf"\coordinate ({p}r) at ({p}l -| pxright);")
        L.append(
            rf"\node[anchor=north west, font=\small\bfseries] at ([xshift=0.22cm,yshift=-0.10cm]{p}l) "
            rf"{{{EXAMPLE_READER_NAME[arm]}}};"
        )
        badge = (
            r"\textcolor{pxhit}{every need resolved}"
            if a["task_success"]
            else r"\textcolor{pxmiss}{needs unresolved}"
        )
        L.append(
            rf"\node[anchor=north east, font=\scriptsize] at ([xshift=-0.22cm,yshift=-0.12cm]{p}r) "
            rf"{{{a['n_resolved']} of {n_needed} needs \quad {badge}}};"
        )
        items: list[tuple[str | None, str, str]] = []  # (badge style, badge label, row text)
        if not rows:
            items.append(("miss", "--", r"\textcolor{pxmiss}{\itshape asks nothing}"))
        for r in rows:
            if r.get("note"):
                k = len(r["folded"])
                what = "rewording" if k == 1 else "rewordings"
                items.append(
                    (
                        "miss",
                        "--",
                        rf"\textcolor{{pxmiss}}{{\itshape {COUNT_WORD.get(k, str(k))} further {what} "
                        r"of the question above, none resolving a need}",
                    )
                )
                continue
            times = (
                rf" \textcolor{{pxmiss}}{{\scriptsize (asked {r['repeats']} times)}}"
                if r["repeats"] > 1
                else ""
            )
            text = rf"``{_tex_escape(r['question'])}''{times}"
            if r["resolved"]:
                items.append(("hit", str(r["resolved_depths"][0]), text))
            else:
                items.append(("miss", "--", text))
        if a["stop_reason"] in ("budget", "max_turns"):
            cap = a["budget_cap"]
            what = (
                "the turn limit"
                if a["stop_reason"] == "max_turns"
                else "the budget" + ("" if cap is None else f" of {COUNT_WORD.get(cap, str(cap))}")
            )
            items.append(
                (None, "", rf"\textcolor{{pxmiss}}{{\scriptsize\itshape stopped by {what}}}")
            )
        for j, (style, label, text) in enumerate(items):
            name = f"{p}q{j}"
            at = (
                rf"([xshift={x_text:.2f}cm,yshift=-0.58cm]{p}l)"
                if j == 0
                else rf"({p}q{j - 1}.south west)"
            )
            L.append(rf"\node[row] ({name}) at {at} {{\strut {text}\strut}};")
            if style:
                L.append(
                    rf"\node[{style}, anchor=center] at "
                    rf"([xshift=-{x_text - x_badge:.2f}cm,yshift=0.10cm]{name}.base west) {{{label}}};"
                )
        last = f"{p}q{len(items) - 1}"
        ans = _clip(a["answer_text"], 150)
        L.append(rf"\coordinate ({p}s) at ([yshift=-0.04cm]{last}.south west);")
        L.append(rf"\draw[draw=pxrule] ({p}s) -- ({p}s -| pxrulend);")
        L.append(
            rf"\node[anchor=north west, text width={text_w:.2f}cm, align=left, font=\scriptsize] "
            rf"({p}a) at ([yshift=-0.04cm]{p}s) {{\textit{{Answer.}} {_tex_escape(ans)}}};"
        )
        L.append(rf"\coordinate ({p}bb) at ({p}a.south -| pxorigin);")
        L.append(rf"\coordinate ({p}b) at ([yshift=-0.06cm]{p}bb);")
        L.append(rf"\node[panel, fit=({p}l)({p}r)({p}b), inner sep=0pt] {{}};")
    L.append(r"\end{tikzpicture}")
    body = "\n".join(L) + "\n"
    if CANARY.search(body):
        raise Mismatch("a canary nonce reached the generated asset. NOT WRITING.")
    curly = sorted({c for c in body if c in "‘’“”"})
    if curly:
        raise Mismatch(
            f"curly quotes {curly} reached the asset, which T1 drops silently. NOT WRITING."
        )
    verdicts = ", ".join(
        f"{EXAMPLE_READER_NAME[p].lower()}: task_success {got['arms'][p]['task_success']:.0f}, "
        f"answer_correct {got['arms'][p]['answer_correct']:.1f}, answer_token_f1 "
        f"{got['arms'][p]['answer_token_f1']:.2f}"
        for p in EXAMPLE_ARMS
    )
    prov = provenance(
        asset_id="examples_transcripts",
        sources=[
            EXAMPLE_STORE / "runs.parquet",
            EXAMPLE_STORE / "calls.parquet",
            EXAMPLE_STORE / "matches.parquet",
            EXAMPLE_STORE / "turns.parquet",
            EXAMPLE_STORE / "scores.parquet",
            EXAMPLE_BY_SEED,
            MUSIQUE_GOLD,
            MUSIQUE_TASKS,
        ],
        query=(
            "held-out MuSiQue, rollout seed 0, ok runs on one task chosen by the selection rule "
            "below: the no-question arm, the 8B base prompted, and the trained questioner at "
            "training seeds 1 and 2 (keyed by the inquirer model on calls.parquet); questions, "
            "resolved nodes and answer scores as the store records them"
        ),
        values=got,
        scorer_hash=got["scorer_hash"],
        graph_version=got["graph_version"],
        run_ids_from={arm: rel(REPO / "runs" / r) for arm, r in sel["run_ids"].items()},
        extra={
            "selection_rule": (
                "Among held-out MuSiQue tasks where every drawn policy has an ok run at rollout "
                "seed 0, take those where EACH training seed of the trained questioner resolves a "
                "required need at depth two or more that the prompted arm does not, and select the "
                "lexicographically first task id. Fixed before any transcript was read. The same "
                "rule keyed to the set-aside run's model reproduces the published figure's "
                "selection (seed0_lock) before this reading is taken; that run is not drawn."
            ),
            "selection_counts": {
                "paired_tasks_all_drawn_policies": sel["n_paired_tasks"],
                "tasks_whose_graph_has_a_depth_two_need": sel["n_tasks_with_a_deep_need"],
                "trained_resolves_a_deep_need_the_prompted_does_not": sel["n_forward"],
                "prompted_resolves_a_deep_need_the_trained_does_not": sel["n_reverse"],
                "each_seed_alone": sel["per_seed"],
                "seed0_lock_published_figure": sel["seed0_lock"],
            },
            "audit_cross_check": (
                f"{rel(EXAMPLE_BY_SEED)} (scripts/seed0_audit/example_by_seed.py) reads the same "
                "rule independently: its seed-0 lock, every per-seed and joint task list, the task "
                "and all four transcripts are asserted equal to this reading before writing."
            ),
            "answer_verdict_is_the_scorer_s": (
                "The panel badge is the scorer's own task_success (every required need resolved), "
                "an evidence quantity. The strict answer match, answer_correct, is not what the "
                f"badge shows. Per panel: {verdicts}."
            ),
            "repeat_collapse": (
                "Consecutive questions with byte-identical text and no resolved need are drawn "
                "once with the number of times they were asked."
            ),
            "rewording_fold": (
                "A question that resolves no need and opens with the same "
                f"{EXAMPLE_REWORDING_WORDS} words as the question drawn above it is counted on "
                "one note row under that question; the folded strings are in drawn_rows. Every "
                "question is accounted for: drawn rows, repeats and folded rewordings sum to the "
                "store's n_asks for each panel."
            ),
            "drawn_rows": drawn,
            "status": (
                "ILLUSTRATION, not evidence. The aggregate reading of this axis is Table 1; the "
                "complementary selection count is printed in the caption."
            ),
            "canary_guard": (
                "Every gold string is stripped of its canary nonce and the emitted body is "
                "asserted canary-free before the file is written."
            ),
            "tex_quotes": (
                "Typographic characters in recorded strings are written as TeX ligatures "
                "(TEX_TYPOGRAPHY); the body is asserted free of curly quotes, which T1 drops."
            ),
            "prov_line": _example_prov_line(got),
        },
    )
    return _emit_tex(out, "examples_transcripts", body, prov)


# --------------------------------------------------------------- held-out method comparison

DECOMP_RESULT = REPO / "artifacts/decomposition_test_20260918/RESULT.md"

# Held-out matched-cost coverage gain for each recipe against its OWN prompted baseline, quoted
# from the record's section "The held-out counterpart of tab:train-coverage". Every string here is
# asserted present in that file before the table is written, so a re-measurement that moves a cell
# breaks the build rather than leaving the paper quietly disagreeing with the record.
TABLE4_ROWS = (
    (
        "imitation reference",
        "filtered demonstrations",
        {
            "musique": ("+0.0858", "+0.0492", "+0.1231"),
            "strategyqa": ("+0.0271", "-0.0070", "+0.0631"),
            "wiki2": ("+0.0587", "+0.0299", "+0.0938"),
        },
    ),
    (
        "stop-weighted imitation",
        "the same, stop rows down-weighted",
        {
            "musique": ("+0.0746", "+0.0410", "+0.1100"),
            "strategyqa": ("+0.0260", "-0.0106", "+0.0638"),
            "wiki2": ("+0.0594", "+0.0325", "+0.0925"),
        },
    ),
    (
        "question pairs",
        "question pairs, from the reference",
        {
            "musique": ("+0.1177", "+0.0848", "+0.1531"),
            "strategyqa": ("+0.0693", "+0.0337", "+0.1067"),
            "wiki2": ("+0.0719", "+0.0456", "+0.1025"),
        },
    ),
    (
        "question pairs, pooled",
        "question pairs, from the pooled imitation checkpoint",
        {
            "musique": ("+0.1110", "+0.0797", "+0.1448"),
            "strategyqa": ("+0.0657", "+0.0319", "+0.1008"),
            "wiki2": ("+0.0413", "+0.0125", "+0.0725"),
        },
    ),
    (
        "stop contrasts only$^{\\ddagger}$",
        "stop contrasts, from the reference",
        {
            "musique": ("+0.0177", "-0.0142", "+0.0496"),
            "strategyqa": ("-0.0623", "-0.0961", "-0.0304"),
            "wiki2": ("+0.0175", "-0.0075", "+0.0469"),
        },
    ),
    (
        "both kinds, from reference",
        "both pair kinds, from the reference",
        {
            "musique": ("+0.0800", "+0.0446", "+0.1165"),
            "strategyqa": ("+0.0127", "-0.0237", "+0.0503"),
            "wiki2": ("+0.0625", "+0.0344", "+0.0975"),
        },
    ),
    (
        "both kinds, from pairs$^{\\dagger}$",
        "both pair kinds, from the question-pair arm",
        {
            "musique": ("+0.1284", "+0.1021", "+0.1593"),
            "strategyqa": ("+0.0799", "+0.0511", "+0.1116"),
            "wiki2": ("+0.0791", "+0.0519", "+0.1113"),
        },
    ),
)
# Rows not quoted from the decomposition record: each is checked against its own JSON record at full
# precision, rounded half-up, instead of by string search. The two question-pair arms are the lane's
# P2 runs at s1/s2's identity (headline-control from the reference, pooled-control from the pooled
# imitation checkpoint); the selected row is the recipe (training seeds 1 and 2 within the task).
P2_ROWS_JSON = REPO / "artifacts/seedrep4_20260924/p2_rows.json"
HELDOUT_METHODS_JSON = REPO / "artifacts/seed0_audit_20260924/heldout_methods_by_seed.json"
TABLE4_JSON_ROWS = {
    "question pairs": (P2_ROWS_JSON, ("rows", "P2a", "cells")),
    "question pairs, pooled": (P2_ROWS_JSON, ("rows", "P2b", "cells")),
    "both kinds, from pairs$^{\\dagger}$": (HELDOUT_METHODS_JSON, ("cells", "s1s2")),
}
TABLE4_SUITES = (
    ("musique", "MuSiQue"),
    ("strategyqa", "StrategyQA"),
    ("wiki2", "2WikiMultiHopQA"),
)


def table4_heldout_methods(out: Path) -> list[Path]:
    """Which recipes still gain once the split they were not selected on is read."""
    require(DECOMP_RESULT)
    record = DECOMP_RESULT.read_text()
    tight = record.replace(" ", "")
    for label, _, cells in TABLE4_ROWS:
        if label in TABLE4_JSON_ROWS:
            path, keys = TABLE4_JSON_ROWS[label]
            require(path)
            node = json.loads(path.read_text())
            for k in keys:
                node = node[k]
            for suite, cell in cells.items():
                src = node[suite]
                want = tuple(_fmt(src[f], 4, sign=True) for f in ("delta", "ci_lo", "ci_hi"))
                if tuple(cell) != want:
                    raise Mismatch(
                        f"{label}/{suite}: printed {cell} but {rel(path)} rounds half-up to {want}. "
                        "NOT WRITING."
                    )
            continue
        for suite, cell in cells.items():
            if cell is None:
                continue
            delta, lo, hi = cell
            if delta not in record:
                raise Mismatch(
                    f"{label}/{suite}: {delta} is not in {rel(DECOMP_RESULT)}. The record has "
                    "moved and this table would disagree with it. NOT WRITING."
                )
            if f"[{lo},{hi}]" not in tight:
                raise Mismatch(
                    f"{label}/{suite}: the interval [{lo}, {hi}] is not in "
                    f"{rel(DECOMP_RESULT)}. NOT WRITING."
                )
    best = {
        key: max(float(c[key][0]) for _, _, c in TABLE4_ROWS if c[key] is not None)
        for key, _ in TABLE4_SUITES
    }
    lines = [
        "% table4_heldout_methods.tex -- generated by " + TOOL + ", do not hand-edit.",
        "% prov: see table4_heldout_methods.provenance.json beside this file. Every value is",
        "% asserted present in artifacts/decomposition_test_20260918/RESULT.md before writing.",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.2cm}" + (COLSEP + "c") * 3 + r"@{}}",
        r"\toprule",
        r"supervision & " + " & ".join(t for _, t in TABLE4_SUITES) + r" \\",
        r"\midrule",
    ]
    values: dict[str, Any] = {}
    for label, desc, cells in TABLE4_ROWS:
        head = label + r" \newline {\scriptsize\color{black!60} " + desc + "}"
        deltas, cis = [], []
        for key, _ in TABLE4_SUITES:
            cell = cells[key]
            if cell is None:
                deltas.append(r"{\scriptsize no held-out run}")
                cis.append("")
                continue
            d, lo, hi = cell
            deltas.append(rf"$\mathbf{{{d}}}$" if float(d) == best[key] else f"${d}$")
            cis.append(rf"{{\scriptsize $[{lo}, {hi}]$}}")
        lines.append(head + " & " + " & ".join(deltas) + r" \\")
        if any(cis):
            lines.append(" & " + " & ".join(cis) + r" \\[2pt]")
        values[label] = cells
    lines += [
        r"\bottomrule",
        r"\multicolumn{4}{@{}p{13.6cm}@{}}{\scriptsize $\dagger$ the selected questioner. "
        r"$\ddagger$ this recipe asks more questions than its comparator, so the rule every other "
        r"row uses is not a matched reading for it, and its cells are the symmetric reading with "
        r"both sides held to the lower question count. Every gain is one recipe against its own "
        r"prompted baseline. \textbf{Bold} marks the largest value in a column.} \\",
        r"\end{tabular}",
    ]
    prov = provenance(
        asset_id="table4_heldout_methods",
        sources=[DECOMP_RESULT, P2_ROWS_JSON, HELDOUT_METHODS_JSON],
        query=(
            "held-out matched-cost coverage gain per training recipe against its own prompted "
            "baseline, 200 tasks per suite, one contrast function, quoted from the record"
        ),
        values=values,
        scorer_hash=["e82c7458ee8efa11660445bbf434b4c1dc74253253e8f190cc36e92fc50f40c7"],
        graph_version=["v1"],
        extra={
            "rule": (
                "BCa, 10,000 resamples, seed 0, paired on (suite, task, seed) with seeds averaged "
                "into the task, and any bound within 0.01 of zero re-read at 50,000 resamples "
                "across three reseeds. The comparator is read at its own prefix of the recipe's "
                "question count, except for the stop-contrast recipe, which asks more than its "
                "comparator and is read with both sides held to the lower count."
            ),
            "absent_row": (
                "The question-pair recipe has no held-out run on any suite in this record, so its "
                "row is empty rather than zero."
            ),
            "withdrawn_cell": (
                "The stop-contrast recipe's 2WikiMultiHopQA gain of +0.0413 under the asymmetric "
                "rule does not survive the symmetric reading (+0.0175, interval containing zero), "
                "and the symmetric value is what is printed."
            ),
        },
    )
    return _emit_tex(out, "table4_heldout_methods", "\n".join(lines) + "\n", prov)


# The published figure drew the set-aside run's three-suite sweep from these records. They are still
# read and the ordering they were drawn for is re-asserted (SHOW_SEED0), but nothing of them is drawn.
CEILING_PANELS = (
    ("MuSiQue", FRONTIER_MUSIQUE, "evidence_coverage"),
    ("StrategyQA", FRONTIER_STRATEGYQA, "evidence_coverage"),
    ("FRAMES (no gold)", FRONTIER_FRAMES, "answer_correct"),
)
# What is drawn: the one-commit record (FRONTIER_RECIPE), keyed by its own arm names, in drawing
# order (the trained questioner on top). The legend follows Figure 2(a): CEILING_LEGEND_ORDER.
CEILING_ARMS = (
    ("teacher", ARM_TERM["teacher"], PALETTE[2]),
    ("base", ARM_TERM["same"], PALETTE[1]),
    ("recipe", ARM_TERM["trained"], PALETTE[0]),
)
CEILING_LEGEND_ORDER = ("recipe", "base", "teacher")
CEILING_SUITES = (("musique", "MuSiQue"), ("strategyqa", "StrategyQA"))
# THE LOCK: what section 5.2 prints from this record. Its prov line gives each endpoint to four
# decimals (musique 0.3387 at cap 4 to 0.0762 at cap 24, strategyqa 0.1850 to 0.0375, the prompted
# comparators' endpoints), and its sentence rounds them to percent ("34 ... 8 percent", "19 and 4",
# "13 to 69 percent" for the prompted questioners over both suites and every ceiling).
CEILING_PRINTED = {
    ("musique", "recipe", 4): "0.3387",
    ("musique", "recipe", 24): "0.0762",
    ("strategyqa", "recipe", 4): "0.1850",
    ("strategyqa", "recipe", 24): "0.0375",
    ("musique", "teacher", 4): "0.5725",
    ("musique", "teacher", 24): "0.3725",
    ("musique", "base", 4): "0.6825",
    ("musique", "base", 24): "0.2200",
    ("strategyqa", "teacher", 4): "0.6925",
    ("strategyqa", "teacher", 24): "0.4350",
    ("strategyqa", "base", 4): "0.6900",
    ("strategyqa", "base", 24): "0.1300",
}
CEILING_PRINTED_PERCENT = {
    ("musique", 4): 34,
    ("musique", 24): 8,
    ("strategyqa", 4): 19,
    ("strategyqa", 24): 4,
}
CEILING_PRINTED_PROMPTED_RANGE = (13, 69)
FRONTIER_RECIPE_SCORER = "3f88ac3527ed6c2fcf78b1538f6c44a3c14838aec93a583c072a770fb70e8162"


def parse_ceiling_hit(text: str, metric: str) -> dict[tuple[str, int], float]:
    """The ceiling-hit share per (arm, cap), off the same level table `parse_level_table` uses.

    Selected by header names rather than position, for the reason that function gives: these
    records carry several pipe tables and more than one of them starts with `arm`.
    """
    want = f"{metric} [bca 95%]"
    hit = None
    for tbl in md_tables(text):
        head = [c.lower() for c in tbl[0]]
        if "arm" in head and "cap" in head and "ceiling-hit" in head and want in head:
            if hit is not None:
                raise Mismatch(f"two ceiling tables for {metric!r}; the record is ambiguous")
            hit = tbl
    if hit is None:
        raise Mismatch(f"no ceiling-hit table for {metric!r} in this record")
    head = [c.lower() for c in hit[0]]
    ia, ic, ih = head.index("arm"), head.index("cap"), head.index("ceiling-hit")
    out: dict[tuple[str, int], float] = {}
    for row in hit[2:]:
        cell = row[ih]
        m = re.search(r"\(([\d.]+)%\)", cell)
        if not m:
            raise Mismatch(f"ceiling-hit cell {cell!r} carries no percentage")
        key = (row[ia].strip("`"), int(row[ic]))
        if key in out:
            raise Mismatch(f"duplicate ceiling cell {key}")
        out[key] = float(m.group(1))
    return out


def _ceiling_seed0_published() -> dict[str, dict[str, float]]:
    """The published figure's cells, re-read and its ordering re-asserted: on every suite and at
    every cap the set-aside run ends at the ceiling less often than EITHER prompted arm."""
    panels = {}
    for title, path, metric in CEILING_PANELS:
        require(path)
        cells = parse_ceiling_hit(path.read_text(), metric)
        for cap in sorted({c for (_a, c) in cells}):
            tr = cells.get(("trained", cap))
            if tr is None:
                raise Mismatch(f"{title}: no trained cell at cap {cap}. NOT WRITING.")
            for arm in ("teacher", "base8b"):
                other = cells.get((arm, cap))
                if other is not None and not tr < other:
                    raise Mismatch(
                        f"{title} cap {cap}: trained ceiling-hit {tr}% is not below {arm}'s "
                        f"{other}%, so the published panel's claim fails. NOT WRITING."
                    )
        panels[title] = {f"{arm}@{cap}": v for (arm, cap), v in sorted(cells.items())}
    return panels


def _ceiling_recipe_cells(rec: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """The drawn cells, each refused unless it is the average of the two seeds, the recipe is below
    both prompted arms, and every value the main text prints reproduces at its printed precision."""
    bad: list[str] = []
    drawn: dict[str, dict[str, dict[str, Any]]] = {}
    for suite, title in CEILING_SUITES:
        lv = rec["levels"][suite]
        drawn[suite] = {}
        for arm, _label, _c in CEILING_ARMS:
            drawn[suite][arm] = {
                str(c): {
                    k: lv[arm][str(c)][k] for k in ("n", "ceiling_hit", "x_mean_retrieval_calls")
                }
                for c in CAPS
            }
        for cap in CAPS:
            r = lv["recipe"][str(cap)]["ceiling_hit"]
            pair = (lv["s1"][str(cap)]["ceiling_hit"] + lv["s2"][str(cap)]["ceiling_hit"]) / 2
            if abs(r - pair) > 1e-9:
                bad.append(f"{title} cap {cap}: the recipe's {r} is not the seeds' average {pair}")
            for arm in ("base", "teacher"):
                other = lv[arm][str(cap)]["ceiling_hit"]
                if not r < other:
                    bad.append(
                        f"{title} cap {cap}: the recipe's share {r} is not below {arm}'s {other}"
                    )
    for (suite, arm, cap), want in CEILING_PRINTED.items():
        got = drawn[suite][arm][str(cap)]["ceiling_hit"]
        if f"{got:.{len(want.split('.')[1])}f}" != want:
            bad.append(f"{suite} {arm} cap {cap}: {got} is not the printed {want}")
    for (suite, cap), want in CEILING_PRINTED_PERCENT.items():
        got = 100 * drawn[suite]["recipe"][str(cap)]["ceiling_hit"]
        if abs(got - want) > 0.5 + 1e-9:
            bad.append(f"{suite} cap {cap}: {got:.2f} percent is not the printed {want}")
    prompted = [
        100 * drawn[s][a][str(c)]["ceiling_hit"]
        for s, _t in CEILING_SUITES
        for a in ("base", "teacher")
        for c in CAPS
    ]
    lo, hi = CEILING_PRINTED_PROMPTED_RANGE
    if abs(min(prompted) - lo) > 0.5 + 1e-9 or abs(max(prompted) - hi) > 0.5 + 1e-9:
        bad.append(
            f"the prompted range {min(prompted):.2f} to {max(prompted):.2f} percent is not the "
            f"printed {lo} to {hi}"
        )
    if bad:
        raise Mismatch("fig10: " + "; ".join(bad) + ". NOT WRITING.")
    return drawn


def fig10_ceiling(out: Path) -> list[Path]:
    """Who ends the run: the share of units stopped by the retrieval ceiling rather than by the
    policy, for the recipe and both prompted comparators at five ceilings and one commit."""
    rec = _frontier_recipe_record()
    drawn = _ceiling_recipe_cells(rec)
    seed0 = _ceiling_seed0_published()

    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(LINEWIDTH_IN - 0.12, 2.05), sharey=True)
    for ax, (suite, title) in zip(axes, CEILING_SUITES):
        for arm, label, colour in CEILING_ARMS:
            ys = [100 * drawn[suite][arm][str(c)]["ceiling_hit"] for c in CAPS]
            ax.plot(
                list(CAPS),
                ys,
                marker="o",
                ms=3.4,
                lw=1.2,
                color=colour,
                label=label,
                mfc=colour if arm == "recipe" else "white",
                mew=1.2,
            )
        ax.set_xticks(list(CAPS))
        ax.set_xlabel("ceiling (retrieval calls)")
        ax.set_title(title, loc="left", fontsize=8)
        ax.grid(axis="x", visible=False)
    # Percent, the unit the caption and the main-text sentence print (34 to 8 percent).
    axes[0].set_ylabel("stopped by the ceiling (%)")
    axes[0].set_ylim(-3, 78)
    drawn_handles, drawn_labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(drawn_labels, drawn_handles))
    label_of = {arm: label for arm, label, _c in CEILING_ARMS}
    labels = [label_of[arm] for arm in CEILING_LEGEND_ORDER]
    handles = [by_label[label] for label in labels]
    fig.tight_layout(pad=0.4, rect=(0, 0.13, 1, 1))
    # A figure-level legend below the panels, because every panel's interior carries data.
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.5,
        columnspacing=1.0,  # the paper's arm names are longer than the old ones
        bbox_to_anchor=(0.5, -0.02),
    )
    prov = provenance(
        asset_id="fig10_ceiling",
        sources=[
            FRONTIER_RECIPE,
            REPO / "scripts" / "seed_identity" / "frontier_recipe.py",
            *(path for _t, path, _m in CEILING_PANELS),
        ],
        query=(
            "levels[suite][arm][cap].ceiling_hit of the record written by "
            "scripts/seed_identity/frontier_recipe.py: the share of units whose stop_reason is "
            "budget or max_turns; the recipe's unit is (task, seed) with training seeds 1 and 2 "
            "averaged within it; drawn as percent"
        ),
        values=drawn,
        scorer_hash=[FRONTIER_RECIPE_SCORER],
        graph_version=["v1"],
        extra={
            "one_code_version": "a61c4f4be3e9",
            "one_base_url_sha": rec["manifests"]["base_url_sha"],
            "caps": list(CAPS),
            "lock": (
                "the record's own lock (all 30 level rows of the published frontier records "
                "reproduced, every cell n=400, one base URL, no dirty manifest) is re-asserted; "
                "every drawn recipe cell is the average of seeds 1 and 2; the recipe is below both "
                "prompted arms at every cap on both suites; and the values section 5.2 prints "
                "reproduce at their printed precision (CEILING_PRINTED, CEILING_PRINTED_PERCENT, "
                "CEILING_PRINTED_PROMPTED_RANGE)"
            ),
            "printed": {f"{s}/{a}/{c}": v for (s, a, c), v in CEILING_PRINTED.items()},
            "seed0_published_sweep_not_drawn": seed0,
            "caveats": (
                "A behavioral reading, not an accuracy one. A low ceiling share is not by itself a "
                "virtue: the same policy also stops on unfinished work (section 5.2 and the "
                "stopping appendix). The two prompted arms are drawn open to keep them distinct "
                "from the trained questioner."
            ),
        },
    )
    return _emit_figure(fig, out, "fig10_ceiling", prov)


TAU2_SOURCE = REPO / "paper" / "results.tex"
TAU2_EFFECT_SOURCE = REPO / "paper" / "appendix_instruments.tex"

# The transfer campaign, decomposed into the two populations inside each arm. The ARMS here are the
# questioner PROTOCOL against a self-ask decomposition comparator, not the trained checkpoint
# against prompted weights: this benchmark is carried unchanged and nothing was trained on it.
#
# (domain, arm, inside_mean, inside_n, over_mean, over_n)
TAU2_STRATA = (
    ("retail", "questioner", 3.58, 89, 7.23, 13),
    ("retail", "comparator", 4.02, 45, 11.79, 57),
    ("airline", "questioner", 4.41, 92, 14.10, 10),
    ("airline", "comparator", 5.00, 60, 8.02, 42),
)
# Retail is the one domain whose interval excludes zero, trace-clustered. Airline's covers zero.
TAU2_RETAIL_EFFECT = (-4.47, -6.28, -3.04)


def _tau2_rows() -> dict[str, Any]:
    """Locked against the two paper sources that carry these numbers.

    These live in another lane's `.tex` rather than in an artifact JSON, which is weaker
    provenance than the other figures here and is recorded as such.
    """
    require(TAU2_SOURCE, TAU2_EFFECT_SOURCE)
    body = TAU2_SOURCE.read_text()
    eff = TAU2_EFFECT_SOURCE.read_text()
    missing = []
    for domain, arm, im, i_n, om, o_n in TAU2_STRATA:
        for v in (f"${im:.2f}$", f"${om:.2f}$", f"${i_n}$", f"${o_n}$"):
            if v not in body:
                missing.append(f"{domain}/{arm}: {v}")
    point, lo, hi = TAU2_RETAIL_EFFECT
    if f"${point:.2f}$" not in eff or f"$[{lo:.2f}, {hi:.2f}]$" not in eff:
        missing.append(f"retail effect {point} [{lo}, {hi}]")
    if missing:
        raise Mismatch("paper sources do not carry " + "; ".join(missing) + ". NOT WRITING.")

    by = {(d, a): (im, i_n, om, o_n) for d, a, im, i_n, om, o_n in TAU2_STRATA}
    for domain in ("retail", "airline"):
        q, c = by[(domain, "questioner")], by[(domain, "comparator")]
        if q[1] + q[3] != c[1] + c[3]:
            raise Mismatch(f"{domain}: the two arms do not have the same run count. NOT WRITING.")
        # What the figure claims REPLICATES: the questioner runs past its cap far less often.
        if not q[3] < c[3]:
            raise Mismatch(
                f"{domain}: the questioner does not overrun less often, so the panel's claim "
                "is wrong. NOT WRITING."
            )
        # And what does NOT replicate, asserted so the figure cannot silently overclaim.
        if not (q[0] < c[0]):
            raise Mismatch(f"{domain}: inside-budget ordering is not as drawn. NOT WRITING.")
    if not by[("retail", "questioner")][2] < by[("retail", "comparator")][2]:
        raise Mismatch("retail's over-budget ordering is not as drawn")
    if not by[("airline", "questioner")][2] > by[("airline", "comparator")][2]:
        raise Mismatch(
            "airline's over-budget cell is asserted to INVERT, and it does not. That inversion is "
            "why that domain cannot decide its own question, so the figure must not lose it."
        )
    return {"by": by, "effect": TAU2_RETAIL_EFFECT}


def fig9_transfer_tail(out: Path) -> list[Path]:
    """On a user-facing benchmark the dialogues run away less often. That is the whole effect."""
    got = _tau2_rows()
    by = got["by"]
    plt = _mpl()
    fig, axes = plt.subplots(
        1, 3, figsize=(LINEWIDTH_IN, 2.05), gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]}
    )
    domains = ("retail", "airline")

    ax = axes[0]
    xs = [0, 1]
    w = 0.34
    for i, domain in enumerate(domains):
        q, c = by[(domain, "questioner")], by[(domain, "comparator")]
        total = q[1] + q[3]
        ax.bar(i - w / 2, 100 * c[3] / total, width=w, color=PALETTE[1], alpha=0.9)
        ax.bar(i + w / 2, 100 * q[3] / total, width=w, color=PALETTE[0])
        for off, val in ((-w / 2, 100 * c[3] / total), (w / 2, 100 * q[3] / total)):
            ax.annotate(
                f"{val:.0f}%",
                (i + off, val),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=GREY,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(domains)
    ax.set_ylim(0, 72)
    ax.set_ylabel("runs that overran (percent)")
    ax.set_title("(a) runaway rate", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="s", ls="none", color=PALETTE[1], label="comparator"),
            plt.Line2D([], [], marker="s", ls="none", color=PALETTE[0], label="questioner"),
        ],
        loc="upper right",
        frameon=False,
        handlelength=0.9,
        borderaxespad=0.15,
    )

    for j, domain in enumerate(domains):
        ax = axes[1 + j]
        q, c = by[(domain, "questioner")], by[(domain, "comparator")]
        for series, colour, label in ((c, PALETTE[1], "comparator"), (q, PALETTE[0], "questioner")):
            ax.plot(
                [0, 1],
                [series[0], series[2]],
                marker="o",
                ms=4.5,
                lw=1.2,
                color=colour,
                label=label,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["inside", "overran"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(0, 15.6)
        if j == 0:
            ax.set_ylabel("follow-up turns needed")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"({'bc'[j]}) {domain}: turns by stratum", loc="left", fontsize=7.5)
        ax.grid(axis="x", visible=False)

    fig.tight_layout(pad=0.4)
    prov = {
        "asset": "fig9_transfer_tail",
        "tool": TOOL,
        "inputs": [
            {"path": rel(TAU2_SOURCE), "sha256": sha256_file(TAU2_SOURCE)},
            {"path": rel(TAU2_EFFECT_SOURCE), "sha256": sha256_file(TAU2_EFFECT_SOURCE)},
        ],
        "strata": [
            {
                "domain": d,
                "arm": a,
                "inside_mean": im,
                "inside_n": i_n,
                "over_mean": om,
                "over_n": o_n,
            }
            for d, a, im, i_n, om, o_n in TAU2_STRATA
        ],
        "retail_effect_trace_clustered": {
            "point": TAU2_RETAIL_EFFECT[0],
            "ci": [TAU2_RETAIL_EFFECT[1], TAU2_RETAIL_EFFECT[2]],
            "n_clusters": 32,
        },
        "arms": (
            "the questioner PROTOCOL against a self-ask decomposition comparator. NOT the trained "
            "checkpoint against prompted weights: nothing was trained on this benchmark."
        ),
        "scorer_hash": "n/a: the benchmark's own evaluator, not this repository's scorer",
        "graph_version": "n/a: this suite carries no need graph",
        "lock": (
            "every level and population count asserted present in the paper sources, the two arms "
            "asserted to have equal run counts per domain, and three shape claims: the questioner "
            "overruns less often on BOTH domains, it is lower inside budget on both, and airline's "
            "over-budget cell INVERTS. That last assertion exists so the figure cannot quietly "
            "present airline as a replication of the magnitude."
        ),
        "caveats": (
            "EXPLORATORY population, flagged as such in the paper, and one domain of three. Only "
            "retail's interval excludes zero, at -4.47 [-6.28,-3.04] trace-clustered over 32 "
            "recorded dialogues. Airline's covers zero at every clustering and panel (c) shows "
            "why: its few runaways are LONGER for the questioner, 14.10 against 8.02, so the mean "
            "cannot separate even though the runaway RATE is 10 against 42. Banking does not join "
            "either and is not drawn. The overrun itself is an artifact of the recorded cap not "
            "being enforced during the dialogue, which is what makes it readable as a runaway. "
            "The inside-budget levels describe two populations and are NOT a second estimate of "
            "the effect. Provenance is another lane's .tex rather than an artifact JSON, which is "
            "weaker than every other figure in this script."
        ),
    }
    return _emit_figure(fig, out, "fig9_transfer_tail", prov)


CONTROLS_RECORD = REPO / "artifacts/killswitch_mechanism_20260918/RESULT.md"

# The full system minus each ablated variant, ALL on the matched basis and ALL from one record's
# pooled population, so the four bars differ in the ablation and in nothing else. The unmatched
# column of the same record is deliberately not mixed in: for the single-model row the two
# numbers +0.0143 and -0.0121 are the matched and unmatched bases of ONE pooled cell, not two
# suites, and reading them as two suites would invent a disagreement that is not there.
CONTROLS = (
    ("random questions", 0.5151, 0.4801, 0.5523, 376, "what the questions say"),
    ("no sequencing", 0.0986, 0.0698, 0.1267, 375, "asking in an order"),
    ("blind to evidence", 0.0662, 0.0440, 0.0886, 376, "conditioning on what came back"),
    ("one model, both roles", 0.0143, -0.0023, 0.0297, 376, "the two-model split"),
)
CONTROLS_MARGIN = 0.05


def _lock_controls() -> str:
    require(CONTROLS_RECORD)
    text = CONTROLS_RECORD.read_text()
    missing = []
    for label, d, lo, hi, n, _what in CONTROLS:
        # The record writes a unicode minus for negative bounds, so both spellings are accepted.
        forms = [f"+{d:.4f}", f"{d:+.4f}"]
        if not any(f in text for f in forms):
            missing.append(f"{label}: {d:+.4f}")
        for b in (lo, hi):
            cands = [f"{b:+.4f}", f"{b:+.4f}".replace("-", "−")]
            if not any(cc in text for cc in cands):
                missing.append(f"{label} bound {b:+.4f}")
        if f"({n})" not in text:
            missing.append(f"{label} n={n}")
    if missing:
        raise Mismatch(
            f"{rel(CONTROLS_RECORD)} does not contain " + "; ".join(missing) + ". NOT WRITING."
        )
    # The figure's claim, in the three states it actually draws. An earlier version asserted that
    # THREE ablations clear the margin by their lower bound; only two do. The evidence-blind
    # control's point estimate is above the margin and its interval reaches into it, so it is
    # drawn as its own state rather than lumped with either neighbour.
    decisive = [c for c in CONTROLS if c[2] > CONTROLS_MARGIN]
    excludes_zero_only = [c for c in CONTROLS if c[2] > 0 and c[2] <= CONTROLS_MARGIN]
    straddles = [c for c in CONTROLS if c[2] <= 0 <= c[3]]
    if (len(decisive), len(excludes_zero_only), len(straddles)) != (2, 1, 1):
        raise Mismatch(
            "the figure draws two decisive ablations, one that excludes zero but reaches into "
            f"the margin, and one equivalence; found {len(decisive)}, "
            f"{len(excludes_zero_only)}, {len(straddles)}"
        )
    single = CONTROLS[-1]
    if not (abs(single[1]) < CONTROLS_MARGIN and single[2] < 0 < single[3]):
        raise Mismatch("the single-model row is not the equivalence the figure draws it as")
    return text


def fig8_controls(out: Path) -> list[Path]:
    """Removing what the questions say costs an order of magnitude more than removing a model."""
    _lock_controls()
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(LINEWIDTH_IN * 0.72, 2.05))
    ys = list(range(len(CONTROLS)))[::-1]

    ax.axvspan(-CONTROLS_MARGIN, CONTROLS_MARGIN, color=GREY, alpha=0.13, lw=0, zorder=0)
    ax.axvline(0.0, color="#444444", lw=0.9, zorder=1)
    for y, (label, d, lo, hi, _n, _what) in zip(ys, CONTROLS):
        # Three states, and the middle one is not collapsed into either neighbour: a filled point
        # clears the margin by its lower bound, an open point excludes zero but reaches into the
        # margin, and grey straddles zero.
        if lo > CONTROLS_MARGIN:
            style = dict(color=PALETTE[0], mfc=PALETTE[0])
        elif lo > 0:
            style = dict(color=PALETTE[0], mfc="white")
        else:
            style = dict(color=GREY, mfc=GREY)
        ax.errorbar(
            [d],
            [y],
            xerr=[[d - lo], [hi - d]],
            marker="o",
            ms=5.0,
            mew=1.4,
            lw=1.2,
            capsize=2.2,
            zorder=3,
            **style,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in CONTROLS])
    ax.set_ylim(-0.9, len(CONTROLS) - 0.4)
    ax.set_xlim(-0.06, 0.60)
    ax.set_xlabel("coverage the full protocol recovers and the variant does not")
    ax.grid(axis="y", visible=False)
    ax.annotate(
        f"equivalence margin, declared in advance ($\\pm{CONTROLS_MARGIN:g}$)",
        (CONTROLS_MARGIN, -0.72),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=7,
        color="#6E6E6E",
    )

    fig.tight_layout(pad=0.4)
    prov = {
        "asset": "fig8_controls",
        "tool": TOOL,
        "inputs": [{"path": rel(CONTROLS_RECORD), "sha256": sha256_file(CONTROLS_RECORD)}],
        "rows": [
            {
                "variant": label,
                "removes": what,
                "delta": d,
                "ci": [lo, hi],
                "n_tasks": n,
                "clears_margin_by_lower_bound": lo > CONTROLS_MARGIN,
                "excludes_zero": lo > 0,
            }
            for label, d, lo, hi, n, what in CONTROLS
        ],
        "margin": CONTROLS_MARGIN,
        "basis": "matched cost (equal retrieval spend), pooled over two suites",
        "scorer_hash": "3f88ac3527ed6c2f",
        "graph_version": "v1",
        "lock": (
            "every point, bound and task count asserted present in the committed record, plus "
            "the three states it draws: exactly two variants clear the declared margin by their "
            "LOWER BOUND, exactly one excludes zero while reaching into the margin, and exactly "
            "one straddles zero."
        ),
        "caveats": (
            "ONE basis and ONE pooled population for all four rows. The record's unmatched "
            "column is not mixed in: for the single-model row its two numbers, +0.0143 and "
            "-0.0121, are the matched and unmatched bases of one pooled cell rather than two "
            "suites. The margin was declared in advance in code with a written consequence, "
            "which is why the fourth row is reported as an equivalence rather than as a null. "
            "The evidence-blind row excludes zero at +0.0662 but its lower bound of +0.0440 "
            "reaches into the margin, so it is drawn open rather than filled and is not "
            "described as clearing the margin. At the fixed budget that same contrast falls "
            "inside the margin at +0.0220, which is one of the four independent places where a "
            "change of cost basis changes a verdict in this paper. "
            "Two further declared controls are structurally zero on this population, with "
            "coverage exactly 0.0 on 181 and 194 tasks, and are named untestable in the text "
            "rather than drawn here as small effects."
        ),
    }
    return _emit_figure(fig, out, "fig8_controls", prov)


CONTROLS_RECIPE_DIR = REPO / "artifacts" / "seed_identity_20260923"
CONTROLS_RECIPE_SEEDS = ("s1", "s2")


def _controls_recipe_rows() -> list[dict[str, Any]]:
    """The recipe's controls, each read under the rule its spend requires, as Section 5.3 reads them.

    Random questions and the single model ask MORE than the recipe, so the published helper, which
    truncates the control to the questioner's spend, matches them; both are pooled over the two suites
    and the two seeds, at the published controls' 1,000 resamples. The depth-one and evidence-blind
    ablations ask FEWER, so they are read with both arms at the lower count, per seed and per suite, at
    10,000 resamples with a verdict at 50,000 under three seeds. Every source record's own lock must
    hold, and the pattern the prose states must hold, or nothing is drawn.
    """
    d = CONTROLS_RECIPE_DIR
    require(
        d / "depth1_matched_recipe.json",
        d / "controls_symmetric_recipe.json",
        d / "controls_recipe.json",
        d / "controls_by_seed.json",
    )
    sym = {
        "no sequencing": (json.loads((d / "depth1_matched_recipe.json").read_text()), "d1"),
        "blind to evidence": (json.loads((d / "controls_symmetric_recipe.json").read_text()), "ne"),
    }
    for name, (rec, _tag) in sym.items():
        bad = [k for k, v in rec["published_lock"].items() if v["verdict"] != "LOCKED"]
        if bad:
            raise Mismatch(f"{name}: published lock not LOCKED on {bad}. NOT DRAWING.")
    cr = json.loads((d / "controls_recipe.json").read_text())
    cb = json.loads((d / "controls_by_seed.json").read_text())
    if not all(v["ok"] for v in cr["lock"].values()) or not all(
        v["ok"] for v in cb["lock"].values()
    ):
        raise Mismatch("a controls record's seed-0 lock failed. NOT DRAWING.")
    rows: list[dict[str, Any]] = []
    rq = cb["cells"]["s1s2"]["random_q::matched"]
    rows.append(
        dict(
            group="random questions",
            suite="both",
            seed="both",
            delta=rq["delta"],
            lo=rq["ci_lo"],
            hi=rq["ci_hi"],
            decided=rq["ci_lo"] > 0,
            n=rq["n"],
            rule="published helper (the control asks more), 1,000 resamples",
        )
    )
    for name, (rec, tag) in sym.items():
        for suite in ("musique", "strategyqa"):
            for s in CONTROLS_RECIPE_SEEDS:
                cell = rec["contrasts"][f"{s}_minus_{tag}_{s}"][f"evidence_coverage::{suite}"]
                r10 = next(x for x in cell["readings"] if x["n_boot"] == 10000)
                rows.append(
                    dict(
                        group=name,
                        suite=suite,
                        seed=s,
                        delta=r10["delta"],
                        lo=r10["ci_lo"],
                        hi=r10["ci_hi"],
                        decided=cell["verdict"] == "DECIDED",
                        n=r10["n"],
                        rule="both arms at the lower question count, 10,000 resamples, "
                        "decided at 50,000 under three seeds",
                    )
                )
    si = cr["cells"]["pooled"]["self_inquire::matched"]
    rows.append(
        dict(
            group="one model, both roles",
            suite="both",
            seed="both",
            delta=si["delta"],
            lo=si["ci_lo"],
            hi=si["ci_hi"],
            decided=si["ci_lo"] > 0,
            n=si["n"],
            rule="published helper (the control asks more), 1,000 resamples",
        )
    )
    # The pattern Section 5.3 states, asserted rather than eyeballed.
    by = {(r["group"], r["suite"], r["seed"]): r for r in rows}
    if not (by[("random questions", "both", "both")]["lo"] > CONTROLS_MARGIN):
        raise Mismatch("random questions no longer clear the margin")
    for g in ("no sequencing", "blind to evidence"):
        for s in CONTROLS_RECIPE_SEEDS:
            if not by[(g, "musique", s)]["decided"]:
                raise Mismatch(f"{g} multi-hop {s} is not decided, as the text says it is")
            if by[(g, "strategyqa", s)]["decided"]:
                raise Mismatch(f"{g} implicit {s} is decided, the text says neither separates")
    one = by[("one model, both roles", "both", "both")]
    if not (one["hi"] < CONTROLS_MARGIN and one["lo"] > -CONTROLS_MARGIN):
        raise Mismatch("the single-model interval is not inside the declared margin")
    return rows


def fig8_controls_recipe(out: Path) -> list[Path]:
    """What each part of the protocol is worth to the recipe, each control read at its own rule."""
    rows = _controls_recipe_rows()
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(LINEWIDTH_IN * 0.92, 2.9))
    suite_name = {"musique": "MuSiQue", "strategyqa": "StrategyQA", "both": "both suites"}
    color = {"musique": PALETTE[0], "strategyqa": PALETTE[1], "both": INK}
    marker = {"s1": "o", "s2": "s", "both": "D"}
    # the names tab:app-controls-pooled and the prose use; the row keys stay the records' own
    shown = {
        "no sequencing": "depth one",
        "blind to evidence": "evidence blind",
        "one model, both roles": "single model",
    }
    xmax = 0.20
    y, ys, labels, last = 0.0, [], [], None
    for r in rows:
        if last is not None and r["group"] != last:
            y -= 0.6
        ys.append(y)
        if r["seed"] == "both":
            labels.append(f"{shown.get(r['group'], r['group'])} (both suites)")
        else:
            labels.append(
                f"{shown.get(r['group'], r['group'])}: {suite_name[r['suite']]}, seed {r['seed'][1]}"
            )
        last = r["group"]
        y -= 1.0
    ax.axvspan(-CONTROLS_MARGIN, CONTROLS_MARGIN, color=GREY, alpha=0.13, lw=0, zorder=0)
    ax.axvline(0.0, color="#444444", lw=0.9, zorder=1)
    clipped = []
    for yy, r in zip(ys, rows):
        c = color[r["suite"]]
        if r["hi"] > xmax:
            # off the axis: an arrow at the edge and the reading beside it
            edge = xmax * 0.985
            ax.plot([edge], [yy], marker=">", ms=5.5, color=c, zorder=3, lw=0)
            ax.annotate(
                f"${_fmt(r['delta'], 3, sign=True)}$ "
                f"$[{_fmt(r['lo'], 3, sign=True)}, {_fmt(r['hi'], 3, sign=True)}]$",
                (edge, yy),
                xytext=(-6, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=7,
                color=c,
            )
            clipped.append({k: r[k] for k in ("group", "suite", "seed", "delta", "lo", "hi")})
            continue
        ax.errorbar(
            [r["delta"]],
            [yy],
            xerr=[[r["delta"] - r["lo"]], [r["hi"] - r["delta"]]],
            marker=marker[r["seed"]],
            ms=4.8,
            mew=1.2,
            lw=1.1,
            capsize=2.0,
            color=c,
            mfc=c if r["decided"] else "white",
            zorder=3,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.7)
    ax.set_xlim(-0.06, xmax)
    ax.set_xlabel("coverage the trained questioner recovers and the variant does not")
    ax.grid(axis="y", visible=False)
    ax.annotate(
        f"equivalence margin, declared in advance ($\\pm{CONTROLS_MARGIN:g}$)",
        (CONTROLS_MARGIN, min(ys) - 0.55),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=7,
        color="#6E6E6E",
    )
    fig.tight_layout(pad=0.4)
    srcs = [
        CONTROLS_RECIPE_DIR / f
        for f in (
            "depth1_matched_recipe.json",
            "controls_symmetric_recipe.json",
            "controls_recipe.json",
            "controls_by_seed.json",
        )
    ]
    prov = provenance(
        asset_id="fig8_controls_recipe",
        sources=srcs,
        query=(
            "random_q::matched (controls_by_seed s1s2) and self_inquire::matched (controls_recipe "
            "pooled) under the published helper; each seed minus its own depth-one and "
            "evidence-blind ablation, evidence_coverage per suite, n_boot=10000 reading and 50,000 "
            "verdict (depth1_matched_recipe, controls_symmetric_recipe)"
        ),
        values=rows,
        scorer_hash=["3f88ac3527ed6c2fcf78b1538f6c44a3c14838aec93a583c072a770fb70e8162"],
        graph_version=["v1"],
        extra={
            "clipped_rows": clipped,
            "xlim": [-0.06, xmax],
            "build": "scripts/paper_figures_iclr.py --only fig8_recipe",
            "margin": CONTROLS_MARGIN,
            "filled": "decided: every 50,000-resample interval excludes zero (ablation rows), or "
            "the 1,000-resample interval excludes zero (the two pooled helper rows)",
            "single_model_stability": "matched +0.02201, lower bound +0.0114 at 1k, +0.0117 at "
            "10k, +0.0116-0.0117 at 50k under seeds 101/202/303 (checked 2026-09-23)",
            "seed0": "the published seed-0 figure (fig8_controls) pooled four controls over two "
            "suites under the helper: +0.5151, +0.0986, +0.0662, +0.0143",
        },
    )
    return _emit_figure(fig, out, "fig8_controls_recipe", prov)


GOLD_GRAPHS = REPO / "data" / "gold" / "graphs"

# The published census this figure must reproduce from the gold itself, per suite:
# (tasks, nodes, tasks with two or more facets, deepest gold chain).
SUITE_CENSUS_LOCK = {
    "musique": (800, 2660, 0, 3),
    "strategyqa": (2290, 6720, 0, 4),
    "wiki2": (12576, 31120, 2663, 1),
}
SUITE_CENSUS_TITLE = {
    "musique": "MuSiQue",
    "strategyqa": "StrategyQA",
    "wiki2": "2WikiMultiHopQA",
}


def _suite_census() -> list[dict[str, Any]]:
    """Recompute the two axes' expressiveness from gold, and lock against the published census.

    Read from the gold rather than from any record, because the claim is about what the CORPUS
    can express. Every field carries a `gold_` prefix; a first version of this read `depth` and
    `facet_id` and got zeros back, which looked exactly like a finding.
    """
    rows = []
    for suite, (want_t, want_n, want_m, want_d) in SUITE_CENSUS_LOCK.items():
        path = GOLD_GRAPHS / suite / "v1.jsonl"
        require(path)
        ntask = multi = nnodes = 0
        maxd = 0
        hist: dict[int, int] = {}
        with path.open() as fh:
            for line in fh:
                g = json.loads(line)
                ntask += 1
                ns = g["gold_nodes"]
                nnodes += len(ns)
                for n in ns:
                    d = n.get("gold_depth")
                    if d is None:
                        continue
                    hist[d] = hist.get(d, 0) + 1
                    maxd = max(maxd, d)
                facets = {
                    n["gold_facet_id"]
                    for n in ns
                    if (n.get("gold_depth") or 0) >= 1 and n.get("gold_facet_id")
                }
                if len(facets) >= 2:
                    multi += 1
        got = (ntask, nnodes, multi, maxd)
        if got != (want_t, want_n, want_m, want_d):
            raise Mismatch(
                f"{suite}: recomputed census {got} against the published "
                f"{(want_t, want_n, want_m, want_d)}. NOT WRITING."
            )
        rows.append(
            {
                "suite": suite,
                "title": SUITE_CENSUS_TITLE[suite],
                "n_tasks": ntask,
                "n_nodes": nnodes,
                "n_multi_facet": multi,
                "share_multi_facet": multi / ntask,
                "max_depth": maxd,
                "depth_histogram": dict(sorted(hist.items())),
            }
        )
    # The figure's claim: no suite is live on both axes, and the two are anti-correlated here.
    both = [r for r in rows if r["share_multi_facet"] > 0 and r["max_depth"] >= 2]
    if both:
        raise Mismatch(
            f"{[r['suite'] for r in both]} are live on BOTH axes, so the empty region the figure "
            "draws is not empty. NOT WRITING."
        )
    if not any(r["share_multi_facet"] > 0 for r in rows):
        raise Mismatch("no suite branches at all, so the horizontal axis is not even definable")
    return rows


def fig7_suite_axes(out: Path) -> list[Path]:
    """What each suite's gold can express, and the region no suite occupies."""
    rows = _suite_census()
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(3.5, 2.45))

    ax.axhspan(1.5, 4.6, xmin=0.0, xmax=1.0, color="#000000", alpha=0.0)
    # The empty quadrant: branching AND chains two or more deep. Drawn as an absence.
    ax.add_patch(
        plt.Rectangle(
            (5.0, 1.5),
            22.0,
            3.1,
            facecolor=GREY,
            alpha=0.10,
            edgecolor=GREY,
            lw=0.7,
            ls=(0, (3, 2)),
            zorder=0,
        )
    )
    ax.annotate(
        "both axes live:\nno suite here",
        (16.0, 3.05),
        ha="center",
        va="center",
        fontsize=7,
        color="#6E6E6E",
        zorder=1,
    )
    for r in rows:
        x = 100 * r["share_multi_facet"]
        ax.plot([x], [r["max_depth"]], marker="o", ms=6.0, color=PALETTE[0], zorder=3)
        # Labels flip inward on the right so none is clipped when the figure is placed narrow.
        right = x > 15.0
        ax.annotate(
            r["title"],
            (x, r["max_depth"]),
            xytext=(-7 if right else 7, 5),
            textcoords="offset points",
            ha="right" if right else "left",
            fontsize=7.5,
            zorder=3,
        )
    ax.set_xlim(-2.0, 29.0)
    ax.set_ylim(0.3, 4.7)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_xlabel("tasks whose needs branch (percent)")
    ax.set_ylabel("deepest prerequisite chain in gold")
    fig.tight_layout(pad=0.4)

    prov = {
        "asset": "fig7_suite_axes",
        "tool": TOOL,
        "inputs": [
            {
                "path": rel(GOLD_GRAPHS / s / "v1.jsonl"),
                "sha256": sha256_file(GOLD_GRAPHS / s / "v1.jsonl"),
            }
            for s in SUITE_CENSUS_LOCK
        ],
        "rows": rows,
        "graph_version": "v1",
        "scorer_hash": "n/a: this is a census of the gold, not a scored reading",
        "lock": (
            "recomputed from the gold graphs and refused unless the per-suite census matches the "
            "published (tasks, nodes, branching tasks, deepest chain) exactly, and unless the "
            "quadrant the figure draws as empty is in fact empty."
        ),
        "caveats": (
            "A census of what the CORPUS can express, not a result about any policy. Facets are "
            "counted over depth-positive nodes, matching the definition the paper uses. The "
            "deepest chain is the gold maximum, not what any arm reached. This is the evidence "
            "behind the paper's statement that the horizontal axis is definable here and not "
            "separately measurable, which is reinforced by two further censuses in the text: 0 "
            "cross-facet prerequisite edges of 19,532, and 0 wholly optional facets among 2,735 "
            "multi-facet tasks."
        ),
    }
    return _emit_figure(fig, out, "fig7_suite_axes", prov)


EFFICIENCY_RECORD = REPO / "artifacts/unreported_metrics_20260919/RESULT.md"
EFFICIENCY_STORE = REPO / "artifacts/testsplit_qa/scores_parquet"

# Tokens spent per required need the run actually found. Lower is better.
#
# The per-arm LEVELS are recomputed from the store over the PAIRED intersection rather than copied
# from the record, because the record's own per-arm levels are means over two different task sets
# and disagree with the paired reading in the third digit (19,354 / 8,555 there against 19,402 /
# 8,365 here on the multi-hop suite). The record's paired DELTAS are correct and are used as the
# cross-check: the recomputation has to land on them.
EFFICIENCY_SUITES = (
    ("musique", "MuSiQue"),
    ("strategyqa", "StrategyQA"),
    ("wiki2", "2WikiMultiHopQA"),
)
EFFICIENCY_DELTAS = {  # from the record, with its intervals; the lock target
    "musique": (-11037, -13029, -9198),
    "strategyqa": (-8839, -9880, -7795),
    "wiki2": (-2189, -2916, -1533),
}


def _efficiency_rows() -> list[dict[str, Any]]:
    """Paired levels from the store, locked against the record's published deltas."""
    require(EFFICIENCY_RECORD, EFFICIENCY_STORE / "runs.parquet")
    text = EFFICIENCY_RECORD.read_text()
    import duckdb

    con = duckdb.connect()
    runs = str(EFFICIENCY_STORE / "runs.parquet")
    scores = str(EFFICIENCY_STORE / "scores.parquet")
    rows = con.execute(
        f"""
        with s as (
          select r.suite_id, r.task_id, r.arm_id, avg(sc.value) v
          from '{runs}' r join '{scores}' sc using (run_id)
          where sc.metric_name='tokens_per_useful_need' and sc.value is not null
          group by 1,2,3
        ), p as (
          select a.suite_id, a.task_id, a.v tr, b.v pr from s a join s b
          on a.suite_id=b.suite_id and a.task_id=b.task_id
          where a.arm_id<>b.arm_id and a.arm_id like '%trained%'
        )
        select suite_id, count(*) n, avg(tr) trained, avg(pr) prompted, avg(tr-pr) delta
        from p group by 1
        """
    ).fetchall()
    by = {r[0]: r for r in rows}
    out = []
    for key, title in EFFICIENCY_SUITES:
        if key not in by:
            raise Mismatch(f"{key}: no paired rows in {rel(EFFICIENCY_STORE)}. NOT WRITING.")
        _s, n, tr, pr, d = by[key]
        want, lo, hi = EFFICIENCY_DELTAS[key]

        # The record writes thousands in LaTeX as 11{,}037, so both spellings are accepted and
        # the needle is built rather than assumed. A plain-comma-only check read as absent here.
        def _forms(v: int) -> tuple[str, str]:
            plain = f"{v:,}"
            return plain, plain.replace(",", "{,}")

        d_plain, d_tex = _forms(want)
        lo_plain, lo_tex = _forms(lo)
        hi_plain, hi_tex = _forms(hi)
        has_delta = d_plain in text or d_tex in text
        has_ci = f"[{lo_plain}, {hi_plain}]" in text or f"[{lo_tex}, {hi_tex}]" in text
        if not (has_delta and has_ci):
            raise Mismatch(
                f"{key}: the record does not carry delta {d_plain} with interval "
                f"[{lo_plain}, {hi_plain}] in either plain or LaTeX spelling. NOT WRITING."
            )
        # The reproduction lock: the store must land on the record's published delta.
        if abs(d - want) > 1.0:
            raise Mismatch(
                f"{key}: recomputed paired delta {d:,.1f} against the record's {want:,}. "
                "A reimplementation that does not reproduce the published number is not a "
                "recomputation. NOT WRITING."
            )
        out.append(
            {
                "suite": key,
                "title": title,
                "trained": tr,
                "prompted": pr,
                "delta": d,
                "ci": [lo, hi],
                "n_paired_tasks": int(n),
            }
        )
    if not all(r["ci"][1] < 0 for r in out):
        raise Mismatch("an interval does not exclude zero on the favourable side")
    if not all(r["prompted"] > r["trained"] for r in out):
        raise Mismatch("the comparator is not more expensive per need on every suite")
    return out


EFFICIENCY_SEED_RECORD = REPO / "artifacts/seed_identity_20260923/tokens_by_seed.json"


def _efficiency_rows_recipe() -> list[dict[str, Any]]:
    """The recipe's two fresh seeds, averaged within the task (tokens_by_seed.json, cells s1s2).

    Two locks: the seed-0 route above must still reproduce the published deltas (it raises if
    not), and the per-seed record's own seed-0 cell must equal that route's delta to within one
    token, so the recipe's row and the published row come from one population and one metric.
    """
    s0_rows = {r["suite"]: r for r in _efficiency_rows()}
    require(EFFICIENCY_SEED_RECORD)
    rec = json.loads(EFFICIENCY_SEED_RECORD.read_text())
    if not all(v["ok"] for v in rec["lock"].values()):
        raise Mismatch("tokens_by_seed.json carries a failed lock. NOT WRITING.")
    out = []
    for key, title in EFFICIENCY_SUITES:
        s0 = rec["cells"]["s0"][key]
        if abs(s0["delta"] - s0_rows[key]["delta"]) > 1.0:
            raise Mismatch(
                f"{key}: per-seed record's seed 0 {s0['delta']:.1f} against the store's "
                f"{s0_rows[key]['delta']:.1f}. NOT WRITING."
            )
        c = rec["cells"]["s1s2"][key]
        out.append(
            {
                "suite": key,
                "title": title,
                "trained": c["level_trained"],
                "prompted": c["level_prompted"],
                "delta": c["delta"],
                "ci": [c["ci_lo"], c["ci_hi"]],
                "n_paired_tasks": c["n"],
                "seed0_delta": s0["delta"],
            }
        )
    if not all(r["ci"][1] < 0 for r in out):
        raise Mismatch("an interval does not exclude zero on the favourable side")
    if not all(r["prompted"] > r["trained"] for r in out):
        raise Mismatch("the comparator is not more expensive per need on every suite")
    return out


def fig6_efficiency(out: Path) -> list[Path]:
    """Cost in the denominator, so no pairing rule enters the reading. Drawn for the recipe."""
    rows = _efficiency_rows_recipe()
    plt = _mpl()
    fig, axes = plt.subplots(
        1, 2, figsize=(LINEWIDTH_IN, 1.95), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    ys = list(range(len(rows)))[::-1]

    ax = axes[0]
    for y, r in zip(ys, rows):
        tr, pr = r["trained"], r["prompted"]
        ax.plot([tr / 1000, pr / 1000], [y, y], color=GREY, lw=1.0, zorder=1)
        ax.plot([pr / 1000], [y], marker="o", ms=5.0, color=PALETTE[1], zorder=3)
        ax.plot([tr / 1000], [y], marker="o", ms=5.0, color=PALETTE[0], zorder=3)
        # Labels point outward from each marker, so the closest pair on any suite cannot collide.
        ax.annotate(
            f"{tr / 1000:.1f}k",
            (tr / 1000, y),
            xytext=(-5, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7,
            color=PALETTE[0],
        )
        ax.annotate(
            f"{pr / 1000:.1f}k",
            (pr / 1000, y),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7,
            color=PALETTE[1],
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([r["title"] for r in rows])
    ax.set_ylim(-1.0, len(rows) - 0.35)
    ax.set_xlim(-1.4, 23.5)
    ax.set_xlabel("thousands of tokens per required need found")
    ax.set_title("(a) what each need costs", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", ls="none", color=PALETTE[1], label="prompted"),
            plt.Line2D([], [], marker="o", ls="none", color=PALETTE[0], label="trained"),
        ],
        loc="lower left",
        frameon=False,
        handlelength=1.0,
        ncol=2,
        borderaxespad=0.2,
    )

    ax = axes[1]
    ax.axvline(0.0, color="#444444", lw=0.9)
    for y, r in zip(ys, rows):
        d, (lo, hi) = r["delta"], r["ci"]
        ax.errorbar(
            [d / 1000],
            [y],
            xerr=[[(d - lo) / 1000], [(hi - d) / 1000]],
            marker="o",
            ms=5.0,
            color=PALETTE[0],
            lw=1.2,
            capsize=2.2,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    ax.set_ylim(-1.0, len(rows) - 0.35)
    ax.set_xlim(right=1.6)
    ax.set_xlabel("paired reduction (thousands)")
    ax.set_title("(b) trained minus prompted", loc="left")
    ax.grid(axis="y", visible=False)

    fig.tight_layout(pad=0.4)
    prov = {
        "asset": "fig6_efficiency",
        "tool": TOOL,
        "inputs": [
            {"path": rel(EFFICIENCY_SEED_RECORD), "sha256": sha256_file(EFFICIENCY_SEED_RECORD)},
            {"path": rel(EFFICIENCY_RECORD), "sha256": sha256_file(EFFICIENCY_RECORD)},
            {"path": rel(EFFICIENCY_STORE / "runs.parquet")},
            {"path": rel(EFFICIENCY_STORE / "scores.parquet")},
        ],
        "rows": rows,
        "drawn": "the recipe's seeds 1 and 2 averaged within the task; seed 0 locked first",
        "metric": "tokens_per_useful_need",
        "scorer_hash": "e82c7458ee8efa11 (seed-replicate store); seed-0 lock route aa18b1fefd3b9b50",
        "graph_version": "v1",
        "resamples": 10000,
        "lock": (
            "the per-arm levels are RECOMPUTED from the store over the paired intersection and "
            "the recomputation must land within 1 token of the record's published paired delta "
            "on every suite, or nothing is written. Plus two shape claims: each interval "
            "excludes zero on the favourable side and the comparator is more expensive per need "
            "on every suite."
        ),
        "caveats": (
            "The per-arm LEVELS are means over the paired intersection, because the record's own "
            "per-arm levels are unpaired over different task sets and differ in the third digit: "
            "19,354 and 8,555 there against 19,402 and 8,365 here on the multi-hop suite. The "
            "PAIRED DELTAS are the estimate and they reproduce the record exactly. On the "
            "same population five other unmatched-cost instruments read null on the multi-hop "
            "suite and worse on the implicit-reasoning one, the largest adverse cell being "
            "latent-need discovery at -0.2798 [-0.4107,-0.1607] over 42 paired tasks, so this is "
            "not the favourable member of a family of unmatched readings and the caption says so."
        ),
    }
    return _emit_figure(fig, out, "fig6_efficiency", prov)


INVERSION_SUITES = (
    ("musique", "MuSiQue"),
    ("strategyqa", "StrategyQA"),
    ("wiki2", "2WikiMultiHopQA"),
)


def _inversion_cells() -> list[dict[str, Any]]:
    """Coverage under both cost bases, same 200 tasks, same scorer, from one record.

    Both bases come out of the SAME file and the same completed population on purpose. Reading
    the fixed-cap column from the short-comparator store and the equal-spend column from the
    completed one would put the population difference inside the comparison the figure is about.
    """
    require(COMPLETED_COVERAGE)
    rec = json.loads(COMPLETED_COVERAGE.read_text())["complete"]
    if rec["rule"] != "matched_cost":
        raise Mismatch(f"{rel(COMPLETED_COVERAGE)}: rule is {rec['rule']!r}. NOT WRITING.")
    out = []
    for key, title in INVERSION_SUITES:
        v = rec["by_suite"][key]
        sym = v["symmetric"]
        if sym["n_tasks"] != v["cap8_n_tasks"]:
            raise Mismatch(
                f"{key}: equal-spend n is {sym['n_tasks']} and fixed-cap n is "
                f"{v['cap8_n_tasks']}. The two bases must be read on ONE population or the "
                "figure compares a basis change with a population change. NOT WRITING."
            )
        out.append(
            {
                "suite": key,
                "title": title,
                "matched": sym["delta"],
                "matched_ci": [sym["ci_lo"], sym["ci_hi"]],
                "cap8": v["cap8_coverage_delta"],
                "n": sym["n_tasks"],
                "matched_rung": sym["mean_k_common"],
                "trained_asks": v["trained_mean_n_asks"],
                "base_asks": v["baseline_mean_n_asks"],
            }
        )
    # The figure's whole claim, asserted over the numbers rather than read off the drawing.
    if not all(c["matched_ci"][0] > 0 for c in out):
        raise Mismatch("an equal-spend interval does not exclude zero; the panel would overclaim")
    flips = [c for c in out if c["cap8"] < 0 < c["matched"]]
    if len(flips) != 2:
        raise Mismatch(
            f"the figure claims two suites change sign between the bases, found {len(flips)}"
        )
    if not all(c["base_asks"] > c["trained_asks"] for c in out):
        raise Mismatch("the comparator does not outspend the trained arm on every suite")
    return out


def fig5_cost_basis(out: Path) -> list[Path]:
    """The same comparison, two cost bases, and the spend that explains the difference."""
    cells = _inversion_cells()
    plt = _mpl()
    fig, axes = plt.subplots(
        1, 2, figsize=(LINEWIDTH_IN, 2.0), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )
    ys = list(range(len(cells)))[::-1]

    ax = axes[0]
    ax.axvline(0.0, color="#444444", lw=0.9)
    for y, c in zip(ys, cells):
        ax.plot(
            [c["cap8"], c["matched"]],
            [y, y],
            color=GREY,
            lw=1.0,
            zorder=1,
            solid_capstyle="butt",
        )
        ax.plot(
            [c["cap8"]],
            [y],
            marker="o",
            ms=5.0,
            mfc="white",
            mec=PALETTE[0],
            mew=1.4,
            zorder=3,
        )
        ax.errorbar(
            [c["matched"]],
            [y],
            xerr=[[c["matched"] - c["matched_ci"][0]], [c["matched_ci"][1] - c["matched"]]],
            marker="o",
            ms=5.0,
            color=PALETTE[0],
            lw=1.2,
            capsize=2.2,
            zorder=3,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([c["title"] for c in cells])
    # An empty band under the data, so the legend sits beside nothing rather than on a row.
    ax.set_ylim(-1.25, len(cells) - 0.45)
    ax.set_xlabel("required-evidence coverage, trained minus prompted")
    ax.set_title("(a) the cost basis decides the sign", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="o",
                ls="none",
                mfc="white",
                mec=PALETTE[0],
                mew=1.4,
                label="fixed cap of 8 calls",
            ),
            plt.Line2D([], [], marker="o", ls="none", color=PALETTE[0], label="equal questions"),
        ],
        loc="lower left",
        frameon=False,
        handlelength=1.0,
        borderaxespad=0.2,
    )

    ax = axes[1]
    h = 0.34
    for y, c in zip(ys, cells):
        ax.barh(y + h / 2, c["base_asks"], height=h, color=PALETTE[1], alpha=0.85)
        ax.barh(y - h / 2, c["trained_asks"], height=h, color=PALETTE[0])
        ax.annotate(
            f"{c['base_asks']:.1f}",
            (c["base_asks"], y + h / 2),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=GREY,
        )
        ax.annotate(
            f"{c['trained_asks']:.1f}",
            (c["trained_asks"], y - h / 2),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=GREY,
        )
        if y == ys[0]:
            ax.annotate(
                "prompted",
                (0.12, y + h / 2),
                va="center",
                fontsize=6.5,
                color="white",
            )
            ax.annotate(
                "trained",
                (0.12, y - h / 2),
                va="center",
                fontsize=6.5,
                color="white",
            )
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    ax.set_ylim(-1.25, len(cells) - 0.45)
    ax.set_xlim(0, max(c["base_asks"] for c in cells) * 1.22)
    ax.set_xlabel("questions asked")
    ax.set_title("(b) because it asks less", loc="left")
    ax.grid(axis="y", visible=False)

    fig.tight_layout(pad=0.4)
    prov = {
        "asset": "fig5_cost_basis",
        "tool": TOOL,
        "inputs": [{"path": rel(COMPLETED_COVERAGE), "sha256": sha256_file(COMPLETED_COVERAGE)}],
        "cells": cells,
        "scorer_hash": "e82c7458ee8efa11",
        "graph_version": "v1",
        "estimand": (
            "paired required-evidence coverage, trained questioner minus the same weights "
            "prompted, on the completed comparator at n=200 per suite. Both bases from one "
            "record and one population."
        ),
        "lock": (
            "refuses unless the two bases carry the same n per suite, unless every equal-spend "
            "interval excludes zero, unless exactly two suites change sign between the bases, "
            "and unless the comparator outspends the trained arm on every suite."
        ),
        "caveats": (
            "THE FIXED-CAP POINTS CARRY NO INTERVAL. The record supplies a point estimate for "
            "that basis and the paper prints it the same way in Table 1, so the panel draws the "
            "sign change in the point estimates and does not assert a significant negative at "
            "the cap. The equal-spend end carries a BCa cluster-bootstrap interval at 10,000 "
            "resamples. Neither basis is subtracted from the other anywhere."
        ),
    }
    return _emit_figure(fig, out, "fig5_cost_basis", prov)


COMPLETED_COVERAGE = REPO / "artifacts/baseline_completion_20260920/three.json"
COMPLETED_PLAN = REPO / "artifacts/plan_metrics_completed_20260920"

# Which record carries each Table 1 row on the COMPLETED comparator, and on which pairing rule.
# SUPERSEDED 2026-09-22: this comment used to say two rows have no symmetric completed reading and
# ride the published asymmetric rule. two_more_symmetric.json below now supplies both, so every row
# is symmetric and the only per-row difference left is how strongly each is locked.
COMPLETED_TWO_MORE = REPO / "artifacts/table1_completed_20260922/two_more_symmetric.json"

# All five rows on ONE pairing rule, which is what a reviewer asked for. Breadth and deepest need
# have no PUBLISHED symmetric float to lock against, so their symmetric readings are SINGLE-locked
# (the asymmetric half only) and the table says so per row rather than presenting them as equal in
# standing to the other three. Taking one rule costs two decided cells on wiki2; see
# artifacts/table1_completed_20260922/RESULT.md for why the weaker table is the better one.
COMPLETED_ROWS = (
    ("evidence_coverage", "required-evidence coverage", "coverage", "symmetric"),
    ("facet_breadth", "breadth", "two_more", "symmetric, single-locked"),
    ("dwr", "depth-weighted recall", "symmetric_completed.json", "symmetric"),
    ("max_depth_reached", "deepest need resolved", "two_more", "symmetric, single-locked"),
    (
        "precedence_violation_rate",
        r"out-of-order rate$^{\dagger}$",
        "symmetric_completed.json",
        "symmetric",
    ),
)
# The ladder name each of those two metrics is read under. Breadth MUST be the scorer's COUNT.
TWO_MORE_KEY = {"facet_breadth": "facet_breadth_scorer", "max_depth_reached": "max_depth_reached"}

# What the paper prints today, per cell. A completed reading is emitted ONLY if the record's own
# lock reproduces these, which is what tells "the population changed" apart from "the reader did".
COMPLETED_PUBLISHED_LOCK = {
    ("evidence_coverage", "musique"): 0.11666666666666667,
    ("evidence_coverage", "strategyqa"): 0.06029761904761905,
    ("evidence_coverage", "wiki2"): 0.06250301204819277,
    ("facet_breadth", "musique"): 0.125,
    ("facet_breadth", "strategyqa"): 0.015,
    ("facet_breadth", "wiki2"): 0.11446,  # the published COUNT, reproduced at 0.0 by the lock
    ("dwr", "musique"): 0.131155303030303,
    ("dwr", "strategyqa"): 0.02656547619047619,
    ("dwr", "wiki2"): 0.06710174029451138,
    ("max_depth_reached", "musique"): 0.2983,
    ("max_depth_reached", "strategyqa"): 0.0425,
    ("max_depth_reached", "wiki2"): 0.0753,
    ("precedence_violation_rate", "musique"): 0.059079601990049746,
    ("precedence_violation_rate", "strategyqa"): 0.0,
    ("precedence_violation_rate", "wiki2"): 0.008823529411764706,
}


def _completed_cell(metric: str, suite: str, source: str) -> dict[str, Any]:
    """One completed cell, with the record's own reproduction lock enforced first."""
    published = COMPLETED_PUBLISHED_LOCK[(metric, suite)]
    if source == "coverage":
        require(COMPLETED_COVERAGE)
        rec = json.loads(COMPLETED_COVERAGE.read_text())["complete"]["by_suite"][suite]
        sym = rec["symmetric"]
        return {
            "delta": sym["delta"],
            "ci": [sym["ci_lo"], sym["ci_hi"]],
            "n": rec["n_tasks"],
            "published": published,
            "verdict": "DECIDED" if sym["ci_lo"] > 0 or sym["ci_hi"] < 0 else "SPANS_ZERO",
            "source": rel(COMPLETED_COVERAGE),
            "resamples": 10000,
        }
    if source == "two_more":
        require(COMPLETED_TWO_MORE)
        rec = json.loads(COMPLETED_TWO_MORE.read_text())
        key = f"{TWO_MORE_KEY[metric]}::{suite}"
        lock = rec["lock"][key]
        if lock["verdict"] != "SINGLE_LOCKED":
            raise Mismatch(f"{metric}/{suite}: lock verdict is {lock['verdict']!r}. NOT WRITING.")
        # The lock here is against the PUBLISHED ASYMMETRIC value, because no published symmetric
        # float exists for this metric. That is weaker and is recorded per cell.
        if abs(lock["published_asymmetric"] - published) > 1e-4:
            raise Mismatch(
                f"{metric}/{suite}: the record locks against "
                f"{lock['published_asymmetric']!r}, the paper prints {published!r}. NOT WRITING."
            )
        at10k = next(r for r in rec["completed"][key] if r["n_boot"] == 10000)
        return {
            "delta": at10k["delta"],
            "ci": [at10k["ci_lo"], at10k["ci_hi"]],
            "n": at10k["n"],
            "published": published,
            "verdict": rec["stability"][key]["verdict"],
            "source": rel(COMPLETED_TWO_MORE),
            "resamples": 10000,
            "lock_kind": "SINGLE (asymmetric half only; no published symmetric float exists)",
        }
    path = COMPLETED_PLAN / source
    require(path)
    rec = json.loads(path.read_text())
    if source == "symmetric_completed.json":
        key = f"{metric}::{suite}"
        lock = rec["lock"][key]
        if abs(lock["published_symmetric_delta"] - published) > 1e-9:
            raise Mismatch(
                f"{metric}/{suite}: the completed record locks against "
                f"{lock['published_symmetric_delta']!r}, the paper prints {published!r}. "
                "Not the same cell. NOT WRITING."
            )
        if lock["verdict"] != "LOCKED":
            raise Mismatch(f"{metric}/{suite}: lock verdict is {lock['verdict']!r}. NOT WRITING.")
        at10k = next(r for r in rec["completed"][key] if r["n_boot"] == 10000)
        return {
            "delta": at10k["delta"],
            "ci": [at10k["ci_lo"], at10k["ci_hi"]],
            "n": at10k["n"],
            "published": published,
            "verdict": rec["stability"][key]["verdict"],
            "source": rel(path),
            "resamples": 10000,
        }
    key = f"{metric}_scorer::{suite}" if metric == "facet_breadth" else f"{metric}::{suite}"
    cell = rec[key]
    if abs(cell["published_delta"] - published) > 1e-4:
        raise Mismatch(
            f"{metric}/{suite}: the completed record locks against {cell['published_delta']!r}, "
            f"the paper prints {published!r}. Not the same cell. NOT WRITING."
        )
    if cell["lock_verdict"] != "LOCKED":
        raise Mismatch(f"{metric}/{suite}: lock verdict is {cell['lock_verdict']!r}. NOT WRITING.")
    at10k = next(r for r in cell["readings"] if r["n_boot"] == 10000)
    return {
        "delta": at10k["delta"],
        "ci": [at10k["ci_lo"], at10k["ci_hi"]],
        "n": at10k["n"],
        "published": published,
        "verdict": cell["verdict"],
        "source": rel(path),
        "resamples": 10000,
    }


def table5_completed(out: Path) -> list[Path]:
    """Table 1's five rows on the completed comparator, matched basis."""
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for metric, _label, source, _rule in COMPLETED_ROWS:
        for suite in TABLE1_SUITES:
            cells[(metric, suite)] = _completed_cell(metric, suite, source)

    # The control that makes every other movement attributable: the one suite whose comparator was
    # already complete must reproduce its published value. Asserted, not assumed.
    for metric, _l, _s, _r in COMPLETED_ROWS:
        c = cells[(metric, "strategyqa")]
        if abs(c["delta"] - c["published"]) > 5e-4:
            raise Mismatch(
                f"{metric}/strategyqa: the comparator was already complete there, so this cell "
                f"must not move, but it reads {c['delta']!r} against {c['published']!r}. "
                "NOT WRITING."
            )
    moved_down = sum(
        1 for (m, s), c in cells.items() if s != "strategyqa" and c["delta"] < c["published"] - 1e-9
    )

    lines = [
        "% table5_completed.tex -- generated by " + TOOL + ", do not hand-edit.",
        "% prov: see table5_completed.provenance.json beside this file.",
        "% note: matched basis only. The completed comparator carries a cap-8 reading for the "
        "coverage row and for no other row, so no second line is emitted rather than mixing two "
        "populations inside one cell.",
        SIZE_NOTE.format(small="391.7", normal="408.2"),
        r"\begin{tabular}{@{}l" + COLSEP + "c" + COLSEP + "c" + COLSEP + r"c@{}}",
        r"\toprule",
        " & " + " & ".join(TABLE1_TITLE[s] for s in TABLE1_SUITES) + r" \\",
        r"\midrule",
    ]
    for metric, label, _source, rule in COMPLETED_ROWS:
        top, bot = [], []
        for suite in TABLE1_SUITES:
            c = cells[(metric, suite)]
            top.append(_cell3(c["delta"], c["ci"][0], c["ci"][1]))
            bot.append(
                r"{\scriptsize $n=" + str(c["n"]) + r"$, was $" + _fmt3(c["published"]) + r"$}"
            )
        lines.append(label + " & " + " & ".join(top) + r" \\")
        lines.append(
            r"{\scriptsize\color{black!60} " + rule + "} & " + " & ".join(bot) + r" \\[1.5pt]"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]

    prov = {
        "asset": "table5_completed",
        "tool": TOOL,
        "inputs": [
            {"path": rel(q), "sha256": sha256_file(q)}
            for q in sorted(
                {COMPLETED_COVERAGE, COMPLETED_TWO_MORE}
                | {
                    COMPLETED_PLAN / s
                    for _m, _l, s, _r in COMPLETED_ROWS
                    if s not in ("coverage", "two_more")
                }
            )
        ],
        "cells": {f"{m}::{s}": c for (m, s), c in sorted(cells.items())},
        "graph_version": "v1",
        "scorer_hash": {
            "evidence_coverage": "e82c7458ee8efa11",
            "plan_metrics": "aa18b1fefd3b9b50",
        },
        "control": (
            "the implicit-reasoning comparator was never short, so all five of its cells must "
            "reproduce their published values and are asserted to within 5e-4 before writing. "
            "That is what makes the movement on the other two suites the completed population "
            "rather than a changed reader."
        ),
        "cells_moved_down_off_the_control_suite": moved_down,
        "caveats": (
            "MATCHED BASIS ONLY. All five rows are on ONE pairing rule, both arms read at the "
            "lower question count. What differs per row is the LOCK, not the rule: breadth and "
            "the deepest need resolved have no published symmetric float to check against, so "
            "their readings reproduce the asymmetric half only and the table marks them "
            "single-locked. The out-of-order row is defined only "
            "where both arms' trajectories define it, so its n is below the column n by "
            "construction. Readings are at 10,000 resamples, with 50,000-resample three-seed "
            "stability verdicts recorded per cell in the source records."
        ),
    }
    return _emit_tex(out, "table5_completed", "\n".join(lines) + "\n", prov)


SEED_IDENTITY = REPO / "artifacts" / "seed_identity_20260923" / "table1_by_seed.json"
# Table 1's metric name in table5 -> the name the per-seed record keys it under.
SEED_KEY = {
    "evidence_coverage": "evidence_coverage",
    "facet_breadth": "facet_breadth_scorer",
    "dwr": "dwr",
    "max_depth_reached": "max_depth_reached",
    "precedence_violation_rate": "precedence_violation_rate",
}


def _table1_recipe_cells() -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    """Table 1's cells for the recipe (seeds 1 and 2 averaged within the task), with both locks:
    the record's own lock flags, and its seed-0 column equal to what table5_completed reads from
    ITS three records. The verdict is recomputed from the three 50k readings, not read."""
    require(SEED_IDENTITY)
    rec = json.loads(SEED_IDENTITY.read_text())
    if not rec["lock"] or not all(v["ok"] for v in rec["lock"].values()):
        raise Mismatch("table1_by_seed.json carries a failed or empty lock. NOT WRITING.")
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for metric, _label, source, _rule in COMPLETED_ROWS:
        for suite in TABLE1_SUITES:
            key = f"{SEED_KEY[metric]}::{suite}"
            s0_here = _completed_cell(metric, suite, source)
            s0_rec = rec["cells"]["s0"][key]
            if abs(s0_rec["delta"] - s0_here["delta"]) > 1e-9:
                raise Mismatch(
                    f"{key}: seed 0 in the per-seed record reads {s0_rec['delta']!r}, the "
                    f"published table's record {s0_here['delta']!r}. NOT WRITING."
                )
            readings = rec["cells"]["s1s2"][key]
            ten = next(r for r in readings if r["n_boot"] == 10000)
            fifty = [r for r in readings if r["n_boot"] == 50000]
            if len(fifty) != 3:
                raise Mismatch(f"{key}: expected three 50k readings, found {len(fifty)}.")
            decided = all(r["ci_lo"] > 0 for r in fifty) or all(r["ci_hi"] < 0 for r in fifty)
            cells[(metric, suite)] = {
                "delta": ten["delta"],
                "ci": [ten["ci_lo"], ten["ci_hi"]],
                "n": ten["n"],
                "levels": [ten.get("level_a"), ten.get("level_b")],
                "verdict": "DECIDED" if decided else "SPANS_ZERO",
                "seed0": s0_here["delta"],
                "per_seed": {k: rec["cells"][k][key]["delta"] for k in ("s1", "s2")},
            }

    return rec, cells


def table1_recipe(out: Path) -> list[Path]:
    """Table 1 on the recipe's two fresh training seeds, with the resumed seed beneath each cell.

    The reported questioner's registered adapter (seed 0) resumed its final preference stage and
    reloaded the stage's starting weights, keeping about a tenth of it (artifacts/
    seed_identity_20260923/RESULT.md, established from the weights). This table prints seeds 1
    and 2 averaged within the task, under Table 1's own rule, against the same comparator runs.
    Two locks before writing: the record's seed-0 column must equal, cell for cell, what
    table5_completed reads from ITS three records, so both tables stand on one population and one
    rule; and the record's own lock flags must all be true.
    """
    rec, cells = _table1_recipe_cells()
    lines = [
        "% table1_recipe.tex -- generated by " + TOOL + ", do not hand-edit.",
        "% prov: see table1_recipe.provenance.json beside this file.",
        SIZE_NOTE.format(small="391.7", normal="408.2"),
        r"\begin{tabular}{@{}l" + COLSEP + "c" + COLSEP + "c" + COLSEP + r"c@{}}",
        r"\toprule",
        " & " + " & ".join(TABLE1_TITLE[s] for s in TABLE1_SUITES) + r" \\",
        r"\midrule",
    ]
    for metric, label, _source, _rule in COMPLETED_ROWS:
        top, bot = [], []
        for suite in TABLE1_SUITES:
            c = cells[(metric, suite)]
            # The printed interval is the 10k one; the verdict is the 50k one. A cell whose 10k
            # interval excludes zero but whose 50k intervals do not would read as decided, so every
            # undecided cell carries a mark (found by lane R on the 2Wiki deepest-need cell).
            mark = "" if c["verdict"] == "DECIDED" else r"$^{\circ}$"
            top.append(_cell3(c["delta"], c["ci"][0], c["ci"][1]) + mark)
            bot.append(
                r"{\scriptsize $n="
                + str(c["n"])
                + r"$, $"
                + _fmt3(c["per_seed"]["s1"])
                + r"$/$"
                + _fmt3(c["per_seed"]["s2"])
                + r"$}"
            )
        lines.append(label + " & " + " & ".join(top) + r" \\")
        lines.append(
            r"{\scriptsize\color{black!60} seed 1/seed 2} & " + " & ".join(bot) + r" \\[1.5pt]"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    prov = {
        "asset": "table1_recipe",
        "tool": TOOL,
        "inputs": [{"path": rel(SEED_IDENTITY), "sha256": sha256_file(SEED_IDENTITY)}],
        "cells": {f"{m}::{s}": c for (m, s), c in sorted(cells.items())},
        "graph_version": "v1",
        "scorer_hash": {
            "evidence_coverage": "e82c7458ee8efa11",
            "plan_metrics": "aa18b1fefd3b9b50 (s0 records), e82c7458ee8efa11 (seed-replicate store)",
        },
        "rule": rec["rule"],
        "why": (
            "seed 0 (adapter 80a56b74) resumed its final preference stage and reloaded the stage's "
            "init; delta-W distance from the init 0.0070 against 0.080 for its own checkpoint-1800 "
            "and for seeds 1 and 2 (LSF 910176; lane S2 2a9f88f). Seeds 1 and 2 are the recipe."
        ),
        "decided_cells": sum(1 for c in cells.values() if c["verdict"] == "DECIDED"),
    }
    return _emit_tex(out, "table1_recipe", "\n".join(lines) + "\n", prov)


def table_seeds(out: Path) -> list[Path]:
    """Every headline-table cell per training seed (tab:app-seeds): seed 1, seed 2 and the two
    averaged within the task, plus seed 0 (resumed) when SHOW_SEED0.

    Point estimates only, each read from the per-seed record its headline cell comes from:
    table1_by_seed.json (Table 1's rows), teacher_by_seed.json (equal spend against gpt-oss-120b),
    cap8_by_seed.json (own-stop coverage and question counts), teacher_ownstop_by_seed.json
    (own-stop coverage against gpt-oss-120b) and answer_f1_by_seed.json (the answer row). Refuses
    unless every source's own lock holds and every pooled column equals the headline table's cell,
    so the two tables cannot drift apart.
    """
    head = _headline_cells()["cells"]
    require(TEACHER_BY_SEED, CAP8_BY_SEED, TEACHER_OWNSTOP, ANSWER_F1)
    t1 = json.loads(SEED_IDENTITY.read_text())
    tb = json.loads(TEACHER_BY_SEED.read_text())
    c8 = json.loads(CAP8_BY_SEED.read_text())
    tos = json.loads(TEACHER_OWNSTOP.read_text())
    ans = json.loads(ANSWER_F1.read_text())
    for name, lock in (
        ("table1_by_seed", t1["lock"]),
        ("teacher_by_seed", tb["lock"]),
        ("cap8_by_seed", c8["lock"]),
        ("teacher_ownstop_by_seed", tos["lock"]),
    ):
        if not lock or not all(v["ok"] for v in lock.values()):
            raise Mismatch(f"{name}.json carries a failed or empty lock. NOT WRITING.")

    def ten(v: Any) -> float:
        return (
            next(r for r in v if r["n_boot"] == 10000)["delta"]
            if isinstance(v, list)
            else v["delta"]
        )

    seeds = ("s0", "s1", "s2", "s1s2")
    readers: dict[str, Callable[[str, str], float]] = {
        "equal::teacher::evidence_coverage": lambda k, s: ten(tb["cells"][k][s]),
        "own::same::evidence_coverage": lambda k, s: c8["cells"][k][s]["delta"],
        "own::teacher::evidence_coverage": lambda k, s: tos["cells"][k]["evidence_coverage"][s][
            "delta"
        ],
        "own::same::answer_token_f1": lambda k, s: ans["cells"][k][s]["delta"],
    }
    for metric, *_ in COMPLETED_ROWS:
        readers[f"equal::same::{SEED_KEY[metric]}"] = lambda k, s, m=metric: ten(
            t1["cells"][k][f"{SEED_KEY[m]}::{s}"]
        )
    cols = [c for c in (("s0", "seed 0"),) if SHOW_SEED0] + [
        ("s1", "seed 1"),
        ("s2", "seed 2"),
        ("s1s2", "seeds 1+2"),
    ]
    ncol = 2 + len(cols)
    lines = [
        "% table_seeds.tex -- generated by " + TOOL + ", do not hand-edit.",
        "% prov: see table_seeds.provenance.json beside this file.",
        r"\begin{tabular}{@{}ll" + "r" * len(cols) + r"@{}}",
        r"\toprule",
        " & suite & " + " & ".join(c[1] for c in cols) + r" \\",
    ]
    cells: dict[str, dict[str, float]] = {}
    block = None
    for blk, key, label in HEADLINE_ROWS:
        if blk != block:
            lines += [
                r"\midrule",
                r"\multicolumn{" + str(ncol) + r"}{@{}l}{" + HEADLINE_BLOCKS[blk] + r"} \\",
            ]
            block = blk
        for i, suite in enumerate(TABLE1_SUITES):
            if key is None:
                # the trained questioner's own mean; the headline row prints all three arms
                label = "questions asked"
                vals = {k: c8["asks"][k][suite]["mean_n_asks"] for k in seeds}
                cells[f"own::asks_trained::{suite}"] = vals
                printed = [_fmt(vals[k], 2) for k, _ in cols]
            else:
                vals = {k: readers[key](k, suite) for k in seeds}
                full = f"{key}::{suite}"
                if abs(vals["s1s2"] - head[full]["delta"]) > 1e-12:
                    raise Mismatch(
                        f"{full}: the per-seed record's pooled cell {vals['s1s2']!r} is not the "
                        f"headline table's {head[full]['delta']!r}. NOT WRITING."
                    )
                cells[full] = vals
                printed = ["$" + _fmt3(vals[k]) + "$" for k, _ in cols]
            lines.append(
                (label if i == 0 else "")
                + " & "
                + TABLE1_TITLE[suite]
                + " & "
                + " & ".join(printed)
                + r" \\"
            )
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    prov = {
        "asset": "table_seeds",
        "tool": TOOL,
        "inputs": [
            {"path": rel(p), "sha256": sha256_file(p)}
            for p in (SEED_IDENTITY, TEACHER_BY_SEED, CAP8_BY_SEED, TEACHER_OWNSTOP, ANSWER_F1)
        ],
        "cells": cells,
        "printed_columns": [c[0] for c in cols],
        "rule": {
            "equal": t1["rule"],
            "own": "each arm at its own stop under the ceiling of eight, per-task means, paired "
            "over tasks, 10,000 resamples",
        },
        "graph_version": "v1",
        "scorer_hash": "e82c7458ee8efa11",
    }
    return _emit_tex(out, "table_seeds", "\n".join(lines) + "\n", prov)


FOUR_SEED = REPO / "artifacts" / "seedrep4_20260924" / "four_seed_table.json"
# (record key, printed label): the two runs the headline averages, then the two trained afterwards
FOUR_RUNS = (("s1", "seed 1"), ("s2", "seed 2"), ("s3", "seed 3"), ("s0fresh", "seed 0, retrained"))
FOUR_SUITES = (("musique", "MuSiQue"), ("strategyqa", "StrategyQA"), ("wiki2", "2WikiMultiHopQA"))


def table_fourseeds(out: Path) -> list[Path]:
    """Coverage at equal spend for the four complete runs of the final preference stage (seedrep4
    RULES sec 4, declared before either new run finished), read from four_seed_table.json after its
    locks: every lock row's reproduced value equals its published one to 1e-9."""
    require(FOUR_SEED)
    raw = FOUR_SEED.read_bytes()
    rec = json.loads(raw)["table1"]
    bad = [
        k
        for k, v in rec["lock"].items()
        if not isinstance(v, dict) or abs(v["reproduced"] - v["published"]) > 1e-9
    ]
    if not rec["lock"] or bad:
        raise Mismatch(f"four-seed record lock failed on {bad[:3]}. NOT DRAWING.")

    def reading(x: Any) -> dict:
        return x[0] if isinstance(x, list) else x

    def cell(c: dict) -> tuple[str, str]:
        return (
            f"${_fmt(c['delta'], 3, sign=True)}$",
            rf"{{\scriptsize $[{_fmt(c['ci_lo'], 3, sign=True)}, {_fmt(c['ci_hi'], 3, sign=True)}]$}}",
        )

    values: list[dict] = []
    lines = [
        "% table_fourseeds.tex -- generated by " + TOOL + ", do not hand-edit.",
        "% prov: see table_fourseeds.provenance.json beside this file.",
        r"\begin{tabular}{@{}l" + "c" * len(FOUR_RUNS) + "c@{}}",
        r"\toprule",
        " & " + " & ".join(lab for _, lab in FOUR_RUNS) + r" & four runs pooled \\",
        r"\midrule",
    ]
    for suite, name in FOUR_SUITES:
        k = f"evidence_coverage::{suite}"
        runs = [reading(rec["cells"][k][seed]) for seed, _ in FOUR_RUNS]
        s = rec["summary"][k]
        pooled = rec["pooled4"][k]
        p10 = next(r for r in pooled if r["n_boot"] == 10000)
        fifty = [r for r in pooled if r["n_boot"] == 50000]
        decided = bool(fifty) and all(r["ci_lo"] > 0 for r in fifty)
        tops = [cell(c)[0] for c in runs]
        subs = [cell(c)[1] for c in runs]
        ptop, psub = cell(p10)
        lines.append(f"{name} & " + " & ".join(tops) + f" & {ptop} \\\\")
        lines.append(" & " + " & ".join(subs) + f" & {psub} \\\\[1pt]")
        values.append(
            {
                "suite": suite,
                "runs": {seed: runs[i] for i, (seed, _) in enumerate(FOUR_RUNS)},
                "between_seed_sd": s["between_seed_sd"],
                "mean_within_seed_bootstrap_se": s["mean_within_seed_bootstrap_se"],
                "pooled4_10k": p10,
                "pooled4_decided_50k_x3": decided,
                "intervals_above_zero": s["intervals_above_zero"],
            }
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    prov = provenance(
        asset_id="table_fourseeds",
        sources=[FOUR_SEED],
        query=(
            "table1.cells['evidence_coverage::<suite>'][run] (10,000 resamples), "
            "table1.summary (between_seed_sd, mean_within_seed_bootstrap_se), table1.pooled4 "
            "(10,000 printed, 50,000 x 3 for the verdict); every lock row reproduced to 1e-9"
        ),
        values=values,
        scorer_hash=["e82c7458ee8efa11660445bbf434b4c1dc74253253e8f190cc36e92fc50f40c7"],
        graph_version=["v1"],
        extra={
            "record_sha256": hashlib.sha256(raw).hexdigest(),
            "rule": "symmetric seed-matched, both arms at min(k_a, k_b), vs the completed cohort's "
            "prompted runs, n=200 tasks per suite per run",
            "declared": "artifacts/seedrep4_20260924/RULES.md sec 4 (5789a10), exploratory; the "
            "headline stays seeds 1+2",
            "s0fresh": "a fresh run of seed 0 trained from the start after the resumed run was set "
            "aside (app:seedzero)",
        },
    )
    return _emit_tex(out, "table_fourseeds", "\n".join(lines) + "\n", prov)


COMPARATOR_DIR = REPO / "artifacts" / "structured_baselines_clean_20260922"
# (record, contrast name, row label, model group, is_structured)
COMPARATOR_ROWS = (
    ("contrasts_C.json", "rec_minus_c_prompted", "prompted", "8B base", False),
    ("contrasts_C.json", "rec_minus_c_par2", "PAR$^2$-RAG", "8B base", True),
    ("contrasts_recipe_vs_A.json", "rec_minus_a_prompted", "prompted", "gpt-oss-120b", False),
    ("contrasts_recipe_vs_A.json", "rec_minus_a_par2", "PAR$^2$-RAG", "gpt-oss-120b", True),
    ("contrasts_D.json", "rec_minus_d_prompted", "prompted", "Claude Opus 5", False),
    ("contrasts_D.json", "rec_minus_d_par2", "PAR$^2$-RAG", "Claude Opus 5", True),
)


# FRAMES panel (gold-free): answer_correct at each policy's own stop under the shared ceiling of 8,
# read by scripts/seed_identity/frames_structured.py. (model group, is_structured) -> comparator label.
FRAMES_STRUCTURED = REPO / "artifacts" / "frames_structured_20260923" / "frames_structured.json"
FRAMES_COMPARATORS = {
    ("8B base", False): "base",
    ("8B base", True): "par2_8b",
    ("gpt-oss-120b", False): "teacher",
    ("gpt-oss-120b", True): "par2_120b",
    ("Claude Opus 5", False): "opus",
    ("Claude Opus 5", True): "par2_opus",
}


def _frames_comparator_cells() -> list[dict[str, Any]]:
    """The six recipe-minus-comparator FRAMES contrasts, refused unless the record's lock against
    frames_recipe.json held and its population check found nothing."""
    require(FRAMES_STRUCTURED)
    rec = json.loads(FRAMES_STRUCTURED.read_text())
    lock = rec.get("lock") or {}
    if lock.get("failures"):
        raise Mismatch(f"FRAMES reader lock failed: {lock['failures'][:3]}. NOT DRAWING.")
    rows = [*(lock.get("levels") or {}).values(), *(lock.get("contrasts") or {}).values()]
    if any(not r.get("ok") for r in rows):
        raise Mismatch("FRAMES reader lock has a row that did not reproduce. NOT DRAWING.")
    if rec.get("population_failures"):
        raise Mismatch(f"FRAMES population refused: {rec['population_failures'][:3]}. NOT DRAWING.")
    reading = rec.get("reading")
    if not reading or not reading.get("contrasts"):
        raise Mismatch("FRAMES record carries no reading. NOT DRAWING.")
    cells = []
    for (group, structured), comp in FRAMES_COMPARATORS.items():
        name = f"recipe-{comp}/cap8"
        con = reading["contrasts"].get(name)
        if con is None:
            raise Mismatch(f"FRAMES record lacks {name}. NOT DRAWING.")
        cells.append(
            {
                "record": rel(FRAMES_STRUCTURED),
                "contrast": name,
                "label": "PAR$^2$-RAG" if structured else "prompted",
                "group": group,
                "structured": structured,
                "suite": "frames",
                "delta": con["delta"],
                "ci": [con["ci_lo"], con["ci_hi"]],
                "n": con["n"],
                "verdict": "DECIDED" if con["decided"] else "SPANS_ZERO",
            }
        )
    return cells


REGIMES = REPO / "artifacts" / "seed_identity_20260923" / "regimes.json"
REGIME_COMPARATORS = (("base", "same weights"), ("teacher", r"$15\times$ larger"))


TEACHER_BY_SEED = REPO / "artifacts" / "seed_identity_20260923" / "teacher_by_seed.json"


def _regimes_record() -> dict[str, Any]:
    """regimes.json, refused unless every cell carries three 50k bounds whose verdict is the one
    the record states (recomputed here, never read), and the Q1 cells equal the Table 1 and the
    teacher records they were read from."""
    require(REGIMES, SEED_IDENTITY, TEACHER_BY_SEED)
    rec = json.loads(REGIMES.read_text())
    t1 = json.loads(SEED_IDENTITY.read_text())
    tb = json.loads(TEACHER_BY_SEED.read_text())
    for suite in TABLE1_SUITES:
        for src, rows in (
            ("table1_by_seed", t1["cells"]["s1s2"][f"evidence_coverage::{suite}"]),
            ("teacher_by_seed", tb["cells"]["s1s2"][suite]),
        ):
            comp = "base" if src == "table1_by_seed" else "teacher"
            ten = next(r for r in rows if r["n_boot"] == 10000)
            got = rec["cells"][comp]["q1"][suite]
            if any(abs(ten[k] - got[k]) > 1e-12 for k in ("delta", "ci_lo", "ci_hi")):
                raise Mismatch(
                    f"Q1 {comp} {suite}: regimes.json disagrees with {src}. NOT WRITING."
                )
    for comp, by_q in rec["cells"].items():
        for q, by_suite in by_q.items():
            for suite, c in by_suite.items():
                if len(c["b50"]) != 3:
                    raise Mismatch(f"{comp} {q} {suite}: expected three 50k bounds. NOT WRITING.")
                dec = all(lo > 0 for _, lo, _ in c["b50"]) or all(hi < 0 for _, _, hi in c["b50"])
                if dec != c["decided"]:
                    raise Mismatch(
                        f"{comp} {q} {suite}: the record says decided={c['decided']}, its bounds "
                        f"say {dec}. NOT WRITING."
                    )
    return rec


def table_regimes(out: Path) -> list[Path]:
    """Two questions side by side: equal realized spend (Q1) and each arm stopping on its own under
    a shared ceiling of eight (Q2), the recipe against two prompted comparators, coverage.

    Every cell comes from artifacts/seed_identity_20260923/regimes.json (scripts/seed_identity/
    regimes.py), whose Q2 cells were recomputed through the published readers and locked to their
    records before the 50k verdicts were added. Refuses unless each cell carries three 50k bounds
    and the Q1 cells equal the Table 1 and teacher records they were read from.
    """
    rec = _regimes_record()
    lines = [
        "% table_regimes.tex -- generated by " + TOOL + ", do not hand-edit.",
        "% prov: see table_regimes.provenance.json beside this file.",
        r"\begin{tabular}{@{}l" + COLSEP + "c" + COLSEP + "c" + COLSEP + r"c@{}}",
        r"\toprule",
        " & " + " & ".join(TABLE1_TITLE[s] for s in TABLE1_SUITES) + r" \\",
    ]
    cells_out: dict[str, Any] = {}
    for q, title in (
        ("q1", r"\emph{Q1: equal realized spend}"),
        ("q2", r"\emph{Q2: each stops on its own, ceiling of eight}"),
    ):
        lines += [r"\midrule", r"\multicolumn{4}{@{}l}{" + title + r"} \\"]
        for comp, label in REGIME_COMPARATORS:
            row = []
            for suite in TABLE1_SUITES:
                c = rec["cells"][comp][q][suite]
                if len(c["b50"]) != 3:
                    raise Mismatch(f"{comp} {q} {suite}: expected three 50k bounds. NOT WRITING.")
                mark = "" if c["decided"] else r"$^{\circ}$"
                row.append(_cell3(c["delta"], c["ci_lo"], c["ci_hi"]) + mark)
                cells_out[f"{comp}::{q}::{suite}"] = c
            lines.append(r"\quad vs " + label + " & " + " & ".join(row) + r" \\")
    asks = rec["asks"]
    lines += [
        r"\midrule",
        r"{\scriptsize questions asked at own stop} & "
        + " & ".join(
            r"{\scriptsize "
            + f"{_fmt(asks[s]['recipe'], 1)} / {_fmt(asks[s]['base'], 1)} / {_fmt(asks[s]['teacher'], 1)}"
            + "}"
            for s in TABLE1_SUITES
        )
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    prov = {
        "asset": "table_regimes",
        "tool": TOOL,
        "inputs": [
            {"path": rel(REGIMES), "sha256": sha256_file(REGIMES)},
            {"path": rel(SEED_IDENTITY), "sha256": sha256_file(SEED_IDENTITY)},
        ],
        "cells": cells_out,
        "asks_recipe_base_teacher": asks,
        "sources": rec["sources"],
        "graph_version": "v1",
        "scorer_hash": {
            "q1_base": "e82c7458ee8efa11",
            "q1_teacher": "see teacher_by_seed.json",
            "q2": "see cap8_by_seed.json and teacher_ownstop_by_seed.json",
        },
        "decided_cells": sum(1 for c in cells_out.values() if c["decided"]),
    }
    return _emit_tex(out, "table_regimes", "\n".join(lines) + "\n", prov)


ANSWER_F1 = REPO / "artifacts" / "seed_identity_20260923" / "answer_f1_by_seed.json"
CAP8_BY_SEED = REPO / "artifacts" / "seed_identity_20260923" / "cap8_by_seed.json"
TEACHER_OWNSTOP = REPO / "artifacts" / "seed_identity_20260923" / "teacher_ownstop_by_seed.json"
# What the paper prints for the answer cells per training seed: section 5.5's prose (MuSiQue and
# 2WikiMultiHopQA) and its prov line (StrategyQA). The answer reader must still reproduce them.
ANSWER_F1_PRINTED = {
    ("s1", "musique"): "+0.031",
    ("s2", "musique"): "+0.022",
    ("s1", "strategyqa"): "+0.004",
    ("s2", "strategyqa"): "-0.011",
    ("s1", "wiki2"): "+0.033",
    ("s2", "wiki2"): "+0.032",
}


def _decided(bounds: Sequence[tuple[float, float]]) -> bool:
    if len(bounds) != 3:
        raise Mismatch(f"expected three 50k bounds, found {len(bounds)}. NOT WRITING.")
    return all(lo > 0 for lo, _ in bounds) or all(hi < 0 for _, hi in bounds)


def _answer_f1_record(regimes: Mapping[str, Any]) -> dict[str, Any]:
    """answer_f1_by_seed.json, refused unless every one of its locks held, it still reproduces the
    per-seed values the paper prints, it stands on the own-stop coverage row's population (its
    coverage lock equals regimes.json's own-stop cell), and each pooled verdict is the one its
    three 50k bounds give (recomputed here)."""
    require(ANSWER_F1)
    rec = json.loads(ANSWER_F1.read_text())
    lk = rec["lock"]
    sizes = {"L1_printed_per_seed": 6, "L2_own_stop_population": 12, "L3_decompose": 3}
    for name, n in sizes.items():
        got = lk.get(name) or {}
        if len(got) != n or not all(v.get("ok") for v in got.values()):
            raise Mismatch(f"answer_f1_by_seed.json: lock {name} is not {n}/{n} ok. NOT WRITING.")
    for (seed, suite), printed in ANSWER_F1_PRINTED.items():
        c = lk["L1_printed_per_seed"][f"{seed}::{suite}"]
        if c["printed"] != printed or _fmt3(c["point"]) != printed:
            raise Mismatch(
                f"answer F1 {seed} {suite}: the reader reproduces {c['point']!r} against "
                f"{c['printed']}; the paper prints {printed}. NOT WRITING."
            )
    for suite in TABLE1_SUITES:
        mine = lk["L2_own_stop_population"][f"s1s2::{suite}"]["point"]
        theirs = regimes["cells"]["base"]["q2"][suite]["delta"]
        if abs(mine - theirs) > 1e-12:
            raise Mismatch(
                f"answer F1 {suite}: read on a population whose own-stop coverage is {mine!r}, "
                f"the table's own-stop row {theirs!r}. NOT WRITING."
            )
        c = rec["cells"]["s1s2"][suite]
        ten = [r for r in c["readings"] if r["n_boot"] == 10000 and r["seed"] == 0]
        if len(ten) != 1 or any(ten[0][k] != c[k] for k in ("delta", "ci_lo", "ci_hi", "n")):
            raise Mismatch(
                f"answer F1 {suite}: the printed cell is not the 10k reading. NOT WRITING."
            )
        dec = _decided([(r["ci_lo"], r["ci_hi"]) for r in c["readings"] if r["n_boot"] == 50000])
        if dec != c["decided"]:
            raise Mismatch(
                f"answer F1 {suite}: the record says decided={c['decided']}, its bounds say "
                f"{dec}. NOT WRITING."
            )
    return rec


def _headline_cells() -> dict[str, Any]:
    """Every interval cell of the headline table, keyed '<regime>::<comparator>::<metric>::<suite>',
    each with its point, 10k interval, n, decided flag and source record."""
    rec_t1, t1 = _table1_recipe_cells()
    reg = _regimes_record()
    ans = _answer_f1_record(reg)
    require(TEACHER_BY_SEED, CAP8_BY_SEED, TEACHER_OWNSTOP)
    tb = json.loads(TEACHER_BY_SEED.read_text())["cells"]["s1s2"]
    c8 = json.loads(CAP8_BY_SEED.read_text())["cells"]["s1s2"]
    tos = json.loads(TEACHER_OWNSTOP.read_text())["cells"]["s1s2"]["evidence_coverage"]
    cells: dict[str, Any] = {}
    for metric, _label, _source, _rule in COMPLETED_ROWS:
        for suite in TABLE1_SUITES:
            c = t1[(metric, suite)]
            cells[f"equal::same::{SEED_KEY[metric]}::{suite}"] = {
                "delta": c["delta"],
                "ci": c["ci"],
                "n": c["n"],
                "decided": c["verdict"] == "DECIDED",
                "levels": c["levels"],
                "source": rel(SEED_IDENTITY) + " cells.s1s2 (levels: level_a, level_b)",
            }
    for suite in TABLE1_SUITES:
        ten = next(r for r in tb[suite] if r["n_boot"] == 10000)
        own = {
            "equal::teacher::evidence_coverage": (
                "teacher",
                "q1",
                [ten.get("level_a"), ten.get("level_b")],
                rel(TEACHER_BY_SEED) + " cells.s1s2 level_a, level_b",
            ),
            "own::same::evidence_coverage": (
                "base",
                "q2",
                [c8[suite].get("level_trained"), c8[suite].get("level_prompted")],
                rel(CAP8_BY_SEED) + " cells.s1s2 level_trained, level_prompted",
            ),
            "own::teacher::evidence_coverage": (
                "teacher",
                "q2",
                [tos[suite].get("level_trained"), tos[suite].get("level_teacher")],
                rel(TEACHER_OWNSTOP) + " cells.s1s2 level_trained, level_teacher",
            ),
        }
        for key, (comp, q, levels, lsrc) in own.items():
            c = reg["cells"][comp][q][suite]
            cells[f"{key}::{suite}"] = {
                "delta": c["delta"],
                "ci": [c["ci_lo"], c["ci_hi"]],
                "n": c["n"],
                "decided": c["decided"],
                "levels": levels,
                "source": rel(REGIMES) + f" cells.{comp}.{q}; levels: " + lsrc,
            }
        c = ans["cells"]["s1s2"][suite]
        cells[f"own::same::answer_token_f1::{suite}"] = {
            "delta": c["delta"],
            "ci": [c["ci_lo"], c["ci_hi"]],
            "n": c["n"],
            "decided": c["decided"],
            "levels": [c["level_trained"], c["level_prompted"]],
            "source": rel(ANSWER_F1) + " cells.s1s2 (levels: level_trained, level_prompted)",
        }
    # A level pair is printed only if it is the pair the delta was taken over: both arms' means over
    # the same paired tasks difference to the delta (a paired mean difference is the difference of
    # the two means over one task set). A level over another task set cannot pass this.
    for key, c in cells.items():
        a, b = c["levels"]
        if a is None or b is None or abs((a - b) - c["delta"]) > 1e-12:
            raise Mismatch(
                f"{key}: levels {a!r} and {b!r} do not difference to the delta {c['delta']!r}. "
                "NOT WRITING."
            )
    return {"cells": cells, "asks": reg["asks"], "t1_rule": rec_t1["rule"], "answer": ans}


# The difference rows: (block, cell key prefix or None for the question counts, printed label).
# Table 2 printed these until 2026-09-25; the appendix copy of its differences
# (table_app_headline_diffs) prints every row but the question line, and table_seeds prints all of
# them per seed. The level line Table 2 printed under the coverage row (HEADLINE_LEVEL_ROWS until
# 2026-09-25) and the question line are gone from the appendix copy: the levels and the question
# counts are Table 2's own cells now (INBOX_FOR_PAPER_LANE section 15).
HEADLINE_ROWS: tuple[tuple[str, str | None, str], ...] = (
    *(
        ("equal", f"equal::same::{SEED_KEY[m]}", label)
        for m, label, _source, _rule in COMPLETED_ROWS
    ),
    ("equal", "equal::teacher::evidence_coverage", "coverage vs GPT-OSS-120B"),
    ("own", "own::same::evidence_coverage", "required-evidence coverage"),
    ("own", "own::teacher::evidence_coverage", "coverage vs GPT-OSS-120B"),
    ("own", None, r"{\scriptsize questions: trained / same / 120B}"),
    ("own", "own::same::answer_token_f1", "answer token F1"),
)
HEADLINE_BLOCKS = {
    "equal": r"Equal spend: both arms at the lower of their two question counts",
    # "..., ceiling of eight calls" until 2026-09-24 night: sec 4 now defines the own stop "with at
    # most eight calls", and the main text no longer says "ceiling" (main-text owner)
    "own": r"Own stop: each arm stops itself, with at most eight calls",
}
# The rows whose quantity is a proportion print in percentage points, their levels in percent (the
# user's comment, 2026-09-24). Breadth counts independent lines and deepest need resolved counts
# levels, so those two rows keep their own units, as do the question counts.
HEADLINE_POINTS_METRICS = frozenset(
    {"evidence_coverage", "dwr", "precedence_violation_rate", "answer_token_f1"}
)


def _headline_prov(got: Mapping[str, Any], asset: str) -> dict[str, Any]:
    """The provenance the headline assets share: the six source records and their sha256, every cell
    (point, 10k interval, n, decided flag, both levels, source), the question counts and the rules."""
    cells, ans = got["cells"], got["answer"]
    return {
        "asset": asset,
        "tool": TOOL,
        "inputs": [
            {"path": rel(p), "sha256": sha256_file(p)}
            for p in (
                SEED_IDENTITY,
                REGIMES,
                TEACHER_BY_SEED,
                CAP8_BY_SEED,
                TEACHER_OWNSTOP,
                ANSWER_F1,
            )
        ],
        "cells": cells,
        "levels_note": "cells[*].levels = [the trained questioner, the comparator], both arms' "
        "means over the same paired tasks as the delta, read from the record named in "
        "cells[*].source and asserted to difference to the delta to 1e-12",
        "asks_trained_same_teacher": got["asks"],
        "decided_cells": sum(1 for c in cells.values() if c["decided"]),
        "graph_version": "v1",
        "scorer_hash": "e82c7458ee8efa11 (every source record; the seed-0 lock of Table 1 also "
        "reads aa18b1fefd3b9b50 plan-metric records)",
        "rules": {
            "equal": got["t1_rule"] + "; the 120B row the same rule (teacher_by_seed.json)",
            "own": "each arm at its own stop, ceiling of eight, no matching; per-task means over "
            "the arm's runs, paired over tasks, 10,000 resamples (regimes.json, "
            "answer_f1_by_seed.json)",
            "decided": "all three 50,000-resample intervals (seeds 101/202/303) exclude zero on "
            "one side; recomputed from the bounds",
        },
        "answer_row": {
            "record": rel(ANSWER_F1),
            "locks": {k: f"{len(v)}/{len(v)} ok" for k, v in ans["lock"].items()},
            "printed_per_seed_reproduced": {
                k: v["printed"] for k, v in ans["lock"]["L1_printed_per_seed"].items()
            },
            "clustered_musique_reading_of_decompose_json": ans["lock"]["L3_decompose"]["musique"],
            "definition_note": ans["definition_note"],
        },
    }


def table_app_headline_diffs(out: Path) -> list[Path]:
    """The differences behind Table 2, for the appendix (tab:app-headline-diffs): the trained
    questioner (seeds 1 and 2 averaged within the task) minus a prompted comparator on the held-out
    splits, at equal spend and at its own stop, each with its 10k interval.

    This is Table 2 as the main text printed it until 2026-09-25 (bc1035f), less its two small
    sub-rows: the user moved the differences here and put both arms' levels in the main table
    (INBOX_FOR_PAPER_LANE section 15), so the level line and the question line would print what Table
    2 now prints. Every cell is read through _headline_cells, the loaders Table 2 uses (Table 1's rows
    from table1_by_seed.json, the gpt-oss-120b and own-stop coverage rows from regimes.json, the
    answer row from answer_f1_by_seed.json, which reproduces the per-seed values section 5.5 prints
    before it reads the pooled cell). A cell whose three 50k intervals do not all exclude zero on one
    side carries a degree sign; the verdicts are recomputed from the bounds.
    """
    got = _headline_cells()
    cells = got["cells"]
    lines = [
        "% table_app_headline_diffs.tex -- generated by " + TOOL + ", do not hand-edit.",
        "% prov: see table_app_headline_diffs.provenance.json beside this file.",
        "% note: Table 2 (table_headline.tex) as printed until 2026-09-25 (bc1035f), without its "
        "level line and question line, which Table 2 now prints as its cells.",
        HEADLINE_DIFFS_SIZE_NOTE,
        r"\begin{tabular}{@{}l" + COLSEP + "c" + COLSEP + "c" + COLSEP + r"c@{}}",
        r"\toprule",
        " & " + " & ".join(TABLE1_TITLE[s] for s in TABLE1_SUITES) + r" \\",
    ]
    printed: dict[str, str] = {}
    block = None
    for blk, key, label in HEADLINE_ROWS:
        if key is None:  # the question counts are Table 2's cells now
            continue
        if blk != block:
            lines += [r"\midrule", r"\multicolumn{4}{@{}l}{" + HEADLINE_BLOCKS[blk] + r"} \\"]
            block = blk
        cell = _cell_pts if key.split("::")[2] in HEADLINE_POINTS_METRICS else _cell3
        for s in TABLE1_SUITES:
            c = cells[f"{key}::{s}"]
            mark = "" if c["decided"] else r"$^{\circ}$"
            printed[f"{key}::{s}"] = cell(c["delta"], c["ci"][0], c["ci"][1]) + mark
        # No \quad before the label: indenting the rows under their block heading costs 9pt, and
        # the block fits \linewidth only without it (measured, HEADLINE_DIFFS_SIZE_NOTE).
        row = [printed[f"{key}::{s}"] for s in TABLE1_SUITES]
        lines.append(label + " & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    prov = _headline_prov(got, "table_app_headline_diffs")
    prov["build"] = TOOL + " --only table_app_headline_diffs"
    prov["printed"] = printed
    prov["printed_from"] = (
        "every printed line is Table 2 (paper/iclr2027 2/figures/table_headline.tex) as committed "
        "at bc1035f, less its two {\\scriptsize} sub-rows (the level line and the question line); "
        "tested byte for byte"
    )
    prov["printed_units"] = {
        "percentage_points": sorted(HEADLINE_POINTS_METRICS),
        "rule": "cells of these metrics print their delta and interval in percentage points, one "
        "decimal: each the 3-decimal half-up print of the value in `cells` with its decimal point "
        "moved two places (_pts). The other rows (breadth, deepest need resolved) print at three "
        "decimals (_fmt). `cells` are the records' fractions, unchanged",
    }
    return _emit_tex(out, "table_app_headline_diffs", "\n".join(lines) + "\n", prov)


# Table 2 since 2026-09-25 (INBOX_FOR_PAPER_LANE section 15, the user's request): both arms' levels
# side by side, Baseline then Q&D under each suite, in two blocks each opened by an italic label
# spanning the table; no differences, intervals or marks (those are table_app_headline_diffs). The
# main-text owner chose this layout (2026-09-25) over a four-block one that set each comparator in a
# block of its own and measured 201.0pt at \small, past the page's room. A row that names GPT-OSS-120B
# compares against it: its Baseline cell is the 120B model's level. (block label, ((row label, row
# id), ...)); a row id is a cell key of _headline_cells without its suite, or
# "<regime>::<comparator>::asks" for a question count.
HEADLINE_LAYOUT: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Equal spend",
        (
            (r"Required-evidence coverage (\%)", "equal::same::evidence_coverage"),
            (r"Depth-weighted recall (\%)", "equal::same::dwr"),
            ("Deepest need resolved (depth)", "equal::same::max_depth_reached"),
            ("Breadth (lines)", "equal::same::facet_breadth_scorer"),
            (r"Out-of-order rate (\%) $\downarrow$", "equal::same::precedence_violation_rate"),
            (r"Coverage vs GPT-OSS-120B (\%)", "equal::teacher::evidence_coverage"),
        ),
    ),
    (
        "Own stop (at most eight calls)",
        (
            (r"Required-evidence coverage (\%)", "own::same::evidence_coverage"),
            (r"Answer token F1 (\%)", "own::same::answer_token_f1"),
            ("Questions per task", "own::same::asks"),
            (r"Coverage vs GPT-OSS-120B (\%)", "own::teacher::evidence_coverage"),
            ("Questions vs GPT-OSS-120B", "own::teacher::asks"),
        ),
    ),
)
# The rows' \arraystretch, set inside a group that closes with the table (HEADLINE_SIZE_NOTE).
HEADLINE_STRETCH = "0.9"
# Lower is better only on the out-of-order rate; a question count has no better side and is never
# bold.
HEADLINE_LOWER_IS_BETTER = frozenset({"precedence_violation_rate"})
# A question row's Baseline arm in asks_trained_same_teacher; its Q&D arm is "recipe".
HEADLINE_ASKS_BASELINE = {"same": "base", "teacher": "teacher"}
# Six fixed-width numeric columns. 2WikiMultiHopQA sets 70.6pt wide at \small and "Baseline" 31.0pt
# (tectonic, the paper's preamble, 2026-09-25): 34 + 4 + 34pt holds the suite name over its two
# columns, where free columns let LaTeX widen that group's last column alone.
HEADLINE_COLS = (
    r"@{}l@{\hspace{8pt}}"
    + r"@{\hspace{10pt}}".join([r"w{c}{34pt}@{\hspace{4pt}}w{c}{34pt}"] * 3)
    + "@{}"
)
HEADLINE_HEADER = (
    r"\toprule",
    " & "
    + " & ".join(r"\multicolumn{2}{c}{" + TABLE1_TITLE[s] + "}" for s in TABLE1_SUITES)
    + r" \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
    " & " + " & ".join([r"Baseline & Q\&D"] * len(TABLE1_SUITES)) + r" \\",
)


def _headline_level_pair(got: Mapping[str, Any], row: str, suite: str) -> dict[str, Any]:
    """One (Baseline, Q&D) pair as Table 2 prints it, with its bold flags.

    Baseline is levels[1] (the comparator) and Q&D levels[0] (the trained questioner). A proportion
    prints in percent, one decimal (_pts: the exact-decimal half-up 3-place print with its point
    moved, which is 100x rounded half-up to one decimal); deepest need resolved and breadth at two
    decimals (_fmt); a question count at one. Bold marks the better level only where the record
    decided the cell (cells[*].decided, as _headline_cells read and asserted it; not recomputed here),
    higher being better except on the out-of-order rate. Question counts are never bold.
    """
    _regime, comp, metric = row.split("::")
    if metric == "asks":
        asks = got["asks"][suite]
        return {
            "Baseline": {"text": _fmt(asks[HEADLINE_ASKS_BASELINE[comp]], 1), "bold": False},
            "Q&D": {"text": _fmt(asks["recipe"], 1), "bold": False},
        }
    c = got["cells"][f"{row}::{suite}"]
    qd, base = c["levels"]
    if metric in HEADLINE_POINTS_METRICS:
        text = {"Baseline": _pts(base), "Q&D": _pts(qd)}
    elif metric in ("max_depth_reached", "facet_breadth_scorer"):
        text = {"Baseline": _fmt(base, 2), "Q&D": _fmt(qd, 2)}
    else:
        raise Mismatch(f"{row}: no printing rule for the metric {metric!r}. NOT WRITING.")
    win = None
    if c["decided"]:
        if text["Baseline"] == text["Q&D"]:
            raise Mismatch(
                f"{row}::{suite}: decided, but both levels print {text['Q&D']}; bold would mark "
                "one of two equal numbers. NOT WRITING."
            )
        win = "Q&D" if (qd > base) != (metric in HEADLINE_LOWER_IS_BETTER) else "Baseline"
    return {k: {"text": text[k], "bold": win == k} for k in ("Baseline", "Q&D")}


def table_headline(out: Path) -> list[Path]:
    """The headline table (tab:heldout): on the held-out splits, the level each arm reaches, the
    trained questioner (Q&D; seeds 1 and 2 averaged within the task) beside a prompted comparator
    (Baseline), at equal spend and at its own stop, against the same model and, in the rows that name
    it, against GPT-OSS-120B.

    One tabular for the main text to \\input, in the layout of HEADLINE_LAYOUT. Every level is the
    pair behind a difference of table_app_headline_diffs, read through _headline_cells, which asserts
    each pair differences to its delta to 1e-12. Bold marks the better level where the cell is decided
    (_headline_level_pair); there are no marks or intervals.
    """
    got = _headline_cells()
    printed = {
        f"{row}::{s}": _headline_level_pair(got, row, s)
        for _block, rows in HEADLINE_LAYOUT
        for _label, row in rows
        for s in TABLE1_SUITES
    }
    lines = [
        "% table_headline.tex -- generated by " + TOOL + ", do not hand-edit.",
        "% prov: see table_headline.provenance.json beside this file.",
        HEADLINE_SIZE_NOTE,
        # grouped, so the stretch ends with the table
        r"{\renewcommand{\arraystretch}{" + HEADLINE_STRETCH + "}%",
        r"\begin{tabular}{" + HEADLINE_COLS + "}",
        *HEADLINE_HEADER,
    ]
    for block, rows in HEADLINE_LAYOUT:
        lines += [r"\midrule", r"\multicolumn{7}{@{}l}{\textit{" + block + r"}} \\"]
        for label, row in rows:
            cells = []
            for s in TABLE1_SUITES:
                for arm in ("Baseline", "Q&D"):
                    p = printed[f"{row}::{s}"][arm]
                    cells.append(r"\textbf{" + p["text"] + "}" if p["bold"] else p["text"])
            lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}%"]
    prov = _headline_prov(got, "table_headline")
    prov["build"] = TOOL + " --only table_headline"
    prov["rows"] = [[block, label, row] for block, rows in HEADLINE_LAYOUT for label, row in rows]
    prov["layout"] = (
        "two blocks, equal spend and own stop (at most eight calls); Baseline is the same model, "
        "prompted, except in the rows that name GPT-OSS-120B, where it is the 120B model (cells "
        "equal::teacher and own::teacher, asks_trained_same_teacher teacher); rows at "
        "\\arraystretch " + HEADLINE_STRETCH + " inside a group that closes with the table. Chosen "
        "by the main-text owner on 2026-09-25 over a four-block layout (201.0pt at \\small)"
    )
    prov["printed"] = printed
    prov["printed_units"] = {
        "percent": sorted(HEADLINE_POINTS_METRICS),
        "rule": "Baseline = cells[*].levels[1] (the comparator), Q&D = cells[*].levels[0] (the "
        "trained questioner). Levels of these metrics print in percent, one decimal: the "
        "3-decimal half-up print of the level on its exact decimal with the point moved two "
        "places (_pts), i.e. 100x rounded half-up to one decimal from full precision. Deepest "
        "need resolved and breadth print at two decimals (_fmt), question counts "
        "(asks_trained_same_teacher: Baseline base or teacher, Q&D recipe) at one decimal "
        "(_fmt). `cells` are the records' fractions, unchanged",
    }
    prov["rules"]["bold"] = (
        "bold = the better level of a pair, only where the cell is decided at 50,000 "
        "resamples x 3 (cells[*].decided: all three intervals, seeds 101/202/303, exclude "
        "zero on one side; the flag as the loaders read and asserted it, not recomputed "
        "here). Better = higher, lower for the out-of-order rate (precedence_violation_rate). "
        "Neither level is bold where the cell is not decided; question counts are never bold"
    )
    return _emit_tex(out, "table_headline", "\n".join(lines) + "\n", prov)


def _comparator_cells() -> list[dict[str, Any]]:
    """Every drawn cell, read from the committed contrast records; each record's own locks enforced."""
    cells = []
    for fname, cname, label, group, structured in COMPARATOR_ROWS:
        path = COMPARATOR_DIR / fname
        require(path)
        rec = json.loads(path.read_text())
        lock = rec["published_lock"]
        if not lock or any(v["verdict"] != "LOCKED" for v in lock.values()):
            raise Mismatch(f"{fname}: the reader's published-cell lock did not hold. NOT DRAWING.")
        for arm, by_suite in rec["instrument"].items():
            for suite, by_metric in by_suite.items():
                c = by_metric["evidence_coverage"]
                if (
                    c["differ"]
                    or c["emitted_here_absent_in_store"]
                    or c["in_store_absent_here"]
                    or not c["agree"]
                ):
                    raise Mismatch(
                        f"{fname}: instrument lock failed for {arm}/{suite}. NOT DRAWING."
                    )
        con = rec["contrasts"][cname]
        if "refused" in con:
            raise Mismatch(f"{fname}: {cname} was refused: {con['refused']}. NOT DRAWING.")
        for suite in ("musique", "strategyqa"):
            cell = con[f"evidence_coverage::{suite}"]
            ten = next(r for r in cell["readings"] if r["n_boot"] == 10000)
            cells.append(
                {
                    "record": rel(path),
                    "contrast": cname,
                    "label": label,
                    "group": group,
                    "structured": structured,
                    "suite": suite,
                    "delta": ten["delta"],
                    "ci": [ten["ci_lo"], ten["ci_hi"]],
                    "n": ten["n"],
                    "verdict": cell["verdict"],
                }
            )
    return cells


# Figure 2(a)'s content width on its page, points, as _pinned_page measures it (figsize
# FIG2_PARTS_W_IN, tight_layout's default pad; fig2_frontier_recipe.provenance.json
# placement.content_pt). Part (b) is laid out to the same width, so the two parts, centred on pages of
# one width, start and end at the same x and their bold letters align. If part (a) changes width, the
# alignment test fails. It was 391.7031 at 5.5in.
FIG2_PARTS_CONTENT_W_PT = 344.1831
# Part (b)'s figure height, inches: its content fills the 125.84bp page (MAIN_PAGE_PT) to within a
# point. It was 2.25 (2.2 at the width fit), then 1.85 for the 9-page trim on the 143bp page.
FIG12_H_IN = 1.61
# Part (b)'s rows at 0.88\linewidth (the user, 2026-09-25). Its page is 17.16bp shorter than the 143bp
# one while its type stays 8pt, and a fit that only shrank the panels set the six rows 7.0pt apart, so
# the descenders of "prompted" touched the superscript of "PAR^2-RAG" below them (0.0pt of white at
# 600 dpi; 2.64pt at 396 x 143). Besides the pads above, the height goes back to the rows from the gap
# between the three model groups, in rows (it was 0.6; the groups keep their colours), and the
# two-line x-labels' line spacing (it was the default 1.2).
FIG12_GROUP_GAP = 0.3
FIG12_XLABEL_LINESPACING = 1.1
FIG12_MS, FIG12_MEW = 5.0, 1.2  # the rows' marker size and edge width (points)
FIG12_MARKER_CLEAR_PT = 0.5  # least white between an outer row's marker edge and its panel's edge


def _fit_content_width(fig, target_pt: float, **tight: Any) -> float:
    """tight_layout into a rect inset left and right until the content a tight save cuts (padded by
    savefig.pad_inches, measured as the PDF sets it) is `target_pt` wide, to a hundredth of a point.
    Returns the width reached; refuses if it does not converge."""
    from matplotlib import rcParams

    pad = rcParams["savefig.pad_inches"] * 72
    fig_w = fig.get_figwidth() * 72
    inset = 0.0
    for _ in range(12):
        fig.tight_layout(rect=(inset / fig_w, 0.0, 1.0 - inset / fig_w, 1.0), **tight)
        with _pdf_renderer(fig) as r:
            got = fig.get_tightbbox(r).width * 72 + 2 * pad
        if abs(got - target_pt) < 0.01:
            return got
        inset += (got - target_pt) / 2
    raise Mismatch(f"the content does not settle at {target_pt}pt wide ({got:.3f}). NOT DRAWING.")


def fig12_comparators(out: Path) -> list[Path]:
    """The recipe against plain prompting and a structured retrieval method, at three model scales.

    Recipe (8B, seeds 1+2 averaged within the task) minus each comparator, required-evidence
    coverage at equal retrieval spend under Table 1's symmetric rule. Filled markers: the comparator
    model prompted plainly. Hollow: the same model running PAR2-RAG. A point right of zero is the
    8B recipe ahead.
    """
    cells = _comparator_cells()
    frames = FRAMES_STRUCTURED.exists()
    if frames:
        cells += _frames_comparator_cells()
    plt = _mpl()
    panels = [("musique", "MuSiQue"), ("strategyqa", "StrategyQA")]
    if frames:
        panels.append(("frames", "FRAMES"))
    fig, axes = plt.subplots(1, len(panels), figsize=(FIG2_PARTS_W_IN, FIG12_H_IN), sharey=True)
    groups = ("8B base", "gpt-oss-120b", "Claude Opus 5")
    ypos = {}
    y = 0
    for g in groups:
        for structured in (False, True):
            ypos[(g, structured)] = y
            y -= 1
        y -= FIG12_GROUP_GAP
    # the same model and GPT-OSS-120B keep the colors panel (a) gives them (ARM_STYLE); blue stays
    # the trained questioner's, and Claude Opus 5 takes a fourth Okabe-Ito color
    colors = {
        "8B base": ARM_STYLE["base8b"]["color"],
        "gpt-oss-120b": ARM_STYLE["teacher"]["color"],
        "Claude Opus 5": PALETTE[3],
    }
    for ax, (suite, title) in zip(axes, panels):
        ax.axvline(0.0, color="#444444", lw=0.9)
        for c in (c for c in cells if c["suite"] == suite):
            yy = ypos[(c["group"], c["structured"])]
            col = colors[c["group"]]
            ax.errorbar(
                [c["delta"]],
                [yy],
                xerr=[[c["delta"] - c["ci"][0]], [c["ci"][1] - c["delta"]]],
                fmt="o" if not c["structured"] else "s",
                ms=FIG12_MS,
                color=col,
                lw=1.2,
                capsize=2.0,
                markerfacecolor=col if not c["structured"] else "white",
                markeredgewidth=FIG12_MEW,
            )
        ax.set_title(title, loc="left", pad=FIG2_PARTS_TITLE_PAD_PT)
        ax.grid(axis="y", visible=False)
        # the axis labels at the rcParams' 8pt, as in part (a), so the two parts set every label at
        # one size (the user, round 2: "matching font sizes"; they were 7pt)
        xlabel = {"labelpad": FIG2_PARTS_LABELPAD_PT, "linespacing": FIG12_XLABEL_LINESPACING}
        if suite == "frames":
            fc = [c for c in cells if c["suite"] == "frames"]
            lo = min(0.0, *(c["ci"][0] for c in fc)) - 0.01
            hi = max(0.0, *(c["ci"][1] for c in fc)) + 0.01
            ax.set_xlim(lo, hi)
            ax.set_xlabel("answer correctness,\nown stop", **xlabel)
        else:
            ax.set_xlim(-0.08, 0.17)
            ax.set_xlabel("coverage,\nequal spend", **xlabel)
        # differences in percentage points (the user's comment, 2026-09-24); drawn as fractions
        _axis_in_points(ax.xaxis)
    # The comparator's reader name (G3): the 8B base is "the same model", the 120B model is written
    # GPT-OSS-120B as the main text writes it. The group keys stay the records' own.
    shown = {
        "8B base": "the same model",
        "gpt-oss-120b": "GPT-OSS-120B",
        "Claude Opus 5": "Claude Opus 5",
    }
    labels = {v: f"{shown[g]}, {'PAR$^2$-RAG' if s else 'prompted'}" for (g, s), v in ypos.items()}
    axes[0].set_yticks(list(labels))
    axes[0].set_yticklabels([labels[k] for k in labels])
    fig.supxlabel("the trained questioner minus each comparator, in percentage points", fontsize=8)
    # Laid out at \linewidth to part (a)'s content width, so the two parts print 1:1 on pages of one
    # width and start and end at the same x (the user, round 2: "the same width, with matching font
    # sizes"; the stacked parts read as misaligned). The spacing inside is as before (pad=0.4); at
    # full width the content was 403.3pt and the page was scaled to fit it.
    _fit_content_width(fig, FIG2_PARTS_CONTENT_W_PT, pad=0.4, w_pad=0.8)
    # The top and bottom rows inside their panels. matplotlib's 5% margin does not know marker sizes,
    # and on the 125.84bp page it set the outer markers' edges 0.62pt past the panels' top and bottom,
    # where the panels clip them (0.15pt at 396 x 143). So the limits are set from the panel height
    # the layout gave, each outer marker's edge FIG12_MARKER_CLEAR_PT inside the panel.
    lo_row, hi_row = min(ypos.values()), max(ypos.values())
    panel_pt = axes[0].get_position().height * fig.get_figheight() * 72
    edge_pt = FIG12_MS / 2 + FIG12_MEW / 2 + FIG12_MARKER_CLEAR_PT
    pitch_pt = (panel_pt - 2 * edge_pt) / (hi_row - lo_row)
    axes[0].set_ylim(lo_row - edge_pt / pitch_pt, hi_row + edge_pt / pitch_pt)
    # the second part of the stacked Figure 2, lettered at its top left, in the titles' line
    _panel_label(fig, axes, "(b)")
    prov = provenance(
        asset_id="fig12_comparators",
        sources=[COMPARATOR_DIR / f for f in sorted({r[0] for r in COMPARATOR_ROWS})]
        + ([FRAMES_STRUCTURED] if frames else []),
        query=(
            "contrasts[<name>]['evidence_coverage::<suite>'].readings[n_boot=10000] for the six "
            "recipe-minus-comparator contrasts; each record's published_lock LOCKED and "
            "instrument lock clean asserted before drawing"
        ),
        values=cells,
        scorer_hash=["3f88ac3527ed6c2fcf78b1538f6c44a3c14838aec93a583c072a770fb70e8162"]
        + (json.loads(FRAMES_STRUCTURED.read_text())["reading"]["scorer_hash"] if frames else []),
        graph_version=["v1"],
        extra={
            "colors": colors,
            "rule": "symmetric seed-matched, both arms at min(k_a, k_b), n=200 tasks per suite",
            "recipe": "qwen3-8b-dpo-stacked-notdone-both-s1 and -s2, averaged within the task",
            "one_code_version": "a61c4f4be3e9",
            "one_base_url_sha": "980ca894983a",
            "units": "differences in percentage points: the tick labels are the tick values "
            "moved two places (_pts_tick); `values` are the records' fractions, unchanged",
        },
    )
    return _emit_figure(fig, out, "fig12_comparators", prov)


TAU2_READER = REPO / "artifacts" / "tau2_rerun_20260923" / "final" / "transfer_rerun.json"
TAU2_READER_120B = (
    REPO / "artifacts" / "tau2_rerun_20260923" / "final_120b" / "transfer_rerun_120b.json"
)
# The prompt variants as the appendix tables name them (tab:tau2-trained: prompt base / stop).
TAU2_ROWS = (
    ("tau2_retail::tau2_base", "retail, base prompt"),
    ("tau2_retail::tau2_stop", "retail, stop prompt"),
    ("tau2_airline::tau2_base", "airline, base prompt"),
    ("tau2_airline::tau2_stop", "airline, stop prompt"),
)
# The two comparators Figure 4 draws: its key (ARM_TERM), the ARM_STYLE entry whose color and marker
# it carries in Figure 2, and the comparator model its record must name (rules.comparator_model).
TAU2_COMPARATORS = (
    ("same", "base8b", "qwen3-8b-base"),
    ("teacher", "teacher", "openai/aws/gpt-oss-120b"),
)
TAU2_OFFSET = {"same": 0.2, "teacher": -0.2}  # rows 1 apart; each comparator's marker off its row
TAU2_FOLLOW_UPS = "follow-up user turns"
TAU2_DIRECTION = "trained minus comparator; a negative delta is fewer turns"
TAU2_ARMS = {"treatment_arm": "inquirer_trained", "control_arm": "inquirer_prompted"}
TAU2_LEAD = f"{ARM_TERM['trained']} minus"  # set before the legend entries, on their row
TAU2_NOTE = "fewer turns"
TAU2_NOTE_PT = 7  # the direction note inside the follow-up panel
TAU2_NOTE_CLEAR_PT = 1.5  # least white space between the note (text or arrow) and any drawn mark
TAU2_LEGEND_GAP_PT = 3.0  # between the legend row and the panels


def _tau2_record(path: Path, comparator_model: str) -> dict[str, Any]:
    """One tau2 reader record (scripts/tau2_concordance/transfer_rerun.py), read as the trained
    questioner MINUS the comparator only where the record's own fields say so. Refuses otherwise.

    Declared: rules.primary names the follow-up estimand "trained minus comparator; a negative delta
    is fewer turns", rules.treatment_arm / rules.control_arm are inquirer_trained / inquirer_prompted,
    and rules.comparator_model is the model this slot draws.

    Checked on the numbers, in every (suite, variant, training seed) cell: a success pair delta is
    trained minus control, because guard.task_pair_deltas holds +1 exactly
    guard.pair_level_NOT_QUOTABLE.trained_only times and -1 exactly its control_only times; a
    follow-up pair is [trained, control], because the first entries of
    secondary["follow-up user turns"].task_pair_values average over tasks to its trained_level and
    the second to its control_level, each task_pair_delta is first minus second, and its mean_delta
    is trained_level minus control_level.

    Pooled: each drawn mean_delta (pooled[*].guard.task_level for success,
    pooled[*].secondary["follow-up user turns"] for follow-ups) is the mean over tasks of the seeds'
    task means of those pair deltas (seed_pooling pair_mean), its task_deltas are those means, and
    the follow-up block is the declared primary (pooled[*].task_level) to the last digit.
    """
    require(path)
    rec = json.loads(path.read_text())
    rules, where = rec.get("rules") or {}, rel(path)
    if TAU2_DIRECTION not in rules.get("primary", ""):
        raise Mismatch(
            f"{where}: rules.primary does not declare '{TAU2_DIRECTION}' "
            f"({rules.get('primary')!r}). NOT DRAWING."
        )
    for k, v in TAU2_ARMS.items():
        if rules.get(k) != v:
            raise Mismatch(f"{where}: rules.{k} is {rules.get(k)!r}, not {v!r}. NOT DRAWING.")
    if rules.get("comparator_model") != comparator_model:
        raise Mismatch(
            f"{where}: rules.comparator_model is {rules.get('comparator_model')!r}, not "
            f"{comparator_model!r}. NOT DRAWING."
        )
    cells = {(c["suite"], c["variant"], c["seed_label"]): c for c in rec["cells"]}
    pooled = {f"{p['suite']}::{p['variant']}": p for p in rec["pooled"]}

    def near(a: float, b: float) -> bool:
        return abs(a - b) <= 1e-9

    def mean(xs: Iterable[float]) -> float:
        xs = list(xs)
        return sum(xs) / len(xs)

    rows: dict[str, Any] = {}
    trained_levels: dict[str, float] = {}
    for key, _label in TAU2_ROWS:
        p = pooled.get(key)
        if p is None or p.get("seed_pooling") != "pair_mean":
            raise Mismatch(f"{where}: no pair_mean pooled row for {key}. NOT DRAWING.")
        by_task: dict[str, dict[str, list[float]]] = {"success": {}, "follow_ups": {}}
        for seed in p["seed_labels"]:
            c = cells.get((p["suite"], p["variant"], seed))
            if c is None:
                raise Mismatch(f"{where}: {key} has no cell for seed {seed}. NOT DRAWING.")
            g, pl = c["guard"], c["guard"]["pair_level_NOT_QUOTABLE"]
            flat = [d for v in g["task_pair_deltas"].values() for d in v]
            if (
                flat.count(1.0) != pl["trained_only"]
                or flat.count(-1.0) != pl["control_only"]
                or not near(mean(flat), pl["mean_delta"])
            ):
                raise Mismatch(
                    f"{where}: {key} seed {seed}: the success pair deltas are not trained minus "
                    f"control (+1 x{flat.count(1.0)} against trained_only {pl['trained_only']}, "
                    f"-1 x{flat.count(-1.0)} against control_only {pl['control_only']}). "
                    "NOT DRAWING."
                )
            f = c["secondary"][TAU2_FOLLOW_UPS]
            vals = f["task_pair_values"]
            first = mean(mean(a for a, _b in v) for v in vals.values())
            second = mean(mean(b for _a, b in v) for v in vals.values())
            pairs_ok = all(
                len(v) == len(f["task_pair_deltas"][t])
                and all(near(a - b, d) for (a, b), d in zip(v, f["task_pair_deltas"][t]))
                for t, v in vals.items()
            )
            if not (
                pairs_ok
                and near(first, f["trained_level"])
                and near(second, f["control_level"])
                and near(f["mean_delta"], f["trained_level"] - f["control_level"])
            ):
                raise Mismatch(
                    f"{where}: {key} seed {seed}: the follow-up pairs are not [trained, control] "
                    f"read as trained minus control (first entries average {first}, trained_level "
                    f"{f['trained_level']}; mean_delta {f['mean_delta']}). NOT DRAWING."
                )
            trained_levels[f"{key}::{seed}"] = f["trained_level"]
            for t, v in g["task_pair_deltas"].items():
                by_task["success"].setdefault(t, []).append(mean(v))
            for t, v in f["task_pair_deltas"].items():
                by_task["follow_ups"].setdefault(t, []).append(mean(v))
        fu = p["secondary"][TAU2_FOLLOW_UPS]
        fu = fu.get("task_level", fu)
        prim = p["task_level"]
        if (prim["mean_delta"], prim["printed_interval"]) != (
            fu["mean_delta"],
            fu["printed_interval"],
        ):
            raise Mismatch(
                f"{where}: {key}: the follow-up block is not the declared primary. NOT DRAWING."
            )
        rows[key] = {}
        for metric, block in (("success", p["guard"]["task_level"]), ("follow_ups", fu)):
            means = {t: mean(v) for t, v in by_task[metric].items()}
            pooled_ok = (
                all(len(v) == len(p["seed_labels"]) for v in by_task[metric].values())
                and set(means) == set(block["task_deltas"])
                and all(near(means[t], block["task_deltas"][t]) for t in means)
                and near(mean(means.values()), block["mean_delta"])
            )
            if not pooled_ok:
                raise Mismatch(
                    f"{where}: {key} {metric}: pooled mean_delta {block['mean_delta']} is not the "
                    f"pool of its seeds' trained-minus-control task deltas "
                    f"({mean(means.values())}). NOT DRAWING."
                )
            pi = block["printed_interval"]
            if (pi["n_boot"], pi["seed"]) != (10000, 0):
                raise Mismatch(f"{where}: {key} {metric}: printed interval {pi}. NOT DRAWING.")
            rows[key][metric] = {
                "point": block["mean_delta"],
                "ci_lo": pi["lo"],
                "ci_hi": pi["hi"],
                "n_boot": pi["n_boot"],
                "boot_seed": pi["seed"],
                "n_tasks": block["n_tasks"],
                "interval_excludes_zero": pi["lo"] > 0 or pi["hi"] < 0,
                "decided_50k": (p["guard"]["task_level"] if metric == "success" else prim).get(
                    "decided"
                ),
                "field": (
                    f"pooled[{key}].guard.task_level"
                    if metric == "success"
                    else f"pooled[{key}].secondary['{TAU2_FOLLOW_UPS}']"
                ),
            }
    meta = {
        "path": where,
        "sha256": sha256_file(path),
        "source_sha256": rec["source"]["sha256"],
        "comparator_model": rules["comparator_model"],
        "code_version": rec["provenance"]["code_version"],
        "run_ids_sha256": rec["provenance"]["run_ids_sha256"],
        "trained": [[t["label"], t["model"], t["sha_prefix"]] for t in rules["trained"]],
        "direction": {
            "declared_by": "rules.primary",
            "declared": rules["primary"],
            "arms": f"rules.treatment_arm={rules['treatment_arm']}, "
            f"rules.control_arm={rules['control_arm']}",
            "checked": "per seed cell: guard.task_pair_deltas +1/-1 counts equal "
            "guard.pair_level_NOT_QUOTABLE.trained_only/control_only; secondary['follow-up user "
            "turns'].task_pair_values first/second entries average to trained_level/control_level, "
            "task_pair_deltas = first - second, mean_delta = trained_level - control_level; pooled "
            "mean_delta and task_deltas = the pair_mean pool of those task deltas (1e-9); the "
            "trained arm's per-seed follow-up levels equal in both records",
        },
    }
    return {"rows": rows, "meta": meta, "trained_levels": trained_levels}


def fig13_tau2(out: Path) -> list[Path]:
    """The trained questioner in a customer-service agent: task success and the customer's follow-up
    turns per dialogue, against the same model, prompted, and against GPT-OSS-120B, prompted.

    Redesigned on the user's round-2 comments (INBOX_FOR_PAPER_LANE section 14, item 2, and the
    redirect after the interim PDF: at most 110bp tall). Two panels, never a second y-axis: task
    success difference in percentage points (left) and the follow-up turns difference, in turns
    (right, the declared primary endpoint, its direction marked). One row per suite and prompt
    variant, two markers per row, offset: trained minus the same model, prompted, in its color and
    shape from Figure 2 (final/transfer_rerun.json), and trained minus GPT-OSS-120B, prompted, in
    its (final_120b/transfer_rerun_120b.json). Each record's pooled reading (training seeds pooled
    within the task, BCa over tasks, 10,000 resamples, seed 0), read as trained minus comparator
    only after _tau2_record has checked it from the record's own fields. No levels, no suptitle, no
    questions panel, nothing per successful task (per-dialogue turns is the measure the rules
    declare). paper_levels.json is no longer read: it carried the success levels, which are gone.
    """
    from matplotlib.ticker import MultipleLocator

    got = {
        who: _tau2_record(TAU2_READER if who == "same" else TAU2_READER_120B, model)
        for who, _style, model in TAU2_COMPARATORS
    }
    # The trained arm is one set of runs in both records: its checkpoints and its per-seed follow-up
    # levels agree, so the column read as "trained" is the same column in both.
    if (
        got["same"]["trained_levels"] != got["teacher"]["trained_levels"]
        or got["same"]["meta"]["trained"] != got["teacher"]["meta"]["trained"]
    ):
        raise Mismatch(
            "the two tau2 records do not share their trained arm (checkpoints or per-seed "
            "follow-up levels differ). NOT DRAWING."
        )
    rows = {
        key: {"label": label, **{who: got[who]["rows"][key] for who in got}}
        for key, label in TAU2_ROWS
    }

    plt = _mpl()
    page_w, page_h = MAIN_PAGE_PT["fig13_tau2"]
    fig, axes = plt.subplots(1, 2, figsize=(LINEWIDTH_IN, page_h / 72), sharey=True)
    ys = {key: -i for i, (key, _) in enumerate(TAU2_ROWS)}
    xlabels = ("task success, percentage points", "follow-up turns per dialogue")
    for ax, metric, xlabel in zip(axes, ("success", "follow_ups"), xlabels):
        ax.axvline(0.0, color="#444444", lw=0.9, zorder=1)
        for who, style_key, _model in TAU2_COMPARATORS:
            st = ARM_STYLE[style_key]
            for key, _ in TAU2_ROWS:
                c = rows[key][who][metric]
                ax.errorbar(
                    [c["point"]],
                    [ys[key] + TAU2_OFFSET[who]],
                    xerr=[[c["point"] - c["ci_lo"]], [c["ci_hi"] - c["point"]]],
                    fmt=st["marker"],
                    ms=st["ms"],
                    color=st["color"],
                    lw=1.0,
                    capsize=1.5,
                    zorder=3,
                )
        lo = min(rows[k][w][metric]["ci_lo"] for k in rows for w in got)
        hi = max(rows[k][w][metric]["ci_hi"] for k in rows for w in got)
        pad = 0.06 * (hi - lo)
        ax.set_xlim(min(lo, 0.0) - pad, max(hi, 0.0) + pad)
        ax.set_xlabel(xlabel, labelpad=2.0)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", length=2.5, pad=2.0)
        ax.tick_params(axis="y", length=0, pad=3.0)
    # success in percentage points (the drawn values stay the records' fractions; only the tick
    # labels move their point), follow-up turns as counts on whole-turn ticks
    _axis_in_points(axes[0].xaxis)
    axes[1].xaxis.set_major_locator(MultipleLocator(1))
    axes[0].set_yticks([ys[k] for k, _ in TAU2_ROWS])
    axes[0].set_yticklabels([lab for _, lab in TAU2_ROWS])
    axes[0].set_ylim(ys[TAU2_ROWS[-1][0]] - 0.5, 0.5)
    # One legend row above the panels: the trained questioner, then what it is set against.
    handles = [
        plt.Line2D(
            [],
            [],
            color=ARM_STYLE[s]["color"],
            marker=ARM_STYLE[s]["marker"],
            ms=ARM_STYLE[s]["ms"],
            lw=1.0,
        )
        for _who, s, _model in TAU2_COMPARATORS
    ]
    legend = fig.legend(
        handles,
        [ARM_TERM[who] for who, _s, _model in TAU2_COMPARATORS],
        loc="upper left",
        ncol=2,
        frameon=False,
        borderpad=0.0,
        borderaxespad=0.0,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    lead = fig.text(0.0, 0.0, TAU2_LEAD, ha="left", va="center", fontsize=8)
    layout = _tau2_layout(fig, axes, legend, lead)
    note = _tau2_direction_note(fig, axes[1])
    prov = provenance(
        asset_id="fig13_tau2",
        sources=[TAU2_READER, TAU2_READER_120B],
        query=(
            "per record: pooled[<suite>::<variant>].guard.task_level (task success) and "
            "pooled[...].secondary['follow-up user turns'] (follow-up user turns per dialogue, "
            "equal to pooled[...].task_level, the declared primary): mean_delta and "
            "printed_interval (BCa over tasks, 10,000 resamples, seed 0; training seeds pooled "
            "within the task, pair_mean); each read as trained minus comparator only after "
            "_tau2_record checked the record's own fields (values.records[*].direction)"
        ),
        values={"rows": rows, "records": {who: got[who]["meta"] for who in got}},
        extra={
            "n": "25 retail and 17 airline tasks; 1,224 units in the same-model record and 1,632 "
            "in the 120B record (its 408 comparator units added), one code version ee9afda2",
            "prompt_labels": "base prompt = tau2_base, stop prompt = tau2_stop",
            "comparators": {
                who: {"legend": ARM_TERM[who], "style": s, "model": model}
                for who, s, model in TAU2_COMPARATORS
            },
            "legend_lead": TAU2_LEAD,
            "offsets": TAU2_OFFSET,
            "xlabels": list(xlabels),
            "layout": layout,
            "direction_note": note,
            "not_drawn": "questions per dialogue and the success levels (both in the prose); "
            "follow-up turns per successful task (neither reader declared or computed it)",
            "units": "task success difference in percentage points: the tick labels are the "
            "tick values moved two places (_pts_tick); `values` are the records' fractions, "
            "unchanged. Follow-up turns are counts per dialogue",
        },
    )
    return _emit_figure(fig, out, "fig13_tau2", prov)


def _tau2_layout(fig, axes, legend, lead) -> dict[str, Any]:
    """Lay Figure 4 out in its short page: the panels by tight_layout below one legend row, the
    row's lead text and legend entries centred over the figure together, the lead's centre on the
    entries' centre line, and the panels' top TAU2_LEGEND_GAP_PT below the row. Measured as the PDF
    sets it (points)."""
    pad = 1.0  # tight_layout's pad, in font sizes: 8pt at every edge, the legend row's included
    with _pdf_renderer(fig) as r:
        w_pt, h_pt = fig.bbox.width, fig.bbox.height
        row_h = legend.get_window_extent(r).height
        lead_w = lead.get_window_extent(r).width
        leg_w = legend.get_window_extent(r).width
    gap = legend.columnspacing * 8.0  # the lead stands off the first entry as the entries do
    x0 = (w_pt - (lead_w + gap + leg_w)) / 2
    top = h_pt - pad * 8.0
    legend.set_bbox_to_anchor(((x0 + lead_w + gap) / w_pt, top / h_pt))  # figure fractions
    fig.tight_layout(pad=pad, rect=(0.0, 0.0, 1.0, (top - row_h - TAU2_LEGEND_GAP_PT) / h_pt))
    with _pdf_renderer(fig) as r:
        entry = legend.get_texts()[0].get_window_extent(r)
        over = max(ax.get_tightbbox(r).y1 - ax.get_window_extent(r).y1 for ax in axes)
        want = legend.get_window_extent(r).y0 - TAU2_LEGEND_GAP_PT - over
    lead.set_position(((x0) / w_pt, (entry.y0 + entry.y1) / 2 / h_pt))
    fig.subplots_adjust(top=want / h_pt)
    return {
        "legend_row_pt": round(row_h, 3),
        "lead_width_pt": round(lead_w, 3),
        "legend_width_pt": round(leg_w, 3),
        "gap_to_panels_pt": TAU2_LEGEND_GAP_PT,
    }


def _tau2_direction_note(fig, ax) -> dict[str, Any]:
    """Mark the follow-up panel's direction: "fewer turns" with an arrow under it pointing toward
    negative, set in the panel where it is TAU2_NOTE_CLEAR_PT clear of every drawn mark (points,
    bars, caps, the zero line), as near the panel's lower left corner as that allows. Placed on the
    PDF renderer's numbers (points); refused if no place fits."""
    import matplotlib.text as mtext
    import numpy as np

    text = ax.text(
        0.0,
        0.0,
        TAU2_NOTE,
        ha="left",
        va="bottom",
        fontsize=TAU2_NOTE_PT,
        color="#444444",
        zorder=4,
    )
    arrow = ax.annotate(
        "",
        xy=(0.0, 0.0),
        xytext=(1.0, 0.0),
        arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8, shrinkA=0.0, shrinkB=0.0),
        zorder=4,
    )
    with _pdf_renderer(fig) as r:
        to_data = ax.transData.inverted().transform
        panel = ax.get_window_extent(r)
        tb = mtext.Text.get_window_extent(text, r)
        tw, th = tb.width, tb.height
        head = 2.0  # the arrow's half height (its head), points
        block_h = th + 1.0 + 2 * head  # text, a point of air, then the arrow line and its head
        segments, marks = _drawn_marks(ax)
        pts = np.concatenate(
            [
                np.linspace(p, q, max(2, int(np.hypot(*(np.subtract(q, p))) * 2) + 1))
                for p, q in segments
            ]
        )
        mk = np.array([m.extents for m in marks]).reshape(-1, 4)
        c = TAU2_NOTE_CLEAR_PT
        best = None
        for dy in np.arange(c, panel.height - block_h - c, 0.5):
            for dx in np.arange(c, panel.width - tw - c, 0.5):
                x0, y0 = panel.x0 + dx, panel.y0 + dy
                e = (x0 - c, y0 - c, x0 + tw + c, y0 + block_h + c)
                hit = (
                    (pts[:, 0] >= e[0])
                    & (pts[:, 0] <= e[2])
                    & (pts[:, 1] >= e[1])
                    & (pts[:, 1] <= e[3])
                ).any() or (
                    (mk[:, 0] <= e[2])
                    & (mk[:, 2] >= e[0])
                    & (mk[:, 1] <= e[3])
                    & (mk[:, 3] >= e[1])
                ).any()
                if not hit:
                    best = (x0, y0)
                    break
            if best is not None:
                break
        if best is None:
            raise Mismatch(
                "no place in the follow-up panel sets the direction note clear. NOT DRAWING."
            )
        x0, y0 = best
        y_arrow = y0 + head
        text.set_position(to_data((x0, y_arrow + head + 1.0)))
        arrow.xy = tuple(to_data((x0, y_arrow)))
        arrow.set_position(tuple(to_data((x0 + tw, y_arrow))))
        if to_data((x0 + tw, 0.0))[0] >= 0.0:
            raise Mismatch("the direction note's arrow does not lie left of zero. NOT DRAWING.")
    return {
        "text": TAU2_NOTE,
        "text_xy": [float(v) for v in text.get_position()],
        "arrow_tail_xy": [float(v) for v in arrow.xyann],
        "arrow_head_xy": [float(v) for v in arrow.xy],
        "fontsize_pt": TAU2_NOTE_PT,
        "clearance_pt": TAU2_NOTE_CLEAR_PT,
    }


CHAIN_LADDER_RECORD = REPO / "artifacts" / "two_axis_learned_arm_20260920" / "RESULT.md"

# The three constructions, read from the "All three shapes at 200 tasks" section of the record.
# Each shape has its OWN corpus, gold root, store and scorer_hash, so the three are three
# separate readings and never one pooled n. `depth` is the required chain length the shape is
# built to; `facets` is its breadth denominator.
CHAIN_LADDER = (
    # facets, depth, matched_k, delta, ci_lo, ci_hi, trained_wins, base_wins, ties,
    # trained_deepest, base_deepest, scorer_hash
    (2, 4, 4.43, 0.0391, 0.0281, 0.0506, 55, 2, 143, 2.06, 2.32, "745df1227f0fd097"),
    (3, 6, 5.04, 0.0147, 0.0103, 0.0194, 47, 2, 151, 1.88, 2.12, "8397616fad028b60"),
    (4, 8, 5.59, 0.0068, 0.0046, 0.0092, 42, 0, 158, 1.65, 2.02, "bbe5ce9abe7bc39e"),
)


def _lock_chain_ladder() -> str:
    """Every drawn value must appear verbatim in the committed record.

    This record carries no JSON, so the lock is textual: each number is searched for as a
    formatted string in the record's own prose. That is weaker than recomputing from a store and
    is stated as such in the provenance. It is not vacuous, because a transcription slip in any
    single field fails it -- which is the failure mode a hand-copied figure actually has.
    """
    require(CHAIN_LADDER_RECORD)
    text = CHAIN_LADDER_RECORD.read_text()
    missing = []
    for facets, depth, k, d, lo, hi, tw, bw, ties, td, bd, sh in CHAIN_LADDER:
        for field, needle in (
            ("shape", f"${facets} \\times {depth}$"),
            ("matched_k", f"${k:.2f}$"),
            ("delta", f"$+{d:.4f}$"),
            ("interval", f"$[+{lo:.4f}, +{hi:.4f}]$"),
            ("split", f"{tw} / {bw}"),
            ("ties", f"{ties}$"),
            ("trained_deepest", f"${td:.2f}$"),
            ("base_deepest", f"${bd:.2f}$"),
            ("scorer_hash", sh),
        ):
            if needle not in text:
                missing.append(f"{facets}x{depth}.{field}: {needle!r}")
    if missing:
        raise Mismatch(
            f"{rel(CHAIN_LADDER_RECORD)} does not contain " + "; ".join(missing) + ". NOT WRITING."
        )
    # The claim the figure makes in its shape, asserted rather than eyeballed off the drawing.
    deltas = [row[3] for row in CHAIN_LADDER]
    if not (deltas[0] > deltas[1] > deltas[2] > 0):
        raise Mismatch(f"the advantage is not monotone decreasing and positive: {deltas}")
    if any(row[4] <= 0 for row in CHAIN_LADDER):
        raise Mismatch("a lower bound does not exclude zero, so the panel's claim is wrong")
    # NOT asserted: that the deepest-need GAP widens monotonically. It does not -- it is 0.26,
    # 0.24, 0.37 -- and an earlier version of this figure claimed it did. What is true is that
    # both arms reach absolutely shallower needs as the chain lengthens and the trained arm sits
    # below the comparator at every length, which is what panel (b) now says.
    trained = [row[9] for row in CHAIN_LADDER]
    base = [row[10] for row in CHAIN_LADDER]
    if not (trained[0] > trained[1] > trained[2]) or not (base[0] > base[1] > base[2]):
        raise Mismatch(f"deepest need is not monotone decreasing: {trained} / {base}")
    if any(tr >= ba for tr, ba in zip(trained, base)):
        raise Mismatch(
            f"the trained arm is not below the comparator at every length: {trained} / {base}"
        )
    return text


TWO_AXIS_STABILITY = REPO / "artifacts" / "two_axis_recipe_20260923" / "stability.json"


def _chain_ladder_recipe() -> dict[str, Any]:
    """Coverage and breadth at equal spend, per chain length, recipe and seed 0 on the SAME corpora.

    Both readings come from one record (scripts/two_axis_recipe/stability.py), which re-reads every
    contrast of the published two-axis analysis after locking all 48 of them to the analysis records at
    their own 2,000 resamples. The figure asserts that lock, then the shape the text states: coverage
    decided and decaying with chain length for the recipe; breadth decided at the two shorter chains and
    not at the longest; seed 0's breadth decided NEGATIVE at the longest.
    """
    require(TWO_AXIS_STABILITY)
    st = json.loads(TWO_AXIS_STABILITY.read_text())
    bad = [k for k, v in st["lock"].items() if not v["ok"]]
    if bad or len(st["lock"]) != 48:
        raise Mismatch(f"{rel(TWO_AXIS_STABILITY)}: lock failed on {bad[:4]}. NOT DRAWING.")
    cells: dict[str, Any] = {}
    for lab in ("rec", "s0"):
        for metric in ("evidence_coverage", "facet_breadth_scorer"):
            for shape, _f, _d in TWO_AXIS_SHAPES:
                c = st["cells"][f"{shape}/{lab}/{metric}"]
                r = c["n10000_s0"]
                cells[f"{lab}/{metric}/{shape}"] = {
                    "delta": r["delta"],
                    "ci_lo": r["ci_lo"],
                    "ci_hi": r["ci_hi"],
                    "decided": c["decided"],
                }
    rc = [cells[f"rec/evidence_coverage/{s}"] for s, _f, _d in TWO_AXIS_SHAPES]
    if not all(c["decided"] and c["delta"] > 0 for c in rc) or not (
        rc[0]["delta"] > rc[1]["delta"] > rc[2]["delta"]
    ):
        raise Mismatch("the recipe's coverage advantage is not decided and decaying. NOT DRAWING.")
    rb = [cells[f"rec/facet_breadth_scorer/{s}"] for s, _f, _d in TWO_AXIS_SHAPES]
    if not (rb[0]["decided"] and rb[1]["decided"] and not rb[2]["decided"]):
        raise Mismatch("the recipe's breadth pattern is not decided/decided/tie. NOT DRAWING.")
    s0b = cells["s0/facet_breadth_scorer/4x8"]
    if not (s0b["decided"] and s0b["delta"] < 0):
        raise Mismatch("seed 0's breadth loss at the longest chain is not decided. NOT DRAWING.")
    return cells


def fig4_chain_ladder_recipe(out: Path) -> list[Path]:
    """The advantage per question decays with chain length. Seed 0 is locked, drawn only if SHOW_SEED0."""
    cells = _chain_ladder_recipe()
    drawn = (("rec", -0.12), ("s0", 0.12)) if SHOW_SEED0 else (("rec", 0.0),)
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(LINEWIDTH_IN, 2.2))
    depths = [d for _s, _f, d in TWO_AXIS_SHAPES]
    style = {
        "rec": dict(color=PALETTE[0], marker="o", label="trained questioner (seeds 1, 2)"),
        "s0": dict(color=PALETTE[0], marker="o", label="seed 0 (resumed run)"),
    }
    panels = (
        ("evidence_coverage", "(a) coverage at equal spend"),
        ("facet_breadth_scorer", "(b) breadth at equal spend"),
    )
    for ax, (metric, title) in zip(axes, panels):
        ax.axhline(0.0, color=GREY, lw=0.8, ls="--")
        for lab, off in drawn:
            pts = [cells[f"{lab}/{metric}/{s}"] for s, _f, _d in TWO_AXIS_SHAPES]
            xs = [d + off for d in depths]
            ys = [p["delta"] for p in pts]
            ax.plot(xs, ys, color=style[lab]["color"], lw=1.0, ls="-" if lab == "rec" else ":")
            for x, p in zip(xs, pts):
                filled = lab == "rec"
                ax.errorbar(
                    [x],
                    [p["delta"]],
                    yerr=[[p["delta"] - p["ci_lo"]], [p["ci_hi"] - p["delta"]]],
                    marker="o",
                    ms=4.8,
                    mew=1.2,
                    lw=1.0,
                    capsize=2.0,
                    color=PALETTE[0],
                    mfc=PALETTE[0] if filled else "white",
                    zorder=3,
                )
        ax.set_xticks(depths)
        ax.set_xticklabels([f"{f} lines,\ndepth {d}" for _s, f, d in TWO_AXIS_SHAPES])
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("trained minus the same\nweights prompted")
    if SHOW_SEED0:  # one series needs no legend: the axis label names it
        handles = [
            plt.Line2D([], [], color=PALETTE[0], marker="o", mfc=PALETTE[0], lw=1.0, ls="-"),
            plt.Line2D([], [], color=PALETTE[0], marker="o", mfc="white", lw=1.0, ls=":"),
        ]
        fig.legend(
            handles,
            [style["rec"]["label"], style["s0"]["label"]],
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, -0.1),
        )
    fig.tight_layout(w_pad=1.2)
    prov = provenance(
        asset_id="fig4_chain_ladder_recipe",
        sources=[TWO_AXIS_STABILITY],
        query=(
            "cells[<shape>/<rec|s0>/<evidence_coverage|facet_breadth_scorer>].n10000_s0 and "
            ".decided from artifacts/two_axis_recipe_20260923/stability.json"
        ),
        values=cells,
        scorer_hash=TWO_AXIS_SCORER.values(),
        graph_version=["v1"],
        extra={
            "build": "scripts/paper_figures_iclr.py --only fig4_recipe",
            "rule": "both arms at the lower of their two question counts, pairing at the task (the "
            "published two-axis analysis), 10,000 resamples; decided = every 50,000-resample "
            "interval under seeds 101/202/303 excludes zero",
            "corpora": "the rebuilt constructed corpora of artifacts/two_axis_matched_20260922, one "
            "code version 1979fb7ef321; seed 0 and the recipe are read against the same prompted runs",
            "replaces": "fig4_chain_ladder (seed 0 on the FIRST build, "
            "artifacts/two_axis_learned_arm_20260920), whose panel (b) does not hold for the recipe",
            "drawn": [lab for lab, _off in drawn],
        },
    )
    return _emit_figure(fig, out, "fig4_chain_ladder_recipe", prov)


def fig4_chain_ladder(out: Path) -> list[Path]:
    """The advantage is monotone in the required chain length, and the reason is visible."""
    _lock_chain_ladder()
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(LINEWIDTH_IN, 2.15))

    depths = [row[1] for row in CHAIN_LADDER]
    deltas = [row[3] for row in CHAIN_LADDER]
    los = [row[3] - row[4] for row in CHAIN_LADDER]
    his = [row[5] - row[3] for row in CHAIN_LADDER]

    ax = axes[0]
    ax.axhline(0.0, color=GREY, lw=0.8, ls="--")
    ax.errorbar(
        depths,
        deltas,
        yerr=[los, his],
        marker="o",
        ms=4.5,
        lw=1.2,
        capsize=2.5,
        color=PALETTE[0],
        zorder=3,
    )
    for facets, depth, _k, d, _lo, _hi, tw, bw, _ties, _td, _bd, _sh in CHAIN_LADDER:
        ax.annotate(
            f"{tw}/{bw}",
            (depth, d),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=7,
            color=GREY,
        )
    ax.set_xticks(depths)
    ax.set_xlabel("required chain length")
    ax.set_ylabel("coverage advantage\nat equal questions")
    ax.set_title("(a) the advantage decays with depth", loc="left")
    ax.set_ylim(-0.004, 0.062)

    ax = axes[1]
    ax.plot(
        depths,
        [row[10] for row in CHAIN_LADDER],
        marker="s",
        ms=4.0,
        lw=1.2,
        color=PALETTE[1],
        label="prompted comparator",
    )
    ax.plot(
        depths,
        [row[9] for row in CHAIN_LADDER],
        marker="o",
        ms=4.0,
        lw=1.2,
        color=PALETTE[0],
        label="trained questioner",
    )
    for depth, td, bd in [(row[1], row[9], row[10]) for row in CHAIN_LADDER]:
        ax.annotate(
            "",
            xy=(depth, bd),
            xytext=(depth, td),
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7, ls=":"),
        )
    ax.set_xticks(depths)
    ax.set_xlabel("required chain length")
    ax.set_ylabel("deepest need reached")
    ax.set_title("(b) both reach shallower, trained lower throughout", loc="left")
    lo = min(row[9] for row in CHAIN_LADDER)
    hi = max(row[10] for row in CHAIN_LADDER)
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.legend(loc="upper right", frameon=False, handlelength=1.4, borderaxespad=0.1)

    fig.tight_layout(pad=0.4)
    prov = {
        "asset": "fig4_chain_ladder",
        "tool": TOOL,
        "inputs": [{"path": rel(CHAIN_LADDER_RECORD), "sha256": sha256_file(CHAIN_LADDER_RECORD)}],
        "rows": [
            {
                "shape": f"{r[0]}x{r[1]}",
                "matched_k": r[2],
                "delta": r[3],
                "ci": [r[4], r[5]],
                "trained_better": r[6],
                "base_better": r[7],
                "ties": r[8],
                "trained_deepest": r[9],
                "base_deepest": r[10],
                "scorer_hash": r[11],
            }
            for r in CHAIN_LADDER
        ],
        "graph_version": "v1",
        "n_tasks_per_shape": 200,
        "n_runs_per_shape": 800,
        "resamples": 50000,
        "suite_role": "calibration",
        "lock": (
            "textual: every drawn value is asserted present verbatim in the committed record, "
            "and the figure's two claims (monotone positive advantage, widening deepest-need "
            "gap) are asserted over the table rather than read off the drawing. The record "
            "carries no JSON, so this is weaker than recomputing from a store and is not "
            "represented as more."
        ),
        "caveats": (
            "synth carries the role calibration in `pi suites map`, NOT an endpoint, so no cell "
            "here is a held-out result. Each shape has its own corpus, gold root, store and "
            "scorer_hash, so the three readings are never pooled into one n. The quantity is "
            "lattice-valued and the claim is about the 57, 49 and 42 tasks where the two arms "
            "differ at all. Unmatched, the trained arm loses every axis in every shape while "
            "asking about half as many questions, which is the ask-scaling confound and not a "
            "quality reading."
        ),
    }
    return _emit_figure(fig, out, "fig4_chain_ladder", prov)


TWO_AXIS = REPO / "artifacts/two_axis_matched_20260922"
# The recipe (training seeds 1 and 2) on the same three shapes: same code (1979fb7ef321), base URL,
# frozen roles, corpora and caches as the campaign above; only the questioner differs. Each shape's
# analysis_rec.json is the published analysis script run on a farm of the ORIGINAL prompted and corner
# runs plus the recipe's 800 runs, so its non-trained levels must equal the published ones exactly.
TWO_AXIS_RECIPE = REPO / "artifacts/two_axis_recipe_20260923"
# (facets, chain depth) per constructed shape. Built at D = 2F so the breadth-first and the
# depth-first corner policies spend the same budget, which is what makes their coverage equal.
TWO_AXIS_SHAPES = (("2x4", 2, 4), ("3x6", 3, 6), ("4x8", 4, 8))
# Read from each shape's scored store (scores.parquet, one distinct value each). The analysis
# records do not carry it, and rule 1 says a drawn number without one is not a result.
TWO_AXIS_SCORER = {
    "2x4": "745df1227f0fd0971f8baac147e13cb35fc9616422a172dc698f5c24b33a54fc",
    "3x6": "8397616fad028b6048251f23cac6af71d828bf37b078d53ab320e21a86fc6b2e",
    "4x8": "bbe5ce9abe7bc39eba539e5d5a2549871e387b6b0b9c5ff6a6955732042cab7c",
}
# What artifacts/two_axis_matched_20260922/RESULT.md prints: breadth, deepest need, coverage.
# The figure must reproduce these from the records that RESULT.md was written from, or it is
# drawing something other than what the paper states.
TWO_AXIS_PUBLISHED = {
    "2x4": {
        "shallow_wide": (2.000, 1.000, 0.5000),
        "deep_narrow": (1.000, 3.000, 0.5000),
        "inquirer_prompted": (1.577, 2.317, 0.6150),
        "inquirer_trained": (1.510, 2.078, 0.5784),
    },
    "3x6": {
        "shallow_wide": (3.000, 1.000, 0.3333),
        "deep_narrow": (1.000, 5.000, 0.3333),
        "inquirer_prompted": (1.890, 2.125, 0.3301),
        "inquirer_trained": (1.518, 1.817, 0.2899),
    },
    "4x8": {
        "shallow_wide": (4.000, 1.000, 0.2500),
        "deep_narrow": (1.000, 7.000, 0.2500),
        "inquirer_prompted": (1.917, 2.027, 0.2051),
        "inquirer_trained": (1.405, 1.683, 0.1845),
    },
}


def fig11_axes_separable(out: Path) -> list[Path]:
    """The two axes are separable, and neither learned questioner separates along them.

    One panel per constructed shape. Each axis is normalised to what the shape makes available:
    breadth as a share of the independent lines, the deepest need reached as a share of the
    deepest level. The two corner policies sit at opposite corners of that plane AND reach
    identical coverage, so coverage cannot tell them apart while breadth and depth can. That is
    what licenses reading the learned arms' positions as a fact about the policies rather than
    about an instrument without range. Both learned arms sit in the interior at every shape.

    LOCKS, each a refusal: every drawn level must reproduce RESULT.md to its printed precision,
    and the two corners' coverage must be equal within every shape. The second is a property of
    the construction, so a violation means the records are not the ones the claim is about.
    """
    plt = _mpl()
    arms = ("shallow_wide", "deep_narrow", "inquirer_prompted", "inquirer_trained")
    field = {"breadth": "facet_breadth_scorer", "depth": "max_depth_reached"}
    sources, values = [], {}
    for shape, _f, _d in TWO_AXIS_SHAPES:
        src = TWO_AXIS / f"analysis_{shape}.json"
        sources.append(src)
        levels = json.loads(src.read_text())["corner_levels"]
        values[shape] = {}
        for arm in arms:
            lv = levels[arm]
            b, d, c = lv[field["breadth"]], lv[field["depth"]], lv["evidence_coverage"]
            pb, pd, pc = TWO_AXIS_PUBLISHED[shape][arm]
            for name, got, want, tol in (
                ("breadth", b, pb, 5e-4),
                ("deepest need", d, pd, 5e-4),
                ("coverage", c, pc, 5e-5),
            ):
                if abs(got - want) > tol:
                    raise Mismatch(
                        f"{shape}/{arm}: {name} {got} does not reproduce RESULT.md {want}"
                    )
            values[shape][arm] = {"breadth": b, "deepest_need": d, "coverage": c}
        cw, cd = values[shape]["shallow_wide"]["coverage"], values[shape]["deep_narrow"]["coverage"]
        if abs(cw - cd) > 1e-9:
            raise Mismatch(
                f"{shape}: corner coverage {cw} != {cd}; the construction's premise fails"
            )
        rsrc = TWO_AXIS_RECIPE / shape / "analysis_rec.json"
        require(rsrc)
        sources.append(rsrc)
        rec = json.loads(rsrc.read_text())
        for arm in ("shallow_wide", "deep_narrow", "inquirer_prompted"):
            if rec["corner_levels"][arm] != levels[arm]:
                raise Mismatch(
                    f"{shape}/{arm}: the recipe farm's level differs from the published analysis, "
                    "so the farm does not hold the published runs. NOT DRAWING."
                )
        if rec["n_runs_by_arm"]["inquirer_trained"] != 800:
            raise Mismatch(
                f"{shape}: recipe has {rec['n_runs_by_arm']['inquirer_trained']} runs, not 800"
            )
        lv = rec["corner_levels"]["inquirer_trained"]
        values[shape]["recipe"] = {
            "breadth": lv[field["breadth"]],
            "deepest_need": lv[field["depth"]],
            "coverage": lv["evidence_coverage"],
            "n_asks_mean": lv["n_asks_mean"],
        }

    style = {
        "shallow_wide": dict(
            marker="s", mfc="white", mec=GREY, color=GREY, label="breadth-first corner"
        ),
        "deep_narrow": dict(
            marker="^", mfc="white", mec=GREY, color=GREY, label="depth-first corner"
        ),
        "inquirer_prompted": dict(
            marker="o",
            mfc=PALETTE[1],
            mec="white",
            color=PALETTE[1],
            label=ARM_TERM["same"],
        ),
        "inquirer_trained": dict(
            marker="o", mfc="white", mec=PALETTE[0], color=PALETTE[0], label="seed 0 (resumed run)"
        ),
        "recipe": dict(
            marker="o",
            mfc=PALETTE[0],
            mec="white",
            color=PALETTE[0],
            label=ARM_TERM["trained"],
        ),
    }
    drawn = tuple(a for a in arms if SHOW_SEED0 or a != "inquirer_trained") + ("recipe",)
    fig, axes = plt.subplots(1, 3, figsize=(LINEWIDTH_IN, 2.25), sharey=True)
    for ax, (shape, f, d) in zip(axes, TWO_AXIS_SHAPES):
        pts = {
            a: (values[shape][a]["breadth"] / f, values[shape][a]["deepest_need"] / (d - 1))
            for a in drawn
        }
        (x0, y0), (x1, y1) = pts["shallow_wide"], pts["deep_narrow"]
        ax.plot([x0, x1], [y0, y1], ls="--", lw=0.8, color=GREY, zorder=1)
        for a in drawn:
            x, y = pts[a]
            ax.plot(
                [x],
                [y],
                ls="none",
                ms=6.5,
                mew=1.4,
                zorder=3,
                **{k: v for k, v in style[a].items() if k != "label"},
            )
        cov = values[shape]["shallow_wide"]["coverage"]
        # Label the coverage AT each corner, so the same number sits at both ends of the line --
        # which is the claim. Offsets point into empty space at every shape (checked by render).
        ax.annotate(
            f"cov. {cov:.3f}",
            (x1, y1),
            xytext=(7, -2),
            textcoords="offset points",
            ha="left",
            va="center",
            color=INK,
        )
        ax.annotate(
            f"cov. {cov:.3f}",
            (x0, y0),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=INK,
        )
        ax.set_title(f"{f} lines, depth {d}")
        ax.set_xlim(0, 1.08)
        ax.set_ylim(0, 1.08)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("breadth reached (share of independent lines)")
    axes[0].set_ylabel("deepest need reached\n(share of chain depth)")
    handles = [plt.Line2D([], [], ls="none", ms=6.5, mew=1.4, **style[a]) for a in drawn]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.13),
        handletextpad=0.3,
        columnspacing=1.0,
    )
    fig.subplots_adjust(wspace=0.28)

    prov = provenance(
        asset_id="fig11_axes_separable",
        sources=sources,
        query=(
            "corner_levels from each shape's analysis record: facet_breadth_scorer (the COUNT the "
            "scorer emits), max_depth_reached and evidence_coverage per arm; x = breadth / facets, "
            "y = deepest need / (depth - 1)"
        ),
        values=values,
        scorer_hash=TWO_AXIS_SCORER.values(),
        graph_version=["v1"],
        extra={
            "locks": "every level reproduces artifacts/two_axis_matched_20260922/RESULT.md to its "
            "printed precision, and the two corners' coverage is equal within every shape",
            "unit": "arm means over 400 runs (200 tasks x 2 seeds) per arm per shape",
            "unmatched": "these are per-arm LEVELS, not matched-cost contrasts. The corners spend "
            "exactly 4/6/8 questions; the learned arms spend their own amounts (prompted 9.62/8.80/"
            "8.46, recipe 5.34/7.44/8.41, seed 0 4.65/5.22/5.92), so a learned arm's position relative to the corner line "
            "reflects spend as well as policy. The paired matched-cost contrast is in Section 2.",
            "code_version": "1979fb7ef321 for all three shapes",
            "recipe": "artifacts/two_axis_recipe_20260923/<shape>/analysis_rec.json: the published analysis on a farm of the ORIGINAL prompted and corner runs plus 800 runs of seeds 1 and 2; its non-trained levels are asserted identical to the published analysis before drawing",
            "scorer_hash_by_shape": TWO_AXIS_SCORER,
        },
    )
    return _emit_figure(fig, out, "fig11_axes_separable", prov)


ASSETS: dict[str, Callable[[Path], list[Path]]] = {
    "fig2": fig2_frontier,
    "fig2_recipe": fig2_frontier_recipe,
    "fig3": fig3_stopping,
    "table1": table1_heldout,
    "table2": table2_mechanism,
    "examples": examples_transcripts,
    "table4": table4_heldout_methods,
    "fig4": fig4_chain_ladder,
    "fig4_recipe": fig4_chain_ladder_recipe,
    "table5": table5_completed,
    "fig5": fig5_cost_basis,
    "fig6": fig6_efficiency,
    "fig7": fig7_suite_axes,
    "fig11": fig11_axes_separable,
    "table1_recipe": table1_recipe,
    "table_seeds": table_seeds,
    "table_fourseeds": table_fourseeds,
    "table_regimes": table_regimes,
    "table_headline": table_headline,
    "table_app_headline_diffs": table_app_headline_diffs,
    "table_mechanism_main": table_mechanism_main,
    "fig12": fig12_comparators,
    "fig13": fig13_tau2,
    "fig8": fig8_controls,
    "fig8_recipe": fig8_controls_recipe,
    "fig9": fig9_transfer_tail,
    "fig10": fig10_ceiling,
}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="output directory for the assets")
    ap.add_argument("--only", choices=sorted(ASSETS), action="append", help="build one asset")
    args = ap.parse_args(argv)
    out = Path(args.out)
    wanted = args.only or sorted(ASSETS)
    rc = 0
    for name in wanted:
        try:
            written = ASSETS[name](out)
        except (Mismatch, SourceMissing) as exc:
            print(f"{name}: REFUSED\n{exc}", file=sys.stderr)
            rc = 1
            continue
        for p in written:
            print(f"{name}: wrote {p.as_posix()} ({p.stat().st_size} bytes)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
