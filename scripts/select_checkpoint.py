#!/usr/bin/env python3
"""N1 SELECTION: the 08:00 stopping-round decision, plan v4 section 4.2, applied to gate dirs.

    .venv/bin/python scripts/select_checkpoint.py --repo "$PWD" \
        --gate-dir artifacts/gate \
        --candidates qwen3-8b-sft,qwen3-8b-sft-headline,qwen3-8b-sft-headline-stop \
        --reference qwen3-8b-sft-headline --base qwen3-8b-base

THE RULE (plans/2026-09-15-execution-plan-v4.md section 4.2), per candidate, per suite:

  (a') per-decision-point P(ASK | not done) must RISE by >= 0.03 against the REFERENCE
      checkpoint, AND P(STOP | done) must stay >= 0.85. Both quoted with n_states.

      NOT "the base's value - 0.05", which this tool applied until the 4B reading. The
      untrained base is a DEGENERATE comparator on the stop axis: it almost never stops when
      done (P(STOP|done) 0.007 at 4B, 0.15 at 8B) and spends nearly the whole cap (7.9 asks of
      8 at 4B), so its P(ASK|not done) is trivially high (0.999 at 4B, 0.96 at 8B) and a margin
      below it is unreachable rather than demanding. The reference checkpoint is a policy that
      already stops; the floor on P(STOP|done) is what stops a candidate from buying
      P(ASK|not done) back by giving up stop-when-done -- the base's own trade, restated as the
      thing the rule refuses. The base row is printed for context and is never a threshold.
  (b) ASK decisions taken at not-done states, PER RUN, must RISE against the reference
      checkpoint. Cells used: `n_not_done_ask / n_runs`, both from `criteria.stop_2x2.value`.
      The plan's other wording -- mean n_asks minus mean_asks_after_done -- is the SAME number
      whenever `n_skipped_no_coverage == 0`, because every ASK decision falls in exactly one of
      the two done cells: mean n_asks = (n_done_ask + n_not_done_ask)/n_runs and
      mean_asks_after_done = n_done_ask/n_runs. It is computed and printed beside as
      `asks_minus_after_done`, and a disagreement is reported, never averaged away.
  (c) matched-cost evidence_coverage delta against the base's own PREFIX LADDER (comparator
      `matched_k`) must not fall below the reference's value.
  (d) the cap-8 coverage delta and mean asks at cap 8 are REPORTED, never gated; so are the
      quality gates as `pi train gate` recorded them (malformed <= baseline, within-task
      distinct-3 >= 0.65, question-length TOST).

VERDICT. `core` is (a') AND (b) AND (c) on BOTH suites -- the plan's own sentence. `quality` is
the three recorded quality gates on both suites. `verdict` is PASS only when both hold, and the
one-line verdict names every condition that failed and on which suite. The two are printed
separately because the plan states the selection rule as (a)/(b)/(c) and lists the quality gates
beside it; a reader who wants either reading can have it without re-running anything.
`distinct3` and `malformed` are printed per checkpoint with the verdict's own pass/fail.

NO `stop_2x2` BLOCK WRITTEN BEFORE COMMIT f348feb MAY BE QUOTED (docs/TRAINING.md 5.3.1). Those
blocks read the done axis off `stop_undershoot`, which is identically 0, so `n_not_done` is 0 and
`p_ask_given_not_done` is NaN by construction on every checkpoint. `_check_live_2x2` refuses such
a block by three independent signals (missing per-state cells, a `done_axis` naming
`stop_undershoot`, an empty not-done cell) and names which one fired. That refusal is the point
of this tool: criterion (a') reads exactly the cell the old format could not measure.

WHERE EACH NUMBER COMES FROM, in the order tried, with the source path and its sha256 recorded
for every value in the output JSON:

  the 2x2      1. <gate>/rerun-stop2x2/<model>.<suite>.json
               2. <gate>/<model>.<suite>.json, if its block survives `_check_live_2x2`
               3. recomputed from <gate>/parquet_<tag>/ with `pinq_train.gate`'s own
                  `_stop_2x2` / `_coverage_ladder` / `_select_runs` (needs duckdb; needs NO
                  gold, and `pinq_train` may not import `pi_eval` by contract 3)
  matched cost 1. a `coverage_rule` / `matched_cost` block in the verdict JSON (the T19 shape)
               2. a recorded matched-cost provenance whose `inputs` sha256 still match the
                  parquet files on disk -- by default figures/P13_matched_cost/provenance.json
               3. `scripts/matched_cost.py --gate <gate> --no-figure --json ...`, run as a
                  subprocess with PI_GOLD_ROOT set IN THAT SUBPROCESS ONLY. It is the one
                  gold-side read here, and it is the repo's own script rather than a
                  reimplementation: its BCa interval, its MechanicalMatcher prefix ladder and
                  its refusal to print a table whose cap-8 column does not reproduce the
                  verdicts are all part of what makes (c) the same measurement the figure is.
  quality, (d) <gate>/<model>.<suite>.json -- the verdict of record.

CROSS-ARM PROVENANCE IS CHECKED, NOT ASSUMED. Every 2x2 quoted must carry the `--base` model id
in `selection.baseline_model_id` -- context, not threshold, but still the check that says the
gate directory is the one you meant -- and every value compared across checkpoints ((a'), (b)
and (c), all three against the reference) must carry the SAME `scorer_hash`.
A scorer mismatch is a refusal for that criterion, not a comparison: the done axis and the
coverage ladder are both scorer output, and the gate README's own pass-1 episode is what
re-scoring rather than cross-scorer comparison looks like.

THIS TOOL MEASURES NOTHING ITSELF apart from arithmetic on cells other tools wrote. It makes no
LLM call, writes only under artifacts/n1/, and never edits a tracked file.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

# The repo root. This file lives INSIDE the checkout, at scripts/select_checkpoint.py (promoted
# from a session scratchpad so the selection rule is reviewable and reproducible from a clean
# checkout, since it decides which checkpoint the paper reports). It still carries no absolute
# path of its own: `--repo`, else $PI_REPO, else the working directory, and the choice is
# checked against scripts/matched_cost.py being there.
REPO = Path.cwd()
SUITES = ("musique", "strategyqa")

# Criterion (a'), revised after the 4B reading: the rise in P(ASK|not done) a candidate must
# show over the REFERENCE, and the floor on P(STOP|done) that keeps it from buying that rise by
# forgetting to stop. Every trained checkpoint measured so far clears the floor (8B rung 1:
# 0.880-0.953); no untrained base comes near it (8B 0.15, 4B 0.007).
ASK_RISE_MIN = 0.03
STOP_FLOOR = 0.85
# Float slack on the >= comparisons only, so a value that is exactly on the threshold passes.
# (b) is a STRICT rise and uses it as a floor the candidate must clear, not as slack.
TOL = 1e-12

MATCHED_METRIC = "evidence_coverage"
MATCHED_COMPARATOR = "matched_k"
CAP8_COMPARATOR = "cap8"
DEFAULT_P13 = "figures/P13_matched_cost/provenance.json"

# `pinq_train.gate._stop_2x2` returns exactly these. A block missing any of them was written
# before f348feb and is the degenerate run-level 2x2; see the module docstring.
LIVE_CELLS = (
    "p_stop_given_done",
    "p_ask_given_not_done",
    "n_done",
    "n_not_done",
    "n_done_stop",
    "n_done_ask",
    "n_not_done_ask",
    "n_not_done_stop",
    "n_states",
    "n_runs",
    "n_forced_stops",
    "n_skipped_no_coverage",
    "mean_asks_after_done",
)

QUALITY_GATES = ("malformed", "distinct3", "length_equivalence")

# docs/TRAINING.md 5.3.1 / 5.3.2, quoted to the digits the doc prints. Used ONLY by
# --reproduce, never by the rule. `tol` is half a unit in the doc's last printed place, so a
# value agrees when it rounds to what the doc says -- not when it is close enough.
RECORDED_2X2 = {
    "base": {
        "p_stop_given_done": (0.15, 0.15),
        "p_ask_given_not_done": (0.96, 0.96),
        "mean_asks_after_done": {"musique": (1.66, 1.66), "strategyqa": (3.42, 3.42)},
    },
    "rung1": {
        "p_stop_given_done": (0.88, 0.95),
        "p_ask_given_not_done": (0.83, 0.91),
        "mean_asks_after_done": (0.02, 0.10),
    },
}
RECORDED_2X2_TOL = 0.005
RECORDED_MATCHED_K = {  # docs/TRAINING.md 5.3.2, and figures/P13_matched_cost/provenance.json
    ("qwen3-8b-sft", "musique"): (0.042, -0.012, 0.094),
    ("qwen3-8b-sft-headline", "musique"): (0.049, 0.003, 0.092),
    ("qwen3-8b-sft-headline-stop", "musique"): (0.065, 0.018, 0.105),
    ("qwen3-8b-sft", "strategyqa"): (0.064, 0.039, 0.089),
    ("qwen3-8b-sft-headline", "strategyqa"): (0.080, 0.050, 0.108),
    ("qwen3-8b-sft-headline-stop", "strategyqa"): (0.059, 0.033, 0.084),
}
RECORDED_MATCHED_TOL = 0.0005


class Refused(RuntimeError):
    """A refusal, never a warning. Exits non-zero so nothing downstream reads a half verdict."""


def set_repo(p: Path) -> Path:
    """Pin the repo root once, for every path this module derives from it."""
    global REPO
    p = Path(p).resolve()
    if not (p / "scripts" / "matched_cost.py").is_file():
        raise Refused(
            f"{p} does not look like the ProactiveInquirer checkout "
            "(no scripts/matched_cost.py). Pass --repo, or set PI_REPO."
        )
    REPO = p
    return REPO


def out_dir() -> Path:
    return REPO / "artifacts" / "n1"


# --------------------------------------------------------------------------- provenance


def sha256_file(p: Path) -> str:
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


class Inputs:
    """Every file any number came from, path -> sha256, read once."""

    def __init__(self) -> None:
        self.sha: dict[str, str] = {}

    def add(self, p: Path) -> str:
        k = rel(p)
        if k not in self.sha:
            self.sha[k] = sha256_file(Path(p))
        return self.sha[k]

    def load(self, p: Path) -> tuple[Any, str]:
        self.add(Path(p))
        return json.loads(Path(p).read_text()), rel(p)


def prov(value: Any, source: str, inputs: Inputs, note: str = "") -> dict[str, Any]:
    """One number with the file it came from and that file's sha256. Rule 1."""
    d: dict[str, Any] = {"value": value, "source": source, "sha256": inputs.sha.get(source, "")}
    if note:
        d["note"] = note
    return d


# --------------------------------------------------------------------------- layout


def pass1_tag_map() -> dict[str, str]:
    """`{model id: parquet dir name}` read OUT OF scripts/matched_cost.py, not copied from it.

    The pass-1 gate named its parquet directories by a short tag (`parquet_headline`) rather
    than by the served model id, and `matched_cost.PASS1_CHECKPOINTS` is where that mapping
    lives. Parsing the literal keeps one source of truth without importing the module, which
    would pull `pi_eval` in through its own imports.
    """
    src = (REPO / "scripts" / "matched_cost.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        targets = getattr(node, "targets", []) or ([node.target] if hasattr(node, "target") else [])
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "PASS1_CHECKPOINTS":
                return {model: tag for tag, model in ast.literal_eval(node.value)}
    raise Refused("scripts/matched_cost.py: no PASS1_CHECKPOINTS literal to read the tag map from")


def parquet_dir_for(gate: Path, model: str) -> Path | None:
    """`<gate>/parquet_<model>/`, or the pass-1 short tag when that is what is on disk."""
    d = gate / f"parquet_{model}"
    if d.is_dir():
        return d
    tag = pass1_tag_map().get(model)
    if tag and (gate / tag).is_dir():
        return gate / tag
    return None


def find_verdict(gates: Sequence[Path], model: str, suite: str) -> tuple[Path, Path]:
    """`(gate dir, verdict path)` for one (model, suite), searched across the gate dirs.

    A model present in two gate dirs is a REFUSAL: two verdicts for one name means the answer
    depends on argument order, and an 08:00 decision cannot depend on argument order.
    """
    hits = [
        (g, g / f"{model}.{suite}.json") for g in gates if (g / f"{model}.{suite}.json").is_file()
    ]
    if not hits:
        raise Refused(
            f"no verdict {model}.{suite}.json in any of {[rel(g) for g in gates]}. "
            "`pi train gate` writes <model id>.<suite>.json; pass the directory holding it."
        )
    if len(hits) > 1:
        raise Refused(
            f"{model}.{suite}.json exists in more than one gate dir: "
            f"{[rel(p) for _g, p in hits]}. Name one."
        )
    return hits[0]


# --------------------------------------------------------------------------- the 2x2


def _check_live_2x2(block: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    """Refuse a pre-f348feb `stop_2x2`. Three signals, each named when it fires."""
    val = block.get("value")
    if not isinstance(val, Mapping):
        raise Refused(f"{where}: criteria.stop_2x2.value is not an object")
    missing = [c for c in LIVE_CELLS if c not in val]
    if missing:
        raise Refused(
            f"{where}: criteria.stop_2x2.value is missing the per-state cells {missing}. This is "
            "the run-level 2x2 written before f348feb; docs/TRAINING.md 5.3.1 forbids quoting it."
        )
    axis = str((block.get("columns") or {}).get("done_axis", ""))
    if "stop_undershoot" in axis:
        raise Refused(
            f"{where}: criteria.stop_2x2.columns.done_axis is {axis!r}. `stop_undershoot` is "
            "identically 0, so the not-done cell is empty by construction."
        )
    if "frontier_q" not in axis:
        raise Refused(
            f"{where}: criteria.stop_2x2.columns.done_axis is {axis!r}; expected the "
            "`frontier_q#t >= 1-1e-12` axis."
        )
    if not val.get("n_not_done") or val.get("p_ask_given_not_done") is None:
        raise Refused(
            f"{where}: criteria.stop_2x2 has n_not_done={val.get('n_not_done')!r} and "
            f"p_ask_given_not_done={val.get('p_ask_given_not_done')!r}; criterion (a) is "
            "exactly that cell, so there is nothing to gate on."
        )
    base = block.get("baseline")
    if not isinstance(base, Mapping) or base.get("p_ask_given_not_done") is None:
        raise Refused(f"{where}: criteria.stop_2x2.baseline carries no p_ask_given_not_done")
    return block


def recompute_2x2(gate: Path, model: str, suite: str, sel: Mapping[str, Any]) -> dict[str, Any]:
    """The 2x2 from the parquets, through `pinq_train.gate`'s own functions.

    Used only when neither `rerun-stop2x2/` nor the verdict carries a live block. `pinq_train`
    reaches gold over HTTP and never by import (contract 3), so this needs no PI_GOLD_ROOT.
    """
    d = parquet_dir_for(gate, model)
    if d is None:
        raise Refused(
            f"{model}.{suite}: no live stop_2x2 anywhere and no parquet_<tag>/ under {rel(gate)} "
            "to recompute one from."
        )
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    from pinq_train.gate import (  # noqa: PLC0415 - imported only on this path
        _con,
        _coverage_ladder,
        _select_runs,
        _stop_2x2,
        _with_stop,
    )

    con = _con(d)
    sh = str(sel["scorer_hash"])

    def side(arm: str, grids: Sequence[str], model_id: str | None) -> dict[str, Any]:
        runs = _with_stop(con, _select_runs(con, arm=arm, grids=list(grids), model_id=model_id))
        return _stop_2x2(con, runs, _coverage_ladder(con, runs, scorer_hash=sh))

    ck = side(sel["checkpoint_arm"], [sel["grid_name"]], None)
    ba = side(sel["baseline_arm"], sel["baseline_grid_names"], sel.get("baseline_model_id"))
    for k in ("p_stop_given_done", "p_ask_given_not_done", "mean_asks_after_done"):
        for cell in (ck, ba):
            if isinstance(cell[k], float) and math.isnan(cell[k]):
                cell[k] = None
    return {
        "value": ck,
        "baseline": ba,
        "columns": {"done_axis": "frontier_q#t >= 1-1e-12 before the decision at t"},
        "n": ck["n_done"] + ck["n_not_done"],
    }


def load_2x2(
    gates: Sequence[Path], model: str, suite: str, inputs: Inputs
) -> tuple[Mapping[str, Any], str, Mapping[str, Any], str]:
    """`(stop_2x2 block, its source, the verdict's selection block, the verdict path)`."""
    gate, vpath = find_verdict(gates, model, suite)
    verdict, vrel = inputs.load(vpath)
    sel = verdict.get("selection") or {}

    rerun = gate / "rerun-stop2x2" / f"{model}.{suite}.json"
    if rerun.is_file():
        doc, r = inputs.load(rerun)
        block = doc.get("criteria", {}).get("stop_2x2")
        if block is None:
            raise Refused(f"{r}: no criteria.stop_2x2")
        _check_live_2x2(block, r)
        sel = doc.get("selection") or sel
        _agree_outside_stop(verdict, doc, vrel, r)
        return block, r, sel, vrel

    why_not = ""
    block = verdict.get("criteria", {}).get("stop_2x2")
    if block is not None:
        try:
            _check_live_2x2(block, vrel)
        except Refused as exc:
            # Not quotable. Remembered rather than swallowed: if nothing else can supply a live
            # 2x2, THIS is the reason the tool stops, and a message about a missing parquet
            # directory would send a reader looking in the wrong place at 08:00.
            why_not = str(exc)
        else:
            return block, vrel, sel, vrel

    try:
        rec = recompute_2x2(gate, model, suite, sel)
    except Refused as exc:
        raise Refused(
            f"{exc}\n  and the verdict's own block is not quotable: {why_not}"
            if why_not
            else str(exc)
        ) from exc
    _check_live_2x2(rec, f"recomputed from {rel(parquet_dir_for(gate, model) or gate)}")
    src = f"{rel(parquet_dir_for(gate, model) or gate)}/ (recomputed via pinq_train.gate._stop_2x2)"
    return rec, src, sel, vrel


def _agree_outside_stop(verdict: Mapping, rerun: Mapping, vrel: str, rrel: str) -> None:
    """The re-run file must differ from the verdict of record in `stop_2x2` AND NOWHERE ELSE.

    docs/TRAINING.md 5.3.1 says the re-run left "every other criterion byte-identical". If that
    is not true of a file handed to this tool, the two are not the same gate run and quoting the
    quality gates from one and the 2x2 from the other would silently mix two measurements.
    """
    diffs = sorted(
        k
        for k in set(verdict.get("criteria", {})) | set(rerun.get("criteria", {}))
        if k != "stop_2x2"
        and verdict.get("criteria", {}).get(k) != rerun.get("criteria", {}).get(k)
    )
    if diffs:
        raise Refused(
            f"{rrel} and {vrel} disagree outside stop_2x2 on {diffs}. The re-run 2x2 is only "
            "quotable beside the verdict it re-ran."
        )


def asks_at_not_done(cells: Mapping[str, Any]) -> tuple[float, float, bool]:
    """(the gated value, the plan's other wording, whether they are exactly equal).

    Gated: `n_not_done_ask / n_runs`. Other wording: mean n_asks - mean_asks_after_done, which
    is `(n_done_ask + n_not_done_ask)/n_runs - mean_asks_after_done`. Equal iff no state was
    skipped for absent coverage, because then every ASK decision sits in exactly one done cell.
    """
    n_runs = int(cells["n_runs"])
    if not n_runs:
        raise Refused("stop_2x2 has n_runs = 0")
    gated = int(cells["n_not_done_ask"]) / n_runs
    other = (int(cells["n_done_ask"]) + int(cells["n_not_done_ask"])) / n_runs - float(
        cells["mean_asks_after_done"]
    )
    return gated, other, abs(gated - other) <= 1e-9


# --------------------------------------------------------------------------- matched cost


def _from_contrast_list(rows: Sequence[Mapping[str, Any]], model: str, suite: str, comparator: str):
    for c in rows:
        if (
            str(c.get("checkpoint")) == model
            and str(c.get("suite_id")) == suite
            and str(c.get("metric")) == MATCHED_METRIC
            and str(c.get("comparator")) == comparator
        ):
            return c
    return None


def _inverdict_matched(verdict: Mapping[str, Any], model: str, suite: str, comparator: str):
    """A `coverage_rule` / `matched_cost` block inside a verdict (the T19 shape), if present.

    Four shapes are accepted, all of them things the branch could plausibly write; anything
    else is a refusal naming what was looked for, because a matched-cost number read out of the
    wrong key would be a wrong published number that nothing downstream could catch.

        1. `{'contrasts': [...]}`, read via `_from_contrast_list`.
        2. `{comparator: {MATCHED_METRIC: {...}}}`, a per-comparator, per-metric block.
        3. a flat `{comparator, metric, delta|value}` block.
        4. `{'by_suite': {suite: {...}}}`, branching again on `comparator` (cap8 vs matched).

    Fallback-tier audit (asked for after a real cross-scorer prose mixup elsewhere, not a bug
    found here): none of the four shapes above, nor the pre-loop `continue` that skips a
    `coverage_rule` that is absent, `null`, or the bare string `pi train gate
    --coverage-rule matched_cost` stamps, ever reads `verdict["criteria"]["evidence_coverage"]`
    -- the quality-style criterion block with its own `.value` / `.passed` / `.baseline`. That
    block shares MATCHED_METRIC's name but is structurally unrelated to any of the four shapes:
    shape 2's `MATCHED_METRIC` lookup is nested *inside* an already-resolved `coverage_rule` /
    `matched_cost` Mapping, never at `verdict["criteria"]`. The distinction matters because
    `criteria.evidence_coverage.value` is exactly where the rule-dependent ambiguity lives: in a
    verdict predating the matched-cost rule (`coverage_rule` is `null`), that `.value` IS the
    fixed-budget (cap8) delta; in one written under the rule, that same `.value` is the
    matched-cost delta and the fixed-budget figure moves to a sibling `cap8_value` field
    instead -- confirmed across every real verdict under artifacts/gate/ with a `criteria`
    block, re-measured 2026-09-17 (supersedes an earlier same-day count of 122 new-shape: the
    corpus is a live target, campaigns are writing new run directories as this is written, and
    the new-shape count moved while old-shape did not): 34 have `coverage_rule: null`, no
    top-level `matched_cost` key, and no `cap8_value` sibling (old shape -- roughly a fifth of
    the 174 checked, not a historical curiosity a reader can skip past); 140 have
    `coverage_rule: "matched_cost"` (a bare string, shapes 1-4 above never seeing it either)
    plus a real top-level `matched_cost` key and a `cap8_value` sibling (new shape); zero have
    `coverage_rule` as a Mapping with `contrasts` the way `_synth_verdict` idealizes it -- that
    shape is still accepted, but only as TOLERANCE: a form the four-shapes list above was
    written to allow, not one any real file has ever needed. A later reader should not treat
    its acceptance as load-bearing production behaviour worth preserving at some cost; it
    survives here because `_synth_verdict` and the `scorer_mismatch` / `bad_coverage_rule`
    fixtures (see _reported_cap8_delta) happen to use it, not because production data does.
    Zero new-shape verdicts are missing the `cap8_value` sibling -- see _reported_cap8_delta,
    which refuses that combination rather than assuming it cannot occur. A
    `coverage_rule: null` verdict therefore returns None from *this* function -- it never
    reaches a shape that could read the ambiguous field -- and `matched_cost()`'s remaining
    tiers (a `gate/matched_cost.json` sidecar, `p13`, or a real subprocess recomputation via
    `_run_matched_cost`, itself gated on a real `--gold-root`) do not read
    `criteria.evidence_coverage` either. See
    tests/fixtures/select_checkpoint/pre_matched_cost_rule and
    test_refuses_coverage_value_predating_matched_cost_rule: a verdict built exactly this way,
    carrying a realistic fixed-budget delta in `criteria.evidence_coverage.value`, is refused
    rather than read -- and, checked by temporarily injecting the fallback this paragraph rules
    out, would instead be silently accepted (criterion (c)'s `delta` becomes that same
    fixed-budget number) if such a fallback were ever added, which is exactly the regression
    that test now guards against.

    One place in this module used to read `criteria.evidence_coverage.value`
    unconditionally, regardless of `coverage_rule`'s shape: the `d_reported.cap8_coverage_delta`
    field built in `evaluate()` (report-only, `"gated": False`, never compared against a
    threshold or used in a `passed` decision). Its label was only correct for a pre-rule
    verdict; for a post-rule one it was quietly the matched-cost delta wearing the cap8 name,
    with no assertion against the independently-computed `cap8_coverage_delta_recomputed`
    sitting right beside it in the same output. `evaluate()` now builds that field through
    `_reported_cap8_delta` (defined just above it), which applies the same rule-dependent
    discriminator documented here -- `.value` for a pre-rule verdict, the `cap8_value` sibling
    for a post-rule one, refusing rather than falling back if that sibling is absent -- and
    then asserts the selected value agrees with `cap8_coverage_delta_recomputed` to within
    `TOL`, refusing (not warning) if they differ. Refusing rather than warning here follows
    `_agree_outside_stop`'s precedent for two independently-sourced descriptions of the same
    quantity that disagree, even though this field, unlike that one, is never gated: it is
    still FORMATTED into the printed report table, where a person reads it and may quote it,
    and this project has now chased that exact failure mode -- a mislabelled number entering a
    draft because nothing checked it at the point it was printed -- more than once in one day.
    See tests/fixtures/select_checkpoint/cap8_report_disagreement and
    test_refuses_cap8_report_disagreeing_with_recomputed_delta for a new-shape verdict whose
    two cap8 readings deliberately differ, confirmed to make the *unfixed* code report the
    matched-cost delta under the fixed-budget label before this refusal existed.
    """
    for key in ("coverage_rule", "matched_cost"):
        block = verdict.get(key) or (verdict.get("criteria") or {}).get(key)
        if not isinstance(block, Mapping):
            # `pi train gate --coverage-rule matched_cost` (main at 0cd1252) stamps the
            # top-level `coverage_rule` as a BARE STRING naming the rule ("matched_cost"), not
            # a block -- the block with the numbers is the separate top-level `matched_cost`
            # key, tried next by this same loop. A value that is neither absent nor a Mapping
            # cannot be any accepted shape, so it is skipped rather than refused on the spot:
            # refusing here would stop the loop before it ever reached the key that matters.
            continue
        if isinstance(block, Mapping) and isinstance(block.get("contrasts"), list):
            hit = _from_contrast_list(block["contrasts"], model, suite, comparator)
            if hit is not None:
                return key, hit
        if isinstance(block, Mapping) and isinstance(block.get(comparator), Mapping):
            inner = block[comparator].get(MATCHED_METRIC)
            if isinstance(inner, Mapping):
                return key, inner
        if (
            isinstance(block, Mapping)
            and str(block.get("comparator", comparator)) == comparator
            and str(block.get("metric", MATCHED_METRIC)) == MATCHED_METRIC
            and ("delta" in block or "value" in block)
        ):
            return key, block
        if isinstance(block, Mapping) and isinstance(block.get("by_suite"), Mapping):
            # `pinq_train.gate._matched_cost`'s own shape (D9; `--coverage-rule matched_cost`
            # is main's default as of 0cd1252): `matched_cost.by_suite[<suite>]` carries the
            # matched-k contrast directly under `delta`/`ci_lo`/`ci_hi`/`n_tasks` -- no
            # `comparator` key at all, because this block IS the matched_k comparator; cap-8
            # rides beside it, per suite, as the REPORT-ONLY `cap8_coverage_delta` /
            # `cap8_n_tasks`, with no CI recorded for that comparator here. The checkpoint's
            # own k is `trained_mean_k` (identically `trained_mean_n_asks` -- gate.py appends
            # the same value to both accumulators in the same loop iteration over matched
            # tasks). The base's side is the PREFIX-LADDER value AT that k --
            # `baseline_mean_k_charged` -- and must not be confused with
            # `baseline_mean_n_asks`, the base's own unmatched, full-episode mean asks; that
            # is a different and larger number answering a different question (the base's
            # cost on its own terms, not charged at the checkpoint's k).
            row = block["by_suite"].get(suite)
            if isinstance(row, Mapping):
                if comparator == MATCHED_COMPARATOR and "delta" in row:
                    return key, {
                        "delta": row.get("delta"),
                        "ci_lo": row.get("ci_lo"),
                        "ci_hi": row.get("ci_hi"),
                        "n_tasks": row.get("n_tasks"),
                        "trained_asks": row.get("trained_mean_k", row.get("trained_mean_n_asks")),
                        "base_asks": row.get("baseline_mean_k_charged"),
                    }
                if comparator == CAP8_COMPARATOR and "cap8_coverage_delta" in row:
                    return key, {
                        "delta": row.get("cap8_coverage_delta"),
                        "n_tasks": row.get("cap8_n_tasks"),
                    }
        raise Refused(
            f"a `{key}` block is present but carries no {comparator}/{MATCHED_METRIC} entry in "
            "any accepted shape: {'contrasts': [{checkpoint, suite_id, metric, comparator, "
            "delta}]}, {'matched_k': {'evidence_coverage': {...}}}, {'by_suite': {<suite>: "
            "{delta, ci_lo, ci_hi, n_tasks, ...}}}, or a flat {metric, comparator, delta}. "
            "Extend scripts/select_checkpoint.py:_inverdict_matched rather than guessing which key "
            "holds the number."
        )
    return None


def _p13_covers(prov_doc: Mapping[str, Any], gate: Path, model: str) -> bool:
    """Is a recorded matched-cost provenance still valid for THIS gate dir's bytes?

    Every input it names that lives under the gate dir must still be on disk with the sha256
    it recorded. A recorded delta is a property of the parquet files it was computed from; the
    moment one of them changes, the record describes a different measurement.
    """
    d = parquet_dir_for(gate, model)
    if d is None:
        return False
    wanted = {rel(d / f"{t}.parquet") for t in ("runs", "turns", "scores")}
    got = prov_doc.get("inputs") or {}
    if not wanted <= set(got):
        return False
    return all((REPO / k).is_file() and sha256_file(REPO / k) == got[k] for k in wanted)


def _run_matched_cost(gate: Path, gold_root: Path, out_json: Path) -> Path:
    """`scripts/matched_cost.py --gate <gate> --no-figure --json <out>`, PI_GOLD_ROOT in the
    subprocess env ONLY. `--no-figure` because figures/ is not ours to write."""
    if out_json.is_file():
        return out_json
    if not gold_root.is_dir():
        raise Refused(f"--gold-root {gold_root} is not a directory; matched-cost cannot be run")
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "matched_cost.py"),
        "--gate",
        str(gate),
        "--no-figure",
        "--json",
        str(out_json),
    ]
    env = dict(os.environ)
    env["PI_GOLD_ROOT"] = str(gold_root)
    print(f"  running: PI_GOLD_ROOT={gold_root} {' '.join(cmd[1:])}", file=sys.stderr)
    r = subprocess.run(cmd, env=env, cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0 or not out_json.is_file():
        raise Refused(
            f"scripts/matched_cost.py exited {r.returncode} for {rel(gate)}:\n{r.stderr.strip()}"
        )
    return out_json


def matched_cost(
    gates: Sequence[Path],
    model: str,
    suite: str,
    comparator: str,
    inputs: Inputs,
    *,
    p13: Path,
    gold_root: Path,
    cache: dict[str, Any],
) -> dict[str, Any]:
    """One matched-cost contrast row, with its source. Sources are tried in a fixed order."""
    gate, vpath = find_verdict(gates, model, suite)
    verdict, vrel = inputs.load(vpath)

    hit = _inverdict_matched(verdict, model, suite, comparator)
    if hit is not None:
        key, row = hit
        return _contrast_record(row, vrel, f"verdict `{key}` block", inputs)

    for cand in (gate / "matched_cost.json", p13):
        if cand.is_file():
            doc, r = inputs.load(cand)
            if _p13_covers(doc, gate, model):
                row = _from_contrast_list(doc.get("contrasts") or [], model, suite, comparator)
                if row is not None:
                    return _contrast_record(row, r, "recorded matched-cost provenance", inputs)

    key = rel(gate).replace("/", "_")
    out = out_dir() / f"matched_cost.{key}.json"
    if key not in cache:
        cache[key] = inputs.load(_run_matched_cost(gate, gold_root, out))
    doc, r = cache[key]
    row = _from_contrast_list(doc.get("contrasts") or [], model, suite, comparator)
    if row is None:
        raise Refused(f"{r}: no {comparator}/{MATCHED_METRIC} contrast for {model}/{suite}")
    return _contrast_record(row, r, "scripts/matched_cost.py", inputs)


def _contrast_record(row: Mapping[str, Any], source: str, how: str, inputs: Inputs) -> dict:
    delta = row.get("delta", row.get("value"))
    if delta is None:
        raise Refused(f"{source}: matched-cost row carries neither `delta` nor `value`")
    return {
        "delta": float(delta),
        "ci_lo": _f(row.get("ci_lo")),
        "ci_hi": _f(row.get("ci_hi")),
        "p_value": _f(row.get("p_value")),
        "n": row.get("n_tasks", row.get("n")),
        "trained_asks": _f(row.get("trained_asks")),
        "base_asks": _f(row.get("base_asks")),
        "source": source,
        "sha256": inputs.sha.get(source, ""),
        "how": how,
    }


def _f(x: Any) -> float | None:
    return None if x is None else float(x)


# --------------------------------------------------------------------------- the rule


def _reported_cap8_delta(verdict_doc: Mapping[str, Any], source: str, recomputed: float) -> float:
    """The fixed-budget (cap8) coverage delta to REPORT in `d_reported` for one verdict, read
    from whichever field holds it under that verdict's shape, then checked against
    `recomputed` -- the same quantity `matched_cost()` computed by an independent route (its
    own fallback tiers: this same in-verdict block first, then a provenance sidecar, then a
    real subprocess recompute against gold).

    `criteria.evidence_coverage.value` is the cap8 delta ONLY in a verdict predating the
    matched-cost rule (`coverage_rule` absent or null) OR in one of the test-only synthetic
    shapes that predate the rule's relabelling regardless of `coverage_rule` (see below); in a
    verdict actually written under the rule, that same field holds the MATCHED-COST delta
    instead, and the cap8 figure moves to a sibling `cap8_value` field -- see
    _inverdict_matched's docstring for the full shape audit and the census this reuses (34
    old-shape, 140 new-shape-with-sibling, 0 new-shape-missing-sibling, checked 2026-09-17
    against every real verdict under artifacts/gate/). A new-shape verdict missing that sibling
    is a combination that should not exist on real data -- checked: 0 of 140 -- so it is
    refused rather than silently read as `.value`, which is exactly the mislabelling this
    function exists to prevent: the matched-cost delta wearing the fixed-budget name.

    "New-shape" here means specifically `coverage_rule` as the bare string
    `pi train gate --coverage-rule matched_cost` actually stamps -- the only shape the census
    above covers. A Mapping-shaped `coverage_rule` (`_inverdict_matched` shape 1) is a
    different, test-only convention with zero occurrences on real data; `_synth_verdict` and
    the `scorer_mismatch` / `bad_coverage_rule` fixtures all use it with `.value` already equal
    to that shape's own cap8 contrast delta by construction, and are read via the old-shape
    branch below rather than refused for a missing sibling the census never says they need.
    `_inverdict_matched` accepts that Mapping shape as TOLERANCE, not because any real verdict
    takes it -- zero do -- and this function's old-shape branch extends the same tolerance for
    the same reason. Neither acceptance is load-bearing; a later reader should not spend effort
    preserving either one beyond what the three fixtures above already require.

    The second check -- agreement with `recomputed` -- exists because the value just selected
    and `recomputed` are two numbers claiming to be the same quantity, taken by two different
    routes (one frozen into the verdict at write time, one just computed fresh by this tool),
    with nothing else in this module that would notice if they diverged. This refuses rather
    than warns on disagreement, for the same reason `_agree_outside_stop` refuses rather than
    warns when a re-run 2x2 disagrees with the verdict of record: `d_reported.cap8_coverage_delta`
    is never gated (no `passed` decision reads it -- see criterion (d) in the module docstring)
    but it IS formatted into the printed report table (render(), the cap8_coverage_delta
    column), where a person reads it and may quote it -- and a mislabelled number a human
    quotes is worse than a wrong number a machine refuses. A warning is a column a busy reader
    can miss; a refusal stops the run before either number reaches print.
    """
    ec = verdict_doc["criteria"]["evidence_coverage"]
    coverage_rule = verdict_doc.get("coverage_rule")
    # The relabelling this function guards against is specific to the REAL production stamp:
    # `pi train gate --coverage-rule matched_cost` writes `coverage_rule` as the bare STRING
    # "matched_cost" (see _inverdict_matched's docstring/comment on the same string) -- that is
    # the only shape the 2026-09-17 census found on real data (140/174), and the only one it
    # confirmed always carries a `cap8_value` sibling (0/140 missing). A Mapping-shaped
    # `coverage_rule` (`_inverdict_matched` shape 1, `{'contrasts': [...]}` embedded directly)
    # is a DIFFERENT, test-only convention -- zero real files use it, per that same census --
    # predating the matched-cost rule's relabelling of `.value`. `_synth_verdict` and the
    # `scorer_mismatch` / `bad_coverage_rule` fixtures all use it, and in every one `.value`
    # already equals that shape's own cap8 contrast delta by construction (checked directly,
    # not assumed), because they were built to exercise other invariants (self-test's core
    # checks, scorer-hash mismatches, unknown-shape refusal) and never needed a `cap8_value`
    # sibling. Treating that shape as "new" here would refuse three pieces of pre-existing,
    # correct test infrastructure over a combination the census never claims is wrong for
    # them -- the "should not exist" finding is about the bare-string shape specifically.
    if isinstance(coverage_rule, str) and coverage_rule:
        value = ec.get("cap8_value")
        if value is None:
            raise Refused(
                f"{source}: coverage_rule is set (post matched-cost-rule shape) but "
                "criteria.evidence_coverage has no cap8_value sibling. Under this rule, "
                "`.value` is the matched-cost delta, not the fixed-budget one, and reading it "
                "as a fallback would silently relabel it -- the exact defect this refusal "
                "exists to prevent."
            )
    else:
        value = ec["value"]
    value = float(value)
    recomputed = float(recomputed)
    if abs(value - recomputed) > TOL:
        raise Refused(
            f"{source}: reported fixed-budget coverage delta ({value!r}) disagrees with the "
            f"independently recomputed cap8 delta ({recomputed!r}) by more than {TOL}. Two "
            "routes to the same quantity that disagree means at least one of them is wrong; "
            "this tool will not publish either number until that is resolved."
        )
    return value


def evaluate(
    gates: Sequence[Path],
    *,
    candidates: Sequence[str],
    reference: str,
    base_model: str,
    suites: Sequence[str],
    p13: Path,
    gold_root: Path,
) -> dict[str, Any]:
    inputs = Inputs()
    cache: dict[str, Any] = {}
    models = list(dict.fromkeys([*candidates, reference]))

    two: dict[tuple[str, str], dict[str, Any]] = {}
    for m in models:
        for s in suites:
            block, src, sel, vrel = load_2x2(gates, m, s, inputs)
            if str(sel.get("baseline_model_id")) != base_model:
                raise Refused(
                    f"{src}: selection.baseline_model_id is "
                    f"{sel.get('baseline_model_id')!r}, not --base {base_model!r}. Criterion (a) "
                    "is a margin below THAT base's own rate; a different base is a different rule."
                )
            gated, other, same = asks_at_not_done(block["value"])
            two[(m, s)] = {
                "block": block,
                "source": src,
                "verdict": vrel,
                "scorer_hash": str(sel.get("scorer_hash", "")),
                "asks_not_done_per_run": gated,
                "asks_minus_after_done": other,
                "definitions_agree": same,
            }

    mc: dict[tuple[str, str, str], dict[str, Any]] = {}
    for m in models:
        for s in suites:
            for comp in (MATCHED_COMPARATOR, CAP8_COMPARATOR):
                mc[(m, s, comp)] = matched_cost(
                    gates, m, s, comp, inputs, p13=p13, gold_root=gold_root, cache=cache
                )

    rows: list[dict[str, Any]] = []
    for m in candidates:
        for s in suites:
            t, tref = two[(m, s)], two[(reference, s)]
            c, cref = mc[(m, s, MATCHED_COMPARATOR)], mc[(reference, s, MATCHED_COMPARATOR)]
            cap8 = mc[(m, s, CAP8_COMPARATOR)]
            cells, bcells = t["block"]["value"], t["block"]["baseline"]

            p_ask = float(cells["p_ask_given_not_done"])
            p_ask_ref = float(tref["block"]["value"]["p_ask_given_not_done"])
            p_stop = float(cells["p_stop_given_done"])
            scorer_ok = (m == reference) or (t["scorer_hash"] == tref["scorer_hash"])
            a_rise = p_ask - p_ask_ref
            a_rise_ok = bool(scorer_ok and a_rise >= ASK_RISE_MIN - TOL)
            a_stop_ok = bool(p_stop >= STOP_FLOOR - TOL)
            a_ok = a_rise_ok and a_stop_ok

            b_val, b_ref = t["asks_not_done_per_run"], tref["asks_not_done_per_run"]
            b_scorer_ok = scorer_ok
            b_ok = bool(b_scorer_ok and b_val > b_ref + TOL)

            c_scorer_ok = _same_scorer(c, cref)
            c_ok = bool(c_scorer_ok and c["delta"] >= cref["delta"] - TOL)

            verdict_doc, _ = inputs.load(find_verdict(gates, m, s)[1])
            vrel = rel(find_verdict(gates, m, s)[1])
            qual = {
                g: {
                    "value": verdict_doc["criteria"][g]["value"],
                    "baseline": verdict_doc["criteria"][g].get("baseline"),
                    "threshold": verdict_doc["criteria"][g].get("threshold"),
                    "passed": bool(verdict_doc["criteria"][g]["passed"]),
                    "n": verdict_doc["criteria"][g]["n"],
                }
                for g in QUALITY_GATES
            }
            mean_asks_cap8 = (int(cells["n_done_ask"]) + int(cells["n_not_done_ask"])) / int(
                cells["n_runs"]
            )
            rows.append(
                {
                    "candidate": m,
                    "suite": s,
                    "a": {
                        "rule": (
                            f"P(ASK|not done) - reference >= {ASK_RISE_MIN} AND "
                            f"P(STOP|done) >= {STOP_FLOOR}. The base is CONTEXT, never the "
                            "threshold: an untrained base almost never stops when done, so its "
                            "P(ASK|not done) is trivially high."
                        ),
                        "p_ask_given_not_done": prov(p_ask, t["source"], inputs),
                        "reference": reference,
                        "reference_p_ask_given_not_done": prov(p_ask_ref, tref["source"], inputs),
                        "rise": a_rise,
                        "rise_min": ASK_RISE_MIN,
                        "rise_passed": a_rise_ok,
                        "p_stop_given_done": prov(p_stop, t["source"], inputs),
                        "stop_floor": STOP_FLOOR,
                        "stop_passed": a_stop_ok,
                        "scorer_match": scorer_ok,
                        "n_states": int(cells["n_states"]),
                        "n_not_done": int(cells["n_not_done"]),
                        "n_done": int(cells["n_done"]),
                        "context_base": {
                            "model_id": base_model,
                            "p_ask_given_not_done": float(bcells["p_ask_given_not_done"]),
                            "p_stop_given_done": float(bcells["p_stop_given_done"]),
                            "mean_asks_after_done": float(bcells["mean_asks_after_done"]),
                            "n_states": int(bcells["n_states"]),
                            "gated": False,
                            "source": t["source"],
                        },
                        "passed": bool(a_ok),
                    },
                    "b": {
                        "cells": "criteria.stop_2x2.value.n_not_done_ask / .n_runs",
                        "asks_not_done_per_run": prov(b_val, t["source"], inputs),
                        "n_not_done_ask": int(cells["n_not_done_ask"]),
                        "n_runs": int(cells["n_runs"]),
                        "asks_minus_after_done": t["asks_minus_after_done"],
                        "definitions_agree": t["definitions_agree"],
                        "n_skipped_no_coverage": int(cells["n_skipped_no_coverage"]),
                        "reference": reference,
                        "reference_value": prov(b_ref, tref["source"], inputs),
                        "rise": b_val - b_ref,
                        "scorer_match": b_scorer_ok,
                        "passed": bool(b_ok),
                    },
                    "c": {
                        "comparator": MATCHED_COMPARATOR,
                        "metric": MATCHED_METRIC,
                        "delta": c,
                        "reference": reference,
                        "reference_delta": cref,
                        "margin": c["delta"] - cref["delta"],
                        "scorer_match": c_scorer_ok,
                        "passed": bool(c_ok),
                    },
                    "d_reported": {
                        "cap8_coverage_delta": {
                            "value": _reported_cap8_delta(verdict_doc, vrel, cap8["delta"]),
                            "ci_lo": verdict_doc["criteria"]["evidence_coverage"].get("ci_lo"),
                            "ci_hi": verdict_doc["criteria"]["evidence_coverage"].get("ci_hi"),
                            "n": verdict_doc["criteria"]["evidence_coverage"]["n"],
                            "source": vrel,
                            "sha256": inputs.sha[vrel],
                        },
                        "cap8_coverage_delta_recomputed": cap8["delta"],
                        "mean_asks_at_cap8": mean_asks_cap8,
                        "mean_asks_at_cap8_source": (
                            "(n_done_ask + n_not_done_ask) / n_runs, " + t["source"]
                        ),
                        "mean_asks_at_cap8_matched_cost": cap8["trained_asks"],
                        "p_stop_given_done": cells["p_stop_given_done"],
                        "mean_asks_after_done": cells["mean_asks_after_done"],
                        "n_forced_stops": int(cells["n_forced_stops"]),
                        "gated": False,
                    },
                    "quality": qual,
                    "quality_passed": all(q["passed"] for q in qual.values()),
                }
            )

    per_cand: list[dict[str, Any]] = []
    for m in candidates:
        mine = [r for r in rows if r["candidate"] == m]
        fails = [
            f"({'a\u2032' if k == 'a' else k}) {r['suite']}"
            for r in mine
            for k in ("a", "b", "c")
            if not r[k]["passed"]
        ]
        qfails = [
            f"{g} {r['suite']}"
            for r in mine
            for g in QUALITY_GATES
            if not r["quality"][g]["passed"]
        ]
        core = not fails
        per_cand.append(
            {
                "candidate": m,
                "core_passed": core,
                "quality_passed": not qfails,
                "verdict": "PASS" if core and not qfails else "FAIL",
                "failed": fails,
                "quality_failed": qfails,
                "line": _one_line(m, core, fails, qfails),
            }
        )

    return {
        "rows": rows,
        "candidates": per_cand,
        "two": two,
        "mc": mc,
        "inputs": inputs,
        "reference": reference,
        "base": base_model,
        "suites": list(suites),
    }


def _same_scorer(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Two matched-cost rows are comparable only if they came from one scorer.

    When both came from the same file the question does not arise. Across files the sources are
    named and the caller is told, because a matched-cost delta is scorer output end to end.
    """
    return a["source"] == b["source"] or a["sha256"] == b["sha256"] or a["how"] == b["how"]


def _one_line(model: str, core: bool, fails: Sequence[str], qfails: Sequence[str]) -> str:
    if core and not qfails:
        return f"{model}: PASS -- (a\u2032), (b) and (c) hold on both suites; quality gates pass."
    bits = []
    if fails:
        bits.append("core " + ", ".join(fails))
    if qfails:
        bits.append("quality " + ", ".join(qfails))
    kind = "FAIL" if not core else "FAIL (core holds, quality does not)"
    return f"{model}: {kind} -- " + "; ".join(bits)


# --------------------------------------------------------------------------- rendering


def table(title: str, cols: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    w = [
        max(len(str(c)), *(len(str(r[i])) for r in rows)) if rows else len(c)
        for i, c in enumerate(cols)
    ]
    head = "  ".join(str(c).ljust(x) for c, x in zip(cols, w))
    rule = "-" * len(head)
    body = ["  ".join(str(v).ljust(x) for v, x in zip(r, w)) for r in rows]
    return "\n".join([title, rule, head, rule, *body, ""])


def _y(b: bool) -> str:
    return "PASS" if b else "FAIL"


def render(res: Mapping[str, Any]) -> str:
    rows = res["rows"]
    out = [
        table(
            f"A  P(ASK | not done) and P(STOP | done) per decision point  "
            f"[GATED: rise >= {ASK_RISE_MIN} vs {res['reference']} AND P(STOP|done) >= {STOP_FLOOR}]",
            (
                "candidate",
                "suite",
                "P(ASK|~done)",
                "ref",
                "rise",
                "rise?",
                "P(STOP|done)",
                "floor?",
                "n_states",
                "verdict",
            ),
            [
                [
                    r["candidate"],
                    r["suite"],
                    f"{r['a']['p_ask_given_not_done']['value']:.6f}",
                    f"{r['a']['reference_p_ask_given_not_done']['value']:.6f}",
                    f"{r['a']['rise']:+.6f}",
                    _y(r["a"]["rise_passed"]),
                    f"{r['a']['p_stop_given_done']['value']:.6f}",
                    _y(r["a"]["stop_passed"]),
                    str(r["a"]["n_states"]),
                    _y(r["a"]["passed"]),
                ]
                for r in rows
            ],
        ),
        table(
            "A0  CONTEXT ONLY, NEVER A THRESHOLD: the prompted base's own stop behaviour",
            ("base", "suite", "P(ASK|~done)", "P(STOP|done)", "asks after done", "n_states"),
            _base_rows(rows),
        ),
        table(
            f"B  ASK decisions at not-done states, per run  [GATED: > {res['reference']}]",
            (
                "candidate",
                "suite",
                "n_notdone_ask",
                "n_runs",
                "asks/run",
                "ref asks/run",
                "rise",
                "alt defn",
                "same",
                "verdict",
            ),
            [
                [
                    r["candidate"],
                    r["suite"],
                    str(r["b"]["n_not_done_ask"]),
                    str(r["b"]["n_runs"]),
                    f"{r['b']['asks_not_done_per_run']['value']:.6f}",
                    f"{r['b']['reference_value']['value']:.6f}",
                    f"{r['b']['rise']:+.6f}",
                    f"{r['b']['asks_minus_after_done']:.6f}",
                    "yes" if r["b"]["definitions_agree"] else "NO",
                    _y(r["b"]["passed"]),
                ]
                for r in rows
            ],
        ),
        table(
            f"C  matched-cost coverage delta vs the base prefix ladder  [GATED: >= {res['reference']}]",
            (
                "candidate",
                "suite",
                "delta",
                "ci95",
                "n",
                "ref delta",
                "margin",
                "source",
                "verdict",
            ),
            [
                [
                    r["candidate"],
                    r["suite"],
                    f"{r['c']['delta']['delta']:+.6f}",
                    _ci(r["c"]["delta"]),
                    str(r["c"]["delta"]["n"]),
                    f"{r['c']['reference_delta']['delta']:+.6f}",
                    f"{r['c']['margin']:+.6f}",
                    r["c"]["delta"]["how"],
                    _y(r["c"]["passed"]),
                ]
                for r in rows
            ],
        ),
        table(
            "D  REPORTED, NEVER GATED: cap-8 and the recorded quality gates",
            (
                "candidate",
                "suite",
                "cap8 dcov",
                "asks@cap8",
                "P(STOP|done)",
                "asks after done",
                "malformed",
                "distinct3",
                "len TOST",
            ),
            [
                [
                    r["candidate"],
                    r["suite"],
                    f"{r['d_reported']['cap8_coverage_delta']['value']:+.6f}",
                    f"{r['d_reported']['mean_asks_at_cap8']:.3f}",
                    f"{float(r['d_reported']['p_stop_given_done']):.6f}",
                    f"{float(r['d_reported']['mean_asks_after_done']):.4f}",
                    f"{r['quality']['malformed']['value']:.6f}<={r['quality']['malformed']['baseline']:.6f} {_y(r['quality']['malformed']['passed'])}",
                    f"{r['quality']['distinct3']['value']:.4f} {_y(r['quality']['distinct3']['passed'])}",
                    _y(r["quality"]["length_equivalence"]["passed"]),
                ]
                for r in rows
            ],
        ),
        table(
            "V  VERDICT",
            ("candidate", "(a\u2032)+(b)+(c)", "quality", "verdict", "failed"),
            [
                [
                    c["candidate"],
                    _y(c["core_passed"]),
                    _y(c["quality_passed"]),
                    c["verdict"],
                    ", ".join([*c["failed"], *(f"q:{q}" for q in c["quality_failed"])]) or "--",
                ]
                for c in res["candidates"]
            ],
        ),
    ]
    out.append("\n".join(c["line"] for c in res["candidates"]) + "\n")
    return "\n".join(out)


def _base_rows(rows: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """One row per suite. The base is printed so a reader can see WHY it is not the comparator:
    it barely stops when done, so its P(ASK|not done) is near 1 for free."""
    seen: set[str] = set()
    out: list[list[str]] = []
    for r in rows:
        b = r["a"]["context_base"]
        if r["suite"] in seen:
            continue
        seen.add(r["suite"])
        out.append(
            [
                b["model_id"],
                r["suite"],
                f"{b['p_ask_given_not_done']:.6f}",
                f"{b['p_stop_given_done']:.6f}",
                f"{b['mean_asks_after_done']:.4f}",
                str(b["n_states"]),
            ]
        )
    return out


def _ci(d: Mapping[str, Any]) -> str:
    if d.get("ci_lo") is None or d.get("ci_hi") is None:
        return "--"
    return f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]"


def render_reproduction(res: Mapping[str, Any]) -> str:
    """docs/TRAINING.md 5.3.1 / 5.3.2 beside what this tool just read. Never gates anything."""
    rows: list[list[str]] = []
    seen_base: set[str] = set()
    for (m, s), t in sorted(res["two"].items()):
        v, b = t["block"]["value"], t["block"]["baseline"]
        if s not in seen_base:
            seen_base.add(s)
            for k in ("p_stop_given_done", "p_ask_given_not_done", "mean_asks_after_done"):
                want = RECORDED_2X2["base"][k]
                if isinstance(want, dict):
                    want = want[s]
                rows.append(
                    _rep_row(f"{res['base']} ({s})", k, float(b[k]), want, RECORDED_2X2_TOL)
                )
        for k in ("p_stop_given_done", "p_ask_given_not_done", "mean_asks_after_done"):
            rows.append(
                _rep_row(f"{m} ({s})", k, float(v[k]), RECORDED_2X2["rung1"][k], RECORDED_2X2_TOL)
            )
    for (m, s, comp), c in sorted(res["mc"].items()):
        if comp != MATCHED_COMPARATOR or (m, s) not in RECORDED_MATCHED_K:
            continue
        d, lo, hi = RECORDED_MATCHED_K[(m, s)]
        rows.append(
            _rep_row(f"{m} ({s})", "matched_k dcov", c["delta"], (d, d), RECORDED_MATCHED_TOL)
        )
        if c["ci_lo"] is not None:
            rows.append(
                _rep_row(f"{m} ({s})", "  ci_lo", c["ci_lo"], (lo, lo), RECORDED_MATCHED_TOL)
            )
            rows.append(
                _rep_row(f"{m} ({s})", "  ci_hi", c["ci_hi"], (hi, hi), RECORDED_MATCHED_TOL)
            )
    return table(
        "R  REPRODUCTION: measured here vs docs/TRAINING.md 5.3.1-5.3.2 as recorded",
        ("arm", "quantity", "measured", "recorded", "agrees"),
        rows,
    )


def _rep_row(arm: str, q: str, got: float, want: tuple[float, float], tol: float) -> list[str]:
    lo, hi = want
    ok = (lo - tol) <= got <= (hi + tol)
    shown = f"{lo:.4g}" if lo == hi else f"{lo:.4g}-{hi:.4g}"
    return [arm, q, f"{got:+.6f}", shown, "yes" if ok else "NO"]


# --------------------------------------------------------------------------- output


def record(res: Mapping[str, Any], argv: Sequence[str], stamp: str) -> dict[str, Any]:
    return {
        "tool": "scripts/select_checkpoint.py (N1 selection, plan v4 4.2); promoted from scratch",
        "tool_path": str(Path(__file__).resolve()),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "generated_at_utc": stamp,
        "argv": list(argv),
        "rule": (
            "plans/2026-09-15-execution-plan-v4.md section 4.2, criterion (a) revised after "
            "the 4B reading (the untrained base is a degenerate comparator on the stop axis, so "
            f"it is context and never the threshold). (a') P(ASK|not done) must RISE by >= "
            f"{ASK_RISE_MIN} against the REFERENCE and P(STOP|done) must stay >= {STOP_FLOOR}; (b) "
            "criteria.stop_2x2.value.n_not_done_ask / .n_runs must RISE against the reference; "
            "(c) matched-cost evidence_coverage delta at comparator matched_k must not fall "
            "below the reference's; (d) cap-8 delta, mean asks at cap 8 and the base row are "
            "reported, never gated; distinct3 and malformed are printed with the verdict's own "
            "pass/fail."
        ),
        "reference": res["reference"],
        "base": res["base"],
        "suites": res["suites"],
        "candidates": res["candidates"],
        "rows": res["rows"],
        "inputs": dict(sorted(res["inputs"].sha.items())),
    }


# --------------------------------------------------------------------------- self-test


def _synth_verdict(
    *,
    p_ask: float,
    base_p_ask: float,
    n_not_done_ask: int,
    n_runs: int,
    mc_delta: float,
    suite: str,
    quality_ok: bool = True,
    degenerate: bool = False,
    p_stop: float = 0.90,
) -> dict[str, Any]:
    n_not = 500
    cells = {
        "p_stop_given_done": p_stop,
        "p_ask_given_not_done": p_ask,
        "n_done": 60,
        "n_not_done": n_not,
        "n_done_stop": 57,
        "n_done_ask": 3,
        "n_not_done_ask": n_not_done_ask,
        "n_not_done_stop": n_not - n_not_done_ask,
        "n_states": 60 + n_not,
        "n_runs": n_runs,
        "n_forced_stops": 0,
        "n_skipped_no_coverage": 0,
        "mean_asks_after_done": 3 / n_runs,
    }
    # The base: the degenerate comparator (a') exists to refuse -- it almost never stops when
    # done, so its P(ASK|not done) is near 1 for free. Printed for context, gated on by nothing.
    base = dict(
        cells,
        p_ask_given_not_done=base_p_ask,
        p_stop_given_done=0.15,
        n_runs=2 * n_runs,
        n_states=4 * n_not,
    )
    stop = {
        "value": cells,
        "baseline": base,
        "columns": {"done_axis": "frontier_q#t >= 1-1e-12 before the decision at t"},
        "gated": True,
        "passed": False,
        "threshold": "both cells >= baseline",
        "n": cells["n_states"],
        "reason": "",
    }
    if degenerate:
        stop = {
            "value": {
                "n_done": n_runs,
                "n_not_done": 0,
                "p_ask_given_not_done": None,
                "p_stop_given_done": 0.9,
            },
            "baseline": {
                "n_done": 2 * n_runs,
                "n_not_done": 0,
                "p_ask_given_not_done": None,
                "p_stop_given_done": 0.47,
            },
            "columns": {"done_axis": "scores.metric_name='stop_undershoot'"},
            "gated": True,
            "passed": False,
            "threshold": "both cells >= baseline",
            "n": n_runs,
            "reason": "one cell has no runs, so it cannot be compared",
        }
    q = 1 if quality_ok else 0
    return {
        "passed": False,
        "criteria": {
            "stop_2x2": stop,
            "malformed": {
                "value": 0.0 if q else 0.01,
                "baseline": 0.0,
                "threshold": "<= baseline",
                "gated": True,
                "passed": bool(q),
                "n": n_runs,
                "reason": "",
            },
            "distinct3": {
                "value": 0.80,
                "baseline": 0.40,
                "threshold": 0.65,
                "gated": True,
                "passed": True,
                "n": n_runs,
                "reason": "",
            },
            "length_equivalence": {
                "value": 12.0,
                "baseline": 12.5,
                "threshold": 0.1,
                "gated": True,
                "passed": True,
                "n": 400,
                "reason": "",
            },
            "evidence_coverage": {
                "value": -0.05,
                "baseline": None,
                "gated": True,
                "passed": False,
                "n": n_runs,
                "reason": "",
                "ci_lo": -0.09,
                "ci_hi": -0.01,
                "threshold": "delta > 0, CI excludes 0",
            },
            "cad_ge2": {
                "value": -0.1,
                "baseline": None,
                "gated": True,
                "passed": False,
                "n": 80,
                "reason": "",
                "ci_lo": -0.2,
                "ci_hi": -0.02,
                "threshold": "delta > 0, CI excludes 0",
            },
            "facet_breadth": {
                "value": -0.07,
                "baseline": None,
                "gated": True,
                "passed": False,
                "n": n_runs,
                "reason": "",
                "ci_lo": -0.1,
                "ci_hi": -0.01,
                "threshold": "CI upper bound >= 0",
            },
            "newly_reachable_share": {
                "value": 0.1,
                "baseline": None,
                "gated": False,
                "passed": True,
                "n": 50,
                "reason": "reported",
                "ci_lo": -0.01,
                "ci_hi": 0.2,
            },
        },
        "coverage_rule": {
            "contrasts": [
                {
                    "checkpoint": "SELF",
                    "suite_id": suite,
                    "metric": "evidence_coverage",
                    "comparator": "matched_k",
                    "delta": mc_delta,
                    "ci_lo": mc_delta - 0.02,
                    "ci_hi": mc_delta + 0.02,
                    "p_value": 0.01,
                    "n_tasks": 132,
                    "trained_asks": 3.0,
                    "base_asks": 3.0,
                },
                {
                    "checkpoint": "SELF",
                    "suite_id": suite,
                    "metric": "evidence_coverage",
                    "comparator": "cap8",
                    "delta": -0.05,
                    "ci_lo": -0.09,
                    "ci_hi": -0.01,
                    "p_value": 0.02,
                    "n_tasks": 132,
                    "trained_asks": 3.0,
                    "base_asks": 6.3,
                },
            ]
        },
        "selection": {
            "checkpoint_arm": "inquirer_trained",
            "baseline_arm": "inquirer_prompted",
            "grid_name": f"dev_select_{suite}",
            "baseline_grid_names": [f"dev_baseline_{suite}"],
            "baseline_model_id": "qwen3-8b-base",
            "baseline_model_id_source": "calls.parquet actor='inquirer'",
            "scorer_hash": "0" * 64,
            "parquet_dir": "synthetic",
        },
        "pairing": {"key": "(suite_id, task_id, seed)"},
        "bootstrap": {"seed": 0, "n_resamples": 1000, "unit": "(suite_id, task_id)"},
        "k_matches_baseline": {suite: None},
        "thresholds": {"length_margin": 0.1, "distinct_floor": 0.65},
    }


def self_test() -> int:
    """A synthetic gate dir: one candidate that passes, and one that fails each of (a), (b), (c).

    Rule 2: each failing case differs from the passing one in exactly ONE cell, so a test that
    goes green cannot be green for a second reason. `--self-test` needs no parquet, no gold and
    no network: the matched-cost number comes from the in-verdict `coverage_rule` block, which
    is also how the T19 shape is exercised.
    """
    ref_ask_per_run, ref_mc, ref_ask = 400 / 132, 0.049242, 0.860
    a = "(\u2032".replace("(", "(a")  # "(a')" -- the label the verdict prints
    specs = {
        # reference: P(ASK|~done) 0.860, P(STOP|done) 0.90, 400 asks at not-done over 132 runs
        "ref": dict(p_ask=ref_ask, n_not_done_ask=400, mc_delta=ref_mc),
        "pass_all": dict(p_ask=ref_ask + 0.040, n_not_done_ask=440, mc_delta=ref_mc + 0.010),
        # each failing case differs from pass_all in EXACTLY ONE cell
        "fail_a_rise": dict(p_ask=ref_ask + 0.020, n_not_done_ask=440, mc_delta=ref_mc + 0.010),
        "fail_a_stop": dict(
            p_ask=ref_ask + 0.040, n_not_done_ask=440, mc_delta=ref_mc + 0.010, p_stop=0.840
        ),
        "fail_b": dict(p_ask=ref_ask + 0.040, n_not_done_ask=400, mc_delta=ref_mc + 0.010),
        "fail_c": dict(p_ask=ref_ask + 0.040, n_not_done_ask=440, mc_delta=ref_mc - 0.001),
        "fail_quality": dict(
            p_ask=ref_ask + 0.040, n_not_done_ask=440, mc_delta=ref_mc + 0.010, quality_ok=False
        ),
    }
    want = {
        "pass_all": ("PASS", [], []),
        "fail_a_rise": ("FAIL", [f"{a}) musique", f"{a}) strategyqa"], []),
        # P(ASK|not done) rises by 0.040 but stop-when-done drops to 0.840: (a') must FAIL
        "fail_a_stop": ("FAIL", [f"{a}) musique", f"{a}) strategyqa"], []),
        "fail_b": ("FAIL", ["(b) musique", "(b) strategyqa"], []),
        "fail_c": ("FAIL", ["(c) musique", "(c) strategyqa"], []),
        "fail_quality": ("FAIL", [], ["malformed musique", "malformed strategyqa"]),
        # the reference against itself: rise 0 on both (a') and (b)
        "ref": ("FAIL", [f"{a}) musique", "(b) musique", f"{a}) strategyqa", "(b) strategyqa"], []),
    }
    fails: list[str] = []
    with tempfile.TemporaryDirectory(prefix="n1-selftest-") as td:
        gate = Path(td) / "gate"
        (gate / "rerun-stop2x2").mkdir(parents=True)
        for name, spec in specs.items():
            for suite in SUITES:
                v = _synth_verdict(base_p_ask=0.96, n_runs=132, suite=suite, **spec)
                for c in v["coverage_rule"]["contrasts"]:
                    c["checkpoint"] = name
                (gate / f"{name}.{suite}.json").write_text(json.dumps(v, indent=2))
                (gate / "rerun-stop2x2" / f"{name}.{suite}.json").write_text(
                    json.dumps(v, indent=2)
                )
        res = evaluate(
            [gate],
            candidates=list(specs),
            reference="ref",
            base_model="qwen3-8b-base",
            suites=SUITES,
            p13=REPO / DEFAULT_P13,
            gold_root=REPO / "data" / "gold",
        )
        print(render(res))
        got = {c["candidate"]: c for c in res["candidates"]}
        for name, (verdict, failed, qfailed) in want.items():
            g = got[name]
            if (g["verdict"], g["failed"], g["quality_failed"]) != (verdict, failed, qfailed):
                fails.append(
                    f"{name}: expected {verdict} failed={failed} quality={qfailed}, got "
                    f"{g['verdict']} failed={g['failed']} quality={g['quality_failed']}"
                )
        # the base is NOT the comparator: it scores 0.96 P(ASK|~done) at 0.15 P(STOP|done), and
        # every candidate here sits below it on the ask axis yet some of them pass (a').
        pa = next(
            x for x in res["rows"] if x["candidate"] == "pass_all" and x["suite"] == "musique"
        )
        if (
            pa["a"]["context_base"]["p_ask_given_not_done"]
            <= pa["a"]["p_ask_given_not_done"]["value"]
        ):
            fails.append("the synthetic base does not out-score the candidates on P(ASK|~done)")
        if pa["a"]["context_base"].get("gated") is not False or not pa["a"]["passed"]:
            fails.append(
                "a candidate below the base's P(ASK|~done) failed (a'); the base is "
                "context, not the threshold"
            )
        fs = next(
            x for x in res["rows"] if x["candidate"] == "fail_a_stop" and x["suite"] == "musique"
        )
        if fs["a"]["rise_passed"] is not True or fs["a"]["stop_passed"] is not False:
            fails.append("fail_a_stop did not isolate the P(STOP|done) floor")

        # the reference's own (b) margin is exactly 0: a rise is STRICT, not >=
        r = next(x for x in res["rows"] if x["candidate"] == "ref" and x["suite"] == "musique")
        if abs(r["b"]["asks_not_done_per_run"]["value"] - ref_ask_per_run) > 1e-12:
            fails.append("ref asks/run is not n_not_done_ask/n_runs")
        if r["b"]["rise"] != 0.0 or r["b"]["passed"]:
            fails.append("a zero rise passed (b); the plan says the count must RISE")
        if not r["b"]["definitions_agree"]:
            fails.append("the two wordings of (b) disagree with n_skipped_no_coverage = 0")

        # a pre-f348feb stop_2x2 must be REFUSED, not quoted
        deg = Path(td) / "degenerate"
        deg.mkdir()
        for suite in SUITES:
            v = _synth_verdict(
                base_p_ask=0.96,
                n_runs=132,
                suite=suite,
                degenerate=True,
                p_ask=0.94,
                n_not_done_ask=440,
                mc_delta=ref_mc,
            )
            (deg / f"old.{suite}.json").write_text(json.dumps(v, indent=2))
        try:
            evaluate(
                [deg],
                candidates=["old"],
                reference="old",
                base_model="qwen3-8b-base",
                suites=SUITES,
                p13=REPO / DEFAULT_P13,
                gold_root=REPO / "data" / "gold",
            )
        except Refused as exc:
            if "stop_undershoot" not in str(exc) and "per-state cells" not in str(exc):
                fails.append(f"degenerate 2x2 refused for the wrong reason: {exc}")
        else:
            fails.append("a pre-f348feb stop_2x2 was QUOTED; docs/TRAINING.md 5.3.1 forbids it")

    for f in fails:
        print(f"SELF-TEST FAIL: {f}", file=sys.stderr)
    print(
        f"self-test: {'PASS' if not fails else f'{len(fails)} FAILURE(S)'} "
        f"({len(want)} candidates + 6 invariants + 1 refusal)"
    )
    return 0 if not fails else 1


# --------------------------------------------------------------------------- cli


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--gate-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="a gate artifact directory; repeat for several",
    )
    ap.add_argument("--candidates", default="", help="comma-separated served model ids")
    ap.add_argument("--reference", default="qwen3-8b-sft-headline")
    ap.add_argument("--base", default="qwen3-8b-base")
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument(
        "--repo",
        default=os.environ.get("PI_REPO") or str(Path.cwd()),
        help="the checkout root; default $PI_REPO, else the working directory",
    )
    ap.add_argument(
        "--p13",
        default=DEFAULT_P13,
        help="recorded matched-cost provenance, relative to --repo unless absolute; "
        "used only while its sha256 still match the parquet files on disk",
    )
    ap.add_argument(
        "--gold-root",
        default=None,
        help="PI_GOLD_ROOT for the scripts/matched_cost.py SUBPROCESS only "
        "(default <repo>/data/gold)",
    )
    ap.add_argument(
        "--out", default=None, help="default <repo>/artifacts/n1/selection.<stamp>.json"
    )
    ap.add_argument(
        "--reproduce",
        action="store_true",
        help="also print measured vs docs/TRAINING.md 5.3.1-5.3.2 (reports only)",
    )
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    try:
        set_repo(Path(a.repo))
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if a.self_test:
        return self_test()
    if not a.gate_dir or not a.candidates:
        ap.error("--gate-dir and --candidates are required")

    gates = [Path(g) if Path(g).is_absolute() else REPO / g for g in a.gate_dir]
    for g in gates:
        if not g.is_dir():
            print(f"REFUSED: {g} is not a directory", file=sys.stderr)
            return 2
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        res = evaluate(
            gates,
            candidates=[c for c in a.candidates.split(",") if c],
            reference=a.reference,
            base_model=a.base,
            suites=[s for s in a.suites.split(",") if s],
            p13=Path(a.p13) if Path(a.p13).is_absolute() else REPO / a.p13,
            gold_root=Path(a.gold_root) if a.gold_root else REPO / "data" / "gold",
        )
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(render(res))
    if a.reproduce:
        print(render_reproduction(res))
    out = Path(a.out) if a.out else out_dir() / f"selection.{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        record(res, ["select_checkpoint.py", *argv], stamp), indent=2, sort_keys=True
    )
    out.write_text(payload + "\n")
    print(f"written: {rel(out)}")
    return 0 if all(c["verdict"] == "PASS" for c in res["candidates"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
