#!/usr/bin/env python3
"""Recover `graph_version`, the third element of the provenance triple CLAUDE.md rule 1
requires (beside `run_id` and `scorer_hash`), for gate verdicts written before
`pinq_train.gate.run_gate` began recording it in `selection` (commit 68f4f49, merged to main in
4dc17c3).

THIS IS A PROMOTION, NOT A RE-SCORE. `scorer_hash` is global over the gold set: scoring a suite
into the shared store re-stamps every run under it. This script never scores anything. It opens
the `scores.parquet` a verdict's own `selection.parquet_dir` already names and reads a column
that scoring already wrote. Nothing under `data/`, `runs/` or a gateway is touched, and
`recover_one` never opens a connection with `PI_GOLD_ROOT` set.

THIS IS A SIDECAR, NOT AN EDIT. CLAUDE.md: "recorded verdicts are immutable." `recover_one`
reads a verdict file and returns a dict describing what its `graph_version` is or would be; it
never writes to the verdict path it was given. `main` writes one row per verdict to a sidecar
JSON/TSV under `--out`, never back into `artifacts/gate/`.

THE RECOVERY REPLICATES `run_gate`'s OWN SELECTION, not a coarser "whole parquet" scan: it
re-selects the checkpoint and baseline runs the verdict named (`gate._select_runs`, unmodified,
imported) and reads `graph_version` filtered to those run ids AND the verdict's own
`scorer_hash`, exactly the query `run_gate` itself runs before writing a fresh verdict today
(`src/pinq_train/gate.py`, the block above `selection["graph_version"]`). A parquet directory
whose selected runs carry more than one `graph_version` raises `GraphVersionAmbiguous` -- the
same refusal `run_gate` would raise -- rather than averaging or guessing.

Usage:
    .venv/bin/python scripts/recover_graph_version.py --gate-root artifacts/gate --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinq_train.gate import GraphVersionAmbiguous, _con, _select_runs  # noqa: E402


def _sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _q(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _resolve_parquet_dir(raw: str, gate_root: Path | None) -> Path:
    """`selection.parquet_dir` is recorded relative to the repo root the verdict was WRITTEN
    from, which need not be the repo root this scan is RUN from (a worktree, for one). If the
    recorded path does not resolve directly, re-root the `artifacts/gate/...` suffix it carries
    under the `gate_root` this scan was actually given, rather than declaring it missing."""
    p = Path(raw)
    if (p / "runs.parquet").exists():
        return p
    if gate_root is not None:
        parts = p.parts
        for i in range(len(parts) - 1):
            if parts[i : i + 2] == ("artifacts", "gate"):
                candidate = Path(gate_root).joinpath(*parts[i + 2 :])
                if (candidate / "runs.parquet").exists():
                    return candidate
    return p


def recover_one(verdict_path: Path, *, gate_root: Path | None = None) -> dict:
    """Recover `graph_version` for one verdict file, without writing to it.

    Returns a dict with `status` in {"already_recorded", "recovered", "gap"}. Raises
    `GraphVersionAmbiguous` (imported from `pinq_train.gate`, not redefined here) if the runs
    this verdict selected carry more than one `graph_version` under its own `scorer_hash`.

    `gate_root`, if given, is used only to re-root a recorded `parquet_dir` that does not
    resolve as-is (see `_resolve_parquet_dir`); it changes nothing about which runs are
    selected or which column is read.
    """
    verdict_path = Path(verdict_path)
    d = json.loads(verdict_path.read_text())
    sel = d.get("selection", {})
    out: dict = {
        "verdict": str(verdict_path),
        "scorer_hash": sel.get("scorer_hash"),
        "grid_name": sel.get("grid_name"),
        "checkpoint_arm": sel.get("checkpoint_arm"),
        "baseline_arm": sel.get("baseline_arm"),
    }
    if sel.get("graph_version"):
        out["status"] = "already_recorded"
        out["graph_version"] = sel["graph_version"]
        return out

    parquet_dir = (
        _resolve_parquet_dir(sel["parquet_dir"], gate_root) if sel.get("parquet_dir") else None
    )
    if parquet_dir is None or not (parquet_dir / "runs.parquet").exists():
        out["status"] = "gap"
        out["gap"] = "parquet_dir_missing"
        return out
    out["parquet_dir"] = str(parquet_dir)

    con = _con(parquet_dir)
    ckpt = _select_runs(con, arm=sel["checkpoint_arm"], grids=[sel["grid_name"]], model_id=None)
    base = _select_runs(
        con,
        arm=sel["baseline_arm"],
        grids=list(sel.get("baseline_grid_names") or []),
        model_id=sel.get("baseline_model_id"),
    )
    ids = sorted(r["run_id"] for r in ckpt) + sorted(r["run_id"] for r in base)
    if not ids:
        out["status"] = "gap"
        out["gap"] = "empty_arm_on_replay"
        return out

    rows = con.execute(
        "SELECT DISTINCT graph_version FROM scores WHERE run_id IN "
        f"({', '.join(_q(i) for i in ids)}) AND scorer_hash = {_q(sel.get('scorer_hash', ''))} "
        "ORDER BY 1"
    ).fetchall()
    versions = [str(r[0]) for r in rows]
    if len(versions) > 1:
        raise GraphVersionAmbiguous(versions=versions, parquet_dir=parquet_dir, n_runs=len(ids))
    if not versions:
        out["status"] = "gap"
        out["gap"] = "no_scored_rows_for_selected_runs"
        return out

    out.update(
        status="recovered",
        graph_version=versions[0],
        n_runs=len(ids),
        run_ids_sha256=_sha(ids),
    )
    return out


def scan_gate_root(gate_root: Path) -> list[dict]:
    """Every `*.json` under `gate_root` that carries a `selection` block, recovered or reported
    as a named gap. Paths in the returned rows are relative to `gate_root`, never absolute, so
    the sidecar this feeds carries no machine-specific path."""
    gate_root = Path(gate_root)
    out = []
    for p in sorted(gate_root.rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(d, dict) or "selection" not in d:
            continue
        row = recover_one(p, gate_root=gate_root)
        row["verdict"] = str(p.relative_to(gate_root))
        if row.get("parquet_dir"):
            try:
                row["parquet_dir"] = str(Path(row["parquet_dir"]).relative_to(gate_root))
            except ValueError:
                row["parquet_dir"] = Path(row["parquet_dir"]).name
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-root", type=Path, default=ROOT / "artifacts" / "gate")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    rows = scan_gate_root(args.gate_root)
    if not rows:
        print(f"no verdicts with a `selection` block found under {args.gate_root}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verdict_graph_version.json").write_text(json.dumps(rows, indent=2, sort_keys=True))

    cols = [
        "verdict",
        "status",
        "graph_version",
        "scorer_hash",
        "grid_name",
        "checkpoint_arm",
        "baseline_arm",
        "n_runs",
        "run_ids_sha256",
        "gap",
    ]
    with (args.out / "verdict_graph_version.tsv").open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    n_already = sum(1 for r in rows if r["status"] == "already_recorded")
    n_recovered = sum(1 for r in rows if r["status"] == "recovered")
    n_gap = sum(1 for r in rows if r["status"] == "gap")
    versions = sorted({r["graph_version"] for r in rows if r.get("graph_version")})
    print(
        f"verdicts scanned: {len(rows)}  already_recorded: {n_already}  "
        f"recovered: {n_recovered}  gap: {n_gap}  distinct graph_version values: {versions}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
