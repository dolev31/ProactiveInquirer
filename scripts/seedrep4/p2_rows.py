"""The question-pair rows of tab:app-heldout-methods, read with the table's own function.

Declared in artifacts/seedrep4_20260924/RULES.md section 5 before any P2 unit existed.

  P2b  qwen3-8b-dpo-pooled-control    fills the printed row "question pairs, from the reference"
  P2a  qwen3-8b-dpo-headline-control  an added row, the recipe's initialization

Both at code 3ae099d0 through http://127.0.0.1:4000 (base_url_sha e6436030), against the comparator
the selected row uses (decomposition_test run_ids baseline_qwen3-8b-base_3ae099d0.txt).

LOCK FIRST, in the same store: the selected row (qwen3-8b-dpo-stacked-notdone-both, run_ids
stacked_test_3ae099d0.txt) must reproduce the printed +0.1169 / +0.0628 / +0.0737 and their intervals
to four decimals before any P2 cell is read. The agreement store must reproduce the decomposition
lane's headline-control reading at code 107ef222 (+0.1313 / +0.0730 / +0.0737) before the two
identities are compared.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "decomposition_test"))

import contrast as dc  # noqa: E402  (the table's own contrast function)

SUITES = ("musique", "strategyqa", "wiki2")
BASE = "qwen3-8b-base"
STACKED = "qwen3-8b-dpo-stacked-notdone-both"
ARMS = {"P2a": "qwen3-8b-dpo-headline-control", "P2b": "qwen3-8b-dpo-pooled-control"}
PRINTED_STACKED = {
    "musique": (0.1169, 0.0823, 0.1544),
    "strategyqa": (0.0628, 0.0254, 0.1008),
    "wiki2": (0.0737, 0.0469, 0.1050),
}
PRINTED_HC_107 = {
    "musique": (0.1313, 0.0990, 0.1671),
    "strategyqa": (0.0730, 0.0395, 0.1091),
    "wiki2": (0.0737, 0.0481, 0.1044),
}
RUN_IDS = REPO / "artifacts" / "decomposition_test_20260918" / "run_ids"


def _ids(p: Path) -> list[str]:
    return [x.strip() for x in p.read_text().splitlines() if x.strip() and not x.startswith("#")]


def _cells(res: dict) -> dict:
    return {
        s: {
            k: res["by_suite"][s][k]
            for k in (
                "n_tasks",
                "delta",
                "ci_lo",
                "ci_hi",
                "trained_mean_k",
                "baseline_mean_n_asks",
                "n_baseline_shorter_than_k",
                "cap8_coverage_delta",
                "outspent_comparator",
            )
        }
        | {"symmetric": res["by_suite"][s]["symmetric"]}
        for s in SUITES
    }


def _lock(cells: dict, printed: dict, what: str) -> dict:
    out = {}
    for s, (d, lo, hi) in printed.items():
        got = (cells[s]["delta"], cells[s]["ci_lo"], cells[s]["ci_hi"])
        ok = all(round(g, 4) == p for g, p in zip(got, (d, lo, hi)))
        out[s] = {"printed": [d, lo, hi], "reproduced": list(got), "ok": ok}
        if not ok:
            raise SystemExit(f"LOCK FAILED {what} {s}: {got} vs printed {(d, lo, hi)}")
    return out


def _read(store: Path, model: str, scorer_hash: str) -> dict:
    res = dc.select_and_contrast(
        store, checkpoint_model_id=model, baseline_model_id=BASE, scorer_hash=scorer_hash
    )
    cells = _cells(res)
    stab = dc.stability_recheck(
        store,
        checkpoint_model_id=model,
        baseline_model_id=BASE,
        by_suite=res["by_suite"],
        scorer_hash=scorer_hash,
    )
    return {"selection": res["selection"], "cells": cells, "stability": stab}


def _agree(a: dict, b: dict) -> dict:
    out = {}
    for s in SUITES:
        x, y = a[s], b[s]
        same_sign = (x["delta"] > 0) == (y["delta"] > 0)
        x_in_y = y["ci_lo"] <= x["delta"] <= y["ci_hi"]
        y_in_x = x["ci_lo"] <= y["delta"] <= x["ci_hi"]
        out[s] = {
            "new_3ae099d": [x["delta"], x["ci_lo"], x["ci_hi"]],
            "old_107ef222": [y["delta"], y["ci_lo"], y["ci_hi"]],
            "same_sign": same_sign,
            "new_inside_old_interval": x_in_y,
            "old_inside_new_interval": y_in_x,
            "agree": same_sign and x_in_y and y_in_x,
        }
    out["cell_for_cell"] = all(out[s]["agree"] for s in SUITES)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--p2-store", type=Path, required=True)
    ap.add_argument("--p2-runs", type=Path, required=True, help="the farm's runs dir")
    ap.add_argument("--agree-store", type=Path, required=True)
    ap.add_argument("--agree-runs", type=Path, required=True)
    ap.add_argument("--p2-ids", type=Path, nargs=2, required=True, help="P2a then P2b id lists")
    ap.add_argument("--scorer-hash", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    out: dict = {"scorer_hash": a.scorer_hash, "lock": {}, "non_vacuity": {}, "rows": {}}

    stacked = _read(a.p2_store, STACKED, a.scorer_hash)
    out["lock"]["stacked_3ae099d"] = _lock(stacked["cells"], PRINTED_STACKED, "stacked")
    out["lock"]["stacked_selection"] = stacked["selection"]
    for (lab, model), idf in zip(ARMS.items(), a.p2_ids):
        ids = _ids(idf)
        out["non_vacuity"][lab] = dc.assert_non_vacuous(
            a.p2_store, a.p2_runs, ids, scorer_hash=a.scorer_hash
        )
        r = _read(a.p2_store, model, a.scorer_hash)
        dagger = any(
            r["cells"][s]["trained_mean_k"] >= r["cells"][s]["baseline_mean_n_asks"] for s in SUITES
        )
        r["dagger_rule_applies"] = dagger
        r["printed_reading"] = "symmetric" if dagger else "matched_cost"
        out["rows"][lab] = {"model": model, **r}

    old = _read(a.agree_store, ARMS["P2a"], a.scorer_hash)
    out["lock"]["headline_control_107ef222"] = _lock(old["cells"], PRINTED_HC_107, "hc107")
    out["non_vacuity"]["hc107"] = dc.assert_non_vacuous(
        a.agree_store,
        a.agree_runs,
        _ids(RUN_IDS / "headline_control_test_107ef222_bonus.txt"),
        scorer_hash=a.scorer_hash,
    )
    out["agreement_hc"] = _agree(out["rows"]["P2a"]["cells"], old["cells"])
    out["agreement_hc"]["old_selection"] = old["selection"]
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
