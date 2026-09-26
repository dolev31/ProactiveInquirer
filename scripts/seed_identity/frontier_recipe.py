"""The retrieval-budget frontier for the recipe (seeds 1 and 2) and both prompted comparators, one commit.

The published frontier (artifacts/frontier/FRONTIER.md, artifacts/frontier_strategyqa/
FRONTIER_STRATEGYQA.md) ran the registered pin, which is seed 0, whose final DPO stage a resume cut to
about a tenth (artifacts/seed_identity_20260923), and its two panels ran at two different commits.
This reads the same five budget ceilings for the recipe and for both prompted comparators at ONE code
version (a61c4f4be3e9) through ONE base URL: caps 4/12/16/24 from artifacts/frontier_recipe_20260923
(grids conf/grids/frontier_recipe_cap*.yaml), cap 8 from artifacts/structured_baselines_clean_20260922,
whose runs share that code, base URL and frozen roles.

Estimator = the published frontier's: a level is `cluster_bootstrap` over template clusters (the
unweighted mean over clusters, BCa), x is the pooled mean of retrieval_calls, ceiling-hit is
stop_reason in {budget, max_turns}. The recipe's unit is (task, seed) with seeds 1 and 2 averaged
within it, so its interval is over tasks and does not include variation between training seeds.
Contrasts are `paired_difference` on (task, seed), clusters on template, at 10,000 resamples. They are
shared-ceiling contrasts, NOT equal-spend ones: at one ceiling each arm stops where it chooses.

LOCK: before any new cell is written, the same level function pointed at the published stores and
run-id lists must reproduce every level row of both published records (value, interval, mean
retrieval calls) to 1e-4 at the published 1,000 resamples.

POPULATION, refused rather than read: a cell must hold 400 eligible runs on 400 distinct (task, seed)
keys, one code_version, one model_pin_hash, the cell's own budget_cap, the named inquirer model on
every inquirer call; across every cell of the new reading, every role's base_url_sha must be one value
and no manifest may be dirty.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

from pi_eval.report import ELIGIBLE
from pi_eval.stats.inference import cluster_bootstrap, paired_difference

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from paper_figures_iclr import (  # noqa: E402
    FRONTIER_MUSIQUE,
    FRONTIER_STRATEGYQA,
    parse_level_table,
    read_run_ids,
)

SB = REPO / "artifacts/structured_baselines_clean_20260922"
FR = REPO / "artifacts/frontier_recipe_20260923"
CAPS = (4, 8, 12, 16, 24)
SUITES = ("musique", "strategyqa")
CODE = "a61c4f4be3e9"
N_CELL = 400
CEILING = ("budget", "max_turns")

# label -> (arm_id, inquirer model, cap-8 store, cap-8 runs root)
ARMS = {
    "teacher": ("inquirer_prompted", "openai/aws/gpt-oss-120b", "scores_parquet", "runs"),
    "base": ("inquirer_prompted", "qwen3-8b-base", "scores_parquet_8b", "runs_8b"),
    "s1": (
        "inquirer_trained",
        "qwen3-8b-dpo-stacked-notdone-both-s1",
        "scores_parquet_s1",
        "runs_s1",
    ),
    "s2": (
        "inquirer_trained",
        "qwen3-8b-dpo-stacked-notdone-both-s2",
        "scores_parquet_s2",
        "runs_s2",
    ),
    "s0": ("inquirer_trained", "qwen3-8b-dpo-stacked-notdone-both", "scores_parquet", "runs"),
}
CAP8_ONLY = {"s0"}


def _stores(label: str, cap: int) -> tuple[Path, Path]:
    _, _, store8, root8 = ARMS[label]
    if cap == 8:
        return SB / store8, SB / root8
    return FR / f"scores_parquet_{label}", FR / f"runs_{label}"


def unit_rows(store: Path, arm: str, model: str, suite: str, cap: int) -> dict[str, dict]:
    """(task|seed) -> one eligible run of this arm, inquirer model, suite and ceiling."""
    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT r.run_id, r.task_id, r.seed, COALESCE(NULLIF(r.template_id, ''), r.task_id),
               r.retrieval_calls, s.value, r.stop_reason, r.code_version, r.model_pin_hash
        FROM read_parquet('{store / "runs.parquet"}') r
        JOIN read_parquet('{store / "scores.parquet"}') s
          ON s.run_id = r.run_id AND s.metric_name = 'evidence_coverage'
        WHERE {ELIGIBLE} AND r.arm_id = ? AND r.suite_id = ? AND r.split = 'test'
          AND r.budget_cap = ?
        """,
        [arm, suite, cap],
    ).fetchall()
    models: dict[str, set[str]] = {}
    for rid, m in con.execute(
        f"SELECT DISTINCT run_id, model FROM read_parquet('{store / 'calls.parquet'}') "
        "WHERE actor = 'inquirer'"
    ).fetchall():
        models.setdefault(rid, set()).add(m)
    out: dict[str, dict] = {}
    for rid, task, seed, cl, rc, y, stop, code, pin in rows:
        if models.get(rid) != {model}:
            continue
        key = f"{task}|{int(seed)}"
        if key in out:
            raise SystemExit(f"REFUSING: two runs at {suite} {key} cap {cap} in {store}")
        out[key] = {
            "run_id": rid,
            "cl": cl,
            "rc": float(rc),
            "y": None if y is None else float(y),
            "stop": stop,
            "code": code,
            "pin": pin,
        }
    return out


def level(units: dict[str, dict], n_boot: int) -> dict:
    groups: dict[str, list[float]] = {}
    for u in units.values():
        if u["y"] is not None:
            groups.setdefault(u["cl"], []).append(u["y"])
    point, lo, hi = cluster_bootstrap(list(groups.values()), n_boot=n_boot, seed=0)
    n = len(units)
    return {
        "n": n,
        "n_clusters": len(groups),
        "y": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "x_mean_retrieval_calls": sum(u["rc"] for u in units.values()) / n,
        "ceiling_hit": sum(1 for u in units.values() if u.get("stop") in CEILING) / n,
    }


def pooled(a: dict[str, dict], b: dict[str, dict]) -> dict[str, dict]:
    """Seeds 1 and 2 averaged within (task, seed); the ceiling-hit share is the mean of the two."""
    keys = sorted(set(a) & set(b))
    if len(keys) != len(a) or len(keys) != len(b):
        raise SystemExit(f"REFUSING: the two seeds cover different units ({len(a)}, {len(b)})")
    out = {}
    for k in keys:
        if a[k]["cl"] != b[k]["cl"]:
            raise SystemExit(f"REFUSING: cluster disagrees across seeds at {k}")
        ya, yb = a[k]["y"], b[k]["y"]
        out[k] = {
            "cl": a[k]["cl"],
            "rc": (a[k]["rc"] + b[k]["rc"]) / 2,
            "y": None if ya is None or yb is None else (ya + yb) / 2,
            "stop": None,
            "hit": ((a[k]["stop"] in CEILING) + (b[k]["stop"] in CEILING)) / 2,
        }
    return out


def contrast(a: dict[str, dict], b: dict[str, dict]) -> dict:
    keys = sorted(k for k in set(a) & set(b) if a[k]["y"] is not None and b[k]["y"] is not None)
    e = paired_difference(
        {k: a[k]["y"] for k in keys},
        {k: b[k]["y"] for k in keys},
        clusters={k: a[k]["cl"] for k in keys},
        n_boot=10_000,
        n_perm=10_000,
        seed=0,
    )
    # The paper's decision rule: DECIDED only if every 50,000-resample interval under seeds
    # 101/202/303 excludes zero. n_perm only feeds the p-value, which is read at 10,000 above.
    fifty = [
        paired_difference(
            {k: a[k]["y"] for k in keys},
            {k: b[k]["y"] for k in keys},
            clusters={k: a[k]["cl"] for k in keys},
            n_boot=50_000,
            n_perm=100,
            seed=s,
        )
        for s in (101, 202, 303)
    ]
    return {
        "delta": e.point,
        "ci_lo": e.ci_lo,
        "ci_hi": e.ci_hi,
        "p": e.p_value,
        "n": len(keys),
        "x_a": sum(a[k]["rc"] for k in keys) / len(keys),
        "x_b": sum(b[k]["rc"] for k in keys) / len(keys),
        "ci_50k": [[f.ci_lo, f.ci_hi] for f in fifty],
        "decided": all(f.ci_lo > 0 for f in fifty) or all(f.ci_hi < 0 for f in fifty),
    }


def _published_units(store: Path, ids: set[str]) -> dict[str, dict]:
    con = duckdb.connect()
    units = {}
    for rid, cl, rc, y in con.execute(
        f"""
        SELECT r.run_id, COALESCE(NULLIF(r.template_id, ''), r.task_id), r.retrieval_calls, s.value
        FROM read_parquet('{store / "runs.parquet"}') r
        JOIN read_parquet('{store / "scores.parquet"}') s
          ON s.run_id = r.run_id AND s.metric_name = 'evidence_coverage'
        """
    ).fetchall():
        if rid in ids:
            units[rid] = {"cl": cl, "rc": float(rc), "y": float(y), "stop": None}
    return units


def lock() -> dict:
    """The level function must reproduce both published records' level rows before anything else."""
    out = {}
    specs = (
        (
            "musique",
            FRONTIER_MUSIQUE,
            lambda a: REPO / f"artifacts/frontier/scores_parquet.{a}",
            lambda a, c: REPO / f"artifacts/frontier/run_ids.{a}.cap{c}.txt",
        ),
        (
            "strategyqa",
            FRONTIER_STRATEGYQA,
            lambda a: REPO / "artifacts/frontier_strategyqa/scores_parquet.strategyqa",
            lambda a, c: REPO / f"artifacts/frontier_strategyqa/run_ids.{a}.cap{c}.tsv",
        ),
    )
    for suite, md, store_of, ids_of in specs:
        rec = parse_level_table(md.read_text(), "evidence_coverage")
        for arm in ("trained", "base8b", "teacher"):
            for cap in CAPS:
                want = rec[(arm, cap)]
                ids = set(read_run_ids(ids_of(arm, cap)))
                got = level(_published_units(store_of(arm), ids), 1000)
                diffs = (
                    got["y"] - want.value,
                    got["ci_lo"] - want.lo,
                    got["ci_hi"] - want.hi,
                    got["x_mean_retrieval_calls"] - want.mean_calls,
                )
                out[f"{suite}/{arm}/cap{cap}"] = {
                    "n": got["n"],
                    "want_n": want.n,
                    "max_abs_diff": max(abs(v) for v in diffs),
                    "ok": got["n"] == want.n and all(abs(v) <= 1e-4 for v in diffs),
                }
    return out


def manifests_ok(cells: dict[tuple[str, str, int], dict[str, dict]]) -> dict:
    """Every role's base_url_sha is one value across the whole reading, and no manifest is dirty."""
    base_urls: set[str] = set()
    frozen: set[tuple[str, str]] = set()
    dirty = 0
    n = 0
    for (label, _suite, cap), units in cells.items():
        _, root = _stores(label, cap)
        for u in units.values():
            m = json.loads((root / u["run_id"] / "manifest.json").read_text())
            n += 1
            dirty += bool(m.get("dirty"))
            for role, pin in (m.get("pins") or {}).items():
                base_urls.add(pin.get("base_url_sha"))
                if role != "inquirer":
                    frozen.add((role, pin.get("model_id")))
    return {
        "manifests_read": n,
        "base_url_sha": sorted(base_urls),
        "frozen_roles": sorted(frozen),
        "dirty": dirty,
        "ok": len(base_urls) == 1
        and dirty == 0
        and {m for _, m in frozen} == {"openai/aws/gpt-oss-120b"},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lock-only", action="store_true")
    args = ap.parse_args(argv)

    out: dict = {"lock": lock()}
    bad = [k for k, v in out["lock"].items() if not v["ok"]]
    if bad or args.lock_only:
        args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        n_ok = len(out["lock"]) - len(bad)
        print(f"lock: {n_ok}/{len(out['lock'])} published cells reproduced; failures {bad[:6]}")
        return 1 if bad else 0

    cells: dict[tuple[str, str, int], dict[str, dict]] = {}
    pop: list[str] = []
    for suite in SUITES:
        for label, (arm, model, _, _) in ARMS.items():
            for cap in CAPS:
                if label in CAP8_ONLY and cap != 8:
                    continue
                store, _ = _stores(label, cap)
                u = unit_rows(store, arm, model, suite, cap)
                codes = {v["code"][:12] for v in u.values()}
                pins = {v["pin"] for v in u.values()}
                if len(u) != N_CELL or codes != {CODE} or len(pins) != 1:
                    pop.append(
                        f"{suite} {label} cap{cap}: n={len(u)} codes={codes} pins={len(pins)}"
                    )
                cells[(label, suite, cap)] = u
    out["manifests"] = manifests_ok(cells)
    if pop or not out["manifests"]["ok"]:
        out["population_failures"] = pop
        args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print(f"POPULATION REFUSED: {pop[:6]} manifests={out['manifests']}")
        return 1

    out["levels"] = {}
    out["contrasts"] = {}
    for suite in SUITES:
        lv = out["levels"].setdefault(suite, {})
        ct = out["contrasts"].setdefault(suite, {})
        for cap in CAPS:
            rec = pooled(cells[("s1", suite, cap)], cells[("s2", suite, cap)])
            lev = level(rec, n_boot=10_000)
            lev["ceiling_hit"] = sum(v["hit"] for v in rec.values()) / len(rec)
            one_k = level(rec, n_boot=1000)
            lev["ci_1k"] = [one_k["ci_lo"], one_k["ci_hi"]]
            lv.setdefault("recipe", {})[str(cap)] = lev
            for label in ("teacher", "base", "s1", "s2") + (("s0",) if cap == 8 else ()):
                lv.setdefault(label, {})[str(cap)] = level(cells[(label, suite, cap)], 10_000)
            for comp in ("teacher", "base"):
                ct.setdefault(f"recipe-{comp}", {})[str(cap)] = contrast(
                    rec, cells[(comp, suite, cap)]
                )
                for s in ("s1", "s2"):
                    ct.setdefault(f"{s}-{comp}", {})[str(cap)] = contrast(
                        cells[(s, suite, cap)], cells[(comp, suite, cap)]
                    )
        # Budget response: the same questioner at the largest ceiling minus the smallest, paired.
        r24 = pooled(cells[("s1", suite, 24)], cells[("s2", suite, 24)])
        r4 = pooled(cells[("s1", suite, 4)], cells[("s2", suite, 4)])
        ct["budget_response"] = {"recipe": contrast(r24, r4)}
        for label in ("teacher", "base"):
            ct["budget_response"][label] = contrast(
                cells[(label, suite, 24)], cells[(label, suite, 4)]
            )
        # Across ceilings, paired on (task, seed): the recipe at its CHEAPEST ceiling against each
        # comparator at its most GENEROUS one. Same units, different ceilings, so this is a contrast
        # of two points on the frontier and not an equal-spend reading; x_a and x_b record the spend.
        ct["recipe_cap4_vs_cap24"] = {
            label: contrast(r4, cells[(label, suite, 24)]) for label in ("teacher", "base")
        }
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}; lock ok on {len(out['lock'])} published cells; population ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
