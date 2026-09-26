"""FRAMES panel of Figure 4: the recipe against six comparators at cap 8, answer_correct at each own STOP.

ESTIMAND. FRAMES is gold-free, so this panel reads `answer_correct` at each policy's OWN STOP under
the shared ceiling of 8 retrieval calls. That is not the MuSiQue/StrategyQA panels' coverage at
equal spend; the figure has to say so.

ARMS, all at cap 8, code a61c4f4be3e9, one base URL (proxy 4030), frozen drafter and answerer
openai/aws/gpt-oss-120b, one root per (arm, questioner) under artifacts/frames_recipe_20260923/:
  recipe      runs_s1 + runs_s2, inquirer_trained @ qwen3-8b-dpo-stacked-notdone-both-s1 / -s2,
              averaged WITHIN the task (frames_recipe.pool_seeds), grid frames_recipe_cap8
  base        runs_base,      inquirer_prompted @ qwen3-8b-base,            grid frames_recipe_cap8
  teacher     runs_teacher,   inquirer_prompted @ openai/aws/gpt-oss-120b,  grid frames_recipe_cap8
  par2_8b     runs_par2_8b,   par2_rag @ qwen3-8b-base,                     grid frames_structured_cap8
  par2_120b   runs_par2_120b, par2_rag @ openai/aws/gpt-oss-120b,           grid frames_structured_cap8
  par2_opus   runs_par2_opus, par2_rag @ openai/aws/claude-opus-5,          grid frames_structured_cap8
  opus        runs_opus,      inquirer_prompted @ openai/aws/claude-opus-5, grid frames_structured_cap8
frames_structured_cap8 (dc62cc3) is frames_recipe_cap8 with only the name and the arms changed;
grid_name is outside semantic_hash, so the new cells pair with the recipe's by (task, seed). The
model ids are the ones the HPC job passes as PI_MODEL_INQUIRER and the ones runs carry in
calls.jsonl (openai/aws/claude-opus-5 is how that model is recorded through this proxy).

ESTIMATOR: frames_recipe.py's, imported, not copied. Read from the FRAMES record (`parse_estimator`),
cluster = task (824), levels `cluster_bootstrap` at 10,000 resamples (seed 0), each contrast
recipe-minus-comparator paired over the SAME unit set (refused otherwise) at 10,000 resamples, also
at 1,000 (whether a bound within 0.01 of zero flips its verdict is recorded), and DECIDED only if
the 50,000-resample interval excludes zero on the same side under each of seeds 101, 202 and 303.

LOCK (every stage, first): on the existing stores, the recipe, s1, s2, base and teacher cap-8 levels
and the recipe-base/cap8 and recipe-teacher/cap8 contrasts are rebuilt by that code path and must
equal artifacts/frames_recipe_20260923/frames_recipe.json field by field (counts and verdicts
exactly, floats to 1e-9). Also refused: a record whose own published-row lock or population check
did not pass, whose estimator differs from the one read now, whose store run ids, scorer_hash or
graph_version differ from the stores on disk, or whose certified unit set is not the recipe's.

POPULATION (`--stage population` and `all`), refused rather than read: a root not on disk is refused
by name. Per new root, frames_recipe.population with that root's (arm_id, inquirer model) and grid
(824 ok runs on 824 distinct tasks, run ids reproduced from their manifests, one code_version, the
grid and arm_id, the named inquirer model on every inquirer call, frozen roles on gpt-oss-120b, no
dirty or dev- manifests, canaries loaded); every cell of a new root is gated, any cap. The existing
roots' cap-8 cells get the same checks and their ok run ids must be the ones frames_recipe.json
certified. Across all eight roots: one base_url_sha over every role, one unit set equal to the
recipe's, one corpus_hash.

READING (`--stage all`): each root's store (scores_parquet_<label>) must hold exactly the
population's run ids, 824 of them, with the named inquirer model; one scorer_hash and one
graph_version across all eight stores. Output: levels (n, clusters, mean n_asks, mean
retrieval_calls, answer_correct with its interval, ceiling-hit share) per arm, the six contrasts,
run_id_sha256 per arm, scorer_hash, graph_version.

RUN (the Mac is short of memory: --workers 1 or 2):
  now, before the roots land:   python scripts/seed_identity/frames_structured.py --stage lock --out X
  after the four roots return and are scored into scores_parquet_{par2_8b,par2_120b,par2_opus,opus}
  the way frames_recipe's were (`pi compact --exclude-dev`, then `pi score --gold-root data/gold
  --corpora-root data/corpora --allow-no-judge`, from the clean a61c4f4 worktree):
                                python scripts/seed_identity/frames_structured.py --workers 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frames_recipe as fr  # noqa: E402

OUT = fr.REPO / "artifacts/frames_structured_20260923"
CAP = 8
GRID = "frames_structured_cap8"
LOCK_TOL = 1e-9
EXISTING = ("teacher", "base", "s1", "s2")
# label -> (arm_id, inquirer model); every one at CAP on GRID
NEW = {
    "par2_8b": ("par2_rag", "qwen3-8b-base"),
    "par2_120b": ("par2_rag", "openai/aws/gpt-oss-120b"),
    "par2_opus": ("par2_rag", "openai/aws/claude-opus-5"),
    "opus": ("inquirer_prompted", "openai/aws/claude-opus-5"),
}
ROOTS = (*EXISTING, *NEW)
COMPARATORS = ("base", "teacher", *NEW)
LOCKED_LEVELS = ("recipe", "s1", "s2", "base", "teacher")
LOCKED_COMPARATORS = ("base", "teacher")
ESTIMAND = (
    "answer_correct at each policy's own STOP under the shared ceiling of 8 retrieval calls "
    "(FRAMES is gold-free: not coverage at equal spend)"
)


def spec(label: str) -> tuple[str, str | None, tuple[int, ...], str | None]:
    """(arm_id, inquirer model, caps of the root, grid); grid None = frames_recipe's own."""
    if label in NEW:
        arm_id, model = NEW[label]
        return arm_id, model, (CAP,), GRID
    return (*fr.ARMS[label], None)


def runs_root(label: str) -> Path:
    return fr.OUT / f"runs_{label}"


def store_dir(label: str) -> Path:
    return fr.OUT / f"scores_parquet_{label}"


def _sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


# --------------------------------------------------------------------------- the code path


def store_cell(label: str) -> tuple[dict[str, dict], list[str], list[str]]:
    """frames_recipe.store_units at CAP with this label's spec; a store not on disk is refused."""
    st = store_dir(label)
    missing = [
        n for n in ("runs.parquet", "scores.parquet", "calls.parquet") if not (st / n).is_file()
    ]
    if missing:
        raise SystemExit(
            f"REFUSING: scored store {fr._rel(st)} lacks {missing}: score "
            f"{fr._rel(runs_root(label))} into it first (see the module docstring)"
        )
    return fr.store_units(label, CAP, store=st, arm=spec(label)[:3])


def levels_and_contrasts(
    cells: dict[str, dict], est: dict, workers: int, comparators: tuple[str, ...]
) -> dict:
    """frames_recipe.read's code path at CAP: the recipe pooled within the task, levels at
    READ_N_BOOT, every recipe-minus-comparator contrast with the 1k and three 50k intervals."""
    lb, ls = fr.READ_N_BOOT, est["level_seed"]
    rec = fr.pool_seeds(cells["s1"], cells["s2"])
    levels = {"recipe": fr.level(rec, lb, ls)}
    for label in ("s1", "s2", *comparators):
        levels[label] = fr.level(cells[label], lb, ls)
    specs = {f"recipe-{c}/cap{CAP}": (rec, cells[c], True) for c in comparators}
    contrasts = fr.run_contrasts(specs, {**est, "contrast_n_boot": fr.READ_N_BOOT}, workers)
    return {"cells": {"recipe": rec}, "levels": levels, "contrasts": contrasts}


# --------------------------------------------------------------------------- the lock


def _diff(got, want, path: str, tol: float, out: list[str]) -> float:
    """Every field of `want` in `got`: same fields and lengths, bools, ints and strings exactly,
    floats to `tol`. Appends each mismatch to `out`; returns the largest float difference."""
    if isinstance(want, dict):
        if not isinstance(got, dict) or set(got) != set(want):
            have = sorted(got) if isinstance(got, dict) else repr(got)
            out.append(f"{path}: fields {have} != record's {sorted(want)}")
            return math.inf
        return max((_diff(got[k], want[k], f"{path}.{k}", tol, out) for k in want), default=0.0)
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            out.append(f"{path}: {got!r} != record's {want!r}")
            return math.inf
        return max(
            (_diff(g, w, f"{path}[{i}]", tol, out) for i, (g, w) in enumerate(zip(got, want))),
            default=0.0,
        )
    exact = (bool, str, type(None))
    if isinstance(want, exact) or isinstance(got, exact) or (type(want), type(got)) == (int, int):
        if type(got) is not type(want) or got != want:
            out.append(f"{path}: {got!r} != record's {want!r}")
            return math.inf
        return 0.0
    d = abs(float(got) - float(want))
    if not d <= tol:
        out.append(f"{path}: {got!r} != record's {want!r} (diff {d:.3e})")
    return d


def _row(got: dict, want: dict | None, name: str) -> dict:
    mism: list[str] = []
    got = json.loads(json.dumps(got))  # the record's own representation (lists, str keys)
    if want is None:
        mism.append(f"{name}: absent from the record")
        d = math.inf
    else:
        d = _diff(got, want, name, LOCK_TOL, mism)
    return {
        "record": want,
        "got": got,
        "max_abs_diff": d if math.isfinite(d) else None,
        "mismatches": mism,
        "ok": not mism,
    }


def record_problems(record: dict, est: dict) -> list[str]:
    """The record must itself stand: its code and metric, its published-row lock, its population."""
    probs = []
    if record.get("code_version") != fr.CODE or record.get("metric") != fr.METRIC:
        probs.append(
            f"record: code {record.get('code_version')} metric {record.get('metric')}, "
            f"expected {fr.CODE} {fr.METRIC}"
        )
    lk = record.get("lock") or {}
    try:
        n_rows = sum(len(lk[s]) for s in ("levels", "contrasts", "budget_response"))
        bad = fr.lock_failures(lk)
    except (KeyError, TypeError) as e:
        n_rows, bad = 0, [f"unreadable ({e!r})"]
    if n_rows == 0 or bad:
        probs.append(f"record: its own published-row lock {n_rows - len(bad)}/{n_rows}, {bad[:6]}")
    if record.get("population_failures") != []:
        probs.append(f"record: population_failures {record.get('population_failures')!r}")
    if lk.get("estimator") != est:
        probs.append(f"record: estimator {lk.get('estimator')} != the one read now {est}")
    return probs


def lock(est: dict, record: dict, workers: int = 1) -> tuple[dict, dict[str, dict]]:
    """Rebuild the recorded cap-8 rows from the stores on disk and compare them to `record`
    (frames_recipe.json). -> (the lock section, the store cells it read)."""
    failures = record_problems(record, est)
    rd = record.get("reading") or {}
    cells: dict[str, dict] = {}
    store_check: dict[str, dict] = {}
    for label in EXISTING:
        u, hashes, graphs = store_cell(label)
        key = f"{label}/cap{CAP}"
        got = {
            "eligible_in_store": len(u),
            "run_id_sha16": fr._sha16(v["run_id"] for v in u.values()),
            "scorer_hash": hashes,
            "graph_version": graphs,
        }
        want = (rd.get("store_check") or {}).get(key) or {}
        mism = [
            f"{key} store {k}: {got[k]} != record's {want.get(k)}"
            for k in got
            if got[k] != want.get(k)
        ]
        store_check[key] = {
            "store": fr._rel(store_dir(label)),
            **got,
            "run_id_sha256": _sha256(v["run_id"] for v in u.values()),
            "mismatches": mism,
        }
        failures += mism
        cells[label] = u
    res = levels_and_contrasts(cells, est, workers, LOCKED_COMPARATORS)
    units = fr._sha16(res["cells"]["recipe"])
    certified = (record.get("across_cells") or {}).get("published_unit_set_sha16")
    if units != certified:
        failures.append(f"recipe unit set {units} != the record's certified unit set {certified}")
    out: dict = {
        "tol": LOCK_TOL,
        "record_problems": [p for p in failures if p.startswith("record:")],
        "store_check": store_check,
        "recipe_unit_set_sha16": units,
        "levels": {},
        "contrasts": {},
    }
    for arm in LOCKED_LEVELS:
        want = ((rd.get("levels") or {}).get(arm) or {}).get(str(CAP))
        out["levels"][arm] = _row(res["levels"][arm], want, f"levels.{arm}.{CAP}")
    for comp in LOCKED_COMPARATORS:
        name = f"recipe-{comp}/cap{CAP}"
        out["contrasts"][name] = _row(
            res["contrasts"][name], (rd.get("contrasts") or {}).get(name), f"contrasts.{name}"
        )
    for sec in ("levels", "contrasts"):
        failures += [m for v in out[sec].values() for m in v["mismatches"]]
    out["failures"] = failures
    return out, cells


# --------------------------------------------------------------------------- population


def populations(labels: tuple[str, ...] = ROOTS) -> dict[str, dict]:
    """frames_recipe.population for every root, each with its own spec; a missing root refuses."""
    missing = [fr._rel(runs_root(lab)) for lab in labels if not runs_root(lab).is_dir()]
    if missing:
        raise SystemExit(
            f"REFUSING: runs root(s) not on disk: {', '.join(missing)}. The four structured roots "
            "are run on HPC and returned here; until all are present only --stage lock can run."
        )
    out = {}
    for lab in labels:
        arm_id, model, caps, grid = spec(lab)
        out[lab] = fr.population(lab, root=runs_root(lab), arm=(arm_id, model, caps), grid=grid)
    return out


def population_gate(pop: dict[str, dict], recipe_units: set[str], record: dict) -> dict:
    """Every cell of a new root, the cap-8 cell of an existing one (whose ok run ids must be the
    record's), and across the eight roots one base_url_sha, one unit set (the recipe's) and one
    corpus_hash."""
    failures: list[str] = []
    kept: dict[str, dict] = {}
    at_cap: dict[str, dict] = {}
    for lab, cells in pop.items():
        gated = (
            cells if lab in NEW else {k: v for k, v in cells.items() if k in (CAP, "unreadable")}
        )
        failures += [f"{lab}/{k}: {c['problems']}" for k, c in gated.items() if c["problems"]]
        kept[lab] = {str(k): v for k, v in gated.items()}
        if CAP in cells:
            at_cap[lab] = {CAP: cells[CAP]}
        if lab in EXISTING:
            want = (((record.get("population") or {}).get(lab) or {}).get(str(CAP)) or {}).get(
                "run_id_sha16"
            )
            got = cells.get(CAP, {}).get("run_id_sha16")
            if got is None or got != want:
                failures.append(
                    f"{lab}/{CAP}: ok run ids {got} != the record's {want} "
                    "(the root moved since frames_recipe.json)"
                )
    across = fr.across_cells(at_cap, recipe_units)
    failures += across["problems"]
    corpus = sorted({str(x) for c in at_cap.values() for x in c[CAP].get("corpus_hash", [])})
    if len(corpus) != 1:
        failures.append(f"{len(corpus)} corpus_hash values across roots: {corpus}")
    return {
        "population": kept,
        "across": {
            **across,
            "recipe_unit_set_sha16": fr._sha16(recipe_units),
            "corpus_hash": corpus,
        },
        "failures": failures,
    }


# --------------------------------------------------------------------------- the reading


def read(pop: dict[str, dict], est: dict, workers: int = 1) -> dict:
    """Every store must be its population; then the levels and the six contrasts."""
    cells: dict[str, dict] = {}
    store_check: dict[str, dict] = {}
    refused: list[str] = []
    for lab in ROOTS:
        u, hashes, graphs = store_cell(lab)
        probs = fr.store_problems(u, set(pop[lab][CAP]["ok_units"].values()))
        store_check[f"{lab}/cap{CAP}"] = {
            "store": fr._rel(store_dir(lab)),
            "eligible_in_store": len(u),
            "run_id_sha16": fr._sha16(v["run_id"] for v in u.values()),
            "scorer_hash": hashes,
            "graph_version": graphs,
            "problems": probs,
        }
        refused += [f"{lab}/cap{CAP}: {p}" for p in probs]
        cells[lab] = u
    refused += fr.store_identity_problems(store_check)
    if refused:
        raise SystemExit(f"REFUSING: scored stores are not the population: {refused[:8]}")
    res = levels_and_contrasts(cells, est, workers, COMPARATORS)
    ids = {lab: [v["run_id"] for v in cells[lab].values()] for lab in ROOTS}
    return {
        "store_check": store_check,
        "scorer_hash": sorted({x for v in store_check.values() for x in v["scorer_hash"]}),
        "graph_version": sorted({x for v in store_check.values() for x in v["graph_version"]}),
        "run_id_sha256": {
            "recipe": _sha256(ids["s1"] + ids["s2"]),
            **{lab: _sha256(ids[lab]) for lab in ROOTS},
        },
        "levels": res["levels"],
        "contrasts": res["contrasts"],
    }


# --------------------------------------------------------------------------- main


def _print_reading(rd: dict) -> None:
    for arm, v in rd["levels"].items():
        print(
            f"  level {arm:10s} n={v['n']} asks={v['mean_n_asks']:.4f} "
            f"calls={v['mean_retrieval_calls']:.4f} {v['y']:.6f} [{v['ci_lo']:.6f}, {v['ci_hi']:.6f}] "
            f"ceiling-hit {v['ceiling_hit_share']:.4f}"
        )
    for name, v in rd["contrasts"].items():
        fifty = "; ".join(f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in v["ci_50k"])
        print(
            f"  {name:24s} n={v['n']} {v['delta']:+.6f} [{v['ci_lo']:+.6f}, {v['ci_hi']:+.6f}] "
            f"50k {fifty} {'DECIDED' if v['decided'] else 'not decided'}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, default=OUT / "frames_structured.json")
    ap.add_argument("--stage", choices=("lock", "population", "all"), default="all")
    ap.add_argument("--workers", type=int, default=1, help="bootstrap processes (Mac: 1 or 2)")
    args = ap.parse_args(argv)

    est = fr.parse_estimator(fr.FRONTIER_FRAMES.read_text())
    if est["cluster"] != fr.normalise_cluster(fr.CLUSTER_SQL):
        raise SystemExit(
            f"REFUSING: record clusters on {est['cluster']}, script on {fr.CLUSTER_SQL}"
        )
    rec_path = fr.OUT / "frames_recipe.json"
    if not rec_path.is_file():
        raise SystemExit(f"REFUSING: no record at {fr._rel(rec_path)} to lock against")
    raw = rec_path.read_bytes()
    record = json.loads(raw)
    out: dict = {
        "code_version": fr.CODE,
        "metric": fr.METRIC,
        "cap": CAP,
        "estimand": ESTIMAND,
        "recipe": "training seeds 1 and 2 averaged within the task (frames_recipe.pool_seeds)",
        "record": fr._rel(rec_path),
        "record_sha256": hashlib.sha256(raw).hexdigest(),
        "estimator": est,
        "arms": {
            lab: {
                "arm_id": spec(lab)[0],
                "inquirer_model": spec(lab)[1],
                "grid": spec(lab)[3] or f"frames_recipe_cap{CAP}",
                "runs_root": fr._rel(runs_root(lab)),
                "store": fr._rel(store_dir(lab)),
                "runs_root_present": runs_root(lab).is_dir(),
                "store_present": (store_dir(lab) / "runs.parquet").is_file(),
            }
            for lab in ROOTS
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    lk, cells = lock(est, record, args.workers)
    out["lock"] = lk
    rows = [*lk["levels"].values(), *lk["contrasts"].values()]
    worst = max((r["max_abs_diff"] for r in rows if r["max_abs_diff"] is not None), default=None)
    print(
        f"lock: {sum(r['ok'] for r in rows)}/{len(rows)} cap-{CAP} rows of {out['record']} "
        f"reproduced to {LOCK_TOL:g} (max abs diff {worst}); store checks "
        f"{sum(not v['mismatches'] for v in lk['store_check'].values())}/{len(lk['store_check'])}; "
        f"failures {lk['failures'][:6]}"
    )
    if lk["failures"]:
        fr._dump(args.out, out)
        print(f"REFUSING: the lock does not hold; wrote {args.out}")
        return 1
    if args.stage == "lock":
        absent = [lab for lab, a in out["arms"].items() if not a["runs_root_present"]]
        print(f"roots not yet on disk: {absent}")
        fr._dump(args.out, out)
        print(f"wrote {args.out}")
        return 0

    recipe_units = set(fr.pool_seeds(cells["s1"], cells["s2"]))
    pop = populations()
    gate = population_gate(pop, recipe_units, record)
    out["population"] = gate["population"]
    out["across"] = gate["across"]
    out["population_failures"] = gate["failures"]
    print(
        f"population: {len(gate['failures'])} refusals; base_url_sha {gate['across']['base_url_sha']}"
    )
    for p in gate["failures"]:
        print("  ", p)
    if gate["failures"] or args.stage == "population":
        fr._dump(args.out, out)
        print(f"wrote {args.out}")
        return 1 if gate["failures"] else 0

    out["reading"] = read(pop, est, args.workers)
    fr._dump(args.out, out)
    print(
        f"scorer_hash {out['reading']['scorer_hash']} graph_version {out['reading']['graph_version']}"
    )
    _print_reading(out["reading"])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
