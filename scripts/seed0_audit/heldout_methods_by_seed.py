"""The selected row of tab:app-heldout-methods, read on the recipe's two fresh training seeds.

WHY. The row "both kinds, from pairs" prints +0.1169 / +0.0628 / +0.0737. Those are the
registered adapter `qwen3-8b-dpo-stacked-notdone-both` (seed 0, adapter 80a56b74), whose final
preference stage resumed from checkpoint-1800 and re-loaded its init, keeping ~9% of the stage
(artifacts/seed_identity_20260923/RESULT.md). Seeds 1 and 2 ran fresh and are the recipe.

WHAT. The table's OWN contrast function, unchanged: `scripts/decomposition_test/contrast.py`
`select_and_contrast`, i.e. `pinq_train.gate._select_runs` + `_by_key` + `_matched_cost`
(checkpoint at its own terminal `evidence_coverage`, comparator at `frontier_q#min(k, its own
n_asks)`, rollout seeds averaged within the task, task-clustered BCa, 10,000 resamples, seed 0).

  s0     the completed cohort store, which holds the population the row was read on (its run-id
         lists equal `artifacts/decomposition_test_20260918/run_ids/{stacked_test,
         baseline_qwen3-8b-base}_3ae099d0.txt`, checked below, not assumed).
  s1, s2 the seed-replicate store (artifacts/seedrep_gate_20260919), paired against THE SAME
         comparator runs as s0. Both stores are `scorer_hash e82c7458`, `code_version 3ae099d0`,
         and the runs they share are checked value for value before any contrast is read.
  s1+s2  both training seeds pooled WITHIN the task, as Table 1 pools them
         (`scripts/seed_identity/table1_by_seed.py::pooled_symmetric`): one flat per-task mean
         over every (training seed, rollout seed) delta, bootstrap over tasks. Done by handing
         `_matched_cost` both seeds' runs under one key, which equals the flat mean exactly when
         every key carries one run of each seed; that equality is asserted against an
         independent flat recompute, not argued.

LOCKS, IN ORDER, EACH A HARD STOP.
  1. population: the s0 and comparator run ids the function selects equal the row's recorded
     run-id lists;
  2. published: s0 reproduces every printed point and bound at four decimals;
  3. store join: the union view, read with seed 0 alone, reproduces lock 2 to 1e-12, and every
     run present in both stores carries the same `evidence_coverage`, `frontier_q#k` and
     `n_asks` in both;
  4. pooling: the merged-key reading equals the independent flat per-task pool to 1e-12;
  5. cross-reader: the symmetric key of the pooled reading reproduces Table 1's own s1+s2
     coverage points (artifacts/seed_identity_20260923/table1_by_seed.json) to 1e-12.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "decomposition_test"))

from contrast import select_and_contrast  # noqa: E402

from pinq_train.gate import (  # noqa: E402
    _by_key,
    _con,
    _coverage_ladder,
    _matched_cost,
    _mean,
    _metric_by_run,
    _select_runs,
    _with_stop,
)

SCORER = "e82c7458ee8efa11660445bbf434b4c1dc74253253e8f190cc36e92fc50f40c7"
GRID = "tier1_trained_qa_base"
BASE = "qwen3-8b-base"
S0 = "qwen3-8b-dpo-stacked-notdone-both"
SEEDS = {"s0": [S0], "s1": [f"{S0}-s1"], "s2": [f"{S0}-s2"], "s1s2": [f"{S0}-s1", f"{S0}-s2"]}
SUITES = ("musique", "strategyqa", "wiki2")
# What the tex prints for the row, (point, lo, hi), from
# paper/iclr2027 2/figures/table4_heldout_methods.provenance.json.
PUBLISHED = {
    "musique": ("+0.1169", "+0.0823", "+0.1544"),
    "strategyqa": ("+0.0628", "+0.0254", "+0.1008"),
    "wiki2": ("+0.0737", "+0.0469", "+0.1050"),
}
RESEEDS = ((50_000, 0), (50_000, 1), (50_000, 2))  # contrast.py's own stability rule


def digest(ids) -> str:
    h = hashlib.sha256()
    for rid in sorted(ids):
        h.update(rid.encode())
        h.update(b"\n")
    return h.hexdigest()


def ids_of(path: Path) -> set[str]:
    return {x.strip() for x in path.read_text().split() if x.strip()}


def union_con(cc: Path, seedrep: Path, keep_from_seedrep: set[str]):
    """One read-only duckdb view over both stores. From the seed-replicate store ONLY the s1 and
    s2 run ids enter, so the s0 and comparator runs it also holds cannot enter twice."""
    import duckdb

    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE keep (run_id VARCHAR)")
    con.executemany("INSERT INTO keep VALUES (?)", [(r,) for r in sorted(keep_from_seedrep)])
    for name in ("runs", "scores", "calls"):
        a, b = (cc / f"{name}.parquet").as_posix(), (seedrep / f"{name}.parquet").as_posix()
        con.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{a}') UNION ALL BY NAME "
            f"SELECT * FROM read_parquet('{b}') WHERE run_id IN (SELECT run_id FROM keep)"
        )
    return con


def cross_store_check(cc: Path, seedrep: Path, shared: set[str]) -> dict:
    """Every run held by both stores must carry identical coverage, ladder and n_asks."""
    ca, cb = _con(cc), _con(seedrep)
    rows = {}
    for tag, con in (("cc", ca), ("sr", cb)):
        cov = _metric_by_run(con, "evidence_coverage", scorer_hash=SCORER)
        lad = _coverage_ladder(con, [{"run_id": r} for r in shared], scorer_hash=SCORER)
        k = {
            str(r["run_id"]): r["n_asks"] for r in _with_stop(con, [{"run_id": r} for r in shared])
        }
        rows[tag] = {r: (cov.get(r), lad.get(r), k.get(r)) for r in shared}
    bad = [r for r in sorted(shared) if rows["cc"][r] != rows["sr"][r]]
    n_rungs = sum(len(v[1] or {}) for v in rows["cc"].values())
    return {
        "n_runs_compared": len(shared),
        "n_ladder_rungs_compared": n_rungs,
        "n_mismatch": len(bad),
    }


def pooled(con, model_ids, *, seed=0, n_resamples=10_000) -> dict:
    base = _select_runs(con, arm="inquirer_prompted", grids=[GRID], model_id=BASE)
    per_model = [
        _select_runs(con, arm="inquirer_trained", grids=[GRID], model_id=m) for m in model_ids
    ]
    ckpt = [r for rs in per_model for r in rs]
    ck_keys, ba_keys = _by_key(ckpt), _by_key(base)
    out = _matched_cost(
        con,
        ckpt=ckpt,
        base=base,
        ck_keys=ck_keys,
        ba_keys=ba_keys,
        scorer_hash=SCORER,
        seed=seed,
        n_resamples=n_resamples,
    )
    out["selection"] = {
        "model_ids": list(model_ids),
        "n_checkpoint_runs": [len(rs) for rs in per_model],
        "n_baseline_runs": len(base),
        "checkpoint_run_ids": [sorted(str(r["run_id"]) for r in rs) for rs in per_model],
        "baseline_run_ids": sorted(str(r["run_id"]) for r in base),
    }
    out["_keys"] = ([_by_key(rs) for rs in per_model], ba_keys)
    return out


def flat_pool(con, per_model_keys, ba_keys) -> list[float]:
    """Table 1's pooling, written out independently: every (training seed, rollout seed) delta
    of a task into ONE list, then the task mean. Returned in (suite, task) order, as `deltas`."""
    cov = _metric_by_run(con, "evidence_coverage", scorer_hash=SCORER)
    all_ids = [r for keys in per_model_keys for v in keys.values() for r in v]
    all_ids += [r for v in ba_keys.values() for r in v]
    n_asks = {
        str(r["run_id"]): int(r["n_asks"] or 0)
        for r in _with_stop(con, [{"run_id": x} for x in all_ids])
    }
    base_ids = sorted({r for v in ba_keys.values() for r in v})
    lad = _coverage_ladder(con, [{"run_id": r} for r in base_ids], scorer_hash=SCORER)
    per_task: dict[tuple, list[float]] = {}
    for keys in per_model_keys:
        for key in sorted(set(keys) & set(ba_keys)):
            for c in keys[key]:
                cv = cov.get(c)
                if cv is None:
                    continue
                k = n_asks.get(c, 0)
                rungs = [(lad.get(b) or {}).get(min(k, n_asks.get(b, 0))) for b in ba_keys[key]]
                rungs = [float(x) for x in rungs if x is not None]
                if rungs:
                    per_task.setdefault((key[0], key[1]), []).append(float(cv) - _mean(rungs))
    return [_mean(per_task[t]) for t in sorted(per_task)]


def cell(r: dict, suite: str) -> dict:
    s = r["by_suite"][suite]
    keep = (
        "n_tasks",
        "delta",
        "ci_lo",
        "ci_hi",
        "trained_mean_k",
        "baseline_mean_k_charged",
        "n_baseline_shorter_than_k",
        "baseline_mean_n_asks",
        "cap8_coverage_delta",
        "outspent_comparator",
    )
    out = {k: s[k] for k in keep}
    out["symmetric"] = {k: s["symmetric"][k] for k in ("n_tasks", "delta", "ci_lo", "ci_hi")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--data-root", type=Path, required=True, help="checkout holding artifacts/")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    A = args.data_root / "artifacts"
    cc = A / "completed_cohort_20260922" / "scores_parquet"
    sr = A / "seedrep_gate_20260919" / "scores_parquet"
    rec = A / "decomposition_test_20260918" / "run_ids"
    res: dict = {"scorer_hash": SCORER, "grid_name": GRID, "baseline_model_id": BASE, "locks": {}}

    # LOCK 1 + 2: the table's function on the row's own population, seed 0.
    s0 = select_and_contrast(
        cc, checkpoint_model_id=S0, baseline_model_id=BASE, scorer_hash=SCORER, seed=0
    )
    con_cc = _con(cc)
    got_ck = {
        str(r["run_id"])
        for r in _select_runs(con_cc, arm="inquirer_trained", grids=[GRID], model_id=S0)
    }
    got_ba = {
        str(r["run_id"])
        for r in _select_runs(con_cc, arm="inquirer_prompted", grids=[GRID], model_id=BASE)
    }
    want_ck, want_ba = (
        ids_of(rec / "stacked_test_3ae099d0.txt"),
        ids_of(rec / "baseline_qwen3-8b-base_3ae099d0.txt"),
    )
    res["locks"]["population"] = {
        "checkpoint_equal": got_ck == want_ck,
        "baseline_equal": got_ba == want_ba,
        "n_checkpoint": len(got_ck),
        "n_baseline": len(got_ba),
        "checkpoint_sha256": digest(got_ck),
        "baseline_sha256": digest(got_ba),
    }
    if got_ck != want_ck or got_ba != want_ba:
        raise SystemExit(f"POPULATION LOCK FAILED: {res['locks']['population']}")
    pub_lock = {}
    for suite in SUITES:
        c = s0["by_suite"][suite]
        printed = tuple(f"{c[k]:+.4f}" for k in ("delta", "ci_lo", "ci_hi"))
        pub_lock[suite] = {
            "published": PUBLISHED[suite],
            "reproduced": printed,
            "raw": [c["delta"], c["ci_lo"], c["ci_hi"]],
            "ok": printed == PUBLISHED[suite],
        }
        print(
            f"LOCK s0 {suite}: printed {PUBLISHED[suite]} reproduced {printed} raw {pub_lock[suite]['raw']}"
        )
    res["locks"]["published"] = pub_lock
    if not all(v["ok"] for v in pub_lock.values()):
        raise SystemExit("PUBLISHED LOCK FAILED")

    # LOCK 3: the union view, and the two stores agree on every run they share.
    seedrep_ids = set()
    for m in (f"{S0}-s1", f"{S0}-s2"):
        for suite in SUITES:
            seedrep_ids |= ids_of(
                A / "seedrep_gate_20260919" / "run_ids" / f"run_ids.{m}.{suite}.txt"
            )
    con_sr = _con(sr)
    sr_s0 = {
        str(r["run_id"])
        for r in _select_runs(con_sr, arm="inquirer_trained", grids=[GRID], model_id=S0)
    }
    sr_ba = {
        str(r["run_id"])
        for r in _select_runs(con_sr, arm="inquirer_prompted", grids=[GRID], model_id=BASE)
    }
    res["locks"]["cross_store"] = {
        "seedrep_baseline_subset_of_cc": sr_ba <= got_ba,
        "seedrep_s0_equal_cc_s0": sr_s0 == got_ck,
        **cross_store_check(cc, sr, (sr_s0 & got_ck) | (sr_ba & got_ba)),
    }
    print("LOCK cross-store:", res["locks"]["cross_store"])
    if (
        res["locks"]["cross_store"]["n_mismatch"]
        or not res["locks"]["cross_store"]["seedrep_baseline_subset_of_cc"]
    ):
        raise SystemExit("CROSS-STORE LOCK FAILED")
    con = union_con(cc, sr, seedrep_ids)
    u0 = pooled(con, SEEDS["s0"])
    join = {
        s: max(
            abs(u0["by_suite"][s][k] - s0["by_suite"][s][k]) for k in ("delta", "ci_lo", "ci_hi")
        )
        for s in SUITES
    }
    res["locks"]["union_reproduces_s0"] = join
    if max(join.values()) > 1e-12:
        raise SystemExit(f"UNION LOCK FAILED {join}")

    # Readings.
    res["cells"], res["arms"], res["stability"] = {}, {}, {}
    for tag, models in SEEDS.items():
        r = u0 if tag == "s0" else pooled(con, models)
        res["cells"][tag] = {s: cell(r, s) for s in SUITES}
        res["arms"][tag] = {
            "model_ids": models,
            "n_checkpoint_runs": r["selection"]["n_checkpoint_runs"],
            "checkpoint_run_ids_sha256": [digest(x) for x in r["selection"]["checkpoint_run_ids"]],
            "n_baseline_runs": r["selection"]["n_baseline_runs"],
            "baseline_run_ids_sha256": digest(r["selection"]["baseline_run_ids"]),
        }
        if tag == "s1s2":
            per_model_keys, ba_keys = r["_keys"]
            shared = [set(k) & set(ba_keys) for k in per_model_keys]
            balanced = shared[0] == shared[1] and all(
                all(len(k[key]) == 1 for key in sh) for k, sh in zip(per_model_keys, shared)
            )
            flat = flat_pool(con, per_model_keys, ba_keys)
            diff = (
                max(abs(a - b) for a, b in zip(flat, r["deltas"]))
                if len(flat) == len(r["deltas"])
                else float("inf")
            )
            res["locks"]["pooling"] = {
                "balanced_one_run_per_seed_per_key": balanced,
                "n_task_values": len(flat),
                "max_abs_diff_vs_flat": diff,
            }
            print("LOCK pooling:", res["locks"]["pooling"])
            if diff > 1e-12:
                raise SystemExit("POOLING LOCK FAILED")
            t1 = json.loads((A / "seed_identity_20260923" / "table1_by_seed.json").read_text())
            xr = {}
            for s in SUITES:
                want = next(
                    x
                    for x in t1["cells"]["s1s2"][f"evidence_coverage::{s}"]
                    if x["n_boot"] == 10000
                )["delta"]
                got = r["by_suite"][s]["symmetric"]["delta"]
                xr[s] = {
                    "table1_s1s2": want,
                    "this_reader_symmetric": got,
                    "abs_diff": abs(want - got),
                }
            res["locks"]["table1_cross_reader"] = xr
            print("LOCK table1 cross-reader:", xr)
            if max(v["abs_diff"] for v in xr.values()) > 1e-12:
                raise SystemExit("TABLE1 CROSS-READER LOCK FAILED")
            for nb, sd in RESEEDS:
                rr = pooled(con, models, seed=sd, n_resamples=nb)
                for s in SUITES:
                    b = rr["by_suite"][s]
                    res["stability"].setdefault(s, []).append(
                        {
                            "n_resamples": nb,
                            "seed": sd,
                            "delta": b["delta"],
                            "ci_lo": b["ci_lo"],
                            "ci_hi": b["ci_hi"],
                            "sym_ci_lo": b["symmetric"]["ci_lo"],
                            "sym_ci_hi": b["symmetric"]["ci_hi"],
                        }
                    )
        for s in SUITES:
            c = res["cells"][tag][s]
            print(
                f"{tag:5s} {s:10s} {c['delta']:+.4f} [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] n={c['n_tasks']} "
                f"k={c['trained_mean_k']:.3f} short={c['n_baseline_shorter_than_k']} "
                f"sym {c['symmetric']['delta']:+.4f} [{c['symmetric']['ci_lo']:+.4f}, {c['symmetric']['ci_hi']:+.4f}]"
            )
    for s, reads in res["stability"].items():
        for x in reads:
            print(
                f"s1s2 {s:10s} {x['n_resamples']} seed {x['seed']}: [{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}]"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
