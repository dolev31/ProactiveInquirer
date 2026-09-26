"""FRAMES at five budget ceilings for the recipe (training seeds 1 and 2) and both prompted comparators.

The published FRAMES record (artifacts/frames_frontier/FRAMES_FRONTIER.md, code 107ef2221a55) ran the
registered pin, which is seed 0 of the selected arm, whose final DPO stage a resume cut to about a
tenth (artifacts/seed_identity_20260923). This reads the same five ceilings for the recipe and both
prompted comparators at ONE code version (a61c4f4be3e9) through ONE base URL (proxy 4030):
artifacts/frames_recipe_20260923/runs_{teacher,base,s1,s2,drafter}, grids
conf/grids/frames_recipe_cap*.yaml, one scored store per root (scores_parquet_<label>).

Metric: `answer_correct` only. FRAMES ships no need graph (gold-free; `pi score` takes the
answers-only branch), so there is no coverage, depth or facet metric here.

ESTIMATOR, READ FROM THE RECORD (`parse_estimator`), not assumed: the record's Reproduction section
states `cluster_bootstrap(n_boot=10000, seed=0)` for a level and `paired_difference(n_boot=10000,
n_perm=10000, seed=0)` for a contrast, over `COALESCE(NULLIF(template_id,''), task_id)`, under
`pi_eval.report.ELIGIBLE` unmodified; its FRAMES budget block states n_boot=10000 too. The parser
refuses a record that states two settings or none, and the script's own cluster SQL must normalise to
the stated expression. template_id is NULL on FRAMES, so clusters are tasks, 824 of them. Ceiling-hit
is stop_reason in {budget, max_turns}.

The recipe's unit is the task (rollout seed 0) with training seeds 1 and 2 averaged WITHIN it, so its
interval is over tasks and does not include variation between training seeds; each seed is also
reported alone. A contrast is paired over the SAME unit set on both sides (refused otherwise, since
`paired_difference` would silently intersect) and is DECIDED only if its 50,000-resample interval
excludes zero on the same side under each of seeds 101, 202 and 303. A bound within 0.01 of zero is
also read at 1,000 resamples and whether its verdict flips is recorded.

LOCK (`--stage lock`): before any new cell is read, the same level and contrast functions, pointed at
the published store (scores_parquet.frames) and run-id lists (run_ids.<arm>.cap<k>.tsv), must
reproduce every level row FRAMES_FRONTIER.md prints (n, clusters, mean n_asks, mean retrieval_calls,
value, interval, ceiling-hit count), every row of its six answer_correct contrast tables and its
cap-24-minus-cap-4 table, to 1e-4.

POPULATION, refused rather than read (`--stage all`, after the lock):
  per cell (arm x cap), from the run directories: 824 status-ok runs on 824 distinct (task, seed)
  units and 824 distinct tasks; zero non-ok run directories, zero directories without a manifest and
  zero duplicate units; every run id reproduced from its own manifest's semantic fields and equal to
  its directory name; one code_version, a61c4f4be3e9; one model_pin_hash; the cell's own grid and
  arm_id; the named inquirer model on every inquirer call and at least one inquirer call per run (no
  inquirer call at all for drafter_only); the frozen drafter/answerer on openai/aws/gpt-oss-120b in
  both pins and calls; no dirty and no dev- manifest; canary nonces loaded on every unit; one prompt
  hash set, grid sha, corpus hash, canary count and inquirer adapter sha.
  across cells: one base_url_sha over every role of every cell; one unit set in every cell, equal to
  the published record's.
  per scored store: the eligible runs at that cap are exactly the population's run ids (824); the
  named inquirer model on every inquirer call; one scorer_hash and one graph_version across all stores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb

from pi_eval.report import ELIGIBLE
from pi_eval.stats.inference import cluster_bootstrap, paired_difference

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from paper_figures_iclr import (  # noqa: E402
    FRONTIER_FRAMES,
    md_tables,
    parse_level_table,
    read_run_ids,
)

PUB = REPO / "artifacts/frames_frontier"
OUT = REPO / "artifacts/frames_recipe_20260923"
CAPS = (4, 8, 12, 16, 24)
CODE = "a61c4f4be3e9"
N_CELL = 824
METRIC = "answer_correct"
CEILING = ("budget", "max_turns")
TOL = 1e-4
FROZEN = "openai/aws/gpt-oss-120b"
CLUSTER_SQL = "COALESCE(NULLIF(r.template_id, ''), r.task_id)"
# The paper's decision rule (fixed by the protocol, not by the FRAMES record).
DECIDE_N_BOOT = 50_000
DECIDE_SEEDS = (101, 202, 303)
READ_N_BOOT = 10_000
NEAR_ZERO = 0.01
# 12:35 IDT, when the local sweep was stopped and its completion moved to HPC: a unit whose
# status.json started_at (epoch seconds, clock-zone free) is later was run by the completion.
LOCAL_STOP_EPOCH = 1790156100.0  # 2026-09-23T09:35:00Z
PUB_ARMS = ("teacher", "base8b", "trained", "drafter_only")

# label -> (arm_id, inquirer model or None for NeverAsk, caps run)
ARMS = {
    "teacher": ("inquirer_prompted", "openai/aws/gpt-oss-120b", CAPS),
    "base": ("inquirer_prompted", "qwen3-8b-base", CAPS),
    "s1": ("inquirer_trained", "qwen3-8b-dpo-stacked-notdone-both-s1", CAPS),
    "s2": ("inquirer_trained", "qwen3-8b-dpo-stacked-notdone-both-s2", CAPS),
    "drafter": ("drafter_only", None, (8,)),
}

_NUM = r"([+-]?\d+\.\d+)"
_BUDGET_ROW = re.compile(
    rf"^\s*(\S+)\s+n=\s*(\d+)\s+{_NUM}\s+\[{_NUM},\s*{_NUM}\]",
)


def _rel(p: Path) -> str:
    """Repo-relative when inside the repo (no home path lands in an artifact), else as given."""
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def _sha16(items) -> str:
    return hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- record parsers


def parse_budget_response(text: str) -> dict[str, tuple[int, float, float, float]]:
    """The `--- FRAMES (metric answer_correct, ...)` block: arm -> (n, point, lo, hi)."""
    out: dict[str, tuple[int, float, float, float]] = {}
    inside = False
    for line in text.splitlines():
        if line.startswith("---"):
            inside = line.split()[1:2] == ["FRAMES"]
            continue
        if not inside:
            continue
        m = _BUDGET_ROW.match(line)
        if m is None:
            inside = False
            continue
        arm, n, pt, lo, hi = m.groups()
        out[arm] = (int(n), float(pt), float(lo), float(hi))
    if not out:
        raise ValueError("no FRAMES budget-response block in this record")
    return out


def normalise_cluster(expr: str) -> str:
    """`COALESCE(NULLIF(r.template_id, ''), r.task_id)` -> `COALESCE(NULLIF(template_id,''),task_id)`."""
    return re.sub(r"\s+", "", expr).replace("r.", "")


def _one(matches: list[tuple], what: str) -> tuple:
    if not matches:
        raise ValueError(f"the record states no {what}")
    if len(set(matches)) != 1:
        raise ValueError(
            f"the record states {len(set(matches))} different {what}: {sorted(set(matches))}"
        )
    return matches[0]


def parse_estimator(text: str) -> dict:
    """The level and contrast settings and the clustering, as the record states them.

    Refuses (ValueError) a record that states none of a setting or two different values of it, and
    a FRAMES budget block whose n_boot disagrees with the stated contrast resample count.
    """
    # the call form (Reproduction section) and the prose form beside the level table
    lv = _one(
        [
            tuple(map(int, m))
            for pat in (
                r"cluster_bootstrap\(n_boot=(\d+),\s*seed=(\d+)\)",
                r"cluster_bootstrap`,\s*`n_boot=(\d+)`,\s*`seed=(\d+)`",
            )
            for m in re.findall(pat, text)
        ],
        "cluster_bootstrap setting",
    )
    ct = _one(
        [
            tuple(map(int, m))
            for pat in (
                r"paired_difference\(n_boot=(\d+),\s*n_perm=(\d+),\s*seed=(\d+)\)",
                r"`n_boot=(\d+)`,\s*`n_perm=(\d+)`,\s*`seed=(\d+)`",
            )
            for m in re.findall(pat, text)
        ],
        "paired_difference setting",
    )
    blk = _one(
        [
            (int(m),)
            for m in re.findall(
                r"^--- FRAMES\s+\(metric answer_correct, n_boot=(\d+)\)", text, re.M
            )
        ],
        "FRAMES budget-block n_boot",
    )
    if blk[0] != ct[0]:
        raise ValueError(f"FRAMES budget block n_boot={blk[0]} but contrasts state n_boot={ct[0]}")
    cl = _one(
        [
            (normalise_cluster(m),)
            for m in re.findall(r"COALESCE\(NULLIF\(template_id,\s*''\),\s*task_id\)", text)
        ],
        "cluster expression",
    )
    return {
        "level_n_boot": lv[0],
        "level_seed": lv[1],
        "contrast_n_boot": ct[0],
        "contrast_n_perm": ct[1],
        "contrast_seed": ct[2],
        "cluster": cl[0],
    }


def parse_contrast_tables(
    text: str, metric: str
) -> dict[tuple[str, str], dict[int, tuple[int, int, float, float, float]]]:
    """`#### `a` minus `b`` headings, each over a table with a `delta <metric>` column.

    -> {(a, b): {cap: (n, clusters, delta, lo, hi)}}
    """
    out: dict[tuple[str, str], dict[int, tuple[int, int, float, float, float]]] = {}
    pair: tuple[str, str] | None = None
    want = f"delta {metric}"
    rows: list[list[str]] = []

    def flush() -> None:
        if pair is None or len(rows) < 3:
            return
        head = [c.lower() for c in rows[0]]
        if want not in head:
            return
        i_cap, i_n, i_cl, i_d, i_ci = (
            head.index(k) for k in ("cap", "n", "clusters", want, "bca 95%")
        )
        cells = {}
        for r in rows[2:]:
            lo, hi = (float(x) for x in r[i_ci].strip("[]").split(","))
            cells[int(r[i_cap])] = (int(r[i_n]), int(r[i_cl]), float(r[i_d]), lo, hi)
        out[pair] = cells

    for raw in text.splitlines():
        line = raw.strip()
        m = re.fullmatch(r"#{2,6}\s+`([^`]+)`\s+minus\s+`([^`]+)`", line)
        if m or line.startswith("#"):
            flush()
            pair, rows = (m.group(1), m.group(2)) if m else None, []
            continue
        if line.startswith("|") and line.endswith("|"):
            rows.append([c.strip() for c in line[1:-1].split("|")])
    flush()
    return out


def parse_level_extras(text: str) -> dict[tuple[str, int], tuple[int, int]]:
    """The level table's `clusters` and `ceiling-hit` columns: (arm, cap) -> (clusters, hits)."""
    out: dict[tuple[str, int], tuple[int, int]] = {}
    for tbl in md_tables(text):
        head = [c.lower() for c in tbl[0]]
        if not {"arm", "cap", "clusters", "ceiling-hit", f"{METRIC} [bca 95%]"} <= set(head):
            continue
        for r in tbl[2:]:
            arm = r[head.index("arm")].strip("`")
            out[(arm, int(r[head.index("cap")]))] = (
                int(r[head.index("clusters")]),
                int(r[head.index("ceiling-hit")].split("/")[0]),
            )
    return out


# --------------------------------------------------------------------------- estimators


def pool_seeds(a: dict[str, dict], b: dict[str, dict]) -> dict[str, dict]:
    """Training seeds 1 and 2 averaged WITHIN the unit (task, rollout seed); refuses a mismatch."""
    keys = sorted(set(a) & set(b))
    if len(keys) != len(a) or len(keys) != len(b):
        raise SystemExit(
            f"REFUSING: the two training seeds cover different units ({len(a)}, {len(b)})"
        )
    out = {}
    for k in keys:
        if a[k]["cl"] != b[k]["cl"]:
            raise SystemExit(f"REFUSING: cluster disagrees across training seeds at {k}")
        out[k] = {
            "cl": a[k]["cl"],
            "y": (a[k]["y"] + b[k]["y"]) / 2,
            "asks": (a[k]["asks"] + b[k]["asks"]) / 2,
            "rc": (a[k]["rc"] + b[k]["rc"]) / 2,
            "hit": (a[k]["hit"] + b[k]["hit"]) / 2,
        }
    return out


def decided(cis: list[tuple[float, float]]) -> bool:
    """Every interval excludes zero, all on the same side."""
    if not cis:
        return False
    return all(lo > 0 for lo, _ in cis) or all(hi < 0 for _, hi in cis)


def level(units: dict[str, dict], n_boot: int, seed: int) -> dict:
    groups: dict[str, list[float]] = {}
    for k in sorted(units):
        groups.setdefault(units[k]["cl"], []).append(units[k]["y"])
    point, lo, hi = cluster_bootstrap(list(groups.values()), n_boot=n_boot, seed=seed)
    n = len(units)
    return {
        "n": n,
        "n_clusters": len(groups),
        "y": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_boot": n_boot,
        "mean_n_asks": sum(u["asks"] for u in units.values()) / n,
        "mean_retrieval_calls": sum(u["rc"] for u in units.values()) / n,
        "ceiling_hit": sum(u["hit"] for u in units.values()),
        "ceiling_hit_share": sum(u["hit"] for u in units.values()) / n,
    }


def _pd(job: tuple) -> tuple[float, float, float, float | None, int]:
    a, b, cl, n_boot, n_perm, seed = job
    e = paired_difference(a, b, clusters=cl, n_boot=n_boot, n_perm=n_perm, seed=seed)
    return e.point, e.ci_lo, e.ci_hi, e.p_value, e.n


def _jobs(a: dict[str, dict], b: dict[str, dict], full: bool, est: dict, name: str) -> list[tuple]:
    if set(a) != set(b):
        raise SystemExit(
            f"REFUSING: contrast {name} is not paired: {len(set(a) - set(b))} units only in a, "
            f"{len(set(b) - set(a))} only in b"
        )
    keys = sorted(a)
    for k in keys:
        if a[k]["cl"] != b[k]["cl"]:
            raise SystemExit(f"REFUSING: contrast {name}: cluster disagrees at {k}")
    ya = {k: a[k]["y"] for k in keys}
    yb = {k: b[k]["y"] for k in keys}
    cl = {k: a[k]["cl"] for k in keys}
    jobs = [(ya, yb, cl, est["contrast_n_boot"], est["contrast_n_perm"], est["contrast_seed"])]
    if full:
        jobs += [(ya, yb, cl, 1_000, 1_000, est["contrast_seed"])]
        jobs += [(ya, yb, cl, DECIDE_N_BOOT, 100, s) for s in DECIDE_SEEDS]
    return jobs


def _pack(res: list[tuple], a: dict[str, dict], b: dict[str, dict]) -> dict:
    keys = sorted(a)
    pt, lo, hi, _p, n = res[0]
    out = {
        "delta": pt,
        "ci_lo": lo,
        "ci_hi": hi,
        "n": n,
        "n_clusters": len({a[k]["cl"] for k in keys}),
        "asks_a": sum(a[k]["asks"] for k in keys) / len(keys),
        "asks_b": sum(b[k]["asks"] for k in keys) / len(keys),
        "rc_a": sum(a[k]["rc"] for k in keys) / len(keys),
        "rc_b": sum(b[k]["rc"] for k in keys) / len(keys),
    }
    if len(res) > 1:
        out["ci_1k"] = [res[1][1], res[1][2]]
        out["ci_50k"] = [[r[1], r[2]] for r in res[2:]]
        out["decided"] = decided([(r[1], r[2]) for r in res[2:]])
        out["bound_within_0.01_of_zero"] = min(abs(lo), abs(hi)) < NEAR_ZERO
        out["zero_verdict_flips_at_1k"] = (lo > 0) != (res[1][1] > 0) or (hi < 0) != (res[1][2] < 0)
    return out


def run_contrasts(
    specs: dict[str, tuple[dict, dict, bool]], est: dict, workers: int = 1
) -> dict[str, dict]:
    """name -> (a, b, full); `full` adds the 1k and the three 50k intervals and the decision."""
    names, jobs, spans = [], [], []
    for name, (a, b, full) in specs.items():
        js = _jobs(a, b, full, est, name)
        spans.append((len(jobs), len(jobs) + len(js)))
        jobs += js
        names.append(name)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_pd, jobs))
    else:
        res = [_pd(j) for j in jobs]
    return {
        n: _pack(res[s:e], specs[n][0], specs[n][1]) for n, (s, e) in zip(names, spans, strict=True)
    }


# --------------------------------------------------------------------------- the lock


def _published_units() -> tuple[dict[str, dict], list[str]]:
    store = PUB / "scores_parquet.frames"
    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT r.run_id, r.task_id, r.seed, {CLUSTER_SQL},
               r.n_asks, r.retrieval_calls, r.stop_reason, s.value, s.scorer_hash
        FROM read_parquet('{store / "runs.parquet"}') r
        JOIN read_parquet('{store / "scores.parquet"}') s
          ON s.run_id = r.run_id AND s.metric_name = '{METRIC}'
        WHERE {ELIGIBLE}
        """
    ).fetchall()
    units = {
        rid: {
            "run_id": rid,
            "key": f"{task}|{int(seed)}",
            "cl": cl,
            "asks": float(asks),
            "rc": float(rc),
            "hit": int(stop in CEILING),
            "y": float(y),
        }
        for rid, task, seed, cl, asks, rc, stop, y, _ in rows
    }
    return units, sorted({r[-1] for r in rows})


def published_cells() -> tuple[dict[tuple[str, int], dict[str, dict]], dict]:
    allu, scorer = _published_units()
    cells = {}
    prov: dict = {
        "store": _rel(PUB / "scores_parquet.frames"),
        "scorer_hash": scorer,
    }
    for arm in PUB_ARMS:
        for cap in CAPS:
            path = PUB / f"run_ids.{arm}.cap{cap}.tsv"
            ids = read_run_ids(path)
            missing = [i for i in ids if i not in allu]
            if missing:
                raise SystemExit(f"LOCK: {len(missing)} listed run ids not eligible in the store")
            cell = {allu[i]["key"]: allu[i] for i in ids}
            if len(cell) != len(ids):
                raise SystemExit(f"LOCK: {path.name} lists two runs on one unit")
            cells[(arm, cap)] = cell
            prov[f"{arm}/cap{cap}"] = {
                "run_ids": _rel(path),
                "n": len(ids),
                "run_id_sha16": _sha16(ids),
            }
    return cells, prov


def lock(est: dict, workers: int = 1) -> dict:
    text = FRONTIER_FRAMES.read_text()
    rec = parse_level_table(text, METRIC)
    extras = parse_level_extras(text)
    cells, prov = published_cells()
    out: dict = {
        "estimator": est,
        "provenance": prov,
        "levels": {},
        "contrasts": {},
        "budget_response": {},
    }
    for (arm, cap), units in cells.items():
        want = rec[(arm, cap)]
        want_cl, want_hit = extras[(arm, cap)]
        got = level(units, est["level_n_boot"], est["level_seed"])
        diffs = {
            "value": got["y"] - want.value,
            "ci_lo": got["ci_lo"] - want.lo,
            "ci_hi": got["ci_hi"] - want.hi,
            "mean_n_asks": got["mean_n_asks"] - want.mean_asks,
            "mean_retrieval_calls": got["mean_retrieval_calls"] - want.mean_calls,
        }
        out["levels"][f"{arm}/cap{cap}"] = {
            "published": [
                want.n,
                want_cl,
                want.mean_asks,
                want.mean_calls,
                want.value,
                want.lo,
                want.hi,
            ],
            "published_ceiling_hit": want_hit,
            "got": got,
            "max_abs_diff": max(abs(v) for v in diffs.values()),
            "ok": got["n"] == want.n
            and got["n_clusters"] == want_cl
            and got["ceiling_hit"] == want_hit
            and all(abs(v) <= TOL for v in diffs.values()),
        }
    tables = parse_contrast_tables(text, METRIC)
    budget = parse_budget_response(text)
    specs = {}
    for (a, b), rows in tables.items():
        for cap in rows:
            specs[f"{a}-{b}/cap{cap}"] = (cells[(a, cap)], cells[(b, cap)], False)
    for arm in budget:
        specs[f"budget/{arm}"] = (cells[(arm, 24)], cells[(arm, 4)], False)
    got = run_contrasts(specs, est, workers)
    for (a, b), rows in tables.items():
        for cap, (n, ncl, d, lo, hi) in rows.items():
            g = got[f"{a}-{b}/cap{cap}"]
            diff = max(abs(g["delta"] - d), abs(g["ci_lo"] - lo), abs(g["ci_hi"] - hi))
            out["contrasts"][f"{a}-{b}/cap{cap}"] = {
                "published": [n, ncl, d, lo, hi],
                "got": [g["n"], g["n_clusters"], g["delta"], g["ci_lo"], g["ci_hi"]],
                "max_abs_diff": diff,
                "ok": g["n"] == n and g["n_clusters"] == ncl and diff <= TOL,
            }
    for arm, (n, d, lo, hi) in budget.items():
        g = got[f"budget/{arm}"]
        diff = max(abs(g["delta"] - d), abs(g["ci_lo"] - lo), abs(g["ci_hi"] - hi))
        out["budget_response"][arm] = {
            "published": [n, d, lo, hi],
            "got": [g["n"], g["delta"], g["ci_lo"], g["ci_hi"]],
            "max_abs_diff": diff,
            "ok": g["n"] == n and diff <= TOL,
        }
    return out


def lock_failures(lk: dict) -> list[str]:
    return [
        f"{sec}/{k}"
        for sec in ("levels", "contrasts", "budget_response")
        for k, v in lk[sec].items()
        if not v["ok"]
    ]


# --------------------------------------------------------------------------- population


def _run_id_reproduces(m: dict) -> bool:
    from pinq.ids import h, semantic_hash

    sem = semantic_hash(m)
    core = h("runid", sem, m.get("counterfactual_kind"), str(m.get("prefix_k")))
    return sem == m.get("semantic_hash") and core[:32] == m.get("run_id")


_COUNTERS = (
    "dirty",
    "dev",
    "no_canaries",
    "id_not_reproduced",
    "wrong_inquirer_model",
    "no_inquirer_call",
    "wrong_frozen_call_model",
    "duplicate_units",
)


def _new_cell() -> dict:
    return {
        "dirs": 0,
        "status": {},
        "ok_units": {},
        "ok_tasks": set(),
        "codes": set(),
        "pins": set(),
        "grids": set(),
        "arms": set(),
        "base_url_sha": set(),
        "concurrency": {},
        "inquirer_calls": 0,
        "inquirer_adapter_sha": set(),
        "prompt_hashes": set(),
        "grid_sha256": set(),
        "corpus_hash": set(),
        "canaries": set(),
        "frozen_pin_models": set(),
        "started_after_local_stop": 0,
        **{k: 0 for k in _COUNTERS},
    }


def cell_problems(
    c: dict,
    cap: int,
    caps: tuple[int, ...],
    arm_id: str,
    model: str | None,
    grid: str | None = None,
) -> list[str]:
    """Every refusal of one (arm, cap) cell, from its accumulated run-directory summary.

    `grid` is the one grid_name every run must carry; None is this script's own
    `frames_recipe_cap<cap>` (frames_structured.py passes its own grid, the check is unchanged)."""
    want_grid = f"frames_recipe_cap{cap}" if grid is None else grid
    n_ok = c["status"].get("ok", 0)
    problems = []
    if cap not in caps:
        problems.append(f"cap {cap} not in this stream's caps")
    if n_ok != N_CELL or len(c["ok_units"]) != N_CELL or len(c["ok_tasks"]) != N_CELL:
        problems.append(
            f"ok={n_ok} distinct ok units={len(c['ok_units'])} tasks={len(c['ok_tasks'])}"
        )
    if c["dirs"] != n_ok:
        problems.append(f"non-ok run dirs: {c['dirs'] - n_ok} {c['status']}")
    if c["codes"] != {CODE}:
        problems.append(f"code_versions {sorted(c['codes'])}")
    if len(c["pins"]) != 1:
        problems.append(f"{len(c['pins'])} model_pin_hashes")
    if c["grids"] != {want_grid} or c["arms"] != {arm_id}:
        problems.append(f"grids {sorted(c['grids'])} arms {sorted(c['arms'])}")
    for k in ("prompt_hashes", "grid_sha256", "corpus_hash", "canaries"):
        if len(c[k]) != 1:
            problems.append(f"{len(c[k])} distinct {k}")
    if model is not None and len(c["inquirer_adapter_sha"]) != 1:
        problems.append(f"{len(c['inquirer_adapter_sha'])} inquirer adapter shas")
    roles = {mdl for _, mdl in c["frozen_pin_models"]}
    if roles != {FROZEN}:
        problems.append(f"frozen roles {sorted(c['frozen_pin_models'])}")
    problems += [f"{k}={c[k]}" for k in _COUNTERS if c[k]]
    return problems


def population(
    label: str,
    root: Path | None = None,
    arm: tuple[str, str | None, tuple[int, ...]] | None = None,
    grid: str | None = None,
) -> dict:
    """Per cap, from the run directories themselves (not from a store). `root` is for tests;
    `arm` = (arm_id, inquirer model, caps) and `grid` default to ARMS[label] and this script's
    grids (frames_structured.py passes its own labels' specs)."""
    arm_id, model, caps = ARMS[label] if arm is None else arm
    root = root or OUT / f"runs_{label}"
    cells: dict[int, dict] = {}
    no_manifest: list[str] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (d / "manifest.json").is_file():
            no_manifest.append(d.name)
            continue
        m = json.loads((d / "manifest.json").read_text())
        st_p = d / "status.json"
        st = json.loads(st_p.read_text()) if st_p.exists() else {"status": "missing"}
        c = cells.setdefault(int(m["budget_cap"]), _new_cell())
        c["dirs"] += 1
        c["status"][st["status"]] = c["status"].get(st["status"], 0) + 1
        c["codes"].add(str(m.get("code_version"))[:12])
        c["pins"].add(m.get("model_pin_hash"))
        c["grids"].add(m.get("grid_name"))
        c["arms"].add(m.get("arm_id"))
        conc = str(m.get("concurrency"))
        c["concurrency"][conc] = c["concurrency"].get(conc, 0) + 1
        c["dirty"] += bool(m.get("dirty"))
        c["dev"] += d.name.startswith("dev-") or str(m.get("run_id")).startswith("dev-")
        c["no_canaries"] += not m.get("canary_nonces_loaded")
        c["canaries"].add(m.get("canary_nonces_loaded"))
        c["prompt_hashes"].add(json.dumps(m.get("prompt_hashes"), sort_keys=True))
        c["grid_sha256"].add(m.get("grid_sha256"))
        c["corpus_hash"].add(m.get("corpus_hash"))
        c["started_after_local_stop"] += float(st.get("started_at") or 0) > LOCAL_STOP_EPOCH
        c["id_not_reproduced"] += not (_run_id_reproduces(m) and m.get("run_id") == d.name)
        for role, pin in (m.get("pins") or {}).items():
            c["base_url_sha"].add(pin.get("base_url_sha"))
            if role == "inquirer":
                c["inquirer_adapter_sha"].add(pin.get("adapter_sha"))
            else:
                c["frozen_pin_models"].add((role, pin.get("model_id")))
        if st["status"] != "ok":
            continue
        unit = f"{m['task_id']}|{int(m['seed'])}"
        if unit in c["ok_units"]:
            c["duplicate_units"] += 1
        c["ok_units"][unit] = m.get("run_id")
        c["ok_tasks"].add(m["task_id"])
        inq = set()
        calls = d / "calls.jsonl"
        for line in calls.read_text().splitlines() if calls.exists() else []:
            call = json.loads(line)
            if call.get("actor") == "inquirer":
                inq.add(call.get("model"))
                c["inquirer_calls"] += 1
            elif call.get("model") != FROZEN:
                c["wrong_frozen_call_model"] += 1
        if model is None:
            c["wrong_inquirer_model"] += bool(inq)
        else:
            c["no_inquirer_call"] += not inq
            c["wrong_inquirer_model"] += bool(inq - {model})
    out: dict = {}
    for cap, c in sorted(cells.items()):
        out[cap] = {
            "run_dirs": c["dirs"],
            "status": c["status"],
            "n_ok_distinct_units": len(c["ok_units"]),
            "n_ok_distinct_tasks": len(c["ok_tasks"]),
            "code_version": sorted(c["codes"]),
            "model_pin_hash": sorted(str(x) for x in c["pins"]),
            "grid": sorted(str(x) for x in c["grids"]),
            "base_url_sha": sorted(str(x) for x in c["base_url_sha"]),
            "concurrency": c["concurrency"],
            "inquirer_calls": c["inquirer_calls"],
            "inquirer_adapter_sha": sorted(str(x) for x in c["inquirer_adapter_sha"]),
            "frozen_pin_models": sorted(c["frozen_pin_models"]),
            "prompt_hashes_sha": sorted(
                hashlib.sha256(x.encode()).hexdigest()[:16] for x in c["prompt_hashes"]
            ),
            "grid_sha256": sorted(str(x) for x in c["grid_sha256"]),
            "corpus_hash": sorted(str(x) for x in c["corpus_hash"]),
            "canary_nonces_loaded": sorted(c["canaries"], key=str),
            "started_after_local_stop": c["started_after_local_stop"],
            **{k: c[k] for k in _COUNTERS},
            "ok_units": c["ok_units"],
            "run_id_sha16": _sha16(str(v) for v in c["ok_units"].values()),
            "problems": cell_problems(c, cap, caps, arm_id, model, grid),
        }
    for cap in caps:
        if cap not in out:
            out[cap] = {
                "problems": ["no run directories at this cap"],
                "ok_units": {},
                "base_url_sha": [],
            }
    if no_manifest:
        out["unreadable"] = {
            "problems": [f"no_manifest={len(no_manifest)}"],
            "dirs": no_manifest[:20],
            "ok_units": {},
            "base_url_sha": [],
        }
    return out


def across_cells(pop: dict[str, dict], published_units: set[str]) -> dict:
    """One base_url_sha over every role of every cell; one unit set, equal to the published one."""
    urls: set[str] = set()
    sets: dict[str, list[str]] = {}
    for label, cells in pop.items():
        for cap, c in cells.items():
            urls.update(c.get("base_url_sha", []))
            if isinstance(cap, int):
                sets.setdefault(_sha16(c["ok_units"]), []).append(f"{label}/cap{cap}")
    problems = []
    if len(urls) != 1:
        problems.append(f"{len(urls)} base_url_shas across cells: {sorted(urls)}")
    if len(sets) != 1:
        problems.append(f"{len(sets)} distinct unit sets across cells: {sets}")
    pub = _sha16(published_units)
    if set(sets) != {pub}:
        problems.append(f"unit set differs from the published record's ({pub}): {sorted(sets)}")
    return {
        "base_url_sha": sorted(urls),
        "unit_set_sha16": sets,
        "published_unit_set_sha16": pub,
        "problems": problems,
    }


# --------------------------------------------------------------------------- the scored stores


def store_units(
    label: str,
    cap: int,
    store: Path | None = None,
    arm: tuple[str, str | None, tuple[int, ...]] | None = None,
) -> tuple[dict[str, dict], list[str], list[str]]:
    """(task|seed) -> unit for one eligible cell of one scored store, its scorer hashes and graph
    versions. Refuses a wrong inquirer model (or any inquirer call for drafter_only) and two
    eligible runs on one unit. `store` is for tests; `arm` defaults to ARMS[label]."""
    arm_id, model, _ = ARMS[label] if arm is None else arm
    store = store or OUT / f"scores_parquet_{label}"
    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT r.run_id, r.task_id, r.seed, {CLUSTER_SQL},
               r.n_asks, r.retrieval_calls, r.stop_reason, s.value, s.scorer_hash, s.graph_version
        FROM read_parquet('{store / "runs.parquet"}') r
        JOIN read_parquet('{store / "scores.parquet"}') s
          ON s.run_id = r.run_id AND s.metric_name = '{METRIC}'
        WHERE {ELIGIBLE} AND r.arm_id = ? AND r.suite_id = 'frames' AND r.budget_cap = ?
        """,
        [arm_id, cap],
    ).fetchall()
    models: dict[str, set[str]] = {}
    for rid, mdl in con.execute(
        f"SELECT DISTINCT run_id, model FROM read_parquet('{store / 'calls.parquet'}') "
        "WHERE actor = 'inquirer'"
    ).fetchall():
        models.setdefault(rid, set()).add(mdl)
    out: dict[str, dict] = {}
    hashes: set[str] = set()
    graphs: set[str] = set()
    for rid, task, seed, cl, asks, rc, stop, y, sh, gv in rows:
        if models.get(rid) != ({model} if model else None):
            raise SystemExit(
                f"REFUSING: {label} cap {cap} run {rid} inquirer models {models.get(rid)}"
            )
        key = f"{task}|{int(seed)}"
        if key in out:
            raise SystemExit(f"REFUSING: two eligible runs at {key} in {label} cap {cap}")
        hashes.add(sh)
        graphs.add(gv)
        out[key] = {
            "run_id": rid,
            "cl": cl,
            "asks": float(asks),
            "rc": float(rc),
            "hit": int(stop in CEILING),
            "y": float(y),
        }
    return out, sorted(hashes), sorted(graphs)


def store_problems(units: dict[str, dict], want_ids: set[str]) -> list[str]:
    """The eligible runs of a scored cell must be exactly the population's run ids, N_CELL of them."""
    got = {v["run_id"] for v in units.values()}
    problems = []
    if len(units) != N_CELL:
        problems.append(f"eligible in store {len(units)} != {N_CELL}")
    if got != want_ids:
        problems.append(
            f"store != population: {len(got - want_ids)} only in store, {len(want_ids - got)} only on disk"
        )
    return problems


def store_identity_problems(store_check: dict[str, dict]) -> list[str]:
    hashes = {h for v in store_check.values() for h in v["scorer_hash"]}
    graphs = {g for v in store_check.values() for g in v["graph_version"]}
    problems = []
    if len(hashes) != 1:
        problems.append(f"{len(hashes)} scorer_hash values across stores: {sorted(hashes)}")
    if len(graphs) != 1:
        problems.append(f"{len(graphs)} graph_version values across stores: {sorted(graphs)}")
    return problems


# --------------------------------------------------------------------------- the new reading


def read(pop: dict, est: dict, workers: int, pub_cells: dict, pub_scorer: list[str]) -> dict:
    cells: dict[tuple[str, int], dict[str, dict]] = {}
    store_check: dict[str, dict] = {}
    refused: list[str] = []
    for label, (_, _, caps) in ARMS.items():
        for cap in caps:
            u, hashes, graphs = store_units(label, cap)
            probs = store_problems(u, set(pop[label][cap]["ok_units"].values()))
            store_check[f"{label}/cap{cap}"] = {
                "store": _rel(OUT / f"scores_parquet_{label}"),
                "eligible_in_store": len(u),
                "run_id_sha16": _sha16(v["run_id"] for v in u.values()),
                "scorer_hash": hashes,
                "graph_version": graphs,
                "problems": probs,
            }
            refused += [f"{label}/cap{cap}: {p}" for p in probs]
            cells[(label, cap)] = u
    refused += store_identity_problems(store_check)
    if refused:
        raise SystemExit(f"REFUSING: scored stores are not the population: {refused[:6]}")
    scorer = store_check["s1/cap8"]["scorer_hash"]
    out: dict = {
        "store_check": store_check,
        "scorer_hash": scorer,
        "scorer_hash_equals_published": scorer == pub_scorer,
        "levels": {},
        "contrasts": {},
    }
    lb, ls = READ_N_BOOT, est["level_seed"]
    rec = {cap: pool_seeds(cells[("s1", cap)], cells[("s2", cap)]) for cap in CAPS}
    for cap in CAPS:
        out["levels"].setdefault("recipe", {})[cap] = level(rec[cap], lb, ls)
        for label in ("teacher", "base", "s1", "s2"):
            out["levels"].setdefault(label, {})[cap] = level(cells[(label, cap)], lb, ls)
    out["levels"]["drafter_only"] = {8: level(cells[("drafter", 8)], lb, ls)}
    out["levels"]["seed0_published"] = {
        cap: level(pub_cells[("trained", cap)], lb, ls) for cap in CAPS
    }
    specs: dict[str, tuple[dict, dict, bool]] = {}
    for cap in CAPS:
        for comp in ("base", "teacher"):
            specs[f"recipe-{comp}/cap{cap}"] = (rec[cap], cells[(comp, cap)], True)
            for s in ("s1", "s2"):
                specs[f"{s}-{comp}/cap{cap}"] = (cells[(s, cap)], cells[(comp, cap)], False)
        specs[f"base-teacher/cap{cap}"] = (cells[("base", cap)], cells[("teacher", cap)], False)
    specs["recipe-drafter_only/cap8"] = (rec[8], cells[("drafter", 8)], True)
    specs["budget/recipe"] = (rec[24], rec[4], True)
    for label in ("teacher", "base", "s1", "s2"):
        specs[f"budget/{label}"] = (cells[(label, 24)], cells[(label, 4)], True)
    out["contrasts"] = run_contrasts(specs, {**est, "contrast_n_boot": READ_N_BOOT}, workers)
    # The comparators re-ran at a new commit and base URL: how many units replay the published
    # trajectory's outcome (same answer_correct AND same n_asks), per cap.
    rep = {}
    for label, pub in (("teacher", "teacher"), ("base", "base8b")):
        for cap in CAPS:
            a, b = cells[(label, cap)], pub_cells[(pub, cap)]
            ks = sorted(set(a) & set(b))
            rep[f"{label}/cap{cap}"] = {
                "n": len(ks),
                "same_answer_correct": sum(a[k]["y"] == b[k]["y"] for k in ks),
                "same_n_asks": sum(a[k]["asks"] == b[k]["asks"] for k in ks),
            }
    out["comparator_replay_vs_published"] = rep
    return out


# --------------------------------------------------------------------------- tables


def _ci(lo: float, hi: float) -> str:
    return f"[{lo:+.4f}, {hi:+.4f}]"


def _zero(lo: float, hi: float) -> str:
    return "excludes 0" if lo > 0 or hi < 0 else "covers 0"


def render(out: dict) -> str:
    """Markdown tables, every number read from the json this script wrote."""
    lines: list[str] = []
    lk = out["lock"]
    lines += [
        "### Lock: every published FRAMES_FRONTIER.md row, reproduced",
        "",
        f"estimator read from the record: {lk['estimator']}",
        "",
        "| row | published | reproduced | max abs diff | ok |",
        "|---|---|---|---|---|",
    ]
    for k, v in lk["levels"].items():
        p, g = v["published"], v["got"]
        lines.append(
            f"| level {k} | n={p[0]} cl={p[1]} asks={p[2]:.4f} calls={p[3]:.4f} {p[4]:.4f} "
            f"[{p[5]:.4f}, {p[6]:.4f}] hit={v['published_ceiling_hit']} | n={g['n']} "
            f"cl={g['n_clusters']} asks={g['mean_n_asks']:.6f} calls={g['mean_retrieval_calls']:.6f} "
            f"{g['y']:.6f} [{g['ci_lo']:.6f}, {g['ci_hi']:.6f}] hit={g['ceiling_hit']} | "
            f"{v['max_abs_diff']:.2e} | {v['ok']} |"
        )
    for k, v in lk["contrasts"].items():
        p, g = v["published"], v["got"]
        lines.append(
            f"| contrast {k} | n={p[0]} cl={p[1]} {p[2]:+.4f} {_ci(p[3], p[4])} | n={g[0]} "
            f"cl={g[1]} {g[2]:+.6f} [{g[3]:+.6f}, {g[4]:+.6f}] | {v['max_abs_diff']:.2e} | {v['ok']} |"
        )
    for k, v in lk["budget_response"].items():
        p, g = v["published"], v["got"]
        lines.append(
            f"| cap24-cap4 {k} | n={p[0]} {p[1]:+.4f} {_ci(p[2], p[3])} | n={g[0]} {g[1]:+.6f} "
            f"[{g[2]:+.6f}, {g[3]:+.6f}] | {v['max_abs_diff']:.2e} | {v['ok']} |"
        )
    if "reading" not in out:
        return "\n".join(lines) + "\n"
    rd = out["reading"]
    lines += [
        "",
        f"scorer_hash {rd['scorer_hash']} (equals published: {rd['scorer_hash_equals_published']})",
        "",
        "### Levels, answer_correct (cluster_bootstrap, 10,000 resamples, seed 0)",
        "",
        "| arm | cap | n | clusters | mean n_asks | mean retrieval_calls | answer_correct [BCa 95%] "
        "| ceiling-hit share |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in ("recipe", "s1", "s2", "base", "teacher", "drafter_only", "seed0_published"):
        for cap, v in sorted(rd["levels"][arm].items(), key=lambda kv: int(kv[0])):
            lines.append(
                f"| {arm} | {cap} | {v['n']} | {v['n_clusters']} | {v['mean_n_asks']:.4f} | "
                f"{v['mean_retrieval_calls']:.4f} | {v['y']:.6f} [{v['ci_lo']:.6f}, "
                f"{v['ci_hi']:.6f}] | {v['ceiling_hit_share']:.4f} |"
            )
    ct = rd["contrasts"]
    lines += [
        "",
        "### Contrasts (paired over tasks; 10k seed 0; 1k seed 0; 50k seeds 101/202/303)",
        "",
        "| contrast | n | delta | 10k BCa 95% | zero | 1k | 50k x3 | DECIDED | mean asks a / b |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def order(name: str) -> tuple:
        head, _, cap = name.partition("/cap")
        return (head, int(cap) if cap.isdigit() else 0, name)

    for name in sorted(ct, key=order):
        v = ct[name]
        fifty = "; ".join(_ci(lo, hi) for lo, hi in v.get("ci_50k", [])) or "-"
        one_k = _ci(*v["ci_1k"]) if "ci_1k" in v else "-"
        dec = {True: "DECIDED", False: "not decided"}.get(v.get("decided"), "-")
        lines.append(
            f"| {name} | {v['n']} | {v['delta']:+.6f} | {_ci(v['ci_lo'], v['ci_hi'])} | "
            f"{_zero(v['ci_lo'], v['ci_hi'])} | {one_k} | {fifty} | {dec} | "
            f"{v['asks_a']:.4f} / {v['asks_b']:.4f} |"
        )
    lines += [
        "",
        "### Comparators re-run at a61c4f4 vs the published 107ef22 runs, per task",
        "",
        "| cell | n | same answer_correct | same n_asks |",
        "|---|---|---|---|",
    ]
    for k, v in rd["comparator_replay_vs_published"].items():
        lines.append(f"| {k} | {v['n']} | {v['same_answer_correct']} | {v['same_n_asks']} |")
    return "\n".join(lines) + "\n"


def _dump(path: Path, out: dict) -> None:
    path.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", choices=("lock", "population", "all"), default="all")
    ap.add_argument("--workers", type=int, default=1, help="processes for the bootstraps")
    ap.add_argument("--render", type=Path, help="print the tables from a written json")
    args = ap.parse_args(argv)
    if args.render:
        print(render(json.loads(args.render.read_text())), end="")
        return 0

    est = parse_estimator(FRONTIER_FRAMES.read_text())
    if est["cluster"] != normalise_cluster(CLUSTER_SQL):
        raise SystemExit(f"REFUSING: record clusters on {est['cluster']}, script on {CLUSTER_SQL}")
    out: dict = {"code_version": CODE, "metric": METRIC, "n_cell": N_CELL}
    out["lock"] = lock(est, args.workers)
    bad = lock_failures(out["lock"])
    n_rows = sum(len(out["lock"][s]) for s in ("levels", "contrasts", "budget_response"))
    print(f"lock: {n_rows - len(bad)}/{n_rows} published rows reproduced; failures {bad[:6]}")
    if bad or args.stage == "lock":
        _dump(args.out, out)
        return 1 if bad else 0

    pub_cells, pub_prov = published_cells()
    pop = {label: population(label) for label in ARMS}
    across = across_cells(pop, set(pub_cells[("trained", 8)]))
    out["population"] = {lab: {str(k): v for k, v in p.items()} for lab, p in pop.items()}
    out["across_cells"] = across
    probs = [
        f"{lab}/{cap}: {c['problems']}"
        for lab, p in pop.items()
        for cap, c in p.items()
        if c["problems"]
    ]
    probs += across["problems"]
    out["population_failures"] = probs
    print(f"population: {len(probs)} refusals; base_url_sha {across['base_url_sha']}")
    for p in probs:
        print("  ", p)
    if probs or args.stage == "population":
        _dump(args.out, out)
        return 1 if probs else 0

    out["reading"] = read(pop, est, args.workers, pub_cells, pub_prov["scorer_hash"])
    _dump(args.out, out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
