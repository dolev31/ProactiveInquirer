"""Lane L3, part A (no model calls): where the recipe's evidence gain is lost before the answer.

Implements `artifacts/answer_loss_20260923/DECLARATION.md` (committed c2bdf77 before any data).

POPULATION. The recipe = `qwen3-8b-dpo-stacked-notdone-both-s1` and `-s2`
(`artifacts/seedrep_gate_20260919`), training seeds averaged WITHIN the task, rollout seeds 0 and
1 averaged into the task. The comparator = the same weights prompted, the completed cohort
(`artifacts/completed_cohort_20260922/cohort/run_ids.prompted.<suite>.txt`). Paired over tasks,
clusters = `template_id` where the population carries one (MuSiQue), else the task.

WHAT IS REUSED, NOT RE-DERIVED. The answer-bearing node is `pi_eval.answer_node` via
`scripts.answer_node_coverage.run.build_answer_nodes_for_suite`, and "covered" is that lane's
`lib.covered_map` (matcher `resolve`/`use` on any answer node), i.e. the definition behind
`artifacts/seed_identity_20260923/answer_node_s{0,1,2}.json` (commit 7bb48a7). Containment is
the scorer's own `pi_eval.metrics.quality.contains_answer`. Every interval is
`pi_eval.stats.inference.paired_difference`.

LOCKS, BEFORE ANY NEW NUMBER.
  A1  per-arm P(answer node covered) reproduces answer_node_s1.json / answer_node_s2.json
      (level_trained, level_prompted, and their difference) on those records' own population
      (the stop_pop_s{1,2} store: the seed's 400 runs against the testsplit_qa prompted runs).
  A2  (a) per-arm unconditional P(answer_correct) and mean answer_token_f1 equal the stored
      metric means; (b) the recipe-minus-comparator answer_token_f1 difference reproduces
      stop_s1.json / stop_s2.json step2 (no pooled s1+s2 answer record exists).

Usage (from the repo root; the analysis process is the only one that sees gold):
    PI_GOLD_ROOT=$PWD/data/gold .venv/bin/python -m scripts.answer_loss.decompose \\
        --out artifacts/answer_loss_20260923/decompose.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

from pi_eval.answer_node import BOOL_ANSWERS
from pi_eval.metrics.quality import contains_answer, normalize
from pinq_expt.components import UNIT_CHARS

REPO = Path(__file__).resolve().parents[2]
SEEDREP = REPO / "artifacts" / "seedrep_gate_20260919"
COHORT = REPO / "artifacts" / "completed_cohort_20260922"
SEED_ID = REPO / "artifacts" / "seed_identity_20260923"
CORPORA = REPO / "data" / "corpora"
SUITES = ("musique", "strategyqa", "wiki2")
RECIPE = {
    "s1": "qwen3-8b-dpo-stacked-notdone-both-s1",
    "s2": "qwen3-8b-dpo-stacked-notdone-both-s2",
}
SCORER_HASH = "e82c7458ee8efa11660445bbf434b4c1dc74253253e8f190cc36e92fc50f40c7"
GRAPH_VERSION = "v1"
RESAMPLES = ((10_000, 0), (50_000, 101), (50_000, 202), (50_000, 303))
NEAR_ZERO = 0.01
LOCK_TOL = 1e-12

# --------------------------------------------------------------------------- pure: answer location


def normalized_tokens(text: str) -> list[tuple[str, int, int]]:
    """`normalize(text).split()`, each token carrying the character span of the whitespace
    chunk it came from. `normalize` never creates whitespace (it deletes ASCII punctuation and
    replaces whole-word articles), so normalizing chunk by chunk yields the same token stream
    as normalizing the whole string -- pinned by a test, because `answer_span` is only
    meaningful if it sees exactly what `contains_answer` sees."""
    out: list[tuple[str, int, int]] = []
    for m in re.finditer(r"\S+", text):
        for tok in normalize(m.group()).split():
            out.append((tok, m.start(), m.end()))
    return out


def answer_span(text: str, gold: str, aliases: Sequence[str] = ()) -> tuple[int, int] | None:
    """Character span of the EARLIEST occurrence of the gold answer or any alias, matched as
    `contains_answer` matches (a token sequence after `normalize`), or None."""
    toks = normalized_tokens(text)
    words = [t for t, _s, _e in toks]
    best: tuple[int, int] | None = None
    for g in (gold, *aliases):
        g_toks = normalize(str(g or "")).split()
        if not g_toks:
            continue
        n = len(g_toks)
        for i in range(len(words) - n + 1):
            if words[i : i + n] == g_toks:
                span = (toks[i][1], toks[i + n - 1][2])
                if best is None or span[0] < best[0]:
                    best = span
                break
    return best


def window_reading(
    text: str, gold: str, aliases: Sequence[str] = (), *, window: int = UNIT_CHARS
) -> dict[str, bool]:
    """One paragraph against the drafter's per-unit window (`render_evidence`'s
    `u.text[:UNIT_CHARS]`).

    `starts_past`: the answer's first occurrence starts at or after character `window` (the
    brief's definition). `visible`: the answer is contained in the rendered window
    `text[:window]` -- stricter, it also fails an answer that straddles the cut."""
    span = answer_span(text, gold, aliases)
    return {
        "contains": span is not None,
        "starts_past": span is not None and span[0] >= window,
        "visible": bool(contains_answer(text[:window], gold, tuple(aliases))),
        "longer": len(text) > window,
    }


def is_boolean_answer(answer: str) -> bool:
    """A yes/no gold answer: `contains_answer` of "yes" in a draft carries no information about
    whether the draft settled the question, so such tasks have no draft stage."""
    return normalize(answer) in BOOL_ANSWERS


# --------------------------------------------------------------------------- pure: the stage chain


def cumulative_stages(
    covered: bool, in_draft: float | None, correct: float
) -> tuple[float, float | None, float | None]:
    """(covered, covered AND in draft, covered AND in draft AND correct) for one run. The draft
    stage is None (excluded, never a miss) when the task has no draft stage."""
    c = 1.0 if covered else 0.0
    if in_draft is None:
        return (c, None, None)
    s2 = c * (1.0 if in_draft else 0.0)
    return (c, s2, s2 * (1.0 if correct else 0.0))


# --------------------------------------------------------------------------- pure: aggregation


def task_means(
    rows: Iterable[tuple[str, float | None]], *, expect_per_task: int | None
) -> dict[str, float]:
    """Per-task mean of run values, keys SORTED (the bootstrap reads dicts in insertion order
    only through `paired_difference`'s own sort, but a sorted input costs nothing and removes
    the question). Every task must carry exactly `expect_per_task` runs (None values count as
    runs and are then skipped), so an unbalanced population raises instead of silently
    re-weighting one training seed."""
    counts: dict[str, int] = {}
    vals: dict[str, list[float]] = {}
    for task, v in rows:
        counts[task] = counts.get(task, 0) + 1
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        vals.setdefault(task, []).append(float(v))
    if expect_per_task is not None:
        bad = {t: n for t, n in counts.items() if n != expect_per_task}
        if bad:
            raise ValueError(
                f"expected {expect_per_task} runs per task, got {dict(list(bad.items())[:5])}"
            )
    return {t: sum(vals[t]) / len(vals[t]) for t in sorted(vals)}


def pair_task_means(
    pairs: Iterable[tuple[str, float | None, float | None]],
) -> tuple[dict[str, float], dict[str, float]]:
    """`pooled_symmetric`'s aggregation: every (task, recipe value, comparator value) pair --
    one per (training seed, rollout seed) -- enters one per-task mean for each arm. A pair with
    either side undefined is dropped whole, so both arms always average over the same pairs."""
    acc: dict[str, tuple[list[float], list[float]]] = {}
    for task, a, b in pairs:
        if a is None or b is None or math.isnan(a) or math.isnan(b):
            continue
        la, lb = acc.setdefault(task, ([], []))
        la.append(float(a))
        lb.append(float(b))
    keys = sorted(acc)
    return (
        {t: sum(acc[t][0]) / len(acc[t][0]) for t in keys},
        {t: sum(acc[t][1]) / len(acc[t][1]) for t in keys},
    )


# --------------------------------------------------------------------------- pure: the primary


def implied_gain(d_covered: float, rate_covered: float, rate_uncovered: float) -> float:
    """The answer gain the coverage gain would carry if the comparator's conditional answer
    rates held: d(P covered) x [rate | covered - rate | not covered]."""
    return d_covered * (rate_covered - rate_uncovered)


def mde(diffs: Sequence[float], *, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable paired effect: (z_{1-alpha/2} + z_power) * SD(diffs) / sqrt(n), the
    SD the observed paired differences carry (sample SD, n - 1)."""
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return z * statistics.stdev(diffs) / math.sqrt(len(diffs))


def stratum(cov_recipe: bool, cov_comparator: bool) -> str:
    if cov_recipe and cov_comparator:
        return "both"
    if cov_recipe:
        return "only_recipe"
    if cov_comparator:
        return "only_comparator"
    return "neither"


def decided(cells: Sequence[Mapping[str, Any]]) -> str:
    """DECIDED only if the 50k interval excludes zero on the same side under EACH of the three
    bootstrap seeds (101, 202, 303)."""
    big = [c for c in cells if int(c["n_boot"]) == 50_000]
    if len(big) != 3:
        raise ValueError(f"need exactly three 50k intervals, got {len(big)}")
    if all(float(c["ci_lo"]) > 0 for c in big):
        return "DECIDED (+)"
    if all(float(c["ci_hi"]) < 0 for c in big):
        return "DECIDED (-)"
    return "not decided"


# --------------------------------------------------------------------------- statistics wrapper


def paired_reading(
    a: Mapping[str, float], b: Mapping[str, float], *, clusters: Mapping[str, str] | None
) -> dict[str, Any]:
    """recipe - comparator over shared tasks: 10k interval (seed 0) printed, 50k at 101/202/303
    for the verdict, and a 1k read wherever any bound sits within 0.01 of zero."""
    from pi_eval.stats.inference import paired_difference

    keys = sorted(set(a) & set(b))
    a = {k: a[k] for k in keys}
    b = {k: b[k] for k in keys}
    cl = {k: clusters[k] for k in keys} if clusters else None
    cells = []
    for nb, sd in RESAMPLES:
        e = paired_difference(a, b, clusters=cl, n_boot=nb, seed=sd)
        cells.append(
            {
                "n_boot": nb,
                "seed": sd,
                "point": e.point,
                "ci_lo": e.ci_lo,
                "ci_hi": e.ci_hi,
                "p_value": e.p_value,
                "n": e.n,
            }
        )
    out: dict[str, Any] = {
        "point": cells[0]["point"],
        "ci_lo": cells[0]["ci_lo"],
        "ci_hi": cells[0]["ci_hi"],
        "n_tasks": cells[0]["n"],
        "n_clusters": len(set(cl.values())) if cl else len(keys),
        "level_recipe": sum(a.values()) / len(a) if a else float("nan"),
        "level_comparator": sum(b.values()) / len(b) if b else float("nan"),
        "resamples": cells,
        "verdict": decided(cells),
    }
    if any(abs(c[x]) <= NEAR_ZERO for c in cells for x in ("ci_lo", "ci_hi")):
        e = paired_difference(a, b, clusters=cl, n_boot=1_000, seed=0)
        ex10 = cells[0]["ci_lo"] > 0 or cells[0]["ci_hi"] < 0
        ex1 = e.ci_lo > 0 or e.ci_hi < 0
        out["check_1k"] = {"ci_lo": e.ci_lo, "ci_hi": e.ci_hi, "flips_vs_10k": ex1 != ex10}
    # the paired per-cluster differences, for the minimum detectable effect
    groups: dict[str, list[float]] = {}
    for k in keys:
        groups.setdefault(cl[k] if cl else k, []).append(a[k] - b[k])
    diffs = [sum(v) / len(v) for v in groups.values()]
    out["paired_sd"] = statistics.stdev(diffs) if len(diffs) > 1 else float("nan")
    out["mde_80_05"] = mde(diffs) if len(diffs) > 1 else float("nan")
    return out


# --------------------------------------------------------------------------- data


def read_ids(path: Path) -> list[str]:
    return [
        ln.split()[0]
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def list_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


class Store:
    """Read-only views over one isolated scored store."""

    def __init__(self, path: Path) -> None:
        import duckdb

        self.path = path
        self.con = duckdb.connect()
        for name in ("runs", "scores", "turns", "matches", "evidence"):
            p = (path / f"{name}.parquet").as_posix()
            self.con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{p}')")

    def rows(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        cur = self.con.execute(sql, list(params))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def runs(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        got = self.rows(
            "SELECT run_id, suite_id, task_id, seed, template_id, n_asks, stop_reason, corpus_dir, "
            "code_version FROM runs WHERE run_id IN (SELECT unnest(?)) ORDER BY run_id",
            [list(ids)],
        )
        if len(got) != len(set(ids)):
            raise SystemExit(f"{self.path}: {len(set(ids)) - len(got)} run ids missing from runs")
        return got

    def metric(self, ids: Sequence[str], name: str) -> dict[str, float]:
        got = self.rows(
            "SELECT run_id, value, scorer_hash FROM scores WHERE metric_name = ? "
            "AND run_id IN (SELECT unnest(?))",
            [name, list(ids)],
        )
        hashes = {r["scorer_hash"] for r in got}
        if hashes != {SCORER_HASH}:
            raise SystemExit(f"{self.path}: {name} carries scorer hashes {hashes}")
        out = {r["run_id"]: float(r["value"]) for r in got}
        if len(out) != len(set(ids)):
            raise SystemExit(f"{self.path}: {name} missing for {len(set(ids)) - len(out)} runs")
        return out

    def stored_mean(self, ids: Sequence[str], name: str) -> float:
        return float(
            self.rows(
                "SELECT avg(value) AS m FROM scores WHERE metric_name = ? AND scorer_hash = ? "
                "AND run_id IN (SELECT unnest(?))",
                [name, SCORER_HASH, list(ids)],
            )[0]["m"]
        )

    def final_drafts(self, ids: Sequence[str]) -> dict[str, str]:
        got = self.rows(
            "SELECT run_id, arg_max(draft_text, turn_idx) AS d, max(turn_idx) AS last, count(*) AS n "
            "FROM turns WHERE run_id IN (SELECT unnest(?)) GROUP BY run_id",
            [list(ids)],
        )
        for r in got:
            if int(r["last"]) != int(r["n"]) - 1:
                raise SystemExit(f"{r['run_id']}: turn_idx not contiguous")
        return {r["run_id"]: str(r["d"] or "") for r in got}


def answer_nodes(suite: str, task_ids: Sequence[str], corpus_dir: str):
    from scripts.answer_node_coverage.run import build_answer_nodes_for_suite

    return build_answer_nodes_for_suite(
        suite=suite,
        task_ids=list(task_ids),
        corpora_root=CORPORA,
        corpus_dir=corpus_dir,
        graph_version=GRAPH_VERSION,
    )


def covered_for(
    store: Store, runs: Sequence[Mapping[str, Any]], results: Mapping[str, Any]
) -> dict[str, bool | None]:
    from scripts.answer_node_coverage import lib as an_lib

    keyed = {(str(r["suite_id"]), str(r["task_id"])): results[str(r["task_id"])] for r in runs}
    return an_lib.covered_map(store.con, runs, keyed)


def _mean_or_nan(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def ratio_level(
    runs: Sequence[Mapping[str, Any]], covered: Mapping[str, bool | None], tasks: set[str]
) -> float:
    num = den = 0.0
    for r in runs:
        if r["task_id"] not in tasks:
            continue
        v = covered.get(r["run_id"])
        if v is None:
            continue
        num += 1.0 if v else 0.0
        den += 1.0
    return num / den


# --------------------------------------------------------------------------- locks


def lock_a1(results_by_suite: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in ("s1", "s2"):
        rec = json.loads((SEED_ID / f"answer_node_{s}.json").read_text())
        pop = SEED_ID / f"stop_pop_{s}"
        store = Store(pop / "scores_parquet")
        for suite in SUITES:
            t_ids = read_ids(pop / f"run_ids.trained.{suite}.txt")
            p_ids = read_ids(pop / f"run_ids.prompted.{suite}.txt")
            t_runs, p_runs = store.runs(t_ids), store.runs(p_ids)
            res = results_by_suite[suite]
            cov = covered_for(store, t_runs + p_runs, res)
            shared = {r["task_id"] for r in t_runs} & {r["task_id"] for r in p_runs}
            lt, lp = ratio_level(t_runs, cov, shared), ratio_level(p_runs, cov, shared)
            want = rec["per_suite"][suite]["p_answer_node_covered"]
            cell = {
                "record": f"artifacts/seed_identity_20260923/answer_node_{s}.json",
                "n_tasks": len(shared),
                "level_trained": lt,
                "record_level_trained": want["level_trained"],
                "level_prompted": lp,
                "record_level_prompted": want["level_prompted"],
                "delta": lt - lp,
                "record_point": want["point"],
            }
            cell["ok"] = (
                abs(lt - want["level_trained"]) <= LOCK_TOL
                and abs(lp - want["level_prompted"]) <= LOCK_TOL
                and abs((lt - lp) - want["point"]) <= LOCK_TOL
                and len(shared) == want["n_tasks"]
            )
            out[f"{s}::{suite}"] = cell
            if not cell["ok"]:
                raise SystemExit(f"LOCK A1 FAILED {s} {suite}: {cell}")
    return out


def lock_a2b() -> dict[str, Any]:
    """stop_s{1,2}.json step2 answer_token_f1, re-derived from this module's own task_means."""
    from scripts.stopping_answer_test import lib as st_lib

    from pi_eval.stats.inference import paired_difference

    out: dict[str, Any] = {}
    for s in ("s1", "s2"):
        rec = json.loads((SEED_ID / f"stop_{s}.json").read_text())["step2_answer_quality"]
        pop = SEED_ID / f"stop_pop_{s}"
        store = Store(pop / "scores_parquet")
        for suite in SUITES:
            t_ids = read_ids(pop / f"run_ids.trained.{suite}.txt")
            p_ids = read_ids(pop / f"run_ids.prompted.{suite}.txt")
            t_runs, p_runs = store.runs(t_ids), store.runs(p_ids)
            f1 = store.metric(t_ids + p_ids, "answer_token_f1")
            a = task_means(((r["task_id"], f1[r["run_id"]]) for r in t_runs), expect_per_task=None)
            b = task_means(((r["task_id"], f1[r["run_id"]]) for r in p_runs), expect_per_task=None)

            def compute(n_boot: int, seed: int, a=a, b=b) -> dict[str, Any]:
                e = paired_difference(a, b, clusters=None, n_boot=n_boot, seed=seed)
                return {"point": e.point, "lo": e.ci_lo, "hi": e.ci_hi, "n": e.n}

            got = st_lib.with_stability(compute, seed=0)
            want = rec[suite]["answer_token_f1"]
            cell = {
                "record": f"artifacts/seed_identity_20260923/stop_{s}.json step2_answer_quality.{suite}.answer_token_f1",
                "point": got["point"],
                "record_point": want["point"],
                "lo": got["lo"],
                "record_lo": want["lo"],
                "hi": got["hi"],
                "record_hi": want["hi"],
                "n": got["n"],
                "record_n": want["n"],
            }
            cell["ok"] = (
                all(abs(cell[k] - cell[f"record_{k}"]) <= LOCK_TOL for k in ("point", "lo", "hi"))
                and cell["n"] == cell["record_n"]
            )
            out[f"{s}::{suite}"] = cell
            if not cell["ok"]:
                raise SystemExit(f"LOCK A2b FAILED {s} {suite}: {cell}")
    return out


# --------------------------------------------------------------------------- the reading


def rendering_stats(
    suite: str, graphs: Mapping[str, Any], results: Mapping[str, Any], corpus_dir: str
) -> dict[str, Any]:
    """Among the answer node's gold paragraphs, where the answer sits against the 1,200-char
    unit window. Paragraph text is the suite adapter's own unit text (`load_suite(...).units`),
    i.e. exactly the string `render_evidence` cuts."""
    from scripts.answerer_strength import lib as as_lib

    su = as_lib.get_suite(suite, corpus_dir, CORPORA)
    n_par = n_long = 0
    ab = {"n": 0, "starts_past": 0, "not_visible": 0, "longer": 0}
    tasks_ab = tasks_all_hidden = tasks_bool = tasks_no_hit = 0
    unresolved = 0
    for task in sorted(results):
        res = results[task]
        graph = graphs[task]
        units = {u.uid: u for u in su.units(task)}
        node_by_id = {n.gold_node_id: n for n in graph.gold_nodes}
        uids = sorted({u for nid in res.node_ids for u in node_by_id[nid].gold_ev_uids})
        texts = []
        for u in uids:
            if u not in units:
                unresolved += 1
                continue
            texts.append(units[u].text)
        n_par += len(texts)
        n_long += sum(len(t) > UNIT_CHARS for t in texts)
        if is_boolean_answer(graph.answer):
            tasks_bool += 1
            continue
        readings = [window_reading(t, graph.answer, graph.gold_aliases) for t in texts]
        bearing = [r for r in readings if r["contains"]]
        if not bearing:
            tasks_no_hit += 1
            continue
        tasks_ab += 1
        ab["n"] += len(bearing)
        ab["starts_past"] += sum(r["starts_past"] for r in bearing)
        ab["not_visible"] += sum(not r["visible"] for r in bearing)
        ab["longer"] += sum(r["longer"] for r in bearing)
        if not any(r["visible"] for r in bearing):
            tasks_all_hidden += 1
    return {
        "n_tasks": len(results),
        "n_answer_node_paragraphs": n_par,
        "n_unresolved_uids": unresolved,
        "share_answer_node_paragraphs_longer_than_window": n_long / n_par
        if n_par
        else float("nan"),
        "n_tasks_boolean_answer_excluded": tasks_bool,
        "n_tasks_no_answer_bearing_paragraph": tasks_no_hit,
        "n_tasks_with_answer_bearing_paragraph": tasks_ab,
        "n_answer_bearing_paragraphs": ab["n"],
        "share_answer_starts_past_window": ab["starts_past"] / ab["n"] if ab["n"] else float("nan"),
        "share_answer_not_visible_in_window": ab["not_visible"] / ab["n"]
        if ab["n"]
        else float("nan"),
        "share_answer_bearing_longer_than_window": ab["longer"] / ab["n"]
        if ab["n"]
        else float("nan"),
        "n_tasks_answer_hidden_in_every_bearing_paragraph": tasks_all_hidden,
        "window_chars": UNIT_CHARS,
    }


def read_suite(suite: str) -> dict[str, Any]:
    rec_store, cmp_store = Store(SEEDREP / "scores_parquet"), Store(COHORT / "scores_parquet")
    rec_ids = {
        s: read_ids(SEEDREP / "run_ids" / f"run_ids.{m}.{suite}.txt") for s, m in RECIPE.items()
    }
    cmp_list = COHORT / "cohort" / f"run_ids.prompted.{suite}.txt"
    cmp_ids = read_ids(cmp_list)
    all_rec = rec_ids["s1"] + rec_ids["s2"]
    rec_runs, cmp_runs = rec_store.runs(all_rec), cmp_store.runs(cmp_ids)
    tseed = {rid: s for s, ids in rec_ids.items() for rid in ids}

    tasks = sorted({r["task_id"] for r in rec_runs})
    if tasks != sorted({r["task_id"] for r in cmp_runs}):
        raise SystemExit(f"{suite}: recipe and comparator task sets differ")
    corpus_dirs = {r["corpus_dir"] for r in rec_runs + cmp_runs}
    if len(corpus_dirs) != 1:
        raise SystemExit(f"{suite}: corpus dirs {corpus_dirs}")
    corpus_dir = corpus_dirs.pop()
    graphs, results = answer_nodes(suite, tasks, corpus_dir)

    templates = {r["task_id"]: r["template_id"] for r in rec_runs + cmp_runs}
    clusters = {t: str(v) for t, v in templates.items() if v} if any(templates.values()) else None
    if clusters is not None and len(clusters) != len(tasks):
        raise SystemExit(f"{suite}: template_id present on only some tasks")

    cov = {**covered_for(rec_store, rec_runs, results), **covered_for(cmp_store, cmp_runs, results)}
    if any(v is None for v in cov.values()):
        raise SystemExit(f"{suite}: a task without an answer node")
    metrics = {}
    for name in ("answer_correct", "answer_token_f1", "evidence_coverage", "answer_hedged"):
        metrics[name] = {**rec_store.metric(all_rec, name), **cmp_store.metric(cmp_ids, name)}
    drafts = {**rec_store.final_drafts(all_rec), **cmp_store.final_drafts(cmp_ids)}

    per_run: dict[str, dict[str, Any]] = {}
    for r in rec_runs + cmp_runs:
        rid, task = r["run_id"], r["task_id"]
        g = graphs[task]
        boolean = is_boolean_answer(g.answer)
        in_draft = None if boolean else contains_answer(drafts[rid], g.answer, g.gold_aliases)
        st = cumulative_stages(bool(cov[rid]), in_draft, metrics["answer_correct"][rid])
        per_run[rid] = {
            "task": task,
            "arm": "recipe" if rid in tseed else "comparator",
            "tseed": tseed.get(rid),
            "rseed": int(r["seed"]),
            "covered": bool(cov[rid]),
            "in_draft": in_draft,
            "stage1": st[0],
            "stage2": st[1],
            "stage3": st[2],
            "correct": metrics["answer_correct"][rid],
            "f1": metrics["answer_token_f1"][rid],
            "coverage": metrics["evidence_coverage"][rid],
            "hedged": metrics["answer_hedged"][rid],
            "n_asks": int(r["n_asks"]),
        }

    def arm_means(arm: str, field: str) -> dict[str, float]:
        rows = [(v["task"], v[field]) for v in per_run.values() if v["arm"] == arm]
        return task_means(rows, expect_per_task=4 if arm == "recipe" else 2)

    readings: dict[str, Any] = {}
    fields = {
        "p_answer_node_covered": "stage1",
        "p_covered_and_in_draft": "stage2",
        "p_covered_in_draft_correct": "stage3",
        "p_answer_in_draft": "in_draft",
        "p_answer_correct": "correct",
        "answer_token_f1": "f1",
        "p_answer_hedged": "hedged",
    }
    for label, field in fields.items():
        readings[label] = paired_reading(
            arm_means("recipe", field), arm_means("comparator", field), clusters=clusters
        )
    # own-stop evidence coverage, both with the population's clusters and as cap8_by_seed read it
    cov_a, cov_b = arm_means("recipe", "coverage"), arm_means("comparator", "coverage")
    readings["evidence_coverage_own_stop"] = paired_reading(cov_a, cov_b, clusters=clusters)
    readings["evidence_coverage_own_stop_unclustered"] = paired_reading(cov_a, cov_b, clusters=None)

    # run-level descriptives: the 2x2 and the conditional rates
    def runs_of(arm: str) -> list[dict[str, Any]]:
        return [v for v in per_run.values() if v["arm"] == arm]

    def cond(rows: Sequence[Mapping[str, Any]], field: str, covered: bool) -> float:
        xs = [v[field] for v in rows if v["covered"] == covered]
        return sum(xs) / len(xs) if xs else float("nan")

    two_by_two: dict[str, Any] = {}
    conditional: dict[str, Any] = {}
    for arm in ("recipe", "comparator"):
        rs = runs_of(arm)
        cells = {
            f"covered={int(c)},correct={int(k)}": sum(
                1 for v in rs if v["covered"] == c and bool(v["correct"]) == k
            )
            for c in (True, False)
            for k in (True, False)
        }
        two_by_two[arm] = {"n_runs": len(rs), **cells}
        nb = [v for v in rs if v["in_draft"] is not None]
        conditional[arm] = {
            "n_runs": len(rs),
            "n_covered": sum(v["covered"] for v in rs),
            "f1_given_covered": cond(rs, "f1", True),
            "f1_given_uncovered": cond(rs, "f1", False),
            "correct_given_covered": cond(rs, "correct", True),
            "correct_given_uncovered": cond(rs, "correct", False),
            "n_runs_with_draft_stage": len(nb),
            "in_draft_given_covered": cond(nb, "in_draft", True),
            "in_draft_given_uncovered": cond(nb, "in_draft", False),
            "correct_given_covered_and_in_draft": _mean_or_nan(
                [v["correct"] for v in nb if v["covered"] and v["in_draft"]]
            ),
            "correct_given_covered_not_in_draft": _mean_or_nan(
                [v["correct"] for v in nb if v["covered"] and not v["in_draft"]]
            ),
            "hedged_given_covered": cond(rs, "hedged", True),
            "hedged_given_uncovered": cond(rs, "hedged", False),
        }

    # principal strata over (training seed, task, rollout seed) pairs
    cmp_by_key = {(v["task"], v["rseed"]): v for v in runs_of("comparator")}
    strata: dict[str, dict[str, Any]] = {}
    for v in runs_of("recipe"):
        c = cmp_by_key[(v["task"], v["rseed"])]
        s = strata.setdefault(stratum(v["covered"], c["covered"]), {"pairs": [], "tasks": set()})
        s["pairs"].append((v, c))
        s["tasks"].add(v["task"])
    strata_out = {}
    for name in ("both", "only_recipe", "only_comparator", "neither"):
        ps = strata.get(name, {"pairs": [], "tasks": set()})
        n = len(ps["pairs"])
        strata_out[name] = {
            "n_pairs": n,
            "n_tasks": len(ps["tasks"]),
            "correct_recipe": sum(a["correct"] for a, _ in ps["pairs"]) / n if n else float("nan"),
            "correct_comparator": sum(b["correct"] for _, b in ps["pairs"]) / n
            if n
            else float("nan"),
            "f1_recipe": sum(a["f1"] for a, _ in ps["pairs"]) / n if n else float("nan"),
            "f1_comparator": sum(b["f1"] for _, b in ps["pairs"]) / n if n else float("nan"),
        }

    # the primary: implied answer gain from the coverage gain, beside observed and MDE
    d_cov = readings["p_answer_node_covered"]["point"]
    cc = conditional["comparator"]
    primary = {
        "d_p_answer_node_covered": d_cov,
        "implied_d_f1": implied_gain(d_cov, cc["f1_given_covered"], cc["f1_given_uncovered"]),
        "implied_d_correct": implied_gain(
            d_cov, cc["correct_given_covered"], cc["correct_given_uncovered"]
        ),
        "observed_d_f1": {
            k: readings["answer_token_f1"][k] for k in ("point", "ci_lo", "ci_hi", "verdict")
        },
        "observed_d_correct": {
            k: readings["p_answer_correct"][k] for k in ("point", "ci_lo", "ci_hi", "verdict")
        },
        "mde_d_f1": readings["answer_token_f1"]["mde_80_05"],
        "mde_d_correct": readings["p_answer_correct"]["mde_80_05"],
    }
    for m, lab in (("f1", "answer_token_f1"), ("correct", "p_answer_correct")):
        imp = primary[f"implied_d_{m}"]
        primary[f"implied_{m}_inside_observed_interval"] = {
            f"{c['n_boot']}/{c['seed']}": c["ci_lo"] <= imp <= c["ci_hi"]
            for c in readings[lab]["resamples"]
        }
    primary["implied_f1_below_mde"] = abs(primary["implied_d_f1"]) < primary["mde_d_f1"]
    primary["implied_correct_below_mde"] = (
        abs(primary["implied_d_correct"]) < primary["mde_d_correct"]
    )

    # LOCK A2(a): per-arm levels equal the stored metric means
    a2a = {}
    for arm, ids, store in (("recipe", all_rec, rec_store), ("comparator", cmp_ids, cmp_store)):
        for name, field in (("answer_correct", "correct"), ("answer_token_f1", "f1")):
            tm = arm_means(arm, field)
            mine = sum(tm.values()) / len(tm)
            stored = store.stored_mean(ids, name)
            a2a[f"{arm}::{name}"] = {
                "task_mean_level": mine,
                "stored_mean": stored,
                "ok": abs(mine - stored) <= LOCK_TOL,
            }
            if abs(mine - stored) > LOCK_TOL:
                raise SystemExit(f"LOCK A2a FAILED {suite} {arm} {name}: {mine} vs {stored}")

    n_bool = sum(1 for t in tasks if is_boolean_answer(graphs[t].answer))
    return {
        "population": {
            "recipe_lists": {
                s: {
                    "path": f"artifacts/seedrep_gate_20260919/run_ids/run_ids.{m}.{suite}.txt",
                    "sha256": file_sha(SEEDREP / "run_ids" / f"run_ids.{m}.{suite}.txt"),
                    "n": len(rec_ids[s]),
                }
                for s, m in RECIPE.items()
            },
            "comparator_list": {
                "path": f"artifacts/completed_cohort_20260922/cohort/run_ids.prompted.{suite}.txt",
                "sha256": file_sha(cmp_list),
                "n": len(cmp_ids),
            },
            "n_tasks": len(tasks),
            "clusters": "template_id" if clusters else "task",
            "n_clusters": len(set(clusters.values())) if clusters else len(tasks),
            "corpus_dir": corpus_dir,
            "code_versions": sorted({r["code_version"] for r in rec_runs + cmp_runs}),
            "n_tasks_boolean_answer": n_bool,
            "draft_stage_population": f"{len(tasks) - n_bool} non-boolean-answer tasks",
            "answer_node_rule_tally": _tally(results),
        },
        "lock_a2a": a2a,
        "readings": readings,
        "two_by_two_runs": two_by_two,
        "conditional_runs": conditional,
        "principal_strata_pairs": strata_out,
        "primary": primary,
        "rendering": rendering_stats(suite, graphs, results, corpus_dir),
    }, results


def _tally(results: Mapping[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results.values():
        out[r.rule] = out.get(r.rule, 0) + 1
    return out


def coverage_records() -> dict[str, Any]:
    cap8 = json.loads((SEED_ID / "cap8_by_seed.json").read_text())["cells"]["s1s2"]
    t1 = json.loads((SEED_ID / "table1_by_seed.json").read_text())["cells"]["s1s2"]
    out = {}
    for suite in SUITES:
        eq = next(c for c in t1[f"evidence_coverage::{suite}"] if c["n_boot"] == 10000)
        out[suite] = {
            "own_stop_cap8": cap8[suite],
            "equal_spend_symmetric": eq,
        }
    return out


def _fmt(x: Any) -> str:
    return (
        "nan"
        if isinstance(x, float) and math.isnan(x)
        else (f"{x:+.6f}" if isinstance(x, float) else str(x))
    )


def markdown(result: Mapping[str, Any]) -> str:
    lines = []
    lines.append(
        "| suite | reading | recipe | comparator | recipe - comparator [10k] | 50k verdict | MDE |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for suite in SUITES:
        rd = result["per_suite"][suite]["readings"]
        for label in (
            "p_answer_node_covered",
            "p_covered_and_in_draft",
            "p_covered_in_draft_correct",
            "p_answer_in_draft",
            "p_answer_correct",
            "answer_token_f1",
            "p_answer_hedged",
            "evidence_coverage_own_stop",
        ):
            c = rd[label]
            lines.append(
                f"| {suite} | {label} (n={c['n_tasks']}) | {c['level_recipe']:.6f} | {c['level_comparator']:.6f} | "
                f"{c['point']:+.6f} [{c['ci_lo']:+.6f}, {c['ci_hi']:+.6f}] | {c['verdict']} | {c['mde_80_05']:.6f} |"
            )
    lines.append("")
    lines.append(
        "| suite | d P(answer node covered) | implied dF1 | observed dF1 [10k] | MDE dF1 | implied d correct | observed d correct [10k] | MDE d correct |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for suite in SUITES:
        p = result["per_suite"][suite]["primary"]
        f, c = p["observed_d_f1"], p["observed_d_correct"]
        lines.append(
            f"| {suite} | {p['d_p_answer_node_covered']:+.6f} | {p['implied_d_f1']:+.6f} | "
            f"{f['point']:+.6f} [{f['ci_lo']:+.6f}, {f['ci_hi']:+.6f}] ({f['verdict']}) | {p['mde_d_f1']:.6f} | "
            f"{p['implied_d_correct']:+.6f} | {c['point']:+.6f} [{c['ci_lo']:+.6f}, {c['ci_hi']:+.6f}] ({c['verdict']}) | {p['mde_d_correct']:.6f} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    result: dict[str, Any] = {
        "provenance": {
            "scorer_hash": SCORER_HASH,
            "graph_version": GRAPH_VERSION,
            "declaration": "artifacts/answer_loss_20260923/DECLARATION.md (c2bdf77)",
            "command": "PI_GOLD_ROOT=$PWD/data/gold .venv/bin/python -m scripts.answer_loss.decompose --out "
            + str(args.out),
            "window_chars": UNIT_CHARS,
            "resamples": [list(r) for r in RESAMPLES],
            "intervals": "pi_eval.stats.inference.paired_difference, recipe task mean (s1+s2, rollout seeds 0/1) minus comparator task mean (rollout seeds 0/1)",
        },
        "per_suite": {},
    }
    results_by_suite = {}
    for suite in SUITES:
        print(f"[{suite}] reading ...", flush=True)
        per, results = read_suite(suite)
        result["per_suite"][suite] = per
        results_by_suite[suite] = results
    # locks on the published records (A1 needs the answer nodes built above; they are a function
    # of the task alone, and the record's tasks are a subset of these)
    print("lock A1 ...", flush=True)
    result["lock_a1"] = lock_a1(results_by_suite)
    print("lock A2b ...", flush=True)
    result["lock_a2b"] = lock_a2b()
    result["coverage_gain_records"] = coverage_records()
    for suite in SUITES:
        mine = result["per_suite"][suite]["readings"]["evidence_coverage_own_stop_unclustered"][
            "point"
        ]
        rec = result["coverage_gain_records"][suite]["own_stop_cap8"]["delta"]
        result["coverage_gain_records"][suite]["own_stop_reproduced_here"] = mine
        result["coverage_gain_records"][suite]["own_stop_ok"] = abs(mine - rec) <= LOCK_TOL
        if abs(mine - rec) > LOCK_TOL:
            raise SystemExit(
                f"own-stop coverage does not reproduce cap8_by_seed s1s2 {suite}: {mine} vs {rec}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True, default=_json_default) + "\n")
    print(markdown(result))
    print(f"wrote {args.out}")
    return 0


def _json_default(o: Any) -> Any:
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {o!r}")


if __name__ == "__main__":
    sys.exit(main())
