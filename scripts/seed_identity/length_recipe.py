"""The dev length control and the length-budget coverage contrast, per training seed of the recipe.

WHY. The section "The gain is not bought with words" reports two development-split readings, both
taken on seed 0 of the selected recipe (`qwen3-8b-dpo-stacked-notdone-both`), whose final training
stage resumed and kept about 9% of its steps (artifacts/seed_identity_20260923/RESULT.md). This
reads the same two things for the recipe's fresh seeds 1 and 2, and for the two pooled.

(a) QUESTION LENGTH. `pinq_train.gate._length`'s bootstrap and one-sided `not_longer` rule, with
    the margin, resample count and seed taken from the gate verdicts, over the questions
    `gate._questions` returns (turns.parquet, action_kind 'ask', question non-null and non-empty),
    in words (len(q.split())) and in Qwen3-8B tokens (len(tok.encode(q, add_special_tokens=False))).
    The trained arm's questions against the prompted comparator's, both of its rollout seeds.
(b) THE LENGTH BUDGET. `scripts/matched_cost.py`'s own `contrast`, the comparator charged the
    trained run's question-token spend (primary) or question-word spend (robustness): it buys
    whole questions in the order it asked them until the next would overrun the allowance, and its
    coverage is read there. Pairs are (suite, task, rollout seed), rollout seeds averaged into the
    task, BCa over tasks (`paired_difference`, 1,000 resamples, 1,000 permutations, seed 0). The
    coverage ladder is the store's own `frontier_q#k`, checked to end at each run's stored
    `evidence_coverage`; no gold is read.

TWO ESTIMATOR ERAS, BOTH REPORTED. Every published seed-0 interval here predates a sort:
  * (a) was taken 2026-09-17 in `_questions`' arrival order; `_length` has since sorted its units
    (2026-09-18). `ci_lo`/`ci_hi` are the current gate's; `arrival_order` holds the published era's.
  * (b) was taken 2026-09-17 19:50, before commit 4b7b24b made `cluster_bootstrap` sort its units
    by value (units used to arrive in `paired_difference`'s sorted-KEY order). `ci_lo`/`ci_hi` are
    the current code's; `ci_*_pre_4b7b24b` are `4b7b24b^:src/pi_eval/stats/inference.py`'s, loaded
    from git history, which is the estimator that produced the printed seed-0 intervals.
Point estimates are the same in both eras by construction and are asserted so.

POOLED s1+s2. (b): each seed's pairs are formed exactly as for that seed alone and then averaged
within the task before the bootstrap, as scripts/seed_identity/table1_by_seed.py pools. (a):
`_length` has no task level (its unit is one question), so the pooled reading is the same
estimator over the union of both seeds' questions; each seed ran every task once, so every task
enters with both seeds' questions.

LOCK FIRST. Nothing about seeds 1 and 2 is written unless seed 0 reproduces:
  * artifacts/length_tokens/cells.json (arm A) exactly, in arrival order: both means in both units,
    both question counts, both intervals and both verdicts; the sorted-order MuSiQue token interval
    paper/appendix_matched_cost.tex records; the 4-decimal values of that record's RESULT.md;
  * artifacts/length_control_tokens/cells.json (selected arm) at its recorded precision, for both
    units and for matched_k: delta, p, n, base_short, allowance, spend, both question counts, and
    the interval under the pre-4b7b24b estimator; the values paper/appendix_matched_cost.tex
    prints; and the current-code intervals that file's 2026-09-20 comment records.
Also, for EVERY arm (instrument checks): the seed-0 and comparator run-id sets equal the published
lists; `gate._length` itself, on the same runs, equals this file's copy of its arithmetic bit for
bit, and that equals the dev_gate verdict's `length_equivalence`; matched_k's delta and current
interval equal the verdict's matched-cost cell; and the pooled reader reproduces `contrast` for
one seed and for two. On any failure the lock report alone is written and the exit code is 1.

    PYTHONPATH=src .venv/bin/python scripts/seed_identity/length_recipe.py \
        --out artifacts/seed_identity_20260923/length_recipe.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Callable

os.environ["HF_HUB_OFFLINE"] = "1"  # the tokenizer is read from the local hub cache or not at all
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import duckdb  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import matched_cost as mc  # noqa: E402  (the length-budget instrument, reused rather than copied)

from pinq_train import gate  # noqa: E402

GATE = REPO / "artifacts/seed_identity_20260923/dev_gate"
STORE = GATE / "store"
SUITES = ("musique", "strategyqa")
BASE_MODEL = "qwen3-8b-base"
ARMS = {
    "s0": "qwen3-8b-dpo-stacked-notdone-both",
    "s1": "qwen3-8b-dpo-stacked-notdone-both-s1",
    "s2": "qwen3-8b-dpo-stacked-notdone-both-s2",
}
TOKENIZER = "Qwen/Qwen3-8B"
BUDGETS = ("matched_question_tokens", "matched_question_words")
BASES = (*BUDGETS, "matched_k")
LEN_PUB = REPO / "artifacts/length_tokens"
BUDGET_PUB = REPO / "artifacts/length_control_tokens"
PRE_FIX = "4b7b24b^"  # parent of "a BCa endpoint depended on the order its units arrived in"
INFERENCE = "src/pi_eval/stats/inference.py"
RND6, RND4 = 5e-7 + 1e-12, 5e-5 + 1e-12  # half a unit in a recorded 6th / 4th decimal

# paper/appendix_matched_cost.tex, comment "CORRECTED 2026-09-18, commit 513c876": seed 0's MuSiQue
# token interval recomputed with the value sort `_length` now applies, 1,000 resamples, seed 0.
SORTED_TOKEN_CI_MUSIQUE = (-1.9378798763460847e-05, 0.08111012458424939)
# artifacts/length_tokens/RESULT.md "The one table", arm A, 4 decimals, and the question count.
PRINTED_LENGTH = {
    "musique": {"words": (13.2315, 13.1000), "tokens": (16.6512, 16.0194), "n": 324},
    "strategyqa": {"words": (18.0690, 12.9930), "tokens": (23.0736, 15.6372), "n": 435},
}
# paper/appendix_matched_cost.tex tab:app-lengthbudget as printed (and sec:length of the ICLR
# draft): allowance, spend, comparator questions, delta, 95% interval, with the printed precision.
PRINTED_BUDGET = {
    ("musique", "matched_question_tokens"): (40.9, 32.6, 2.386, 0.1799, 0.1324, 0.2274),
    ("musique", "matched_question_words"): (32.5, 25.6, 2.341, 0.1875, 0.1376, 0.2384),
    ("strategyqa", "matched_question_tokens"): (30.1, 22.4, 1.778, 0.0669, 0.0407, 0.0946),
    ("strategyqa", "matched_question_words"): (23.6, 17.5, 1.724, 0.0772, 0.0493, 0.1056),
}
PRINTED_TOL = (0.05 + 1e-12, 0.05 + 1e-12, 5e-4 + 1e-12, RND4, RND4, RND4)
# paper/appendix_matched_cost.tex, comment "CORRECTED 2026-09-20", the intervals it lists as the
# replaced "old" set; these are what the current `paired_difference` returns.
COMMENT_2026_09_20 = {
    ("musique", "matched_question_tokens"): (0.1390, 0.2313),
    ("musique", "matched_question_words"): (0.1420, 0.2425),
    ("strategyqa", "matched_question_tokens"): (0.0398, 0.0970),
    ("strategyqa", "matched_question_words"): (0.0469, 0.1092),
}


def _ids(p: Path) -> list[str]:
    return [
        ln.split()[0] for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")
    ]


def _digest(ids: list[str]) -> str:
    return hashlib.sha256("".join(f"{r}\n" for r in sorted(ids)).encode()).hexdigest()


class Lock:
    """Every published value against its reproduction, with the tolerance its source's recorded
    precision allows (0 for exact records, half a unit in the last digit for rounded ones)."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, name: str, published: Any, got: Any, tol: float = 0.0) -> None:
        if isinstance(published, (bool, str)) or isinstance(got, (bool, str)):
            ok, diff = published == got, None
        else:
            diff = abs(float(got) - float(published))
            ok = diff <= tol
        self.rows.append(
            {"name": name, "published": published, "reproduced": got, "abs_diff": diff, "tol": tol}
            | {"ok": ok}
        )

    @property
    def ok(self) -> bool:
        return all(r["ok"] for r in self.rows)


# --------------------------------------------------------------------------- inputs


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for t in ("runs", "turns", "scores", "calls"):
        con.execute(
            f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{(STORE / t).as_posix()}.parquet')"
        )
    return con


def populations(con) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Runs by (suite, Inquirer model), the model read off calls.parquet actor='inquirer'. arm_id
    `inquirer_trained` spans three weights in this store, so it selects nothing on its own."""
    rows = con.execute(
        "WITH m AS (SELECT run_id, list(DISTINCT model) AS models FROM calls "
        "WHERE actor = 'inquirer' GROUP BY 1) "
        "SELECT r.run_id, r.suite_id, r.task_id, r.seed, r.split, r.status, r.code_version, "
        "m.models FROM runs r LEFT JOIN m USING (run_id) ORDER BY r.run_id"
    ).fetchall()
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run_id, suite, task, seed, split, status, cv, models in rows:
        if not models or len(models) != 1 or split != "dev" or status != "ok":
            raise SystemExit(f"{run_id}: models={models} split={split} status={status}")
        out.setdefault((suite, models[0]), []).append(
            {"run_id": run_id, "task_id": task, "seed": int(seed), "code_version": cv}
        )
    return out


def tokenizer_counts(con) -> tuple[dict[str, int], dict[str, Any]]:
    """Qwen3-8B token count of every distinct question string in the store, by `transformers`
    (the length control's own call), cross-checked against `tokenizers.Tokenizer.from_file` on
    the same tokenizer.json (matched_cost.py's call). One disagreement refuses."""
    from huggingface_hub import try_to_load_from_cache
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    path = try_to_load_from_cache(TOKENIZER, "tokenizer.json")
    mc_path = mc.find_question_tokenizer()
    if not isinstance(path, str) or mc_path is None or Path(path).resolve() != mc_path.resolve():
        raise SystemExit(f"tokenizer.json resolves two ways: {path!r} and {mc_path!r}")
    raw = Tokenizer.from_file(str(mc_path))
    qs = [
        str(q)
        for (q,) in con.execute("SELECT DISTINCT coalesce(question, '') FROM turns").fetchall()
    ]
    counts = {q: len(tok.encode(q, add_special_tokens=False)) for q in qs}
    bad = [q for q in qs if len(raw.encode(q, add_special_tokens=False).ids) != counts[q]]
    if bad:
        raise SystemExit(f"transformers and tokenizers disagree on {len(bad)} questions: {bad[:3]}")
    return counts, {
        "name": TOKENIZER,
        "tokenizer_json_sha256": mc.sha256_file(mc_path),
        "source": "local HF hub cache, HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
        "call": "len(AutoTokenizer.encode(q, add_special_tokens=False))",
        "cross_check": f"tokenizers.Tokenizer.from_file agrees on all {len(qs)} distinct questions",
    }


def pre_fix_paired_difference() -> tuple[Callable[..., Any], str]:
    """`paired_difference` as it stood before 4b7b24b, from git history, and its source sha256."""
    src = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{PRE_FIX}:{INFERENCE}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    mod = types.ModuleType("inference_pre_4b7b24b")
    sys.modules[mod.__name__] = mod  # @dataclass resolves its module through sys.modules
    exec(compile(src, f"{PRE_FIX}:{INFERENCE}", "exec"), mod.__dict__)
    return mod.paired_difference, hashlib.sha256(src.encode()).hexdigest()


# --------------------------------------------------------------------------- (a) length


def length_reading(
    ck: list[float], ba: list[float], *, margin: float, seed: int, n_resamples: int, sort: bool
) -> dict[str, Any]:
    """`gate._length`'s arithmetic line for line, over any per-question scalar. `sort=False` is
    the arrival-order variant `_length` ran before it sorted its units (2026-09-18)."""
    m_ck, m_ba = gate._mean(ck), gate._mean(ba)
    if sort:
        ck, ba = sorted(ck), sorted(ba)
    rng = random.Random(seed)
    rels = []
    for _ in range(n_resamples):
        a = gate._mean([ck[rng.randrange(len(ck))] for _ in range(len(ck))])
        b = gate._mean([ba[rng.randrange(len(ba))] for _ in range(len(ba))])
        rels.append((a - b) / b if b else float("nan"))
    rels.sort()
    lo = rels[max(0, int(round(0.05 * (len(rels) - 1))))]
    hi = rels[min(len(rels) - 1, int(round(0.95 * (len(rels) - 1))))]
    return {
        "value": m_ck,
        "baseline": m_ba,
        "rel": (m_ck - m_ba) / m_ba,
        "n": len(ck),
        "n_baseline": len(ba),
        "ci_lo": lo,
        "ci_hi": hi,
        "passed_not_longer": bool(hi <= margin),
        "tost_equivalent": bool(lo >= -margin and hi <= margin),
    }


# --------------------------------------------------------------------------- (b) length budget


def ladders(con, runs: list[dict[str, Any]], tok: dict[str, int]) -> list[mc.Ladder]:
    """`matched_cost.Ladder`s whose coverage rungs are the store's `frontier_q#k`, k = 0..n_asks,
    refused unless every rung is present and the last equals the run's `evidence_coverage`. The
    question curves are matched_cost's own rule: every turn of the run, in turn order."""
    ids = [r["run_id"] for r in runs]
    meta = con.execute(
        "SELECT run_id, suite_id, task_id, template_id, arm_id, seed, n_asks, stop_reason "
        "FROM runs WHERE run_id IN (SELECT unnest(?)) ORDER BY run_id",
        [ids],
    ).fetchall()
    stored: dict[str, dict[str, float]] = {}
    for rid, name, val in con.execute(
        "SELECT run_id, metric_name, value FROM scores WHERE run_id IN (SELECT unnest(?)) "
        "AND (metric_name = 'evidence_coverage' OR metric_name LIKE 'frontier_q#%')",
        [ids],
    ).fetchall():
        stored.setdefault(rid, {})[name] = float(val)
    qs: dict[str, list[str]] = {}
    for rid, _idx, q in con.execute(
        "SELECT run_id, turn_idx, question FROM turns WHERE run_id IN (SELECT unnest(?)) "
        "ORDER BY run_id, turn_idx",
        [ids],
    ).fetchall():
        qs.setdefault(rid, []).append(str(q or ""))
    out = []
    for rid, suite, task, template, arm, seed, n_asks, stop in meta:
        q, s = qs.get(rid, []), stored.get(rid, {})
        if len(q) != n_asks:
            raise SystemExit(f"{rid}: {len(q)} turn rows against n_asks={n_asks}")
        try:
            cov = tuple(s[f"frontier_q#{k}"] for k in range(n_asks + 1))
        except KeyError as e:
            raise SystemExit(f"{rid}: no stored {e.args[0]}") from None
        if abs(cov[-1] - s["evidence_coverage"]) > 1e-12:
            raise SystemExit(f"{rid}: frontier_q#{n_asks} {cov[-1]} != evidence_coverage")
        out.append(
            mc.Ladder(
                run_id=rid,
                suite_id=suite,
                task_id=task,
                cluster_id=str(template or task),
                arm_id=arm,
                seed=int(seed),
                n_asks=int(n_asks),
                stop_reason=str(stop or ""),
                cov=cov,
                cad2_hit=(0,) * (n_asks + 1),
                cad2_n=0,
                cad_hit={},
                cad_n={},
                q_words=tuple(len(x.split()) for x in q),
                q_tok=tuple(tok[x] for x in q),
            )
        )
    return out


@contextlib.contextmanager
def estimator(fn: Callable[..., Any]):
    """Run `matched_cost` with `fn` as its `paired_difference`, restoring the current one after."""
    was = mc.paired_difference
    mc.paired_difference = fn
    try:
        yield
    finally:
        mc.paired_difference = was


def contrast(lads: list[mc.Ladder], base: dict, suite: str, basis: str) -> dict[str, Any]:
    c = mc.contrast(
        lads,
        base,
        suite_id=suite,
        checkpoint="",
        metric="evidence_coverage",
        comparator=basis,
        offset=0,
    )
    return {
        "delta": c.delta,
        "ci_lo": c.ci_lo,
        "ci_hi": c.ci_hi,
        "p_value": c.p_value,
        "n_tasks": c.n_tasks,
        "allowance": c.trained_cost,
        "spend": c.base_cost,
        "comparator_questions": c.base_asks,
        "trained_questions": c.trained_asks,
        "trained_coverage": c.trained_mean,
        "comparator_coverage": c.base_mean,
        "n_base_short": c.n_base_short,
    }


def pooled(trained_sets: list[list[mc.Ladder]], base: dict, basis: str) -> dict[str, Any]:
    """Several TRAINING seeds against one comparator, averaged WITHIN the task. Each seed's pairs
    are `matched_cost.contrast`'s pairs exactly (key (suite, task, rollout seed), the comparator
    charged that seed's own spend); every pair of a task then enters one per-task mean, and the
    bootstrap is over tasks, so the interval reflects task sampling only."""
    acc: dict[str, list[dict[str, float]]] = {}
    n_short = 0
    budget = basis in mc.COST_BASES
    for lads in trained_sets:
        for lad in lads:
            peer = base.get((lad.suite_id, lad.task_id, lad.seed))
            if peer is None:
                continue
            if budget:
                spend = lad.cost_at(lad.n_asks, basis)
                k = peer.k_within(spend, basis)
            else:
                spend, k = float("nan"), lad.n_asks
            va, vb = lad.coverage_at(lad.n_asks), peer.coverage_at(k)
            if math.isnan(va) or math.isnan(vb):
                continue
            acc.setdefault(lad.key, []).append(
                {
                    "va": va,
                    "vb": vb,
                    "trained_asks": float(lad.n_asks),
                    "base_asks": float(peer.asks_at(k)),
                    "trained_cost": spend,
                    "base_cost": peer.cost_at(k, basis) if budget else float("nan"),
                }
            )
            if budget:
                n_short += int(spend > peer.cost_at(peer.n_asks, basis) + 1e-9)
            else:
                n_short += int(peer.n_asks < k)
    per = {t: {f: mc._fmean(r[f] for r in rs) for f in mc.PAIR_FIELDS} for t, rs in acc.items()}
    est = mc.paired_difference(
        {t: r["va"] for t, r in per.items()},
        {t: r["vb"] for t, r in per.items()},
        clusters=None,
        n_boot=mc.N_BOOT,
        n_perm=mc.N_PERM,
        seed=mc.SEED,
    )

    def avg(f: str) -> float:
        return mc._fmean(r[f] for r in per.values())

    return {
        "delta": est.point,
        "ci_lo": est.ci_lo,
        "ci_hi": est.ci_hi,
        "p_value": est.p_value,
        "n_tasks": est.n,
        "allowance": avg("trained_cost"),
        "spend": avg("base_cost"),
        "comparator_questions": avg("base_asks"),
        "trained_questions": avg("trained_asks"),
        "trained_coverage": avg("va"),
        "comparator_coverage": avg("vb"),
        "n_base_short": n_short,
        "n_pairs": sum(len(v) for v in acc.values()),
    }


def both_eras(read: Callable[[], dict[str, Any]], pre: Callable[..., Any]) -> dict[str, Any]:
    """One reading under the current estimator, its interval also under the pre-4b7b24b one."""
    now = read()
    with estimator(pre):
        old = read()
    for f in ("delta", "p_value", "n_tasks", "allowance", "spend", "comparator_questions"):
        a, b = now[f], old[f]
        if not ((isinstance(a, float) and math.isnan(a) and math.isnan(b)) or a == b):
            raise SystemExit(f"{f} moved between estimator eras: {a!r} vs {b!r}")
    return now | {"ci_lo_pre_4b7b24b": old["ci_lo"], "ci_hi_pre_4b7b24b": old["ci_hi"]}


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    con = connect()
    pops = populations(con)
    tok, tok_meta = tokenizer_counts(con)
    pre, pre_sha = pre_fix_paired_difference()
    lock = Lock()
    len_pub = {
        c["suite"]: c for c in json.loads((LEN_PUB / "cells.json").read_text()) if c["arm"] == "A"
    }
    bud_pub = json.loads((BUDGET_PUB / "cells.json").read_text())[
        "marker_1_2_selected_arm_primary_table"
    ]
    (scorer_hash,) = {
        h for (h,) in con.execute("SELECT DISTINCT scorer_hash FROM scores").fetchall()
    }
    (graph_version,) = {
        g for (g,) in con.execute("SELECT DISTINCT graph_version FROM scores").fetchall()
    }
    result: dict[str, Any] = {
        "what": __doc__.split("\n", 1)[0],
        "store": STORE.relative_to(REPO).as_posix(),
        "scorer_hash": scorer_hash,
        "graph_version": graph_version,
        "split": "dev",
        "tool": "scripts/seed_identity/length_recipe.py",
        "git_head": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "tokenizer": tok_meta,
        "estimators": {
            "length": "pinq_train.gate._length (value-sorted units); `arrival_order` = the "
            "pre-sort variant the published seed-0 intervals were taken with",
            "length_budget": "scripts/matched_cost.py contrast + current "
            "pi_eval.stats.inference.paired_difference; `*_pre_4b7b24b` = "
            f"{PRE_FIX}:{INFERENCE} (sha256 {pre_sha}), the estimator of the printed seed-0 "
            "intervals",
        },
        "populations": {},
        "lock": lock.rows,
    }
    length: dict[str, dict[str, Any]] = {}
    budget: dict[str, dict[str, Any]] = {}

    for suite in SUITES:
        base_runs = pops[(suite, BASE_MODEL)]
        arm_runs = {a: pops[(suite, m)] for a, m in ARMS.items()}
        arm_runs["s1s2"] = arm_runs["s1"] + arm_runs["s2"]
        for tag, runs, lists in (
            ("s0", arm_runs["s0"], ("run_ids.A.checkpoint", "run_ids.selected")),
            ("comparator", base_runs, ("run_ids.A.baseline", "run_ids.baseline")),
        ):
            for root, stem in zip((LEN_PUB, BUDGET_PUB), lists):
                p = root / f"{stem}.{suite}.txt"
                ids = _ids(p)
                lock.rows.append(
                    {
                        "name": f"run-id set {tag} {suite} == {p.relative_to(REPO).as_posix()}",
                        "published": _digest(ids),
                        "reproduced": _digest([r["run_id"] for r in runs]),
                        "ok": sorted(ids) == sorted(r["run_id"] for r in runs),
                    }
                )
        s0_tasks = {r["task_id"] for r in arm_runs["s0"]}
        for tag, runs in [("comparator", base_runs), *arm_runs.items()]:
            tasks = {r["task_id"] for r in runs}
            result["populations"].setdefault(tag, {})[suite] = {
                "model": BASE_MODEL if tag == "comparator" else ARMS.get(tag, "s1 + s2"),
                "n_runs": len(runs),
                "n_tasks": len(tasks),
                "rollout_seeds": sorted({r["seed"] for r in runs}),
                "code_version": sorted({r["code_version"] for r in runs}),
                "run_id_sha256": _digest([r["run_id"] for r in runs]),
                "tasks_equal_seed0": tasks == s0_tasks,
            }

        # ---- (a) length, gate._length's rule, parameters from the verdicts
        verdicts = {
            a: json.loads((GATE / f"{m}.{suite}.json").read_text()) for a, m in ARMS.items()
        }
        params = {
            (
                v["criteria"]["length_equivalence"]["threshold"],
                v["bootstrap"]["seed"],
                v["bootstrap"]["n_resamples"],
                v["length_rule"],
            )
            for v in verdicts.values()
        }
        if len(params) != 1:
            raise SystemExit(f"{suite}: the verdicts disagree on the length parameters {params}")
        ((margin, bseed, n_res, rule),) = params
        if rule != "not_longer":
            raise SystemExit(f"{suite}: verdict length_rule {rule!r}")
        kw = {"margin": margin, "seed": bseed, "n_resamples": n_res}
        qa_ba = gate._questions(con, base_runs)
        units_ba = {
            "words": [len(q.split()) for _, q in qa_ba],
            "tokens": [tok[q] for _, q in qa_ba],
        }
        for arm, runs in arm_runs.items():
            qa = gate._questions(con, runs)
            units = {"words": [len(q.split()) for _, q in qa], "tokens": [tok[q] for _, q in qa]}
            cell: dict[str, Any] = {}
            for u in ("words", "tokens"):
                cell[u] = length_reading(units[u], units_ba[u], sort=True, **kw)
                cell[u]["arrival_order"] = length_reading(units[u], units_ba[u], sort=False, **kw)
            cell["tokens_per_word"] = sum(units["tokens"]) / sum(units["words"])
            cell["comparator_tokens_per_word"] = sum(units_ba["tokens"]) / sum(units_ba["words"])
            cell["params"] = kw | {"rule": rule}
            length.setdefault(arm, {})[suite] = cell
            ref = gate._length(con, runs, base_runs, rule=rule, **kw)
            for f, g in (
                ("value", "value"),
                ("baseline", "baseline"),
                ("n", "n"),
                ("ci_lo", "ci_lo"),
                ("ci_hi", "ci_hi"),
                ("passed", "passed_not_longer"),
                ("tost_equivalent", "tost_equivalent"),
            ):
                lock.check(
                    f"instrument {arm} {suite} words {f}: gate._length", ref[f], cell["words"][g]
                )
                if arm in verdicts:
                    lock.check(
                        f"instrument {arm} {suite} words {f}: dev_gate verdict length_equivalence",
                        verdicts[arm]["criteria"]["length_equivalence"][f],
                        cell["words"][g],
                    )
            if arm != "s0":
                continue
            pub = len_pub[suite]
            for u in ("words", "tokens"):
                old = cell[u]["arrival_order"]
                for f in (
                    "value",
                    "baseline",
                    "ci_lo",
                    "ci_hi",
                    "passed_not_longer",
                    "tost_equivalent",
                ):
                    lock.check(
                        f"published s0 {suite} {u} {f} (length_tokens/cells.json, arrival order)",
                        pub[f"{u}_{f}"],
                        old[f],
                    )
                for i, f in enumerate(("value", "baseline")):
                    lock.check(
                        f"published s0 {suite} {u} {f} (length_tokens/RESULT.md, 4 dp)",
                        PRINTED_LENGTH[suite][u][i],
                        cell[u][f],
                        RND4,
                    )
            for f, pf, g in (
                ("n questions trained", "n_questions_checkpoint", "n"),
                ("n questions comparator", "n_questions_baseline", "n_baseline"),
            ):
                lock.check(
                    f"published s0 {suite} {f} (length_tokens/cells.json)",
                    pub[pf],
                    cell["words"][g],
                )
            lock.check(
                f"published s0 {suite} n questions trained (RESULT.md)",
                PRINTED_LENGTH[suite]["n"],
                cell["tokens"]["n"],
            )
            if suite == "musique":
                for i, f in enumerate(("ci_lo", "ci_hi")):
                    lock.check(
                        f"published s0 musique tokens {f} (sorted, appendix_matched_cost.tex)",
                        SORTED_TOKEN_CI_MUSIQUE[i],
                        cell["tokens"][f],
                    )

        # ---- (b) the length budget, matched_cost.contrast
        base_by: dict[tuple[str, str, int], mc.Ladder] = {}
        for lad in ladders(con, base_runs, tok):
            key = (lad.suite_id, lad.task_id, lad.seed)
            if key in base_by:
                raise SystemExit(f"comparator: two runs at {key}")
            base_by[key] = lad
        lads = {a: ladders(con, arm_runs[a], tok) for a in ARMS}
        for arm in ARMS:
            for basis in BASES:
                budget.setdefault(arm, {}).setdefault(suite, {})[basis] = both_eras(
                    lambda arm=arm, basis=basis: contrast(lads[arm], base_by, suite, basis), pre
                )
            mk, vmc = (
                budget[arm][suite]["matched_k"],
                verdicts[arm]["matched_cost"]["by_suite"][suite],
            )
            for f in ("delta", "ci_lo", "ci_hi"):
                lock.check(
                    f"instrument {arm} {suite} matched_k {f}: dev_gate verdict",
                    vmc[f],
                    mk[f],
                    1e-12,
                )
            lock.check(
                f"instrument {arm} {suite} matched_k n: dev_gate verdict",
                vmc["n_tasks"],
                mk["n_tasks"],
            )
        for basis in BASES:
            both = both_eras(
                lambda basis=basis: pooled([lads["s1"], lads["s2"]], base_by, basis), pre
            )
            solo = both_eras(lambda basis=basis: pooled([lads["s1"]], base_by, basis), pre)
            cat = both_eras(
                lambda basis=basis: contrast(lads["s1"] + lads["s2"], base_by, suite, basis), pre
            )
            for f in (
                "delta",
                "ci_lo",
                "ci_hi",
                "ci_lo_pre_4b7b24b",
                "allowance",
                "spend",
                "n_base_short",
            ):
                for tag, want, got in (
                    ("one seed == contrast(s1)", budget["s1"][suite][basis][f], solo[f]),
                    ("two seeds == contrast(s1 + s2 ladders)", cat[f], both[f]),
                ):
                    lock.rows.append(
                        {
                            "name": f"pooled reader {suite} {basis} {f}: {tag}",
                            "ok": (isinstance(want, float) and math.isnan(want) and math.isnan(got))
                            or want == got,
                        }
                    )
            budget.setdefault("s1s2", {}).setdefault(suite, {})[basis] = both
        for basis in BASES:
            got, pub = budget["s0"][suite][basis], bud_pub[suite][basis]
            src = "(length_control_tokens/cells.json)"
            for f, tol in (("delta", RND6), ("p_value", RND4), ("n_base_short", 0.0)):
                lock.check(f"published s0 {suite} {basis} {f} {src}", pub[f], got[f], tol)
            for i, f in enumerate(("ci_lo", "ci_hi")):
                lock.check(
                    f"published s0 {suite} {basis} {f} {src}, pre-4b7b24b estimator",
                    pub["ci"][i],
                    got[f"{f}_pre_4b7b24b"],
                    RND6,
                )
            for f, g in (
                ("base_asks", "comparator_questions"),
                ("trained_asks", "trained_questions"),
                ("base_cost", "spend"),
                ("trained_cost", "allowance"),
            ):
                if f in pub:
                    lock.check(f"published s0 {suite} {basis} {g} {src}", pub[f], got[g], RND4)
            lock.check(
                f"published s0 {suite} {basis} n_tasks {src}",
                bud_pub[suite]["n_tasks"],
                got["n_tasks"],
            )
            if basis not in BUDGETS:
                continue
            fields = ("allowance", "spend", "comparator_questions", "delta", "ci_lo", "ci_hi")
            for i, f in enumerate(fields):
                g = f"{f}_pre_4b7b24b" if f.startswith("ci_") else f
                lock.check(
                    f"published s0 {suite} {basis} {g} (paper, as printed)",
                    PRINTED_BUDGET[(suite, basis)][i],
                    got[g],
                    PRINTED_TOL[i],
                )
            for i, f in enumerate(("ci_lo", "ci_hi")):
                lock.check(
                    f"published s0 {suite} {basis} {f} (appendix comment 2026-09-20, 'old' set)",
                    COMMENT_2026_09_20[(suite, basis)][i],
                    got[f],
                    RND4,
                )

    result["lock_ok"] = lock.ok
    if not lock.ok:
        result["status"] = "LOCK FAILED: no seed-1/2 value written"
        result["failed"] = [r for r in lock.rows if not r["ok"]]
        args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        for r in result["failed"]:
            print(f"LOCK FAILED {r['name']}: {r.get('published')!r} vs {r.get('reproduced')!r}")
        return 1
    result["n_lock_checks"] = len(lock.rows)
    result["length"] = length
    result["length_budget"] = budget
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"lock: {len(lock.rows)} checks, all ok; wrote {args.out}")
    for arm in (*ARMS, "s1s2"):
        for suite in SUITES:
            w, t = length[arm][suite]["words"], length[arm][suite]["tokens"]
            print(
                f"{arm:5} {suite:10} n={w['n']}/{w['n_baseline']} | "
                + " | ".join(
                    f"{u} {c['value']:.4f}/{c['baseline']:.4f} rel {c['rel']:+.4f} "
                    f"[{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}] "
                    f"{'pass' if c['passed_not_longer'] else 'FAIL'}"
                    for u, c in (("words", w), ("tokens", t))
                )
            )
            for basis in BASES:
                b = budget[arm][suite][basis]
                print(
                    f"      {basis:24} allow {b['allowance']:7.2f} spend {b['spend']:7.2f} "
                    f"q {b['comparator_questions']:.3f}/{b['trained_questions']:.3f} "
                    f"{b['delta']:+.4f} [{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}] "
                    f"pre-fix [{b['ci_lo_pre_4b7b24b']:+.4f},{b['ci_hi_pre_4b7b24b']:+.4f}] "
                    f"n={b['n_tasks']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
