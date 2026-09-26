#!/usr/bin/env python
"""Measure sigma_J -- the noise floor of the judge -- by TEST-RETEST, and freeze it.

`pi_eval.prereg` names this script, `SIGMA_J_MIN_PAIRS` sets its minimum design, and
`load_sigma_j` reads the `prereg/sigma_j.json` it writes. Until now it did not exist and
nothing wrote that file, so every judge-derived metric carried a noise floor of +inf and
`pi_eval.report` flagged every effect on it `below_noise_floor`. That is the CORRECT
conservative behaviour with no measurement in hand -- and it means no judged result could ever
be claimed, including the primary endpoint P3.

WHAT sigma_J IS. The standard deviation of a SINGLE judge measurement of one item, estimated
from repeated measurement of the same items. `pi_eval.report` uses it as a dead band: an
effect smaller than the instrument's own noise is not reported as an effect.

TWO RETEST CONDITIONS, BECAUSE THEY MEASURE DIFFERENT NOISE.

  identical    The same item, the same prompt, A DIFFERENT SEED. This is the one that is easy
               to fake: with a deterministic cached judge at temperature 0, re-judging at the
               SAME seed replays the same cached bytes and the difference is exactly 0 -- a
               fabricated zero that would set the dead band to nothing and let every judged
               effect through. Varying the seed changes the request bytes, so the provider is
               genuinely re-queried and what comes back is real nondeterminism.

  paraphrase   A surface-form-preserving rewrite of the report, same seed. Measures how much
               of the verdict is about the content and how much is about the wording. This is
               the larger of the two in every published estimate, and it is the one that
               matters for a comparison between two arms whose answers differ in style.

sigma_J is reported per metric as the MAXIMUM of the two conditions. Taking the smaller, or
their mean, would narrow the dead band using the condition that flatters the instrument.

DIFFERENCES, NOT RAW SDs. For two independent measurements of the same item,
Var(x1 - x2) = 2 * sigma^2, so sigma = SD(differences) / sqrt(2). Using SD(differences)
directly would overstate the floor by 41% and silently suppress real effects.

    python scripts/measure_sigma_j.py --parquet scores/parquet --runs-root runs --n 200
    python scripts/measure_sigma_j.py ... --write        # only then is prereg/sigma_j.json written
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi_eval.prereg import SIGMA_J_FILE, SIGMA_J_MIN_PAIRS  # noqa: E402
from pi_eval.score import BY_NAME  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Pinned, and hashed into the provenance sidecar. A paraphrase prompt that drifted between the
# measurement and the runs it gates would make sigma_J an estimate of a different instrument.
PARAPHRASE_PROMPT = (
    "Rewrite the following report so that every factual claim, number, entity and citation is "
    "preserved exactly, but the wording and sentence order differ. Do not add, remove or "
    "reinterpret any claim. Reply with the rewritten report and nothing else.\n\n{report}"
)
PARAPHRASE_MAX_TOKENS = 1200


def judge_metric_names() -> tuple[str, ...]:
    return tuple(sorted(n for n, d in BY_NAME.items() if d.judge_derived))


# --------------------------------------------------------------------------- the estimate


def sigma_from_pairs(pairs: Sequence[tuple[float, float]]) -> float | None:
    """SD of a SINGLE measurement, from paired repeats.

    Var(x1 - x2) = 2*sigma^2 for independent repeats of the same item, so the /sqrt(2). Pairs
    where either side is NaN are dropped rather than imputed: an unjudged item is missing, and
    imputing it at the mean would shrink the estimated noise toward zero -- in the direction
    that lets more effects through.
    """
    diffs = [a - b for a, b in pairs if a == a and b == b]
    if len(diffs) < 2:
        return None
    return statistics.stdev(diffs) / (2**0.5)


def measure(
    *,
    judge: Any,
    runs: Sequence[Any],
    judge_model: str,
    judge_family: str,
    seed: int = 0,
    retest_seed: int | None = None,
    paraphrase: bool = True,
) -> dict[str, Any]:
    """Judge every run three times: baseline, identical-retest, paraphrase-retest."""
    from dataclasses import replace as _replace

    from pi_eval.judges import harness

    retest_seed = seed + 1 if retest_seed is None else retest_seed
    if retest_seed == seed:
        raise SystemExit(
            "--retest-seed must differ from --seed. At temperature 0 with a content-addressed "
            "cache, re-judging at the same seed replays the same bytes and sigma_identical is "
            "0 by construction -- a fabricated zero that sets the dead band to nothing."
        )

    # PROGRESS, FLUSHED TO STDERR. This script printed only at the very end, so a stall was
    # indistinguishable from slow work: a run on 2026-08-28 sat for 61 minutes with 19.5s of
    # CPU, 0.0% utilisation and two ESTABLISHED-but-dead sockets, and emitted nothing. On a
    # job that bills $5-8 and takes hours, "no output" must not be the normal state.
    _pass = {"n": 0}

    def _run(items, s):
        _pass["n"] += 1
        print(
            f"  [sigma_j] pass {_pass['n']}/3: judging {len(items)} items at seed {s} ...",
            file=sys.stderr,
            flush=True,
        )
        t0 = time.time()
        out = harness.judge_runs(
            judge,
            items,
            judge_model=judge_model,
            judge_family=judge_family,
            arm_pairs=(),  # per-run absolute metrics only; pairwise is not what sigma_J is over
            seed=s,
        )
        print(
            f"  [sigma_j] pass {_pass['n']}/3 done in {time.time() - t0:.0f}s"
            f" ({len(getattr(out, 'judgments', ()) or ())} judgments)",
            file=sys.stderr,
            flush=True,
        )
        return out

    base = _run(runs, seed)
    ident = _run(runs, retest_seed)

    para = None
    n_paraphrased = 0
    if paraphrase:
        rewritten = []
        for r in runs:
            text = _paraphrase(judge, r.report, seed=seed)
            if text:
                n_paraphrased += 1
                rewritten.append(_replace(r, report=text))
        if rewritten:
            para = _run(rewritten, seed)

    out: dict[str, Any] = {"per_metric": {}, "n_runs": len(runs), "n_paraphrased": n_paraphrased}
    for metric in judge_metric_names():
        conditions: dict[str, Any] = {}
        for label, other in (("identical", ident), ("paraphrase", para)):
            if other is None:
                continue
            pairs = [
                (
                    float(base.metrics.get(rid, {}).get(metric, float("nan"))),
                    float(other.metrics.get(rid, {}).get(metric, float("nan"))),
                )
                for rid in base.metrics
                if rid in other.metrics
            ]
            usable = [(a, b) for a, b in pairs if a == a and b == b]
            conditions[label] = {"n_pairs": len(usable), "sigma": sigma_from_pairs(usable)}
        sigmas = [c["sigma"] for c in conditions.values() if c["sigma"] is not None]
        out["per_metric"][metric] = {
            "conditions": conditions,
            # The MAXIMUM. The smaller condition, or their mean, would narrow the dead band
            # using whichever repeat flattered the instrument.
            "sigma_j": max(sigmas) if sigmas else None,
            "n_pairs": min((c["n_pairs"] for c in conditions.values()), default=0),
        }
    return out


def _paraphrase(judge: Any, report: str, *, seed: int) -> str:
    if not report.strip():
        return ""
    try:
        text, _tel = judge.complete(
            role="judge",
            messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(report=report)}],
            seed=seed,
            max_tokens=PARAPHRASE_MAX_TOKENS,
            actor="judge",
        )
    except Exception:  # noqa: BLE001 - a failed paraphrase drops one item, never the estimate
        return ""
    return text.strip()


# --------------------------------------------------------------------------- provenance


def provenance(
    result: Mapping[str, Any], *, judge_model: str, judge_family: str, seed: int, retest_seed: int
) -> dict[str, Any]:
    from pinq.ids import h

    return {
        "judge_model": judge_model,
        "judge_family": judge_family,
        "seed": seed,
        "retest_seed": retest_seed,
        "paraphrase_prompt_sha": h("paraphrase", PARAPHRASE_PROMPT),
        "n_runs": result["n_runs"],
        "n_paraphrased": result["n_paraphrased"],
        "min_pairs_required": SIGMA_J_MIN_PAIRS,
        "per_metric": result["per_metric"],
        "note": (
            "sigma_J is the SD of a SINGLE judge measurement: SD(differences)/sqrt(2) over "
            "paired repeats, taken as the MAXIMUM over the identical-retest and "
            "paraphrase-retest conditions. Read by pi_eval.prereg.load_sigma_j and used by "
            "pi_eval.report as a dead band."
        ),
    }


def write_sigma_j(result: Mapping[str, Any], prov: Mapping[str, Any], root: Path) -> list[str]:
    """Write only the metrics that MET the minimum design. Returns what was refused.

    A metric measured on 40 pairs is an anecdote, and writing it would replace an honest
    +inf floor -- which flags every effect -- with a small number that lets everything
    through. Absent stays absent.
    """
    out: dict[str, float] = {}
    refused: list[str] = []
    for metric, m in sorted(result["per_metric"].items()):
        if m["sigma_j"] is None:
            continue
        if m["n_pairs"] < SIGMA_J_MIN_PAIRS:
            refused.append(f"{metric}: {m['n_pairs']} pairs < {SIGMA_J_MIN_PAIRS}")
            continue
        out[metric] = round(float(m["sigma_j"]), 6)

    d = root / "prereg"
    d.mkdir(parents=True, exist_ok=True)
    # The map load_sigma_j reads carries NOTHING but metric -> float: it ignores non-numeric
    # entries, and a bool would be read as 1.0 and silently raise the floor everywhere.
    (d / SIGMA_J_FILE).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    (d / "sigma_j.provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")
    return refused


# --------------------------------------------------------------------------- cli


def load_inputs(
    parquet: Path, runs_root: Path, corpora_root: Path, *, n: int, graph_version: str = "v1"
):
    """Assemble judge inputs the SAME way `pi score` does, from the same parquet and gold.

    Reusing `score.load_answers` / `score.load_questions` / `gold.load_graphs` rather than
    re-reading the files here is the point: sigma_J has to be the noise of the instrument
    measuring THESE items. An estimate built from a differently-assembled pool is an estimate
    of a different instrument, and the number it produces would gate metrics it never saw.
    """
    from pi_eval.gold import load_graphs
    from pi_eval.judges import harness
    from pi_eval.score import _read, load_answers, load_questions, question_cache_key

    runs = _read(parquet / "runs.parquet")
    if not runs:
        raise SystemExit(f"no runs in {parquet / 'runs.parquet'}: run `pi compact` first")
    answers = load_answers(runs_root)
    graphs = {s: load_graphs(s, graph_version) for s in sorted({str(r["suite_id"]) for r in runs})}

    inputs: list[harness.RunInput] = []
    qcache: dict[tuple[str, str], dict[str, str]] = {}
    for r in sorted(runs, key=lambda x: str(x["run_id"])):
        rid, suite_id, task_id = str(r["run_id"]), str(r["suite_id"]), str(r["task_id"])
        ans = answers.get(rid)
        g = graphs.get(suite_id, {}).get(task_id)
        if ans is None or g is None or not ans.text.strip():
            continue
        # THE SCORER'S OWN RULE, imported. This read `corpus_hash` alone, which names no
        # path on disk (0 of 8 resolve), so every run loaded zero questions and every judged
        # item would have been graded with question="" -- while sigma_J is the gate deciding
        # whether any judge-derived metric is reportable, at ~13,400 judge calls to measure.
        key = question_cache_key(r)
        if key not in qcache:
            qcache[key] = load_questions(corpora_root, key[0], key[1])
        inputs.append(
            harness.RunInput(
                run_id=rid,
                arm_id=str(r["arm_id"]),
                report=ans.text,
                task=harness.TaskInput(
                    suite_id=suite_id,
                    task_id=task_id,
                    question=qcache[key].get(task_id, ""),
                    key_points=tuple(
                        (node.gold_node_id, node.gold_text)
                        for node in g.required()
                        if node.gold_text
                    ),
                ),
            )
        )
        if len(inputs) >= n:
            break
    return inputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default="scores/parquet")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--corpora-root", default="data/corpora")
    ap.add_argument("--gold-root", default=str(ROOT / "data" / "gold"))
    ap.add_argument("--n", type=int, default=SIGMA_J_MIN_PAIRS, help="items to measure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retest-seed", type=int, default=None)
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument(
        "--write",
        action="store_true",
        help="write prereg/sigma_j.json. Without it the estimate is printed and nothing is "
        "frozen -- sealing is a commitment and should not be a side effect of measuring.",
    )
    a = ap.parse_args()

    from pi_eval.score import judge_client_from_env

    judge, model, family, reason = judge_client_from_env()
    if judge is None:
        print(json.dumps({"ok": False, "error": reason}, indent=2))
        return 1

    # setdefault is NOT enough: .env ships `PI_GOLD_ROOT=` (empty) so the rollout workers
    # inherit an unset-equivalent, and setdefault sees the key present and does nothing --
    # leaving gold_root() to raise the firewall error in the one process that IS allowed
    # to read gold.
    if not os.environ.get("PI_GOLD_ROOT"):
        os.environ["PI_GOLD_ROOT"] = a.gold_root
    inputs = load_inputs(Path(a.parquet), Path(a.runs_root), Path(a.corpora_root), n=a.n)
    if not inputs:
        print(json.dumps({"ok": False, "error": "no judgeable runs found"}, indent=2))
        return 1

    retest = a.seed + 1 if a.retest_seed is None else a.retest_seed
    try:
        result = measure(
            judge=judge,
            runs=inputs,
            judge_model=model,
            judge_family=family,
            seed=a.seed,
            retest_seed=retest,
            paraphrase=not a.no_paraphrase,
        )
    except Exception as exc:  # noqa: BLE001 - a runbook step reports; it does not traceback
        # The common one by far is a replay-only judge with nothing cached. That is the client
        # doing its job -- a miss there means this path WOULD have called a provider -- but a
        # traceback reads like a bug in the estimator rather than a missing prerequisite.
        name = type(exc).__name__
        hint = ""
        if name == "CacheMiss":
            hint = (
                "the judge is replay-only and these items are not cached. Measure once with "
                "PI_JUDGE_CLIENT=pi_run.judge_client:judge (which dispatches and caches), then "
                "every later re-measurement replays for free."
            )
        print(json.dumps({"ok": False, "error": f"{name}: {exc}", "hint": hint}, indent=2))
        return 1
    prov = provenance(
        result, judge_model=model, judge_family=family, seed=a.seed, retest_seed=retest
    )
    report = {"ok": True, "spend": getattr(judge, "spend", dict)(), **prov}
    if a.write:
        refused = write_sigma_j(result, prov, ROOT)
        report["written"] = str(ROOT / "prereg" / SIGMA_J_FILE)
        report["refused_below_min_pairs"] = refused
    else:
        report["written"] = None
        report["note_write"] = "nothing frozen; re-run with --write"
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
