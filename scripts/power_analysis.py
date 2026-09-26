#!/usr/bin/env python
"""Minimum detectable effect for every preregistered endpoint, THROUGH THE ACTUAL TEST.

WHY THIS EXISTS. Before this script the repository's only statement about design sensitivity
was one sentence in a YAML comment -- `conf/grids/tier2_confirmatory.yaml`, "MDE about 12.6pp
on a binary endpoint". That figure is the textbook McNemar normal approximation at roughly 20
discordant pairs. It is not the sensitivity of the test that actually runs, which is a
CLUSTER-LEVEL PERMUTATION over seed-rounded outcomes. Simulated through
`pi_eval.stats.inference.mcnemar_exact` with tau2's measured template sizes, a 12.6pp lift has
38% power, and the true 80%-power MDE is about 22pp -- optimistic by a factor of 1.8.

A power figure computed from a different test than the one you run is not a power figure. This
computes it from the same functions `pi_eval.report` calls, on the same grid parameters the
sweep will use, so the number cannot drift from the analysis.

WHAT IT IS FOR. Two audiences:

  * BEFORE the spend. An endpoint whose MDE exceeds any plausible effect cannot succeed, and
    finding that out after the campaign is the most expensive way to learn it. A null result on
    an underpowered endpoint is uninterpretable: it cannot distinguish "no effect" from "not
    enough data", which is the single worst outcome for a paid run.
  * IN the paper. A reviewer asks what effect the design could detect. The honest answer is a
    table, and it should be the same table the preregistration froze.

IT ALSO COVERS THE KILL SWITCH, which is a different question. Those five comparators are
judged by ACCEPTING THE NULL, so what matters is not an MDE but an EQUIVALENCE margin and the
n at which the pilot can actually conclude equivalence. That section exists because this
docstring used to promise "every preregistered endpoint" while `evidence_coverage` -- the one
metric carrying a STOP consequence -- appeared nowhere in the file.

WHAT IT DOES NOT DO. It does not choose effect sizes for you. For the continuous endpoints the
PAIRED difference sd is the quantity that sets sensitivity, and it is not measurable until an
arm has actually run against a comparator on shared tasks -- so those rows are reported ACROSS
A RANGE of sds, with the observed per-run sd printed alongside as an upper bound (pairing on a
task with common random numbers can only reduce it). When a real paired sd becomes available
from a pilot, pass --paired-sd and the range collapses to a number.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi_eval.stats.inference import mcnemar_exact, paired_difference  # noqa: E402

# tau2's MEASURED template sizes, from Tau2Suite.template_id over all 97 tasks. Hardcoded
# rather than imported because this script must run without TAU2_DATA_DIR, and because a
# power figure should be reproducible from the paper alone.
TAU2_TEMPLATE_SIZES = (27, 12, 10, 8, 7, 6, 4, 4, 4, 3, 2, 2, 2, 2, 1, 1, 1, 1)

TARGET_POWER = 0.80
ALPHA = 0.05


def tau2_clusters() -> dict[str, str]:
    out: dict[str, str] = {}
    t = 0
    for ci, n in enumerate(TAU2_TEMPLATE_SIZES):
        for _ in range(n):
            out[f"t{t}"] = f"c{ci}"
            t += 1
    return out


def binary_power(
    lift: float,
    *,
    base: float = 0.45,
    seeds: int = 3,
    icc_sd: float = 0.15,
    trials: int = 400,
    n_perm: int = 800,
    seed: int = 0,
) -> float:
    """Power of the tau2 primary, through `mcnemar_exact` with the real cluster map.

    Mirrors `report.estimate`'s binary path exactly: average the seeds, DROP a task sitting at
    exactly 0.5 (the 2-seed tie -- unreachable at 3 seeds, kept for fidelity), round the rest,
    and hand the cluster map to the test. `icc_sd` is a per-template shift in success
    probability, which is what "tasks from one template are not independent" means here.
    """
    clusters = tau2_clusters()
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        a: dict[str, float] = {}
        b: dict[str, float] = {}
        k = 0
        for n in TAU2_TEMPLATE_SIZES:
            shift = rng.gauss(0, icc_sd)
            for _ in range(n):
                key = f"t{k}"
                k += 1
                p_b = min(0.98, max(0.02, base + shift))
                p_a = min(0.98, max(0.02, base + lift + shift))
                b[key] = sum(rng.random() < p_b for _ in range(seeds)) / seeds
                a[key] = sum(rng.random() < p_a for _ in range(seeds)) / seeds
        keep = [x for x in a if abs(a[x] - 0.5) > 1e-9 and abs(b[x] - 0.5) > 1e-9]
        est = mcnemar_exact(
            {x: int(round(a[x])) for x in keep},
            {x: int(round(b[x])) for x in keep},
            clusters=clusters,
            n_perm=n_perm,
            seed=1,
        )
        if est.p_value is not None and est.p_value < ALPHA:
            hits += 1
    return hits / trials


def paired_power(
    lift: float,
    *,
    sd_diff: float,
    n_tasks: int,
    trials: int = 300,
    n_perm: int = 800,
    seed: int = 0,
) -> float:
    """Power of a continuous endpoint, through `paired_difference` (sign-flip + BCa).

    Singleton clusters, because only tau2 mints a template_id -- so this is the per-task
    sign-flip, which is what those suites actually get.
    """
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        a = {f"t{i}": lift + rng.gauss(0, sd_diff) for i in range(n_tasks)}
        b = {f"t{i}": 0.0 for i in range(n_tasks)}
        est = paired_difference(a, b, n_boot=400, n_perm=n_perm, seed=1)
        if est.p_value is not None and est.p_value < ALPHA:
            hits += 1
    return hits / trials


def equivalence_power(
    *,
    margin: float,
    sd_diff: float,
    n_tasks: int,
    true_effect: float = 0.0,
    trials: int = 300,
    n_perm: int = 400,
    seed: int = 0,
) -> float:
    """P(the kill switch returns EQUIVALENT) -- the whole CI inside +/- margin.

    THE ENDPOINT THAT CARRIES A STOP CONSEQUENCE, which this script's docstring promised to
    cover and did not: `grep -c evidence_coverage scripts/power_analysis.py` was 0 while main()
    covered only the three primaries. `report.KILLSWITCH_METRIC` is `evidence_coverage`, and
    five comparators are judged against it with consequences up to "STOP; no reframe survives".

    Two numbers matter and they are different questions:
      * at true_effect = 0, this is the chance the pilot can CORRECTLY conclude equivalence.
        Low means the pilot is too small to stop the paper even when it should.
      * at true_effect > 0, it is the chance of a FALSE stop. That is the expensive error.
    """
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        a = {f"t{i}": true_effect + rng.gauss(0, sd_diff) for i in range(n_tasks)}
        b = {f"t{i}": 0.0 for i in range(n_tasks)}
        est = paired_difference(a, b, n_boot=400, n_perm=n_perm, seed=1)
        if est.ci_lo >= -margin and est.ci_hi <= margin:
            hits += 1
    return hits / trials


def mde(power_of: Callable[[float], float], grid: Sequence[float]) -> float | None:
    """Smallest lift on `grid` reaching TARGET_POWER, by linear interpolation between the
    bracketing points. None when the grid never reaches it -- which is itself the finding."""
    prev_lift, prev_p = 0.0, 0.0
    for lift in grid:
        p = power_of(lift)
        if p >= TARGET_POWER:
            if p == prev_p:
                return lift
            frac = (TARGET_POWER - prev_p) / (p - prev_p)
            return round(prev_lift + frac * (lift - prev_lift), 4)
        prev_lift, prev_p = lift, p
    return None


def observed_sd(parquet_dir: Path, metric: str) -> float | None:
    """Per-run sd of a metric from whatever has already run. An UPPER BOUND on the paired sd:
    pairing on a task with common random numbers can only reduce it."""
    f = parquet_dir / "scores.parquet"
    if not f.exists():
        return None
    try:
        import duckdb
    except ImportError:
        return None
    con = duckdb.connect()
    row = con.execute(
        "SELECT stddev(value) FROM read_parquet(?) WHERE metric_name = ?", [str(f), metric]
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=400, help="simulated experiments per point")
    ap.add_argument("--n-perm", type=int, default=800)
    ap.add_argument("--quick", action="store_true", help="fewer trials; for a smoke check")
    ap.add_argument("--base-rate", type=float, default=0.45, help="tau2 comparator success rate")
    ap.add_argument("--seeds", type=int, default=3, help="tau2 seeds per task")
    ap.add_argument("--paired-sd", type=float, default=None, help="measured paired-difference sd")
    ap.add_argument("--parquet-dir", default="scores/parquet")
    ap.add_argument("--out", default=None, help="write JSON here")
    a = ap.parse_args(argv)
    trials = 60 if a.quick else a.trials
    n_perm = 200 if a.quick else a.n_perm

    pq = Path(a.parquet_dir)
    report: dict[str, object] = {
        "target_power": TARGET_POWER,
        "alpha": ALPHA,
        "trials_per_point": trials,
        "note": (
            "Computed by simulation through pi_eval.stats.inference, the same functions "
            "pi_eval.report calls. A power figure from a different test than the one you run "
            "is not a power figure."
        ),
    }

    # ---- PRIMARY 1: tau2, binary, clustered
    lifts = [0.05, 0.10, 0.126, 0.15, 0.20, 0.25, 0.30]
    curve = {
        f"{x:.3f}": round(
            binary_power(x, base=a.base_rate, seeds=a.seeds, trials=trials, n_perm=n_perm), 3
        )
        for x in lifts
    }
    tau2_mde = mde(
        lambda x: binary_power(x, base=a.base_rate, seeds=a.seeds, trials=trials, n_perm=n_perm),
        lifts,
    )
    report["tau2_tau_reward"] = {
        "design": f"n=97 tasks in {len(TAU2_TEMPLATE_SIZES)} template clusters, "
        f"{a.seeds} seeds, base rate {a.base_rate}",
        "test": "mcnemar_exact with cluster_by=template_id (cluster-level sign-flip)",
        "power_curve": curve,
        "mde_at_80pct": tau2_mde,
        "documented_mde_in_grid_yaml": 0.126,
        "documented_mde_power": curve.get("0.126"),
    }

    # ---- PRIMARY 2 and 3, and the pooled secondaries: continuous
    for key, metric, n_tasks in (
        ("musique_answer_token_f1", "answer_token_f1", 200),
        ("drgym_kpr_incremental", "kpr_incremental", 100),
        # THE VERTICAL CLAIM. Judge-free, so no sigma_J floor applies and the only thing
        # standing between an effect and a claim is n. n=200 matches tier1_confirmatory's
        # musique slice, which is the grid this endpoint would be read from.
        #
        # `latent_discovery_rate` and not `cad_ge2`: cad_ge2 is a DERIVED pull built at report
        # time from the cad#d / cad_n#d families, so it has no rows of its own and
        # `observed_sd` -- which queries `WHERE metric_name = ?` -- returns None for it. The
        # sd would then be guessed rather than measured, which is the failure this whole
        # script exists to end.
        ("musique_latent_discovery_rate", "latent_discovery_rate", 200),
    ):
        obs = observed_sd(pq, metric)
        sds = [a.paired_sd] if a.paired_sd else [0.10, 0.15, 0.22, 0.30]
        block: dict[str, object] = {
            "design": f"n={n_tasks} tasks, per-task sign-flip (singleton clusters)",
            "test": "paired_difference (sign-flip p, task-clustered BCa CI)",
            "observed_per_run_sd": None if obs is None else round(obs, 4),
            "observed_sd_is": "an UPPER BOUND on the paired sd; pairing can only reduce it",
            "by_paired_sd": {},
        }
        for sd in sds:
            grid = [0.02, 0.05, 0.08, 0.12, 0.20]
            block["by_paired_sd"][f"{sd:.2f}"] = {  # type: ignore[index]
                "power_curve": {
                    f"{x:.2f}": round(
                        paired_power(x, sd_diff=sd, n_tasks=n_tasks, trials=trials, n_perm=n_perm),
                        3,
                    )
                    for x in grid
                },
                "mde_at_80pct": mde(
                    lambda x, _sd=sd: paired_power(
                        x, sd_diff=_sd, n_tasks=n_tasks, trials=trials, n_perm=n_perm
                    ),
                    grid,
                ),
            }
        report[key] = block

    # ---- THE KILL SWITCH, which is an EQUIVALENCE decision and not a difference one
    from pi_eval.prereg import KILL_SWITCH_MARGINS  # noqa: PLC0415

    ks: dict[str, object] = {
        "metric": "evidence_coverage",
        "note": (
            "A kill switch ACCEPTS THE NULL, so it needs an equivalence margin. Reported as "
            "P(EQUIVALENT) at a true zero effect -- the chance the pilot can correctly stop -- "
            "and at true effects above zero, which is the chance of a FALSE stop."
        ),
        "by_arm": {},
    }
    for arm, margin in sorted(KILL_SWITCH_MARGINS.items()):
        ks["by_arm"][arm] = {  # type: ignore[index]
            "margin": margin,
            "p_equivalent_by_true_effect_and_n": {
                f"sd={sd}": {
                    f"n={n}": {
                        f"{eff:.2f}": round(
                            equivalence_power(
                                margin=margin,
                                sd_diff=sd,
                                n_tasks=n,
                                true_effect=eff,
                                trials=max(60, trials // 3),
                                n_perm=max(200, n_perm // 2),
                            ),
                            3,
                        )
                        for eff in (0.0, 0.02, 0.04)
                    }
                    for n in (120, 400)
                }
                for sd in (0.20,)
            },
        }
    report["kill_switch_equivalence"] = ks

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
        print(f"\nwrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
