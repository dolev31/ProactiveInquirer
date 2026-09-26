"""Paired tau2 fork contrasts, keyed on fork identity by pi_eval's own pairing.

SUPERSEDES `concordance.py`, `cost_given_success.py` and `cluster_sign.py`, all three of which
keyed pairs on (task_id, seed, prompt_variant). That is wrong: the campaign is 34 recovered FORK
POINTS x 3 seeds and several fork points come from the SAME task at different depths, so that key
paired a trained unit forked at `foreign_prefix_k = 8` against a prompted unit forked at 14 and
called them matched, while a dict key silently dropped whatever it overwrote (213 ok airline units
collapsed to 113 cells). It destroyed the premise the design rests on -- both arms resume the SAME
recorded prefix, so difficulty is shared.

WHY THIS CALLS pi_eval INSTEAD OF KEYING AGAIN. `pi_eval.fork_report._pair_forks` already keys on
(foreign_trace_sha, foreign_prefix_k, seed), already excludes runs with no foreign prefix because
two of them would collide on the empty key, and already REFUSES a second run of one arm at a key
rather than picking from it -- "which one counts would be a choice the reader cannot see". A hand
copy of run-unit logic has already cost this repository a silent four-field divergence, so this
script imports that pairing rather than reproducing it.

WHY IT IS CALLED ONCE PER PROMPT VARIANT. This campaign adds a prompt-variant dimension the paper's
fork campaign does not have, so (trace, k, seed) is NOT unique here -- the same fork and seed is run
under `tau2_base` and `tau2_stop`. Partitioning by variant first makes the key unique again
(verified: 0 duplicated slots within a variant on both suites) and needs no change to pi_eval, whose
published numbers depend on that pairing. It also enforces the rule this programme already had:
the 2x2 is arm x variant, so the two variants are different observations and are never pooled.

THE UNIT OF INDEPENDENCE IS NOT THE PAIR. One fork point yields up to 3 pairs (seeds) and one task
yields several fork points, so a pair count is not an n. Every contrast is reported at pair,
fork_point and task resolution, and only the coarsest is quotable. An 8-0 pair-level split on 2
tasks read p = 0.0078 when the honest reading was p = 0.50 -- a factor of 64.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pi_eval.fork_report import _pair_forks, sign_test_p  # noqa: E402

#: coarsest last; only the coarsest is quotable
LEVELS = ("pair", "fork_point", "task")


def cluster_id(key: tuple, run: dict, level: str) -> str:
    if level == "pair":
        return repr(key)
    if level == "fork_point":
        return repr(key[:2])
    if level == "task":
        return str(run.get("task_id"))
    raise ValueError(f"unknown level {level!r}")


def cp_lower_bound(k: int, n: int, alpha: float = 0.025) -> float | None:
    """Clopper-Pearson one-sided LOWER bound on the success probability from k of n.

    THE ESTIMAND WITH RANGE -- AND ITS RANGE IS ONLY IN THE VALUE, NOT IN THE TEST. For a UNANIMOUS
    split, "this bound clears 0.5" and "the two-sided sign p is below 0.05" are the SAME statement:
    p = 2 * 2**-n < 0.05 <=> 2**-n < 0.025, and L = alpha**(1/n) > 0.5 <=> 0.025 > 0.5**n <=> the
    identical inequality. Both hold iff n >= 6 (verified for n = 1..9). So the bound adds no
    inferential content over the test; what it adds is a NUMBER. It prices the unanimity, and at
    n=6 the price is 0.5407 -- barely above a coin. "The trained arm wins all six" sounds strong;
    "the win rate among such fork points is at least 0.54" is what six unanimous clusters license.

    `alpha` defaults to 0.025, NOT 0.05, because the sign test printed beside it is TWO-SIDED at 0.05
    and so spends 0.025 per tail. A 95% one-sided bound is looser than that test and would let a cell
    appear to exclude chance while the test calls it non-significant. At n=5 that choice decides the
    answer: 0.5493 at alpha=0.05 clears chance, 0.4782 at alpha=0.025 does not.

    Defined for k < n as well, by solving P(X >= k | p = L) = alpha, because the counter-example case
    is where the bound is MOST informative and returning None there would hide it. The outcomes are
    sharply asymmetric at n=6: a seventh cluster agreeing moves the bound to 0.5904, a seventh
    disagreeing drops it to 0.4213, and one of the six flipping drops it to 0.3588 -- the latter two
    below chance.
    """
    if n <= 0 or not 0 <= k <= n:
        return None
    if k == 0:
        return 0.0
    if k == n:
        return alpha ** (1.0 / n)

    def upper_tail(prob: float) -> float:
        return sum(comb(n, i) * prob**i * (1.0 - prob) ** (n - i) for i in range(k, n + 1))

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if upper_tail(mid) > alpha:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def endpoint_calls(r: dict) -> float:
    v = r.get("n_env_calls")
    if v is None:
        raise ValueError(f"run {r.get('run_id')} has no n_env_calls")
    return float(v)


def endpoint_asks(r: dict) -> float:
    return float(r.get("n_asks") or 0)


TOOL_TYPES_PATH = Path(__file__).resolve().parents[2] / "conf" / "tau2" / "tool_types.json"


class UnmappedTool(Exception):
    """A tool the environment's declared types do not cover. Never counted as zero."""


def load_tool_types() -> dict:
    return json.loads(TOOL_TYPES_PATH.read_text())


def domain_of(r: dict) -> str:
    k = r.get("key") or [""]
    return str(k[0]).removeprefix("tau2_")


def typed_calls(tool_types: dict, kind: str):
    """Endpoint: the downstream agent's env calls of one declared type (READ / WRITE / GENERIC).

    Classified by tau2's own @is_tool(ToolType.*) declarations, NOT by the record's `mutating`
    field, which is True on every tau2 call. An unmapped tool name raises: silently counting it
    as zero is the read-as-absence failure this directory has already committed three times.
    """

    def ep(r: dict) -> float:
        counts = r.get("_env_tool_counts")
        if counts is None:
            raise KeyError("_env_tool_counts")
        types = tool_types.get(domain_of(r))
        if types is None:
            raise UnmappedTool(f"no tool-type map for domain {domain_of(r)!r}")
        total = 0
        for name, n in counts.items():
            if name not in types:
                raise UnmappedTool(f"{domain_of(r)}: tool {name!r} has no declared type")
            if types[name] == kind:
                total += n
        return float(total)

    return ep


def succeeded(r: dict) -> bool:
    return float((r.get("native") or {}).get("tau_reward") or 0.0) > 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", required=True, help="JSON array of status.json contents")
    ap.add_argument("--label", default="")
    ap.add_argument("--treatment", default="inquirer_trained")
    ap.add_argument("--control", default="inquirer_prompted")
    ap.add_argument("--out")
    a = ap.parse_args()

    runs = [r for r in json.loads(Path(a.records).read_text()) if r.get("status") == "ok"]

    # TWO ROUTES TO ONE QUANTITY, kept permanently rather than as a one-off. The keying defect
    # this script replaces was found only because two counts of one population disagreed, and no
    # check had failed. A disagreement here means `n_env_calls` and the recorded call list are not
    # the same measurement, so the cost contrast below would be reporting an unknown mixture.
    checked = [r for r in runs if r.get("_env_calls_from_outcome") is not None]
    bad = [r for r in checked if r.get("n_env_calls") != r["_env_calls_from_outcome"]]
    if bad:
        print(
            f"  REFUSING: n_env_calls disagrees with len(outcome.env_calls) on {len(bad)} of "
            f"{len(checked)} runs, e.g. {bad[0].get('run_id')} "
            f"({bad[0].get('n_env_calls')} vs {bad[0]['_env_calls_from_outcome']})"
        )
        return 3
    print(
        f"  endpoint cross-check: n_env_calls == len(outcome.env_calls) on {len(checked)} of "
        f"{len(runs)} runs" + ("" if checked else "  [NOT CHECKED: field absent]")
    )

    # A THIRD ROUTE, and the unmapped-tool refusal, before any typed endpoint is computed.
    tool_types = load_tool_types()
    typed = [r for r in runs if r.get("_env_tool_counts") is not None]
    bad_sum = [r for r in typed if sum(r["_env_tool_counts"].values()) != r.get("n_env_calls")]
    if bad_sum:
        print(
            f"  REFUSING: per-tool counts do not sum to n_env_calls on {len(bad_sum)} runs, "
            f"e.g. {bad_sum[0].get('run_id')}"
        )
        return 3
    unmapped = sorted(
        {
            (domain_of(r), name)
            for r in typed
            for name in r["_env_tool_counts"]
            if name not in tool_types.get(domain_of(r), {})
        }
    )
    if unmapped:
        print(
            f"  REFUSING: tools with no declared type, which would otherwise count as zero: {unmapped}"
        )
        return 3
    print(
        f"  typed-call check: per-tool counts sum to n_env_calls and every tool is mapped on "
        f"{len(typed)} of {len(runs)} runs" + ("" if typed else "  [NOT RECORDED: re-extract]")
    )
    have_types = len(typed) == len(runs) and len(runs) > 0
    variants = sorted({(r.get("key") or ["?"] * 10)[9] for r in runs})
    cvs = sorted({str(r.get("code_version"))[:8] for r in runs})

    # AN EMPTY STORE READS LIKE A CLEAN ONE. Zero ok runs is not a result, and a report printed
    # over it is indistinguishable from a report over a healthy population that found nothing.
    if not runs:
        print(f"  REFUSING: {a.records} holds no runs with status ok -- nothing to pair.")
        return 3

    print(f"===== {a.label or a.records}: {len(runs)} ok runs =====")
    print(f"  code_version(s): {cvs}" + ("   <-- POOLS MORE THAN ONE" if len(cvs) > 1 else ""))
    report: dict = {"label": a.label, "code_versions": cvs, "variants": {}}

    union_task: dict[str, list[float]] = defaultdict(list)
    union_fork: dict[str, list[float]] = defaultdict(list)

    for variant in variants:
        sub = [r for r in runs if (r.get("key") or ["?"] * 10)[9] == variant]
        pairs, n_unpaired, n_not_forks = _pair_forks(sub, treatment=a.treatment, control=a.control)
        print(
            f"\n  --- variant {variant}: {len(sub)} runs -> {len(pairs)} pairs "
            f"({n_unpaired} unpaired, {n_not_forks} non-fork) ---"
        )

        # A 2x2 OF ZEROS IS NOT A 2x2. With no pairs, `prompted_only = 0` prints identically to
        # the real finding, where it means the comparator never won; here it means there is
        # nothing to win. Refuse the cell rather than emit numbers a reader will interpret.
        if not pairs:
            print(
                "      2x2  REFUSED: no complete pairs in this variant, so every cell would be "
                "0 and would read as a finding"
            )
            report["variants"][variant] = {
                "n_runs": len(sub),
                "n_pairs": 0,
                "n_unpaired": n_unpaired,
                "refused": "no complete pairs; 2x2 and contrasts not computed",
            }
            continue

        both = t_only = p_only = neither = 0
        for v in pairs.values():
            t, c = succeeded(v[a.treatment]), succeeded(v[a.control])
            both += t and c
            t_only += t and not c
            p_only += c and not t
            neither += not t and not c
        marg = [succeeded(r) for r in sub]
        p_hat = (sum(marg) / len(marg)) if marg else 0.0
        exp_disc = 2 * p_hat * (1 - p_hat) * len(pairs)
        print(
            f"      2x2  both={both} trained_only={t_only} prompted_only={p_only} neither={neither}"
            f"   marginal p={p_hat:.3f}  discordant obs={t_only + p_only} exp={exp_disc:.1f}"
        )

        # THE DIRECTION TEST BELONGS IN THIS BLOCK, NOT FURTHER DOWN. `prompted_only = 0` reads as a
        # WIN; whether it is one is decided by a clustered test, and a reader who sees the 2x2 without
        # it will take the stronger claim. Null: given a discordant pair, P(favours treatment) = 0.5.
        # A cluster counts once, by the sign of its mean, so seeds and prompt variants of one fork do
        # not each count as evidence.
        disc = {
            k: (1.0 if succeeded(v[a.treatment]) else -1.0)
            for k, v in pairs.items()
            if succeeded(v[a.treatment]) != succeeded(v[a.control])
        }
        for k, d in disc.items():
            union_task[str(pairs[k][a.treatment].get("task_id"))].append(d)
            union_fork[repr(k[:2])].append(d)
        dlv: dict = {}
        if not disc:
            print("      direction  no discordant pairs, so prompted_only=0 carries no direction")
        for level in LEVELS:
            if not disc:
                break
            groups: dict[str, list[float]] = defaultdict(list)
            for k, d in disc.items():
                groups[cluster_id(k, pairs[k][a.treatment], level)].append(d)
            means = [statistics.fmean(v) for v in groups.values()]
            fav = sum(x > 0 for x in means)
            against = sum(x < 0 for x in means)
            pv = sign_test_p(fav, against)
            floor = sign_test_p(len(means), 0)
            cap = (
                ""
                if floor <= 0.05
                else f"  [CANNOT reach 0.05 at n={len(means)}; floor {floor:.4f}]"
            )
            mark = "  <-- quotable" if level == LEVELS[-1] else ""
            lb = cp_lower_bound(fav, fav + against)
            if lb is None:
                wp = ""
            else:
                wp = f"  winprob>={lb:.4f}" + ("" if lb > 0.5 else "  [does NOT clear chance]")
            print(
                f"      direction  {level:<10} {fav}-{against} over {len(means)}  p={pv:.4f}"
                f"  floor={floor:.4f}{wp}{cap}{mark}"
            )
            dlv[level] = {
                "n_clusters": len(means),
                "favours_treatment": fav,
                "against": against,
                "sign_test_p": round(pv, 6),
                "achievable_min_p": round(floor, 6),
                "win_prob_lower_97_5": round(lb, 4) if lb is not None else None,
            }
        vrep_direction = dlv

        vrep: dict = {
            "n_runs": len(sub),
            "n_pairs": len(pairs),
            "n_unpaired": n_unpaired,
            "both": both,
            "treatment_only": t_only,
            "control_only": p_only,
            "neither": neither,
            "marginal_success": round(p_hat, 4),
            "observed_discordant": t_only + p_only,
            "expected_discordant_independence": round(exp_disc, 2),
            "direction_test": vrep_direction,
            "endpoints": {},
        }

        endpoints = [
            ("n_env_calls (COST)", endpoint_calls),
            ("n_asks (decisions)", endpoint_asks),
        ]
        if have_types:
            # n_env_calls sums two behaviours that move in OPPOSITE directions -- lookups and
            # state-changing actions -- so its paired delta can hide both. Report them apart.
            endpoints += [
                ("READ calls (agent lookups)", typed_calls(tool_types, "READ")),
                ("WRITE calls (agent actions)", typed_calls(tool_types, "WRITE")),
            ]
        else:
            print("      [READ/WRITE calls] not recorded in these records -- re-extract; NOT zero")
        for ep_name, ep in endpoints:
            if not pairs:
                print(f"      [{ep_name}] no pairs in this variant -- nothing to contrast")
                vrep["endpoints"][ep_name] = None
                continue
            diffs = {k: ep(v[a.treatment]) - ep(v[a.control]) for k, v in pairs.items()}
            print(
                f"      [{ep_name}] mean {statistics.fmean(diffs.values()):+.2f} "
                f"median {statistics.median(diffs.values()):+.1f}"
            )
            lv: dict = {}
            for level in LEVELS:
                groups: dict[str, list[float]] = defaultdict(list)
                for k, d in diffs.items():
                    groups[cluster_id(k, pairs[k][a.treatment], level)].append(d)
                means = [statistics.fmean(v) for v in groups.values()]
                fewer, more = sum(x < 0 for x in means), sum(x > 0 for x in means)
                p = sign_test_p(fewer, more)
                quotable = "  <-- quotable" if level == LEVELS[-1] else ""
                # PRINT THE ACHIEVABLE MINIMUM, do not merely know it. With zero counter-examples
                # the smallest p this many clusters can produce is sign_test_p(n, 0): n=5 gives
                # 0.0625, so such a cell CANNOT reach 0.05 whatever the data does. A reader who
                # sees four cells and one significant one otherwise applies a multiplicity
                # correction that does not apply, and is wrong in the conservative direction --
                # the unusual case where an unstatable quantity costs a real result rather than
                # inflating one.
                floor = sign_test_p(len(means), 0)
                cap = (
                    ""
                    if floor <= 0.05
                    else f"  [CANNOT reach 0.05 at n={len(means)}; floor {floor:.4f}]"
                )
                print(
                    f"         {level:<10} n={len(means):<3} fewer={fewer} more={more} "
                    f"mean={statistics.fmean(means):+.2f} p={p:.4f}  floor={floor:.4f}{cap}{quotable}"
                )
                lv[level] = {
                    "n_clusters": len(means),
                    "fewer": fewer,
                    "more": more,
                    "diff_mean": round(statistics.fmean(means), 4) if means else None,
                    "sign_test_p": round(p, 6),
                }
            vrep["endpoints"][ep_name] = lv
        report["variants"][variant] = vrep

    # UNION OVER VARIANTS, DIRECTION ONLY. Each task counts ONCE however many variants it appears
    # in, which is the correct unit for a direction claim: the two variants of one task are not
    # independent replications (measured: 4 of 6 discordant tasks shared, union 8) and neither are
    # they one observation twice. This merge is legitimate ONLY because every discordant pair agrees
    # in sign, so no magnitude is being averaged across two scoring bases -- the never-pool rule
    # still forbids it for any cost endpoint, and none is computed here.
    if union_task:
        print("\n  --- UNION over prompt variants, DIRECTION ONLY (a task counts once) ---")
        for name, groups in (
            ("task  (the correct unit)", union_task),
            ("fork_point (less conservative)", union_fork),
        ):
            means = [statistics.fmean(v) for v in groups.values()]
            fav = sum(x > 0 for x in means)
            ag = sum(x < 0 for x in means)
            pv = sign_test_p(fav, ag)
            lb = cp_lower_bound(fav, fav + ag)
            wp = (
                ""
                if lb is None
                else f"  winprob>={lb:.4f}" + ("" if lb > 0.5 else "  [does NOT clear chance]")
            )
            floor = sign_test_p(len(means), 0)
            cap = (
                ""
                if floor <= 0.05
                else f"  [CANNOT reach 0.05 at n={len(means)}; floor {floor:.4f}]"
            )
            print(
                f"      {name:<32s} {fav}-{ag} over {len(means)}  p={pv:.4f}  floor={floor:.4f}{wp}{cap}"
            )
            report.setdefault("union_direction", {})[name.split()[0]] = {
                "n_clusters": len(means),
                "favours_treatment": fav,
                "against": ag,
                "sign_test_p": round(pv, 6),
                "achievable_min_p": round(floor, 6),
                "win_prob_lower_97_5": round(lb, 4) if lb is not None else None,
            }

    total_pairs = sum(v.get("n_pairs") or 0 for v in report["variants"].values())
    if total_pairs == 0:
        print(
            f"\n  REFUSING: {len(runs)} ok runs but 0 complete pairs in any variant -- "
            "the arms are not paired, so no contrast exists to report."
        )
        return 3

    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
