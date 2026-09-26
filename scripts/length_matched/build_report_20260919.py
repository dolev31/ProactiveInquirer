"""Build the length-matched contrast report for preference-8B vs. imitation-{8B,32B} from the
three `*.paired.json` files `score_checkpoint_paired.py` writes on the cluster.

Reads three local files (pulled by scp from
`/proj/pinq/user/artifacts/length_matched_stratum_20260919/eval_paired/`), and for each rung:
  - reports each checkpoint's matched accuracy, n, and Wilson 95% CI (via the
    `pair_accuracy_crosscheck` block, which is `eo.pair_accuracy`'s own output, unmodified);
  - runs the direct paired contrast (McNemar exact + paired bootstrap CI) between preference-8B
    and each imitation checkpoint, using `paired_contrast.join_by_pair_id`;
  - flags rungs below the pre-established power threshold (n<339, this project's own rule for
    detecting a 0.60-vs-0.50 effect at alpha=0.05, power>=0.80) as reportable but not
    conclusion-bearing.

Does not re-score anything; this is pure aggregation over already-computed per-pair JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_contrast import join_by_pair_id, mcnemar_contrast, paired_bootstrap_ci  # noqa: E402

Z = 1.959963984540054
POWERED_RUNGS = {"tol10", "tol20", "tol30", "unmatched_ask_ask_all"}
RUNG_ORDER = ["unmatched_ask_ask_all", "tol0", "tol2", "tol5", "tol10", "tol20", "tol30"]


def wilson_ci(x: int, n: int, z: float = Z) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    phat = x / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    half = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)
    return ((center - half) / denom, (center + half) / denom)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: build_report_20260919.py <imitation8b.paired.json> "
            "<preference8b.paired.json> <imitation32b.paired.json>",
            file=sys.stderr,
        )
        return 2
    imit8, pref8, imit32 = (load(Path(p)) for p in sys.argv[1:4])

    report: dict = {"rungs": {}}
    for rung in RUNG_ORDER:
        rung_out: dict = {"per_checkpoint": {}, "powered": rung in POWERED_RUNGS}
        for label, data in (
            ("imitation_8b", imit8),
            ("preference_8b", pref8),
            ("imitation_32b", imit32),
        ):
            res = data["results"].get(rung)
            if res is None:
                continue
            pa = res["pair_accuracy_crosscheck"]
            n = pa["n"]
            x = round(pa["acc"] * n) if n else 0
            lo, hi = wilson_ci(x, n)
            rung_out["per_checkpoint"][label] = {
                "checkpoint_label": data["label"],
                "n": n,
                "x_correct": x,
                "acc": pa["acc"],
                "wilson_ci": [lo, hi],
                "len_sign_gap": pa.get("len_sign_gap"),
                "acc_by_len_sign": pa.get("acc_by_len_sign"),
            }

        # Direct paired contrasts: preference_8b vs each imitation checkpoint, on the SAME
        # pair_ids of this rung (join_by_pair_id refuses a mismatch rather than guessing).
        contrasts = {}
        pref_pp = pref8["results"].get(rung, {}).get("paired", {}).get("per_pair")
        for other_label, other_data in (("imitation_8b", imit8), ("imitation_32b", imit32)):
            other_pp = other_data["results"].get(rung, {}).get("paired", {}).get("per_pair")
            if pref_pp is None or other_pp is None:
                continue
            joined = join_by_pair_id(other_pp, pref_pp)  # a=imitation, b=preference
            mc = mcnemar_contrast(joined)
            boot = paired_bootstrap_ci(joined, n_boot=10000, seed=20260919)
            contrasts[f"preference_8b_minus_{other_label}"] = {
                "mcnemar": mc,
                "paired_bootstrap_95ci": boot,
            }
        rung_out["paired_contrasts"] = contrasts
        report["rungs"][rung] = rung_out

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
