"""Human anchor: what the returned rater sheets say, and what they do NOT license.

WHY THIS EXISTS. Every A7 label in this project is a model's. The paper states that as a
limitation and a reviewer will open with it. This reads the returned human sheets against three
references: each other, the mechanical rule, and the LLM panel majority.

TWO THINGS THE READER MUST NOT DO, both of which this file is arranged to avoid.
A rater's `preference` is an A/B answer, and the slot that holds the rule's CHOSEN candidate
varies per item (`order` is 'ab' on 58 items and 'ba' on 52). Comparing raw A/B to the rule
would therefore measure slot layout, not judgement, so every comparison here maps A/B through
that item's own `order` into chosen/rejected first. And the 13 planted items are NOT part of any
agreement number: they are scored only as quality control, each against the field its own
`attention_check.expected` names, which is `reaches_B` on 4 of them and `reaches_A` on the rest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    z = 1.96
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, centre - half, centre + half


def kappa(pairs: list[tuple[str, str]], labels: tuple[str, ...]) -> tuple[int, float, float]:
    n = len(pairs)
    if n == 0:
        return 0, float("nan"), float("nan")
    po = sum(1 for x, y in pairs if x == y) / n
    c1, c2 = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum((c1[k] / n) * (c2[k] / n) for k in labels)
    return n, po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument(
        "--sheets", nargs="+", required=True, help="returned answers.csv, in rater order"
    )
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    key_path = next((a.bundle_dir / "key").glob("*.key.json"))
    K = json.loads(key_path.read_text())["items"]
    IDENT = json.loads((a.bundle_dir / "key" / "anchor_identity.json").read_text())["items"]
    SEL = {
        json.loads(line)["pair_id"]: json.loads(line)
        for line in (a.bundle_dir / "select" / "selected_pairs.jsonl").read_text().splitlines()
        if line.strip()
    }
    R = {n + 1: {r["item_id"]: r for r in csv.DictReader(open(p))} for n, p in enumerate(a.sheets)}
    raters = sorted(R)
    norm = lambda s: (s or "").strip().lower()  # noqa: E731
    real = [i for i in K if i in IDENT]
    planted = [i for i in K if i not in IDENT]

    def side(n: int, i: str) -> str | None:
        """A/B mapped through this item's own slot order into chosen/rejected."""
        pref, order = norm(R[n][i]["preference"]), K[i].get("order")
        if pref not in ("a", "b") or order not in ("ab", "ba"):
            return None
        return "chosen" if (pref == "a") == (order == "ab") else "rejected"

    out: dict = {
        "composition": {"items": len(K), "real": len(real), "planted": len(planted)},
        "raters": {},
    }

    for n in raters:
        rec: dict = {}
        ok = 0
        misses = []
        for i in planted:
            exp = K[i]["attention_check"]["expected"]
            field = "reaches_A"
            if exp != "not_a_need":
                field = "reaches_A" if list(exp)[0].startswith("a") else "reaches_B"
            if norm(R[n][i][field]) == "no":
                ok += 1
            else:
                misses.append({"item": i, "field": field, "got": norm(R[n][i][field])})
        rec["quality_control"] = {"correct": ok, "n": len(planted), "misses": misses}

        prefs = [norm(R[n][i]["preference"]) for i in real]
        na, nb = prefs.count("a"), prefs.count("b")
        p, lo, hi = wilson(na, na + nb)
        rec["slot_bias"] = {
            "A": na,
            "B": nb,
            "p_A": p,
            "lo": lo,
            "hi": hi,
            "covers_half": lo <= 0.5 <= hi,
        }

        d = [s for s in (side(n, i) for i in real) if s]
        p, lo, hi = wilson(d.count("chosen"), len(d))
        rec["vs_rule"] = {"agree": d.count("chosen"), "n": len(d), "p": p, "lo": lo, "hi": hi}

        ag = tot = 0
        for i in real:
            h, pm = side(n, i), SEL[K[i]["pair_id"]].get("a7_majority")
            if h and pm in ("chosen", "rejected"):
                tot += 1
                ag += h == pm
        p, lo, hi = wilson(ag, tot)
        rec["vs_panel"] = {"agree": ag, "n": tot, "p": p, "lo": lo, "hi": hi}
        out["raters"][f"rater{n}"] = rec

    if len(raters) >= 2:
        x, y = raters[0], raters[1]
        pr = [(side(x, i), side(y, i)) for i in real]
        pr = [(u, v) for u, v in pr if u and v]
        n, po, kp = kappa(pr, ("chosen", "rejected"))
        out["human_human"] = {"preference_decided": {"n": n, "observed": po, "kappa": kp}}
        for f in ("reaches_A", "reaches_B"):
            q = [(norm(R[x][i][f]), norm(R[y][i][f])) for i in real]
            q = [(u, v) for u, v in q if u in ("yes", "no") and v in ("yes", "no")]
            n2, po2, kp2 = kappa(q, ("yes", "no"))
            out["human_human"][f] = {"n": n2, "observed": po2, "kappa": kp2}

        b = c = 0
        for i in real:
            h1, h2 = side(x, i), side(y, i)
            pm = SEL[K[i]["pair_id"]].get("a7_majority")
            if None in (h1, h2) or pm not in ("chosen", "rejected"):
                continue
            if (h1 == pm) and not (h2 == pm):
                b += 1
            elif (h2 == pm) and not (h1 == pm):
                c += 1
        pval = (
            min(1.0, sum(math.comb(b + c, k) for k in range(min(b, c) + 1)) * 2 / 2 ** (b + c))
            if b + c
            else float("nan")
        )
        out["panel_closeness_paired_sign_test"] = {"r1_only": b, "r2_only": c, "p_two_sided": pval}

    out["notes_written"] = {
        f"rater{n}": sum(1 for i in K if norm(R[n][i].get("notes"))) for n in raters
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for n in raters:
        r = out["raters"][f"rater{n}"]
        print(
            f"  rater{n}: QC {r['quality_control']['correct']}/{r['quality_control']['n']}  "
            f"P(A)={r['slot_bias']['p_A']:.1%}  vs_rule {r['vs_rule']['p']:.1%}  "
            f"vs_panel {r['vs_panel']['p']:.1%} (n={r['vs_panel']['n']})"
        )
    if "human_human" in out:
        h = out["human_human"]
        print(
            f"  human-human: preference kappa {h['preference_decided']['kappa']:+.3f} "
            f"(n={h['preference_decided']['n']}), reaches_A kappa {h['reaches_A']['kappa']:+.3f}, "
            f"reaches_B kappa {h['reaches_B']['kappa']:+.3f}"
        )
        print(
            f"  paired sign test on panel closeness: p={out['panel_closeness_paired_sign_test']['p_two_sided']:.4f}"
        )
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
