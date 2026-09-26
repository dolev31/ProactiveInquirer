"""Render the seedrep4 and P2 records into Markdown tables and LaTeX rows, so no number is typed by
hand. Every printed value is read from the JSON the reader wrote.

  render.py p2   artifacts/seedrep4_20260924/p2_rows.json
  render.py four artifacts/seedrep4_20260924/four_seed_table.json
"""

from __future__ import annotations

import json
import sys

SUITES = ("musique", "strategyqa", "wiki2")
LABEL = {"musique": "MuSiQue", "strategyqa": "StrategyQA", "wiki2": "2WikiMultiHopQA"}


def f(x: float, d: int = 4) -> str:
    """The paper's convention: round the float's shortest decimal repr, half-up (away from zero) on
    an exact-decimal tie. `format(x, '.4f')` would instead round the binary value, which sends
    0.1025 to 0.102 at three decimals because the float sits just below the decimal."""
    from decimal import ROUND_HALF_UP, Decimal

    q = Decimal(repr(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    if q == 0:  # a float-noise zero such as -3.8e-18 prints as an unsigned zero, never "-0.000"
        return f"{Decimal(0):.{d}f}"
    return f"{q:+.{d}f}"


def fpy(x: float, d: int = 4) -> str:
    """Python's binary rounding, kept only to show where it differs from `f` on a tie."""
    return f"{x:+.{d}f}"


def iv(c: dict, d: int = 4) -> str:
    return f"[{f(c['ci_lo'], d)}, {f(c['ci_hi'], d)}]"


def p2(path: str) -> None:
    r = json.load(open(path))
    print(f"scorer_hash {r['scorer_hash']}\n")
    print("| row | reading | " + " | ".join(LABEL[s] for s in SUITES) + " |")
    print("|---|---|" + "---|" * len(SUITES))
    for lab in ("P2b", "P2a"):
        row = r["rows"][lab]
        for reading in ("matched_cost", "symmetric"):
            cells = []
            for s in SUITES:
                c = row["cells"][s] if reading == "matched_cost" else row["cells"][s]["symmetric"]
                st = row["stability"][s]["verdict"] if reading == "matched_cost" else ""
                cells.append(
                    f"{f(c['delta'])} {iv(c)} n={c['n_tasks']}" + (f" ({st})" if st else "")
                )
            mark = " (printed)" if row["printed_reading"] == reading else ""
            print(f"| {lab} `{row['model']}` | {reading}{mark} | " + " | ".join(cells) + " |")
    print("\nasks (checkpoint mean k / comparator mean asks / comparator shorter than k):")
    for lab in ("P2b", "P2a"):
        row = r["rows"][lab]
        print(
            f"  {lab}: "
            + "; ".join(
                f"{s} {row['cells'][s]['trained_mean_k']:.3f} / {row['cells'][s]['baseline_mean_n_asks']:.3f} / "
                f"{row['cells'][s]['n_baseline_shorter_than_k']}"
                for s in SUITES
            )
            + f"; dagger rule applies: {row['dagger_rule_applies']}"
        )
    print("\ncap-8 coverage delta (unmatched, recorded not gated):")
    for lab in ("P2b", "P2a"):
        row = r["rows"][lab]
        print(f"  {lab}: " + " / ".join(f(row["cells"][s]["cap8_coverage_delta"]) for s in SUITES))
    a = r["agreement_hc"]
    print(f"\nagreement of the two headline-control identities: cell_for_cell={a['cell_for_cell']}")
    for s in SUITES:
        x = a[s]
        print(
            f"  {s}: new {f(x['new_3ae099d'][0])} [{f(x['new_3ae099d'][1])}, {f(x['new_3ae099d'][2])}]"
            f" old {f(x['old_107ef222'][0])} [{f(x['old_107ef222'][1])}, {f(x['old_107ef222'][2])}]"
            f" same_sign={x['same_sign']} new_in_old={x['new_inside_old_interval']}"
            f" old_in_new={x['old_inside_new_interval']} agree={x['agree']}"
        )
    print("\nLaTeX rows (the table's four-decimal format):")
    for lab, name in (
        (
            "P2b",
            "question pairs \\newline {\\scriptsize\\color{black!60} question pairs, from the reference}",
        ),
        (
            "P2a",
            "question pairs, from the recipe's start \\newline {\\scriptsize\\color{black!60} "
            "question pairs, from the headline imitation checkpoint}",
        ),
    ):
        row = r["rows"][lab]
        key = row["printed_reading"]
        cs = [
            row["cells"][s] if key == "matched_cost" else row["cells"][s]["symmetric"]
            for s in SUITES
        ]
        print(f"{name} & " + " & ".join(f"${f(c['delta'])}$" for c in cs) + " \\\\")
        print(
            " & "
            + " & ".join(f"{{\\scriptsize $[{f(c['ci_lo'])}, {f(c['ci_hi'])}]$}}" for c in cs)
            + " \\\\[2pt]"
        )


def four(path: str) -> None:
    r = json.load(open(path))
    t = r["table1"]
    seeds = r["seeds"]
    print(
        "| metric::suite | "
        + " | ".join(seeds)
        + " | sign +/4 | CI>0 | between-seed SD | range | mean boot SE | crude t (df 3) | 4-seed pooled (10k) |"
    )
    print("|---|" + "---|" * (len(seeds) + 7))
    for key, cells in t["cells"].items():
        sm = t["summary"][key]
        pool = next(x for x in t["pooled4"][key] if x["n_boot"] == 10000)
        ct = sm["crude_t_interval"]
        print(
            f"| {key} | "
            + " | ".join(
                f"{f(cells[s]['delta'])} {iv(cells[s])}"
                + ("" if cells[s]["stability"]["verdict"] == "as computed" else " UNDECIDED")
                for s in seeds
            )
            + f" | {sm['sign_count_positive']}/{sm['n_seeds']} | {sm['intervals_above_zero']}"
            f" | {sm['between_seed_sd']:.4f} | {sm['range']:.4f} | {sm['mean_within_seed_bootstrap_se']:.4f}"
            f" | {f(ct['mean'])} [{f(ct['lo'])}, {f(ct['hi'])}] | {f(pool['delta'])} {iv(pool)} |"
        )
    print("\npairwise (evidence_coverage, symmetric, row minus column seed):")
    for suite, pairs in t["pairwise"].items():
        print(
            f"  {suite}: "
            + "; ".join(
                f"{k} {f(v['delta'])} {iv(v)}"
                + ("" if v["stability"]["verdict"] == "as computed" else " UNDECIDED")
                for k, v in pairs.items()
            )
        )
    for part in ("cap8", "teacher", "ownstop"):
        if part not in r:
            continue
        print(f"\n{part}:")
        for s in SUITES:
            cells = r[part]["cells"][s]
            sm = r[part]["summary"][s]
            pooled = cells.get("pooled4") or next(
                x for x in r[part]["pooled4"][s] if x["n_boot"] == 10000
            )
            print(
                f"  {s}: "
                + "; ".join(f"{k} {f(cells[k]['delta'])} {iv(cells[k])}" for k in seeds)
                + f"; sign +{sm['sign_count_positive']}/{sm['n_seeds']}; SD {sm['between_seed_sd']:.4f};"
                f" range {sm['range']:.4f}; pooled4 {f(pooled['delta'])} {iv(pooled)}"
            )


def tie(x: float, d: int) -> bool:
    """True when the shortest decimal repr of x sits exactly halfway at d decimals (e.g. 0.07375 at
    4): its printed rounding then depends on the float's binary value, not on the decimal."""
    from decimal import Decimal

    q = Decimal(repr(x)).scaleb(d)
    return abs(q - q.to_integral_value(rounding="ROUND_FLOOR")) == Decimal("0.5")


def p2prec(path: str) -> None:
    """Every P2 value that is printed, rounded from the JSON's own float, with that float beside it."""
    r = json.load(open(path))
    rows = [
        (f"{lab} {r['rows'][lab]['model']}", r["rows"][lab]["cells"]) for lab in ("P2b", "P2a")
    ] + [("hc107 (agreement only, not printed)", None)]
    print(
        "| row | suite | field | full-precision float | 4 dp (half-up) | 3 dp (half-up) "
        "| tie at 4 dp | tie at 3 dp |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for name, cells in rows:
        for s in SUITES:
            if cells is None:
                x = r["agreement_hc"][s]["old_107ef222"]
                src = dict(zip(("delta", "ci_lo", "ci_hi"), x))
            else:
                src = {k: cells[s][k] for k in ("delta", "ci_lo", "ci_hi", "cap8_coverage_delta")}
                src |= {
                    f"symmetric.{k}": cells[s]["symmetric"][k] for k in ("delta", "ci_lo", "ci_hi")
                }
            for k, v in src.items():
                t4 = f"TIE (python {fpy(v, 4)})" if tie(v, 4) else ""
                t3 = f"TIE (python {fpy(v, 3)})" if tie(v, 3) else ""
                print(f"| {name} | {s} | {k} | {v!r} | {f(v, 4)} | {f(v, 3)} | {t4} | {t3} |")


SEED_LABEL = {"s1": "seed 1", "s2": "seed 2", "s0fresh": "seed 0 (fresh)", "s3": "seed 3"}
METRIC_LABEL = {
    "evidence_coverage": "required-evidence coverage",
    "facet_breadth_scorer": "breadth",
    "dwr": "depth-weighted recall",
    "max_depth_reached": "deepest need resolved",
    "precedence_violation_rate": "out-of-order rate",
}


def fourprec(path: str) -> None:
    """Every four-seed value a table or sentence may print, with its float and tie flags."""
    r = json.load(open(path))
    t = r["table1"]
    print("| quantity | full-precision float | 4 dp (half-up) | 3 dp (half-up) | ties |")
    print("|---|---|---|---|---|")

    def row(name: str, v: float) -> None:
        ties = ", ".join(f"{d} dp (python {fpy(v, d)})" for d in (4, 3) if tie(v, d))
        print(f"| {name} | {v!r} | {f(v, 4)} | {f(v, 3)} | {ties} |")

    for key, cells in t["cells"].items():
        for sd in r["seeds"]:
            for k in ("delta", "ci_lo", "ci_hi"):
                row(f"table1 {key} {sd} {k}", cells[sd][k])
        sm = t["summary"][key]
        for k in ("between_seed_sd", "range", "mean_within_seed_bootstrap_se"):
            row(f"table1 {key} {k}", sm[k])
        for k in ("mean", "lo", "hi"):
            row(f"table1 {key} crude_t.{k}", sm["crude_t_interval"][k])
        pool = next(x for x in t["pooled4"][key] if x["n_boot"] == 10000)
        for k in ("delta", "ci_lo", "ci_hi"):
            row(f"table1 {key} pooled4 {k}", pool[k])
    for suite, pairs in t["pairwise"].items():
        for pk, v in pairs.items():
            for k in ("delta", "ci_lo", "ci_hi"):
                row(f"pairwise evidence_coverage::{suite} {pk} {k}", v[k])
    for part in ("cap8", "teacher", "ownstop"):
        for s in SUITES:
            cells = r[part]["cells"][s]
            for sd in r["seeds"]:
                for k in ("delta", "ci_lo", "ci_hi"):
                    row(f"{part} {s} {sd} {k}", cells[sd][k])
            pooled = cells.get("pooled4") or next(
                x for x in r[part]["pooled4"][s] if x["n_boot"] == 10000
            )
            for k in ("delta", "ci_lo", "ci_hi"):
                row(f"{part} {s} pooled4 {k}", pooled[k])
            for k in ("between_seed_sd", "range"):
                row(f"{part} {s} {k}", r[part]["summary"][s][k])


def fourtex(path: str) -> None:
    """The appendix table: every Table 1 metric per training seed, three decimals, half-up."""
    r = json.load(open(path))
    t = r["table1"]
    seeds = r["seeds"]
    print("\\begin{tabular}{@{}ll" + "c" * len(seeds) + "cccc@{}}")
    print("\\toprule")
    print(
        " & & "
        + " & ".join(SEED_LABEL[x] for x in seeds)
        + " & positive & between-seed SD & bootstrap SE & crude interval \\\\"
    )
    print("\\midrule")
    for metric, mlabel in METRIC_LABEL.items():
        for i, s in enumerate(SUITES):
            key = f"{metric}::{s}"
            cells, sm = t["cells"][key], t["summary"][key]
            ct = sm["crude_t_interval"]
            pts = " & ".join(
                f"${f(cells[x]['delta'], 3)}$"
                + ("" if cells[x]["stability"]["verdict"] == "as computed" else "$^{\\circ}$")
                for x in seeds
            )
            ivs = " & ".join(
                f"{{\\scriptsize $[{f(cells[x]['ci_lo'], 3)}, {f(cells[x]['ci_hi'], 3)}]$}}"
                for x in seeds
            )
            lab = mlabel if i == 0 else ""
            ns = sorted({cells[x]["n"] for x in seeds})
            n = f"{ns[0]}" if len(ns) == 1 else f"{ns[0]}$ to ${ns[-1]}"
            print(
                f"{lab} & {LABEL[s]} {{\\scriptsize $n={n}$}} & {pts} & "
                f"{sm['sign_count_positive']} of {sm['n_seeds']} & {f(sm['between_seed_sd'], 3)[1:]} & "
                f"{f(sm['mean_within_seed_bootstrap_se'], 3)[1:]} & "
                f"$[{f(ct['lo'], 3)}, {f(ct['hi'], 3)}]$ \\\\"
            )
            print(f" & & {ivs} & & & & \\\\[1pt]")
        print("\\midrule" if metric != list(METRIC_LABEL)[-1] else "\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    {"p2": p2, "four": four, "p2prec": p2prec, "fourprec": fourprec, "fourtex": fourtex}[
        sys.argv[1]
    ](sys.argv[2])
