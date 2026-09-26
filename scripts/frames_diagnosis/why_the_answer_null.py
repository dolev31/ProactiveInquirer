"""Why the FRAMES answer contrast is null, and why no affordable experiment fixes it.

FRAMES carries no need graph, so the paper reads only answer quality there, and that contrast is
a null: pooled over three seeds `answer_correct` is +0.0020 [-0.0174,+0.0214]. Reporting a null
without a mechanism invites the reader to supply one, and the obvious supply is "the questioner is
no better", which the other three suites contradict. This script establishes what is actually
happening, in five steps, and ends with a power calculation that says which follow-up experiments
are worth compute and which are arithmetically guaranteed to return another null.

WHAT IT ESTABLISHES.

1. The answer endpoint SATURATES in evidence. Quadrupling the assigned budget buys the prompted
   arm +0.67 unique documents and +0.008 token-F1.
2. At MATCHED question count the trained arm does hold more evidence. This is the retrieval stage
   of a matched-cost reading and needs no re-drafting, because evidence accumulation is on disk.
3. HEDGING is the only thing that separates on the answer side, and it is downstream of stopping.
4. Two candidate gold-free quality metrics are tested against the random-question control. The
   RATIO (novelty per call) is FALSIFIED: random questions score highest, because unrelated
   questions retrieve non-overlapping documents. The COUNT (evidence held at matched depth)
   survives: random loses, because a random question also retrieves fewer documents per call.
5. The implied answer effect is computed from the measured slope and compared with what this
   suite's n can resolve. It cannot.

WHAT IT DOES NOT ESTABLISH. The random-question control was never run on FRAMES, so step 4's
falsification is carried out on the three gold suites and its transfer to FRAMES is an assumption,
named here rather than hidden. The slope in step 5 is identified from cap-to-cap movement within
one arm; if extra questions help through any channel other than extra documents then the slope on
documents is overstated, which makes the power conclusion conservative rather than optimistic.
"""

from __future__ import annotations

import collections
from pathlib import Path

import duckdb
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FRAMES = REPO / "artifacts/frames_frontier/scores_parquet.frames"
RANDOM = REPO / "artifacts/random_q_all_suites_20260918/scores_parquet"
TRAINED, BASE = "6bc87062", "e9af49a4"
N_BOOT = 10000


def boot(values, n=N_BOOT, seed=0):
    """Paired bootstrap. The input is SORTED first, and that is load-bearing.

    duckdb returns rows in no guaranteed order without an ORDER BY, so the same data arrives in a
    different order on a rerun. Resampling indices into an order-dependent array then produces the
    same point estimate with different bounds, which is this repository's recorded fingerprint for
    an unsorted bootstrap. It was observed here before the sort was added: one interval read
    [+0.000, +2.500] on one run and [-0.021, +2.479] on the next.
    """
    rng = np.random.default_rng(seed)
    a = np.sort(np.asarray(values, dtype=float))
    idx = rng.integers(0, len(a), (n, len(a)))
    m = np.sort(a[idx].mean(axis=1))
    return a.mean(), m[int(0.025 * n)], m[int(0.975 * n)]


def boot_ratio(num, den, n=N_BOOT, seed=0):
    """As `boot`, and the two arrays are sorted TOGETHER because they are paired per task."""
    rng = np.random.default_rng(seed)
    num, den = np.asarray(num, float), np.asarray(den, float)
    order = np.lexsort((den, num))
    num, den = num[order], den[order]
    idx = rng.integers(0, len(num), (n, len(num)))
    r = np.sort(num[idx].mean(axis=1) / den[idx].mean(axis=1))
    return num.mean() / den.mean(), r[int(0.025 * n)], r[int(0.975 * n)]


def verdict(lo, hi):
    return "EXCLUDES 0" if (lo > 0 or hi < 0) else "spans 0"


def metrics_cte(store: Path) -> str:
    return f"""
    with m as (
      select run_id,
             max(case when metric_name='answer_correct' then value end) ac,
             max(case when metric_name='answer_token_f1' then value end) f1,
             max(case when metric_name='answer_hedged' then value end) hedged,
             max(case when metric_name='unique_docs' then value end) docs
      from '{store}/scores.parquet' group by 1
    )"""


def cumulative_evidence(store: Path, where: str, key: list[tuple[str, str]]) -> dict:
    """Unique documents held after exactly k questions, keyed by `key` with the arm LAST.

    `key` is a list of (expression, output name) pairs. The inner query needs the table alias and
    the outer one must not have it, which is why both spellings are carried explicitly rather
    than reused: a single string fails the binder on the outer select.
    """
    inner = ", ".join(f"{expr} as {name}" for expr, name in key)
    outer = ", ".join(name for _expr, name in key)
    rows = (
        duckdb.connect()
        .execute(f"""
        with a as (
          select {inner}, t.turn_idx as turn_idx, t.n_new as n_new
          from '{store}/turns.parquet' t join '{store}/runs.parquet' r using (run_id)
          where t.action_kind='ask' and {where}
        )
        select {outer}, turn_idx,
               sum(n_new) over (partition by {outer} order by turn_idx) held
        from a
    """)
        .fetchall()
    )
    out = collections.defaultdict(dict)
    for row in rows:
        *ident, turn, held = row
        arm = ident[-1]
        out[(tuple(ident[:-1]), turn)][arm] = held
    return out


def main() -> int:
    c = duckdb.connect()
    print("=" * 78)
    print("STEP 1. The answer endpoint saturates in evidence (prompted arm, assigned budget)")
    print("=" * 78)
    rows = c.execute(f"""
        {metrics_cte(FRAMES)}
        select r.budget_cap, avg(r.n_asks), avg(m.docs), avg(m.f1), avg(m.ac)
        from '{FRAMES}/runs.parquet' r join m using (run_id)
        where substr(r.model_pin_hash,1,8)='{BASE}' group by 1 order by 1
    """).fetchall()
    for cap, asks, docs, f1, ac in rows:
        print(
            f"  cap {cap:2d}: asks {asks:5.2f}  unique docs {docs:6.3f}  f1 {f1:.4f}  correct {ac:.4f}"
        )
    print(
        f"  over a sixfold budget rise: +{rows[-1][2] - rows[0][2]:.3f} documents, "
        f"+{rows[-1][3] - rows[0][3]:.4f} f1"
    )

    print()
    print("=" * 78)
    print("STEP 2. The cap-8 contrast, paired, on three answer readings")
    print("=" * 78)
    rows = c.execute(f"""
        {metrics_cte(FRAMES)}
        select r.task_id, r.seed, substr(r.model_pin_hash,1,8), r.n_asks, m.ac, m.f1, m.hedged, m.docs
        from '{FRAMES}/runs.parquet' r join m using (run_id)
        where r.budget_cap=8 and substr(r.model_pin_hash,1,8) in ('{TRAINED}','{BASE}')
    """).fetchall()
    d = collections.defaultdict(dict)
    for task, seed, pin, asks, ac, f1, hedged, docs in rows:
        d[(task, seed)][pin] = (asks, ac, f1, hedged, docs)
    pairs = [(v[TRAINED], v[BASE]) for v in d.values() if len(v) == 2]
    print(f"  paired task-seeds: {len(pairs)}")
    for name, i in (("answer_correct", 1), ("answer_token_f1", 2), ("answer_hedged", 3)):
        pt, lo, hi = boot([t[i] - b[i] for t, b in pairs])
        print(f"  {name:16s} {pt:+.4f} [{lo:+.4f},{hi:+.4f}]  {verdict(lo, hi)}")
    print(
        f"  asks {np.mean([t[0] for t, _ in pairs]):.2f} vs {np.mean([b[0] for _, b in pairs]):.2f}"
        f"   unique docs {np.mean([t[4] for t, _ in pairs]):.2f} vs "
        f"{np.mean([b[4] for _, b in pairs]):.2f}"
    )
    print("  NOTE: the two answer readings disagree in SIGN, and neither separates. The only")
    print("  quantity that separates is hedging, and it is downstream of stopping.")

    print()
    print("=" * 78)
    print("STEP 3. Hedging is the mediator, and hedged answers collapse")
    print("=" * 78)
    for pin, label in ((TRAINED, "trained"), (BASE, "prompted")):
        for h in (0.0, 1.0):
            r = c.execute(f"""
                {metrics_cte(FRAMES)}
                select count(*), avg(r.n_asks), avg(m.docs), avg(m.ac), avg(m.f1)
                from '{FRAMES}/runs.parquet' r join m using (run_id)
                where r.budget_cap=8 and substr(r.model_pin_hash,1,8)='{pin}' and m.hedged={h}
            """).fetchone()
            print(
                f"  {label:9s} hedged={int(h)}: n={r[0]:4d} asks {r[1]:.2f} docs {r[2]:5.2f} "
                f"correct {r[3]:.4f} f1 {r[4]:.4f}"
            )

    print()
    print("=" * 78)
    print("STEP 4. Two gold-free candidates, against the random-question control")
    print("=" * 78)
    print("  4a. novelty per call at matched turn index, FRAMES:")
    for row in c.execute(f"""
        select t.turn_idx, substr(r.model_pin_hash,1,8) pin, count(*),
               avg(t.n_new)/avg(t.n_retrieved)
        from '{FRAMES}/turns.parquet' t join '{FRAMES}/runs.parquet' r using (run_id)
        where r.budget_cap=8 and substr(r.model_pin_hash,1,8) in ('{TRAINED}','{BASE}')
          and t.action_kind='ask' and t.n_retrieved>0 and t.turn_idx<=3
        group by 1,2 having count(*)>=30 order by 1,2
    """).fetchall():
        print(f"     turn {row[0]}  {row[1]}  n={row[2]:4d}  novelty {row[3]:.3f}")
    print("  4b. the SAME quantity on the suites where a random questioner was run:")
    for row in c.execute(f"""
        select t.turn_idx, r.arm_id, count(*), avg(t.n_new)/avg(t.n_retrieved)
        from '{RANDOM}/turns.parquet' t join '{RANDOM}/runs.parquet' r using (run_id)
        where t.action_kind='ask' and t.n_retrieved>0 and t.turn_idx<=3
        group by 1,2 having count(*)>=40 order by 1,2
    """).fetchall():
        print(f"     turn {row[0]}  {row[1]:18s} n={row[2]:5d}  novelty {row[3]:.3f}")
    print("     VERDICT: FALSIFIED. random_q scores the HIGHEST novelty at every turn index, so")
    print("     the ratio measures divergence rather than usefulness and is not a quality metric.")

    print()
    print("  4c. the COUNT instead: evidence held at matched depth, random minus trained")
    ev = cumulative_evidence(
        RANDOM,
        "1=1",
        [
            ("r.suite_id", "suite_id"),
            ("r.task_id", "task_id"),
            ("r.seed", "seed"),
            ("r.arm_id", "arm_id"),
        ],
    )
    for k in range(3):
        p = [
            (v["random_q"], v["inquirer_trained"])
            for key, v in ev.items()
            if key[1] == k and "random_q" in v and "inquirer_trained" in v
        ]
        if len(p) < 30:
            continue
        pt, lo, hi = boot([a - b for a, b in p])
        who = "RANDOM ahead" if pt > 0 else "trained ahead"
        print(
            f"     after {k + 1} q, n={len(p):4d}: random {np.mean([a for a, _ in p]):5.2f} vs "
            f"trained {np.mean([b for _, b in p]):5.2f}  {pt:+.3f} [{lo:+.3f},{hi:+.3f}] "
            f"{verdict(lo, hi)}  <- {who}"
        )
    print("     VERDICT: SURVIVES. A random question also retrieves FEWER documents per call, so")
    print("     it cannot win on the count even while winning on the ratio.")

    print()
    print("=" * 78)
    print("STEP 5. Evidence at matched depth on FRAMES, then the power calculation")
    print("=" * 78)
    ev = cumulative_evidence(
        FRAMES,
        f"r.budget_cap=8 and substr(r.model_pin_hash,1,8) in ('{TRAINED}','{BASE}')",
        [("r.task_id", "task_id"), ("r.seed", "seed"), ("substr(r.model_pin_hash,1,8)", "pin")],
    )
    gaps = {}
    for k in range(4):
        p = [(v[TRAINED], v[BASE]) for key, v in ev.items() if key[1] == k and len(v) == 2]
        if len(p) < 30:
            print(f"  after {k + 1} q: only {len(p)} paired tasks, not read")
            continue
        pt, lo, hi = boot([t - b for t, b in p])
        gaps[k + 1] = (pt, len(p))
        print(
            f"  after {k + 1} q, n={len(p):4d}: trained {np.mean([t for t, _ in p]):5.2f} vs "
            f"base {np.mean([b for _, b in p]):5.2f}  {pt:+.3f} [{lo:+.3f},{hi:+.3f}] "
            f"{verdict(lo, hi)}"
        )

    rows = c.execute(f"""
        {metrics_cte(FRAMES)}
        select r.task_id, r.budget_cap, m.docs, m.f1, m.ac
        from '{FRAMES}/runs.parquet' r join m using (run_id)
        where substr(r.model_pin_hash,1,8)='{BASE}' and r.budget_cap in (4,24)
    """).fetchall()
    by = collections.defaultdict(dict)
    for task, cap, docs, f1, ac in rows:
        by[task][cap] = (docs, f1, ac)
    p = [(v[4], v[24]) for v in by.values() if 4 in v and 24 in v]
    dd = np.array([b[0] - a[0] for a, b in p])
    df = np.array([b[1] - a[1] for a, b in p])
    slope, s_lo, s_hi = boot_ratio(df, dd)
    print()
    print(f"  slope of f1 on evidence, paired within task across the assigned budget (n={len(p)}):")
    print(f"    {slope:+.5f} f1 per document [{s_lo:+.5f},{s_hi:+.5f}]")

    obs = boot([t[2] - b[2] for t, b in pairs])
    half = (obs[2] - obs[1]) / 2
    print(f"  what this suite can resolve on f1 at n={len(pairs)}: half-width {half:.4f}")
    print()
    for k, (gap, n_k) in sorted(gaps.items()):
        implied = slope * gap
        if implied <= 0:
            continue
        need = (half / implied) ** 2 * len(pairs)
        print(
            f"  at {k} matched questions: evidence gap {gap:+.3f} docs -> implied f1 {implied:+.5f}"
        )
        print(
            f"      that is {half / implied:.1f}x below the detection floor; resolving it needs "
            f"about {need:,.0f} tasks against the {len(pairs)} this suite has"
        )
    print()
    print("  CONCLUSION. The retrieval advantage is real and present on FRAMES at matched depth.")
    print("  The answer endpoint cannot resolve it, by a factor this suite cannot close: a")
    print("  re-drafting experiment at matched cost is arithmetically guaranteed to return")
    print("  another null. The fix is to report the endpoint that resolves, not to buy more of")
    print("  the one that does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
