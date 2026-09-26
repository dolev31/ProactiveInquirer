"""Per-task depth->=2 coverage: each label variant against the prompted base, StrategyQA test.

Reproduces the deltas in RESULT.md sections 6-9 and computes the discordant-set INTERSECTION,
which decides whether three equal p-values are three findings or one.

THE BASE MUST BE PINNED. arm_id='inquirer_prompted' alone spans 16 pins on this suite: five
different base models (qwen3-8b, qwen3-4b, granite33-8b, gpt-oss-120b) across five budget
ceilings (cap 4/8/12/16/24), whose mean asks run from 3.42 to 14.72. Pooling them is not a
base. The comparator here is the one pin sharing the arms' grid and code_version.

Run: .venv/bin/python scripts/strategyqa_gold_audit/depth_discordance.py
"""

import itertools
from math import comb

import duckdb

SNAP = "artifacts/horizontal_axis_instrument_20260919/snapshot_20260919/main_scores_parquet"
R, S = f"{SNAP}/runs.parquet", f"{SNAP}/scores.parquet"
CV = "107ef2221a55"  # the arms' code_version
GRID = "tier1_trained_qa_base"
BASE = "d63d85f1c5be4755"  # qwen3-8b-base, same grid + code_version, 6.00 asks
ARMS = {
    "312fbbbed3836843": "control",
    "98f60f4e1310d7ca": "rater",
    "df0f97fc99afb84c": "reaches",
    "34819b5800b85037": "stopcontrast",
}
c = duckdb.connect()


def per_task(pin):
    """depth->=2 coverage per task = sum(hit)/sum(gold n) over depth buckets 2..4, seeds averaged."""
    q = f"""
    with rr as (select run_id, task_id, seed from '{R}'
                where suite_id='strategyqa' and split='test' and status='ok'
                  and model_pin_hash like '{pin}%' and code_version like '{CV}%' and grid_name='{GRID}'),
    hits as (select rr.task_id, rr.seed,
               sum(cv.value * cn.value) num, sum(cn.value) den
             from rr
             join '{S}' cv on cv.run_id=rr.run_id and cv.metric_name in ('cad#2','cad#3','cad#4')
             join '{S}' cn on cn.run_id=rr.run_id
                          and cn.metric_name='cad_n#'||substr(cv.metric_name,5)
             group by 1,2)
    select task_id, avg(num/den) v from hits where den>0 group by 1"""
    return {t: v for t, v in c.execute(q).fetchall()}


base = per_task(BASE)
arms = {lbl: per_task(p) for p, lbl in ARMS.items()}
tasks = sorted(set(base) & set.intersection(*[set(a) for a in arms.values()]))
print(f"base pin {BASE} (qwen3-8b-base), {len(base)} tasks with a depth->=2 row")
print(f"tasks defined for the base and all four arms: {len(tasks)}\n")


def sign_p(pos, neg):
    n, k = pos + neg, min(pos, neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0


disc = {}
print(f"{'arm':<14}{'delta':>9}{'ties':>6}{'disc':>6}{'pos':>5}{'neg':>5}{'sign p':>9}")
for lbl, a in arms.items():
    d = [a[t] - base[t] for t in tasks]
    pos = sum(1 for x in d if x > 1e-12)
    neg = sum(1 for x in d if x < -1e-12)
    disc[lbl] = {t for t in tasks if abs(a[t] - base[t]) > 1e-12}
    print(
        f"{lbl:<14}{sum(d) / len(d):>+9.4f}{len(d) - pos - neg:>6}{pos + neg:>6}{pos:>5}{neg:>5}{sign_p(pos, neg):>9.4f}"
    )

print("\n=== are the three low-ask arms' discordant sets the SAME tasks? ===")
low = ["control", "rater", "reaches"]
inter = set.intersection(*[disc[lbl] for lbl in low])
union = set.union(*[disc[lbl] for lbl in low])
print(
    "  pairwise: "
    + ", ".join(f"{a}&{b}={len(disc[a] & disc[b])}" for a, b in itertools.combinations(low, 2))
)
print(
    f"  three-way intersection {len(inter)}, union {len(union)}, "
    f"sizes {[len(disc[lbl]) for lbl in low]}, identical: {all(disc[lbl] == disc[low[0]] for lbl in low)}"
)
agree = sum(1 for t in inter if len({(arms[lbl][t] - base[t]) > 0 for lbl in low}) == 1)
print(f"  of the {len(inter)} shared tasks, all three agree in DIRECTION on {agree}")
print(
    f"  stopcontrast: {len(disc['stopcontrast'] & inter)} shared with that intersection, "
    f"{len(disc['stopcontrast'] - inter)} unique"
)

print("\n=== ask counts, each labelled with its population ===")
lab = dict(ARMS)
lab[BASE] = "base"
tl = "','".join(tasks)
for pop, extra in (
    ("all 200 test tasks", ""),
    (f"the {len(tasks)} depth-defined tasks", f" and task_id in ('{tl}')"),
):
    q = f"""select substr(model_pin_hash,1,16) k, avg(n_asks) a, count(*) n from '{R}'
      where suite_id='strategyqa' and split='test' and status='ok' and code_version like '{CV}%'
        and grid_name='{GRID}' and substr(model_pin_hash,1,16) in ('{"','".join(list(ARMS) + [BASE])}'){extra}
      group by 1"""
    print(f"  over {pop}:")
    for k, a, n in sorted(c.execute(q).fetchall(), key=lambda r: r[1]):
        print(f"    {lab[k]:<14} {a:.2f} asks   (n={n} runs)")

print("\n=== the common core: signs on the tasks discordant for ALL FOUR arms ===")
core = set.intersection(*[disc[lbl] for lbl in list(ARMS.values())])
print(f"  tasks discordant for all four arms: {len(core)}")
for lbl in list(ARMS.values()):
    pos = sum(1 for t in core if arms[lbl][t] - base[t] > 0)
    print(
        f"    {lbl:<14} {pos} favour the arm / {len(core) - pos} favour the base"
        f"   sign p = {sign_p(pos, len(core) - pos):.4f}"
    )
print("  (the same tasks, so this IS a paired comparison between the arms)")
