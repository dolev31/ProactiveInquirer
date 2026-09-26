"""Per-population reasoning-token shares and dollar-overstatement factors.

Ground truth for what "reasoning-token share" even means comes from two places, not from
the appendix (grepped for "1.3925", "1.1523", "reasoning.*share", "multiplier" across every
committed `.tex`/`.md`/`.json`/`.py` file on 2026-09-20: zero literal matches for either
figure or for a stated share formula):

  - `src/pinq_adapters/llm/pricing.py`, `PriceTable.usd()` docstring: measured, on the store
    as it stood 2026-09-19, "openai/aws/gpt-oss-120b ... a factor of 1.3924 over 2,668,114
    calls" between the pre-fix (double-charged) and post-fix (single-charged) dollar total.
  - `tests/test_reasoning_tokens_are_a_subset_of_completion.py` docstring: the same
    mechanism, plus a smaller-scale figure (1.2952x) on 408 rescued tau2 units.

Both of those are a DOLLAR OVERSTATEMENT FACTOR: usd_old / usd_new, where usd_old is the
pre-fix formula (reasoning tokens billed on top of the full completion charge) and usd_new
is the current formula in `PriceTable.usd()` (reasoning tokens are a subset of completion,
billed once). This script recomputes exactly that ratio, disaggregated per population,
directly from `scores/parquet/calls.parquet` -- read-only, no `pi compact`/`pi score`, no
store lock taken.

It ALSO reports a second, different quantity the task asked for by name -- the raw
TOKEN-COUNT share `sum(tok_reasoning) / sum(tok_completion)` -- because the two are not
the same thing and an appendix reader could mean either. `tok_reasoning` nests inside
`tok_completion` (confirmed by `pricing.py` and the test docstring above, following the
OpenAI usage schema's `completion_tokens_details.reasoning_tokens`), the same way
`tok_cached` nests inside `tok_prompt`; a "share" is therefore a fraction of the
completion, not an addition to it.

Verified against `calls.parquet`'s own recorded `usd` column: for openai/aws/gpt-oss-120b,
SUM(usd) as currently stored is $1913.300592, matching this script's `usd_old` (pre-fix,
double-charged) to the cent. The store's recorded `usd` has NOT been retroactively
corrected for the pricing fix -- it is still the buggy pre-fix value. That is why a
recomputation is needed at all: reading `calls.parquet.usd` directly would silently
reproduce the bug's own overstatement, in the shape of a real column.

Rates are read directly from `scripts/price_tables/2026-09.json` (the current table;
`2026-08.json`'s `aws/gpt-oss-120b` and `openai/aws/gpt-oss-120b` entries are byte-identical
to 2026-09's, confirmed by diff) rather than through `PriceTable.rates()`, because that
method is `strict=True` by default and raises `LLMConfigError` for
`openai/gcp/gemini-3.7-flash`, which has NO entry in either committed price table (checked
both). That model's calls carry a nonzero recorded `usd` ($4.312468 total, 948 calls)
despite this, so some other, uncommitted rate priced them historically; this script cannot
recompute a dollar factor for that population and says so explicitly rather than guessing
a rate. Its token-count share is still reported, since that needs no price at all.

Usage:
    PYTHONPATH=$PWD/src .venv/bin/python scripts/cost_reasoning_shares/compute_shares.py
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CALLS_PARQUET = REPO_ROOT / "scores/parquet/calls.parquet"
RUNS_PARQUET = REPO_ROOT / "scores/parquet/runs.parquet"
PRICE_TABLE = REPO_ROOT / "scripts/price_tables/2026-09.json"

# Models that report ANY reasoning tokens anywhere in the store (established by a
# GROUP BY model scan: 46 distinct models total, exactly these two have
# SUM(tok_reasoning) > 0; every other model -- the qwen3/granite/etc. training
# checkpoints -- has tok_reasoning == 0 on every call).
REASONING_BEARING_MODELS = ("openai/aws/gpt-oss-120b", "openai/gcp/gemini-3.7-flash")

# Rates pulled from scripts/price_tables/2026-09.json for the models this script prices.
# openai/gcp/gemini-3.7-flash is deliberately absent: it has no entry in the price table
# (see module docstring) and PRICED_MODELS below is what actually gets a dollar factor.
_price_json = json.loads(PRICE_TABLE.read_text())
_PT_MODELS = _price_json["models"]


def _rates_for(model: str) -> dict[str, float] | None:
    entry = _PT_MODELS.get(model)
    if entry is None:
        return None
    rate_out = float(entry.get("output") or 0.0)
    reasoning = entry.get("reasoning")
    return {
        "rate_in": float(entry.get("input") or 0.0),
        "rate_out": rate_out,
        "rate_cached": float(entry.get("cached_input") or 0.0),
        # PriceTable.usd() falls back to rate_out when the table prices no reasoning rate.
        "rate_reasoning": float(reasoning) if reasoning is not None else rate_out,
    }


PRICED_MODELS = {m: r for m in REASONING_BEARING_MODELS if (r := _rates_for(m)) is not None}
# Also price every zero-reasoning model that has a table entry, purely so the "pooled over
# all models" figure can be reproduced exactly as the whole store would compute it -- for a
# model with tok_reasoning == 0 everywhere, usd_old == usd_new, so including it changes
# nothing about the ratio but proves the ratio was not built by omission.
for _m, _entry in _PT_MODELS.items():
    if _m not in PRICED_MODELS:
        PRICED_MODELS[_m] = _rates_for(_m)  # type: ignore[assignment]


def _rates_values_sql() -> str:
    rows = ", ".join(
        f"('{m}', {r['rate_in']}, {r['rate_out']}, {r['rate_cached']}, {r['rate_reasoning']})"
        for m, r in PRICED_MODELS.items()
    )
    return rows


PER_ROW_CTE = f"""
WITH rates AS (
  SELECT * FROM (VALUES {_rates_values_sql()})
    AS t(model, rate_in, rate_out, rate_cached, rate_reasoning)
),
c AS (
  SELECT c.*, r.rate_in, r.rate_out, r.rate_cached, r.rate_reasoning
  FROM read_parquet('{CALLS_PARQUET.as_posix()}') c
  JOIN rates r ON c.model = r.model
),
per_row AS (
  SELECT
    model, tok_prompt, tok_completion, tok_reasoning, tok_cached,
    rate_in, rate_out, rate_cached, rate_reasoning,
    GREATEST(0, tok_prompt - tok_cached) AS uncached,
    LEAST(GREATEST(0, tok_reasoning), GREATEST(0, tok_completion)) AS reasoning_tok_clamped
  FROM c
)
"""

# usd_old: the pre-fix formula -- completion charged in full, PLUS raw (unclamped)
#          tok_reasoning billed again at rate_reasoning. This is what calls.parquet's own
#          `usd` column still records (verified to the cent for gpt-oss-120b).
# usd_new: the current, fixed PriceTable.usd() formula -- completion charged once; the
#          reasoning subset (clamped to completion, guarding the 17,579 rows where
#          reasoning > completion) is billed at rate_reasoning, the remainder at rate_out.
USD_OLD_EXPR = "uncached*rate_in + tok_cached*rate_cached + tok_completion*rate_out + tok_reasoning*rate_reasoning"
USD_NEW_EXPR = (
    "uncached*rate_in + tok_cached*rate_cached "
    "+ (tok_completion - reasoning_tok_clamped)*rate_out + reasoning_tok_clamped*rate_reasoning"
)


def per_model(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = (
        PER_ROW_CTE
        + f"""
    SELECT model,
           COUNT(*) AS n_calls,
           SUM(tok_completion) AS sum_completion,
           SUM(tok_reasoning) AS sum_reasoning_raw,
           SUM({USD_OLD_EXPR}) AS usd_old,
           SUM({USD_NEW_EXPR}) AS usd_new
    FROM per_row
    GROUP BY model
    """
    )
    df = con.execute(sql).fetchdf()
    df["usd_factor"] = df["usd_old"] / df["usd_new"]
    df["token_share"] = df["sum_reasoning_raw"] / df["sum_completion"]
    return df.sort_values("sum_reasoning_raw", ascending=False).reset_index(drop=True)


def per_suite_arm_split(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Restricted to REASONING_BEARING_MODELS that also have a price entry (gpt-oss-120b)."""
    sql = f"""
    WITH rates AS (
      SELECT * FROM (VALUES ('openai/aws/gpt-oss-120b',
        {PRICED_MODELS["openai/aws/gpt-oss-120b"]["rate_in"]},
        {PRICED_MODELS["openai/aws/gpt-oss-120b"]["rate_out"]},
        {PRICED_MODELS["openai/aws/gpt-oss-120b"]["rate_cached"]},
        {PRICED_MODELS["openai/aws/gpt-oss-120b"]["rate_reasoning"]}))
        AS t(model, rate_in, rate_out, rate_cached, rate_reasoning)
    ),
    c AS (
      SELECT c.*, r.rate_in, r.rate_out, r.rate_cached, r.rate_reasoning,
             run.suite_id, run.arm_id, run.split
      FROM read_parquet('{CALLS_PARQUET.as_posix()}') c
      JOIN rates r ON c.model = r.model
      LEFT JOIN read_parquet('{RUNS_PARQUET.as_posix()}') run ON c.run_id = run.run_id
    ),
    per_row AS (
      SELECT suite_id, arm_id, split, tok_prompt, tok_completion, tok_reasoning, tok_cached,
             rate_in, rate_out, rate_cached, rate_reasoning,
             GREATEST(0, tok_prompt - tok_cached) AS uncached,
             LEAST(GREATEST(0, tok_reasoning), GREATEST(0, tok_completion)) AS reasoning_tok_clamped
      FROM c
    )
    SELECT suite_id, arm_id, split,
           COUNT(*) AS n_calls,
           SUM(tok_completion) AS sum_completion,
           SUM(tok_reasoning) AS sum_reasoning_raw,
           SUM({USD_OLD_EXPR}) AS usd_old,
           SUM({USD_NEW_EXPR}) AS usd_new
    FROM per_row
    GROUP BY suite_id, arm_id, split
    """
    df = con.execute(sql).fetchdf()
    df["usd_factor"] = df["usd_old"] / df["usd_new"]
    df["token_share"] = df["sum_reasoning_raw"] / df["sum_completion"]
    return df.sort_values("sum_reasoning_raw", ascending=False).reset_index(drop=True)


def anomaly_count(con: duckdb.DuckDBPyConnection) -> int:
    sql = f"SELECT COUNT(*) FROM read_parquet('{CALLS_PARQUET.as_posix()}') WHERE tok_reasoning > tok_completion"
    return int(con.execute(sql).fetchone()[0])


def pooled(df_model: pd.DataFrame) -> dict[str, float]:
    o = df_model["usd_old"].sum()
    n = df_model["usd_new"].sum()
    c = df_model["sum_completion"].sum()
    r = df_model["sum_reasoning_raw"].sum()
    return {
        "usd_old": o,
        "usd_new": n,
        "usd_factor": o / n,
        "sum_completion": c,
        "sum_reasoning_raw": r,
        "token_share": r / c,
    }


def main() -> None:
    con = duckdb.connect()
    df_model = per_model(con)
    df_pop = per_suite_arm_split(con)
    n_anom = anomaly_count(con)
    pooled_all = pooled(df_model)
    pooled_reasoning_only = pooled(
        df_model[
            df_model["model"].isin(REASONING_BEARING_MODELS)
            | (df_model["usd_old"] != df_model["usd_new"])
        ]
    )

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)

    print("=== per model (all models with a price-table entry) ===")
    print(df_model.to_string(index=False))
    print()
    print(
        f"rows in calls.parquet with tok_reasoning > tok_completion (clamped in usd_new): {n_anom}"
    )
    print()
    print("=== pooled over every priced model (incl. zero-rate local checkpoints) ===")
    print(pooled_all)
    print()
    print("=== pooled restricted to models where usd_old != usd_new anywhere ===")
    print(pooled_reasoning_only)
    print()
    print("=== gpt-oss-120b calls joined to runs.parquet, by (suite_id, arm_id, split) ===")
    print(df_pop.to_string(index=False))
    print()
    print("=== gpt-oss-120b, aggregated per suite ===")
    g = df_pop.groupby("suite_id")[
        ["n_calls", "sum_completion", "sum_reasoning_raw", "usd_old", "usd_new"]
    ].sum()
    g["usd_factor"] = g["usd_old"] / g["usd_new"]
    g["token_share"] = g["sum_reasoning_raw"] / g["sum_completion"]
    print(g.sort_values("usd_factor", ascending=False).to_string())
    print()
    print("=== gpt-oss-120b, aggregated per arm ===")
    g = df_pop.groupby("arm_id")[
        ["n_calls", "sum_completion", "sum_reasoning_raw", "usd_old", "usd_new"]
    ].sum()
    g["usd_factor"] = g["usd_old"] / g["usd_new"]
    g["token_share"] = g["sum_reasoning_raw"] / g["sum_completion"]
    print(g.sort_values("usd_factor", ascending=False).to_string())


if __name__ == "__main__":
    main()
