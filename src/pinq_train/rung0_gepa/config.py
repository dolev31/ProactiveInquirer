"""Rung 0's knobs, and the cost it will spend before it spends any of it.

WHY A COST PROJECTION IS PART OF THE CONFIG AND NOT PART OF THE CLI. The number a researcher
needs is `n_generations * n_candidates * n_tasks * n_suites * n_seeds`, and that product is
the config. Computing it in the CLI would mean the library could be driven past the budget by
anything that imported it directly, and a search that discovers its own price after eight
hours has already spent it.

WHY THE RATES COME FROM THE PINNED PRICE TABLE. `scripts/price_tables/2026-08.json` is the
same table `pi cost estimate` and the re-pricing path use. Re-declaring rates here would make
two sources of truth for what a token costs, and the two would disagree on the first day a
provider moved. It is read with stdlib `json` rather than through
`pinq_adapters.llm.pricing`, because `pinq_train` must stay importable in a training container
whose dependency set is dictated by torch and vllm pins.

THE PROJECTION IS A CEILING, NOT A FORECAST. `tok_in_per_rollout` is the plan's naive draw and
takes no credit for the response cache, which in practice serves a large share of a repeated
candidate sweep for free. `cache_discount` is applied and reported SEPARATELY so the two
numbers -- what it costs if nothing is cached, and what it plausibly costs -- are both
visible. A single blended number would let a cache miss look like a budget overrun.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

from pinq.ids import canon, h

# The pinned table `pi cost estimate` reads. Same file, same version, one source of truth.
DEFAULT_PRICE_TABLE = "scripts/price_tables/2026-08.json"

# Rates for the pinned open model, used only when no table is reachable (a training container
# with no repo checkout). Kept equal to the table's `openai/aws/gpt-oss-120b` row; the test
# `test_rung0_default_rates_match_the_pinned_table` holds the two together.
FALLBACK_USD_PER_MTOK_IN = 0.15
FALLBACK_USD_PER_MTOK_OUT = 0.75

# Prompt caching on the append-only prefix, as measured for the main sweep. Reported as a
# separate line, never blended into the headline number.
CACHE_DISCOUNT = 0.27

# Measured headroom (scripts/probe_retrieval.py, n=100): the share of tasks holding gold
# evidence a sub-question reaches and the full question misses. A single grid-wide k=5 makes
# strategyqa unmeasurable, so k is per suite here exactly as it is in conf/grids/.
K_BY_SUITE: dict[str, int] = {"musique": 5, "strategyqa": 2, "wiki2": 3}


class Rung0ConfigError(ValueError):
    """A search that cannot produce a comparable prompt. Raised at construction time."""


@dataclass(frozen=True, slots=True)
class CostProjection:
    """What the search will spend, itemised. Printed BEFORE the first rollout."""

    n_rollouts: int
    tok_in: int
    tok_out: int
    usd_uncached: float
    usd_cached: float
    usd_per_mtok_in: float
    usd_per_mtok_out: float
    price_table: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def render(self) -> str:
        return "\n".join(
            [
                "rung 0 (GEPA prompt search) -- PROJECTED COST, before any rollout",
                f"  rollouts          {self.n_rollouts:,}",
                f"  prompt tokens     {self.tok_in:,}",
                f"  completion tokens {self.tok_out:,}",
                f"  rates             ${self.usd_per_mtok_in}/Mtok in, "
                f"${self.usd_per_mtok_out}/Mtok out  [{self.price_table}]",
                f"  USD, nothing cached   ${self.usd_uncached:,.2f}",
                f"  USD, {int(CACHE_DISCOUNT * 100)}% prefix cached  ${self.usd_cached:,.2f}",
                "  (a ceiling: the naive per-rollout token draw takes no credit for the "
                "response cache)",
            ]
        )


@dataclass(frozen=True, slots=True)
class Rung0Config:
    """The search, as plain data.

    `suites` DELIBERATELY EXCLUDES tau2 and pare. They are the zero-shot transfer targets;
    optimising a prompt against them would make the transfer claim a within-suite result
    wearing a transfer claim's name. `pinq_train.split.EVAL_ONLY_SUITES` is the same list and
    `validate()` refuses either of them here as well, because two independent refusals are
    what makes the guarantee survive one of them being edited.

    `n_val_tasks` is a HELD-OUT slice of the same train split, used only to pick the winner
    among Pareto-front survivors. It is not the paper's dev set: the seam refuses any task
    outside `train` (`pinq.wire.ScoreRequest.split_assert`), so a prompt search driven through
    it cannot touch dev even by accident. The winner is then validated on dev through the
    ordinary eval path, which runs on the gold side and never inside a trainer.
    """

    suites: tuple[str, ...] = ("musique", "wiki2")
    n_tasks: int = 40  # scoring tasks per suite per candidate evaluation
    n_val_tasks: int = 20  # held-out train tasks, used only to break Pareto ties
    n_seeds: int = 1
    n_candidates: int = 8  # candidates alive per generation
    n_generations: int = 6
    minibatch: int = 8  # traces shown to the reflector per mutation
    base_prompt: str = "inquirer_prompted"
    # The parity budget in tokens. "family" derives it from the base prompt's PARITY_FAMILY
    # (the default, and what keeps a searched prompt shippable into an arm comparison); an int
    # sets it explicitly; None removes the constraint entirely, which is only correct when the
    # goal is the best-performing prompt rather than a comparable one. An unconstrained search
    # optimises length -- MEASURED on 2026-09-01, see artifacts/rung0/lenctl/RESULT.md.
    max_prompt_tokens: int | str | None = "family"
    arm_id: str = "inquirer_prompted"
    max_turns: int = 16
    budget_cap: int | None = 8
    k_by_suite: Mapping[str, int] = field(default_factory=lambda: dict(K_BY_SUITE))
    k_default: int = 5
    seed: int = 0
    # Cost model. The plan's naive per-rollout draw for grid A; every REPORTED dollar comes
    # from the ledger, never from this.
    tok_in_per_rollout: int = 253_000
    tok_out_per_rollout: int = 8_800
    price_table: str = DEFAULT_PRICE_TABLE
    policy_model: str = ""
    policy_base_url: str = ""

    def __post_init__(self) -> None:
        self.validate()

    @property
    def sha(self) -> str:
        """Provenance: the winning prompt must name the search that produced it."""
        return h("rung0", canon(asdict(self)))

    def k_for(self, suite_id: str) -> int:
        return int(self.k_by_suite.get(suite_id, self.k_default))

    @property
    def n_rollouts(self) -> int:
        """Every rollout the search will issue, including generation 0's baseline sweep."""
        per_eval = self.n_tasks * len(self.suites) * self.n_seeds
        # generation 0 evaluates the seed prompt alone; each later generation evaluates
        # n_candidates mutants. Tie-breaking adds one held-out pass per surviving candidate,
        # PLUS ONE for the seed itself: the baseline has to be measured on the same tasks as
        # the winner or the reported gain is a task-set contrast. See search.run_search.
        held_out = self.n_val_tasks * len(self.suites) * self.n_seeds
        return (
            per_eval * (1 + self.n_generations * self.n_candidates)
            + held_out * self.n_candidates
            + held_out
        )

    def validate(self) -> None:
        from pinq_train.split import EVAL_ONLY_SUITES

        bad = sorted(set(self.suites) & EVAL_ONLY_SUITES)
        if bad:
            raise Rung0ConfigError(
                f"rung 0 may not optimise against {bad}: those suites are the zero-shot "
                "transfer targets, and a prompt tuned on them turns the transfer claim into a "
                "within-suite result reported under a transfer claim's name."
            )
        if not self.suites:
            raise Rung0ConfigError("no suites: a prompt optimised on nothing is not a prompt")
        for name, value in (
            ("n_tasks", self.n_tasks),
            ("n_candidates", self.n_candidates),
            ("n_generations", self.n_generations),
            ("minibatch", self.minibatch),
        ):
            if value < 1:
                raise Rung0ConfigError(f"{name}={value}: must be >= 1")
        if self.n_val_tasks < 0:
            raise Rung0ConfigError(f"n_val_tasks={self.n_val_tasks}: must be >= 0")

    def rates(self, root: str | Path = ".") -> tuple[float, float, str]:
        """(USD per Mtok in, USD per Mtok out, table name). Falls back, and SAYS it fell back.

        A silent fallback would put a plausible dollar figure in front of a researcher with no
        indication that the pinned table was never read.
        """
        p = Path(self.price_table)
        if not p.is_absolute():
            p = Path(root) / p
        model = self.policy_model or "openai/aws/gpt-oss-120b"
        try:
            table = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return FALLBACK_USD_PER_MTOK_IN, FALLBACK_USD_PER_MTOK_OUT, "built-in fallback rates"
        row = table.get("models", {}).get(model) or table.get("fallback") or {}
        rin = row.get("input")
        rout = row.get("output")
        if rin is None or rout is None:
            return (
                FALLBACK_USD_PER_MTOK_IN,
                FALLBACK_USD_PER_MTOK_OUT,
                f"built-in fallback rates ({model} unpriced in {table.get('version', '?')})",
            )
        return float(rin), float(rout), f"{p.name} v{table.get('version', '?')}"

    def projected_cost(self, root: str | Path = ".") -> CostProjection:
        rin, rout, table = self.rates(root)
        n = self.n_rollouts
        tok_in = n * self.tok_in_per_rollout
        tok_out = n * self.tok_out_per_rollout
        usd = (tok_in / 1_000_000.0) * rin + (tok_out / 1_000_000.0) * rout
        cached = (tok_in / 1_000_000.0) * rin * (1.0 - CACHE_DISCOUNT) + (
            tok_out / 1_000_000.0
        ) * rout
        return CostProjection(
            n_rollouts=n,
            tok_in=tok_in,
            tok_out=tok_out,
            usd_uncached=usd,
            usd_cached=cached,
            usd_per_mtok_in=rin,
            usd_per_mtok_out=rout,
            price_table=table,
        )
