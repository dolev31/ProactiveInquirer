"""The deterministic price table. Separate from litellm_client on purpose.

Costing a sweep must not require a retry library or a provider SDK. `pi cost estimate` is
the command you run BEFORE deciding whether to install anything, and re-pricing a finished
sweep from parquet is pure arithmetic over stored token counts. Keeping the table here means
both work in a minimal install — and it is the split the repo layout already declares
(pinq_adapters/llm/{litellm_client.py, pricing.py}).

USD is tokens x these rates, ALWAYS. A provider's own dollar field depends on account tier,
credits and billing day, so it is not recomputable from a stored artifact; this is.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DEFAULT_PRICE_TABLE = "scripts/price_tables/2026-08.json"

_MILLION = 1_000_000.0


class LLMConfigError(RuntimeError):
    """A role has no model pin, or a pinned model has no price. Both are fatal at t=0.

    Raised eagerly rather than defaulting: a silent default would produce a sweep whose arms
    ran on different models, and nothing downstream could detect it.
    """


@dataclass(frozen=True, slots=True)
class PriceTable:
    """Rates in USD per 1M tokens, pinned by version.

    `strict` exists because an unpriced model is a silent zero in every cost table in the
    paper. In a real sweep it must raise; in a smoke test a zero-cost stub is fine.
    """

    version: str
    models: Mapping[str, Mapping[str, float | None]]
    fallback: Mapping[str, float | None]
    path: str = ""
    strict: bool = True

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None, *, strict: bool = True) -> PriceTable:
        p = Path(path or os.environ.get("PI_PRICE_TABLE") or DEFAULT_PRICE_TABLE)
        if not p.is_absolute():
            p = Path.cwd() / p
        raw = json.loads(p.read_text())
        return cls(
            version=raw["version"],
            models=raw["models"],
            fallback=raw.get("fallback", {"input": 0.0, "output": 0.0, "cached_input": 0.0}),
            path=str(p),
            strict=strict,
        )

    def rates(self, model: str) -> Mapping[str, float | None]:
        if model in self.models:
            return self.models[model]
        if self.strict:
            raise LLMConfigError(
                f"no price for {model!r} in price table {self.version} ({self.path}). "
                "An unpriced model is a silent $0 in every cost table, so this raises."
            )
        return self.fallback

    def usd(
        self,
        model: str,
        *,
        tok_prompt: int,
        tok_completion: int,
        tok_reasoning: int = 0,
        tok_cached: int = 0,
    ) -> float:
        """Both nested counters are SUBSETS, and each is billed once.

        `tok_cached` is a subset of `tok_prompt`; only the remainder is billed at the input rate.
        `tok_reasoning` is a subset of `tok_completion` in exactly the same way: the OpenAI usage
        schema counts reasoning tokens INSIDE `completion_tokens`, and `_read_response` copies both
        fields straight across (`completion_tokens` and `completion_tokens_details.reasoning_tokens`).
        Measured 2026-09-19 on aws/gpt-oss-120b: completion 110 with reasoning 101 on the same call.

        This function used to add `tok_reasoning * rate_reasoning` on top of the full
        `tok_completion * rate_out`, which billed every reasoning token twice. Over the runs store as
        it stood on 2026-09-19 that overstated openai/aws/gpt-oss-120b by $540.39 on $1917.54
        recorded -- a factor of 1.3924 over 2,668,114 calls -- while agreeing with the gateway's own
        per-call charge only after the correction (median ratio 1.052 on the calls the gateway
        posted, against ~1.36 before). That total spans the compacted parquet and the 74,349 run
        directories it had not yet seen; it is not a closed account, because the store moves.

        Reasoning tokens still fall back to the output rate when the table does not price them
        separately, so an unpriced reasoning subset costs exactly what the rest of the completion
        costs and this function is unchanged for every model that reports no reasoning tokens.
        """
        r = self.rates(model)
        rate_in = float(r.get("input") or 0.0)
        rate_out = float(r.get("output") or 0.0)
        rate_cached = float(r.get("cached_input") or 0.0)
        reasoning = r.get("reasoning")
        rate_reasoning = float(reasoning) if reasoning is not None else rate_out
        uncached = max(0, tok_prompt - tok_cached)
        # Clamped, so an over-reported reasoning count cannot bill more than the completion it
        # is part of -- the same defence `uncached` gives tok_prompt.
        reasoning_tok = min(max(0, tok_reasoning), max(0, tok_completion))
        visible = max(0, tok_completion - reasoning_tok)
        total = (
            uncached * rate_in
            + tok_cached * rate_cached
            + visible * rate_out
            + reasoning_tok * rate_reasoning
        )
        return total / _MILLION
