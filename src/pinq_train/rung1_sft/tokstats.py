"""Real-token percentiles over an exported dataset. The measurement that sets `max_seq_len`.

WHY THIS IS A COMMAND AND NOT A CONSTANT. `SFTConfig.max_seq_len` is 12,288, a number carried over
from the plan rather than measured on this corpus. Both ways of being wrong are silent. Too low
and the templated path REFUSES the longest rows -- which are the evidence-heavy states, the ones
where deciding is actually hard, so the dataset loses precisely the examples the rung exists to
learn from and the loss curve looks fine. Too high and the window wastes memory the rented 80 GB
card does not have, which shows up as an OOM three hours into a paid run.

WHY NEAREST-RANK RATHER THAN INTERPOLATED PERCENTILES. An interpolated p99.9 reports a token count
that no row in the corpus actually has. `max_seq_len` is a decision about real rows, so every
number this prints is a length some row really was. `n` is reported beside them because p99.9 over
a few hundred rows is an anecdote with a decimal point.

WHAT THE NUMBER DOES NOT INCLUDE. The default field is `state_text`, so the reported maximum is the
STATE's contribution alone: the action and the chat template's own header and end-of-turn tokens
are on top of it. Size `max_seq_len` with headroom over `covering_multiple_of_512`, or run this
twice and add the two fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

# The granularity `max_seq_len` is usually chosen at: attention kernels and KV-cache blocks are
# happier on a round window, and a 512-multiple is the coarsest rounding that never costs more
# than half a kilotoken of waste.
WINDOW_MULTIPLE = 512


class EncodingTokenizer(Protocol):
    """The one method this module needs. `AutoTokenizer` satisfies it, and so does a fake."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


def percentile(values: Sequence[int], q: float) -> int:
    """The NEAREST-RANK percentile: the smallest value at or above rank `ceil(q/100 * n)`.

    Integer arithmetic on purpose. `0.9 * 10` is `9.000000000000002` in IEEE double, so a
    float-rounded rank crosses into the next element and the reported p90 is the p100 -- a
    one-element error that is invisible on a large sample and wrong on a small one.
    """
    if not values:
        raise ValueError(
            "no lengths to summarise: an empty corpus has no percentiles, and reporting 0 for "
            "one would set max_seq_len from a file that was never read."
        )
    if not 0 < q <= 100:
        raise ValueError(f"q={q}: a percentile is in (0, 100]")
    ordered = sorted(values)
    milli = int(round(q * 1000))
    rank = -(-(milli * len(ordered)) // 100_000)  # ceil, in integers
    return int(ordered[min(max(rank, 1), len(ordered)) - 1])


def covering_window(max_tokens: int, multiple: int = WINDOW_MULTIPLE) -> int:
    """The smallest multiple of `multiple` that COVERS `max_tokens`. 512 covers 512."""
    if max_tokens <= 0:
        return multiple
    return -(-int(max_tokens) // multiple) * multiple


@dataclass(frozen=True, slots=True)
class TokStats:
    n: int
    p50: int
    p90: int
    p99: int
    p999: int
    max_tokens: int
    covering_multiple_of_512: int

    def as_dict(self) -> dict[str, int]:
        return {
            "n": self.n,
            "p50": self.p50,
            "p90": self.p90,
            "p99": self.p99,
            "p99_9": self.p999,
            "max_tokens": self.max_tokens,
            "covering_multiple_of_512": self.covering_multiple_of_512,
        }

    def render(self) -> str:
        return (
            f"n={self.n}  p50={self.p50}  p90={self.p90}  p99={self.p99}  "
            f"p99.9={self.p999}  max={self.max_tokens}  "
            f"-> max_seq_len {self.covering_multiple_of_512} covers every row "
            "(the state alone; add the action and the chat header)"
        )


def token_lengths(texts: Iterable[str], tok: EncodingTokenizer) -> list[int]:
    """Real tokens per text, WITHOUT special tokens.

    `add_special_tokens=False` because the BOS and the chat header are added by the template on
    top of this, and counting a BOS here would be counting one of them twice while still missing
    the other six.
    """
    return [len(tok.encode(t, add_special_tokens=False)) for t in texts]


def token_stats(texts: Iterable[str], tok: EncodingTokenizer) -> TokStats:
    """The pure function: an iterator of texts and a tokenizer in, the percentiles out."""
    lengths = token_lengths(texts, tok)
    if not lengths:
        raise ValueError(
            "no texts: an empty corpus produces no percentiles. Check --dataset and --field."
        )
    top = max(lengths)
    return TokStats(
        n=len(lengths),
        p50=percentile(lengths, 50),
        p90=percentile(lengths, 90),
        p99=percentile(lengths, 99),
        p999=percentile(lengths, 99.9),
        max_tokens=top,
        covering_multiple_of_512=covering_window(top),
    )


def _field_of(path: Path, field: str) -> Iterable[str]:
    """Stream one field out of a jsonl file. The real one is 415 MB; nothing here holds it."""
    with path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if field not in row:
                raise KeyError(
                    f"{path}:{i} has no field {field!r}. Present: {sorted(row)[:12]}. A silently "
                    "skipped row would report percentiles over a subset of the corpus."
                )
            yield str(row[field])


def stats_from_jsonl(
    path: str | Path, tok: EncodingTokenizer, *, field: str = "state_text"
) -> TokStats:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist. Build it with `pi train export --kind sft`.")
    return token_stats(_field_of(p, field), tok)


def load_tokenizer(name_or_path: str) -> Any:
    """`AutoTokenizer.from_pretrained`, imported INSIDE the function.

    `transformers` lives behind the `[train]` extra. A module-level import here would make
    `pi train tokstats --help` -- and every test that touches this file -- require a 2 GB wheel.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name_or_path)
