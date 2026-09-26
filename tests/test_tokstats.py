"""`pi train tokstats` -- the measurement that sets `max_seq_len`.

`max_seq_len` is currently 12,288, which is a number from the plan rather than from this corpus.
Setting it too low silently drops the evidence-heavy states -- exactly the states where deciding
is hard, since the templated path REFUSES an over-length row rather than truncating it. Setting it
too high wastes memory that the rented card does not have. Neither failure announces itself, so
the percentiles are measured rather than guessed.

Percentiles are NEAREST-RANK (ceil on the sorted list), not interpolated: an interpolated p99.9
reports a token count that no row actually has, and `max_seq_len` is a decision about real rows.
`n` is reported beside them because p99.9 over a few hundred rows is an anecdote.
"""

from __future__ import annotations

import json

import pytest

from pinq_train.rung1_sft.tokstats import TokStats, percentile, stats_from_jsonl, token_stats


class FakeTokenizer:
    """`n` whitespace tokens in, `n` ids out. The one method `tokstats` uses."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [7] * len(text.split())


def _text(n: int) -> str:
    return " ".join(["w"] * n)


def test_percentile_is_nearest_rank_and_never_invents_a_value():
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(vals, 50) == 5
    assert percentile(vals, 90) == 9
    assert percentile(vals, 99) == 10
    assert percentile(vals, 100) == 10
    assert percentile([42], 99.9) == 42
    # Every reported number is a value that some row actually had.
    assert all(percentile(vals, q) in vals for q in (50, 90, 99, 99.9))


def test_percentile_refuses_an_empty_sample():
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)


def test_token_stats_reports_the_percentiles_and_the_covering_window():
    st = token_stats((_text(n) for n in [10, 20, 30, 600]), FakeTokenizer())
    assert isinstance(st, TokStats)
    assert st.n == 4
    assert st.max_tokens == 600
    assert st.p50 == 20 and st.p90 == 600
    # 600 -> the next multiple of 512 that covers it.
    assert st.covering_multiple_of_512 == 1024
    assert "600" in st.render() and "1024" in st.render()


@pytest.mark.parametrize(
    "max_tokens,expected", [(1, 512), (511, 512), (512, 512), (513, 1024), (6144, 6144)]
)
def test_the_covering_window_is_the_smallest_multiple_of_512_that_fits(max_tokens, expected):
    """An exact multiple must not round UP to the next one: 512 covers 512."""
    st = token_stats([_text(max_tokens)], FakeTokenizer())
    assert st.covering_multiple_of_512 == expected


def test_token_stats_refuses_an_empty_corpus():
    with pytest.raises(ValueError, match="empty|no texts"):
        token_stats([], FakeTokenizer())


def test_stats_from_jsonl_reads_the_named_field_streaming(tmp_path):
    p = tmp_path / "sft.jsonl"
    p.write_text(
        "\n".join(
            json.dumps({"state_text": _text(n), "action_json": _text(3)}) for n in (5, 50, 500)
        )
        + "\n\n"  # a blank line, which real exports have and which must not count as a row
    )
    st = stats_from_jsonl(p, FakeTokenizer())
    assert st.n == 3 and st.max_tokens == 500

    other = stats_from_jsonl(p, FakeTokenizer(), field="action_json")
    assert other.n == 3 and other.max_tokens == 3


def test_stats_from_jsonl_names_the_field_it_could_not_find(tmp_path):
    p = tmp_path / "sft.jsonl"
    p.write_text(json.dumps({"state_text": "a b c"}) + "\n")
    with pytest.raises(KeyError, match="nope"):
        stats_from_jsonl(p, FakeTokenizer(), field="nope")


# ------------------------------------------------------------------------------------ the CLI


def test_pi_train_tokstats_is_registered_and_parses():
    from pi_run.cli import build_parser

    args = build_parser().parse_args(
        ["train", "tokstats", "--dataset", "d.jsonl", "--tokenizer", "Qwen/Qwen3-8B"]
    )
    assert args.fn.__name__ == "cmd_train_tokstats"
    assert args.dataset == "d.jsonl" and args.tokenizer == "Qwen/Qwen3-8B"
    assert args.field == "state_text"


def test_the_tokstats_module_is_reachable_by_name_not_by_import():
    """`pi_run` may not import `pinq_train` -- contract 4. The dynamic loader is the only door,
    and `test_train_cli.py::test_cmd_train_never_statically_imports_the_trainer` pins the rest."""
    from pi_run import cmd_train

    assert cmd_train._train("rung1_sft.tokstats").percentile([1, 2, 3], 50) == 2
