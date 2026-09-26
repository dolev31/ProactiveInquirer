"""A criterion with no rankable candidate must not record the same thing as one with no candidates.

`_stop2x2` returning None for a non-finite cell (commit ed03455) stops the selector picking by
dict order, but it drops the candidate SILENTLY: `by_stop2x2` then comes back None and the record
reads `status: "none"` — which is exactly what a run with zero verdicts records. Two opposite
situations printing the same string:

  - the three granite runs: `n_verdicts: 0`, nothing was ever scored, nothing could be selected;
  - E39-8b s0/s1/s2: three verdicts existed and were scored, but every `p_stop_given_done` was
    NaN because `PI_REFERENCE_ASK` was unset, so nothing could be RANKED.

This repo already holds the rule these two must obey, in `pair_accuracy`'s `_finite` convention:
an empty cell prints None and not 0.0, because "no pairs were scored" and "every pair was ordered
wrong" are opposite readings and must not print the same. The same applies here, and it matters
more because the next person will not know the environment variable exists — the record has to say
why there is no winner, not merely that there is none.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "scripts" / "hpc" / "keep_best.py"


def _mod():
    spec = importlib.util.spec_from_file_location("keep_best_rank_undertest", _SRC)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _verdict(p_stop, p_ask, nll):
    return {
        "stop_confusion": {"p_stop_given_done": p_stop, "p_ask_given_not_done": p_ask},
        "sft_nll": {"nll_per_token": nll},
    }


def _write(eval_dir: Path, run: str, tag: str, v: dict) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"{run}.ckpt-{tag}.tierA.json").write_text(json.dumps(v))


def test_unrankable_is_distinguishable_from_no_candidates(tmp_path: Path) -> None:
    m = _mod()
    nan = float("nan")
    run = "run-nan"
    ev = tmp_path / "eval"
    for tag, nll in (("200", 4.90), ("400", 4.80), ("600", 4.70)):
        _write(ev, run, tag, _verdict(nan, 1.0, nll))

    sel = m.keep(run=run, eval_dir=ev, keep_dir=tmp_path / "best", sources=[tmp_path / "src"])
    stop = sel["by_stop2x2"]

    assert stop["tag"] is None, stop
    # THE POINT: it must say WHY, and name what it could not rank.
    assert stop["status"] != "none", (
        "an unrankable criterion recorded the same status as one with no candidates at all"
    )
    assert "unrankable_tags" in stop, stop
    assert sorted(stop["unrankable_tags"]) == ["200", "400", "600"], stop["unrankable_tags"]
    # by_nll is unaffected and still decides.
    assert sel["by_nll"]["tag"] == "600", sel["by_nll"]


def test_no_candidates_records_plain_none(tmp_path: Path) -> None:
    """Non-vacuity: the empty case must still read as empty, not as unrankable."""
    m = _mod()
    sel = m.keep(
        run="run-empty",
        eval_dir=tmp_path / "eval-empty",
        keep_dir=tmp_path / "best",
        sources=[tmp_path / "src"],
    )
    stop = sel["by_stop2x2"]
    assert stop["tag"] is None
    assert stop["status"] == "none", stop
    assert not stop.get("unrankable_tags"), stop


def test_a_finite_criterion_is_untouched(tmp_path: Path) -> None:
    """Non-vacuity: a real metric must still select, with no unrankable annotation."""
    m = _mod()
    run = "run-ok"
    ev = tmp_path / "eval"
    _write(ev, run, "200", _verdict(0.4, 0.8, 4.90))
    _write(ev, run, "400", _verdict(0.9, 0.9, 4.80))
    sel = m.keep(run=run, eval_dir=ev, keep_dir=tmp_path / "best", sources=[tmp_path / "src"])
    stop = sel["by_stop2x2"]
    assert stop["tag"] == "400", stop
    assert not stop.get("unrankable_tags"), stop
