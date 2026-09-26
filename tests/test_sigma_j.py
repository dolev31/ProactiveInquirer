"""sigma_J: the judge's own noise floor, and the producer that measures it.

`pi_eval.prereg` named `scripts/measure_sigma_j.py` and `load_sigma_j` read
`prereg/sigma_j.json` -- and neither the script nor the file existed. So every judge-derived
metric carried a noise floor of +inf and `pi_eval.report` flagged every effect on it
`below_noise_floor`. That is the right conservative behaviour with nothing measured, and it
means no judged result could ever have been claimed, including primary endpoint P3.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "measure_sigma_j", Path(__file__).resolve().parents[1] / "scripts" / "measure_sigma_j.py"
)
msj = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(msj)


def test_the_estimator_recovers_a_known_sigma():
    """Var(x1 - x2) = 2*sigma^2 for independent repeats, so sigma = SD(diffs)/sqrt(2).

    Using SD(diffs) directly -- the obvious thing -- overstates the floor by 41% and silently
    suppresses real effects, which is a failure in the direction nobody checks.
    """
    rng = random.Random(0)
    true_sigma = 0.10
    pairs = [(rng.gauss(0.5, true_sigma), rng.gauss(0.5, true_sigma)) for _ in range(4000)]
    est = msj.sigma_from_pairs(pairs)
    assert est == pytest.approx(true_sigma, rel=0.06)
    # And the naive version really is off by sqrt(2), i.e. this is not an academic distinction.
    import statistics

    naive = statistics.stdev([a - b for a, b in pairs])
    assert naive == pytest.approx(true_sigma * math.sqrt(2), rel=0.06)


def test_a_perfectly_repeatable_judge_yields_zero_and_too_few_pairs_yield_none():
    assert msj.sigma_from_pairs([(0.4, 0.4)] * 300) == 0.0
    assert msj.sigma_from_pairs([(0.4, 0.5)]) is None
    assert msj.sigma_from_pairs([]) is None


def test_unjudged_items_are_dropped_rather_than_imputed():
    """Imputing a missing judgement at the mean shrinks the estimated noise toward zero -- in
    the direction that lets more effects through the dead band."""
    nan = float("nan")
    clean = [(0.2, 0.4), (0.6, 0.5), (0.1, 0.3), (0.9, 0.7)]
    with_holes = clean + [(nan, 0.5), (0.3, nan), (nan, nan)]
    assert msj.sigma_from_pairs(with_holes) == msj.sigma_from_pairs(clean)


def test_the_retest_seed_must_differ_from_the_baseline_seed():
    """THE FABRICATED ZERO THIS GUARD EXISTS FOR. At temperature 0 with a content-addressed
    cache, re-judging at the SAME seed replays the same bytes, every difference is exactly 0,
    sigma_identical is 0, and the dead band becomes nothing -- so every judged effect passes.
    A measurement that cannot fail is not a measurement."""
    with pytest.raises(SystemExit, match="must differ"):
        msj.measure(
            judge=object(), runs=[], judge_model="m", judge_family="f", seed=3, retest_seed=3
        )


def test_sigma_is_the_max_of_the_conditions_not_the_mean():
    """The two conditions measure different noise and the paraphrase one is larger in every
    published estimate. Taking the smaller, or their mean, narrows the dead band using
    whichever repeat flattered the instrument."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "measure_sigma_j.py").read_text()
    assert "max(sigmas) if sigmas else None" in src
    assert "statistics.mean(sigmas)" not in src


def test_a_metric_measured_on_too_few_pairs_is_refused_rather_than_written(tmp_path):
    """An honest +inf floor flags every effect. Replacing it with a small number measured on
    40 pairs lets everything through, and looks like diligence."""
    from pi_eval.prereg import SIGMA_J_MIN_PAIRS, load_sigma_j

    result = {
        "n_runs": 40,
        "n_paraphrased": 40,
        "per_metric": {
            "keypoint_recall": {"sigma_j": 0.11, "n_pairs": SIGMA_J_MIN_PAIRS, "conditions": {}},
            "citation_support": {"sigma_j": 0.02, "n_pairs": 40, "conditions": {}},
            "quality_depth": {"sigma_j": None, "n_pairs": 0, "conditions": {}},
        },
    }
    refused = msj.write_sigma_j(result, {"judge_model": "m"}, tmp_path)
    written = json.loads((tmp_path / "prereg" / "sigma_j.json").read_text())

    assert written == {"keypoint_recall": 0.11}
    assert any("citation_support" in r for r in refused)
    assert "quality_depth" not in written
    # And the file the frozen reader sees carries nothing but metric -> float.
    got = load_sigma_j(tmp_path)
    assert got == {"keypoint_recall": 0.11}
    assert all(isinstance(v, float) for v in got.values())


def test_the_written_file_never_carries_a_boolean():
    """`isinstance(True, int)` is True in Python, so a stray flag in this file would be read as
    a sigma_J of 1.0 and silently raise the floor on every judge metric. Provenance goes in the
    sidecar, which is why there are two files."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "measure_sigma_j.py").read_text()
    assert "sigma_j.provenance.json" in src
    assert "json.dumps(out, indent=2, sort_keys=True)" in src


def test_every_judge_derived_metric_is_a_target():
    """A judge metric absent from the measurement keeps its +inf floor forever, so it can never
    be claimed and nothing says why."""
    from pi_eval.score import BY_NAME

    declared = {n for n, d in BY_NAME.items() if d.judge_derived}
    assert set(msj.judge_metric_names()) == declared
    assert "kpr_incremental" in declared, "the primary endpoint P3 is judge-derived"


def test_the_paraphrase_prompt_is_pinned_into_the_provenance():
    """A paraphrase prompt that drifted between the measurement and the runs it gates would
    make sigma_J an estimate of a different instrument."""
    prov = msj.provenance(
        {"n_runs": 1, "n_paraphrased": 1, "per_metric": {}},
        judge_model="m",
        judge_family="f",
        seed=0,
        retest_seed=1,
    )
    assert prov["paraphrase_prompt_sha"]
    assert prov["seed"] != prov["retest_seed"]
    assert prov["min_pairs_required"] == 200
