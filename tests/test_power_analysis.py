"""The design's sensitivity, computed from the test that actually runs.

Before scripts/power_analysis.py the repository's only statement about design sensitivity was
one sentence in a YAML comment: "MDE about 12.6pp on a binary endpoint". That is the textbook
McNemar normal approximation at roughly 20 discordant pairs -- not the sensitivity of the
CLUSTER-LEVEL PERMUTATION over seed-rounded outcomes that `pi_eval.report` runs. A power figure
computed from a different test than the one you run is not a power figure.
"""

import importlib.util
import pathlib

import pytest

SPEC = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "power_analysis.py"


@pytest.fixture(scope="module")
def pa():
    spec = importlib.util.spec_from_file_location("_power_analysis", SPEC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_cluster_map_matches_tau2s_measured_templates(pa):
    """97 tasks in 18 templates, the sizes measured from Tau2Suite.template_id."""
    assert sum(pa.TAU2_TEMPLATE_SIZES) == 97
    assert len(pa.TAU2_TEMPLATE_SIZES) == 18
    cl = pa.tau2_clusters()
    assert len(cl) == 97 and len(set(cl.values())) == 18


def test_power_rises_with_effect_size(pa):
    """The sanity property. If this ever inverts, the simulation is wrong, not the design."""
    lo = pa.binary_power(0.05, trials=40, n_perm=150)
    hi = pa.binary_power(0.30, trials=40, n_perm=150)
    assert hi > lo


def test_a_zero_effect_rejects_at_about_alpha(pa):
    """Calibration: with no true effect the test must reject at roughly 5%, or every power
    number computed from it is meaningless."""
    p = pa.binary_power(0.0, trials=200, n_perm=400)
    assert p <= 0.12, f"type-I rate {p:.1%} is far above alpha=0.05"


def test_the_documented_mde_is_not_the_real_one(pa):
    """The finding this script exists to record: at the 12.6pp lift the grid used to claim as
    its MDE, the actual test has well under 80% power."""
    p = pa.binary_power(0.126, trials=200, n_perm=400)
    assert p < 0.60, f"12.6pp reached {p:.0%} power; the grid claimed it was the 80% MDE"


def test_the_grid_states_the_simulated_mde_and_not_the_approximation():
    """conf/grids/tier2_confirmatory.yaml is a preregistration document, so what it ASSERTS
    matters.

    Asserted on the CLAIM, not on the absence of a string. A naive `"12.6pp" not in y` cannot
    tell a claim from a correction that quotes it, and the note deliberately records what it
    used to say -- so that check would force the document to delete its own history in order
    to pass. (I wrote the naive version first and it failed on exactly that.)
    """
    y = (
        pathlib.Path(__file__).resolve().parents[1] / "conf/grids/tier2_confirmatory.yaml"
    ).read_text()
    assert "MDE = 22.1pp at 80% power" in y, "the simulated figure must be the stated one"
    assert "power_analysis.py" in y, "and it must say what produced it"
    # every mention of the old number must be part of the correction, not a live claim
    for line in y.splitlines():
        if "12.6pp" in line:
            assert "previously read" in line or "real power is" in line, line


def test_mde_returns_none_when_the_grid_never_reaches_target(pa):
    """An endpoint that cannot reach 80% power on any effect worth claiming must say so
    rather than returning the largest value it tried."""
    assert pa.mde(lambda _x: 0.10, [0.05, 0.10, 0.20]) is None
    assert pa.mde(lambda x: 1.0 if x >= 0.10 else 0.0, [0.05, 0.10]) is not None


def test_continuous_power_is_monotone_in_sd(pa):
    """More noise, less power -- at a fixed effect and n."""
    tight = pa.paired_power(0.05, sd_diff=0.10, n_tasks=200, trials=40, n_perm=150)
    loose = pa.paired_power(0.05, sd_diff=0.40, n_tasks=200, trials=40, n_perm=150)
    assert tight >= loose
