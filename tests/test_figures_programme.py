"""The P-series generator, run once on the REAL data into a tmp directory.

This is a smoke test with one job: catch a figure that stopped being drawn, or a figure whose
provenance stopped naming what it was drawn from. It deliberately does NOT re-derive the
numbers -- a test that recomputes the figure's arithmetic from the same rows with the same
code passes whether or not the arithmetic is right, which is the tautological-reconciliation
failure `test_turn_usage_is_real_not_a_zero_stamp` exists to prevent. What is checkable here
without restating the implementation is the CONTRACT:

  * every measured figure produces a PDF, a PNG, a provenance record and a caption;
  * every provenance names at least one source file WITH ITS SHA256, so rule 1 holds --
    a figure whose provenance names no input is a number without provenance;
  * the recorded `inputs_hash` is a function of that map, so a moved input is detectable;
  * a figure whose result file does not exist is NOT DRAWN -- `NOT_YET.md` and no image.

It runs against the real corpus rather than a fixture because the thing most likely to break
is a column disappearing from `pairs.jsonl` or `runs.parquet`, which a fixture would hide. It
therefore SKIPS, with the missing path in the reason, on any checkout that does not carry the
corpus; a skip that says which file is absent is honest, and an invented fixture is not.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "scripts" / "figures_programme.py"


def _load():
    """`scripts/` is not an importable package, so load the file by path.

    The module must be in `sys.modules` BEFORE it executes: `@dataclass` under
    `from __future__ import annotations` resolves its string annotations through
    `sys.modules[cls.__module__]`, and a module that is not registered there raises at class
    creation rather than at use.
    """
    spec = importlib.util.spec_from_file_location("figures_programme", GEN)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    pytest.importorskip("duckdb")
    pytest.importorskip("matplotlib")
    if not GEN.exists():
        pytest.skip(f"generator absent: {GEN}")
    return _load()


@pytest.fixture(scope="module")
def rendered(gen, tmp_path_factory):
    """Run the whole generator once. Every test below reads the same output tree."""
    missing = [
        p for p in (gen.PAIRS, gen.PAIRS_DEV, gen.SFT, gen.COHORT, gen.RUNS) if not p.exists()
    ]
    if missing:
        pytest.skip("corpus absent: " + ", ".join(str(p) for p in missing))
    out = tmp_path_factory.mktemp("figures")
    rc = gen.main(["--out", str(out)])
    # P7 IS REFUSED ON THIS CORPUS, AND THAT IS THE POINT. Every gate verdict's bootstrap
    # endpoint was computed before `4b7b24b` canonicalised the order the cluster bootstrap
    # resamples in, and none has been reconciled since (`gen.BCA_ORDER_RECONCILED` is empty),
    # so P7 refuses rather than typeset an interval whose order no record states. The earlier
    # `rc == 0` here encoded the belief that every DRAWN figure can be built from whatever is
    # on disk; that belief is false while P7's inputs are known-defective, so the assertion is
    # corrected rather than dropped -- nothing ELSE may refuse, and P7's refusal is pinned by
    # name in `test_p7_refuses_the_real_pre_sort_gate_corpus`.
    p7_refused = not any(
        gen.sha256_file(p) in gen.BCA_ORDER_RECONCILED for p in gen.P7_GATE_ROOT.rglob("*.json")
    )
    unbuilt = sorted(f for f in DRAWN + HOOKS + REFUSED if not (out / f).exists())
    assert unbuilt == (list(REFUSED) if p7_refused else []), (
        f"unexpected refusal(s): {unbuilt}; only P7's pre-sort interval refusal is expected"
    )
    assert rc == (1 if p7_refused else 0)
    return out


def _needs(gen, fid: str) -> None:
    """Skip THIS figure when one of its own sources is absent, naming the file."""
    needed = {
        "P1_rater_panel": [gen.PAIRS],
        "P2_corpus_by_cohort": [gen.PAIRS, gen.PAIRS_DEV, gen.COHORT],
        "P3_target_depth": [gen.SFT, gen.COHORT],
        "P4_margin_vs_agreement": [gen.PAIRS],
        "P5_spend": [gen.RUNS, gen.ANNOT_ROOT],
        "P6_dev_and_test_pools": [gen.RUNS, gen.GRIDS],
        "P7_scaling_by_model_size": [gen.CHECKPOINTS, gen.P7_GATE_ROOT],
        "P10_programme_map": [gen.PROGRAMME],
        "P11_scale_ladder": [gen.PROGRAMME],
    }[fid]
    absent = [p for p in needed if not p.exists()]
    if absent:
        pytest.skip(f"{fid}: source absent: " + ", ".join(str(p) for p in absent))


DRAWN = (
    "P1_rater_panel",
    "P2_corpus_by_cohort",
    "P3_target_depth",
    "P4_margin_vs_agreement",
    "P5_spend",
    "P6_dev_and_test_pools",
    "P10_programme_map",
    "P11_scale_ladder",
)
HOOKS = ("P8_frontier_trained", "P9_domain_matrix")
# P7 is NOT in DRAWN: it produces no output on this corpus at all. See the `rendered` fixture.
REFUSED = ("P7_scaling_by_model_size",)


@pytest.mark.parametrize("fid", DRAWN)
def test_figure_is_drawn_with_provenance(gen, rendered, fid):
    _needs(gen, fid)
    d = rendered / fid
    for name in (f"{fid}.pdf", f"{fid}.png", "provenance.json", "caption.md"):
        assert (d / name).exists(), f"{fid}: missing {name}"
        assert (d / name).stat().st_size > 0, f"{fid}: {name} is empty"


@pytest.mark.parametrize("fid", DRAWN)
def test_provenance_names_a_source_file_with_its_sha256(gen, rendered, fid):
    """Rule 1, at the artifact level. A figure whose provenance names no input is a number
    with no provenance, and this is the assertion that makes that unpublishable."""
    _needs(gen, fid)
    prov = json.loads((rendered / fid / "provenance.json").read_text())
    inputs = prov.get("inputs") or {}
    assert inputs, f"{fid}: provenance names no source file"
    for path, digest in inputs.items():
        assert isinstance(digest, str) and len(digest) == 64, f"{fid}: {path} has no sha256"
        assert int(digest, 16) >= 0, f"{fid}: {path}'s digest is not hex"
    assert prov["inputs_hash"] == gen.h("inputs", gen.canon(sorted(inputs.items()))), (
        f"{fid}: inputs_hash does not digest the inputs map it was written with, so a moved "
        "input would not be detectable"
    )
    assert prov["table_id"] == fid
    assert prov["provenance_digest"]


@pytest.mark.parametrize(
    "fid", ("P1_rater_panel", "P2_corpus_by_cohort", "P3_target_depth", "P4_margin_vs_agreement")
)
def test_corpus_figures_carry_run_identity(gen, rendered, fid):
    """The three identities CLAUDE.md rule 1 names, for figures whose rows carry them.

    P5 and P6 read the run ledger and the grid files, which carry no `scorer_hash`; their
    provenance records the empty list rather than omitting the key, which is a fact about
    the source rather than a gap.
    """
    _needs(gen, fid)
    prov = json.loads((rendered / fid / "provenance.json").read_text())
    assert prov["n_run_ids"] > 0, f"{fid}: names no run_id"
    assert prov["scorer_hash"], f"{fid}: names no scorer_hash"
    assert prov["graph_version"], f"{fid}: names no graph_version"


@pytest.mark.parametrize("fid", DRAWN)
def test_planned_figures_say_so_and_measured_ones_do_not(gen, rendered, fid):
    _needs(gen, fid)
    prov = json.loads((rendered / fid / "provenance.json").read_text())
    caption = (rendered / fid / "caption.md").read_text().lower()
    planned = fid in ("P10_programme_map", "P11_scale_ladder")
    assert prov["measured"] is not planned
    if planned:
        assert "planned, not measured" in caption, f"{fid}: a plan drawn without saying so"


@pytest.mark.parametrize("fid", HOOKS)
def test_result_hooks_produce_no_image_until_their_file_exists(gen, rendered, fid):
    """The rule-3 assertion. If the awaited table appears, this test tells you to write the
    drawing code rather than silently passing on a directory that now holds a stale NOT_YET."""
    awaited = {
        "P8_frontier_trained": gen.P8_PATH,
        "P9_domain_matrix": gen.P9_PATH,
    }[fid]
    d = rendered / fid
    if awaited.exists():
        pytest.fail(f"{fid}: {awaited} now exists; write the figure and drop it from HOOKS")
    assert (d / "NOT_YET.md").exists(), f"{fid}: no NOT_YET.md"
    assert str(awaited.relative_to(REPO)) in (d / "NOT_YET.md").read_text()
    for forbidden in (f"{fid}.pdf", f"{fid}.png", "provenance.json"):
        assert not (d / forbidden).exists(), (
            f"{fid}: {forbidden} was written for a figure whose data does not exist"
        )


def test_readme_lists_every_figure(gen, rendered):
    readme = (rendered / "README.md").read_text()
    for fid in (*DRAWN, *HOOKS, "P12_rung1_curves"):
        assert fid in readme, f"README does not list {fid}"


def test_readme_explains_every_status_it_can_print(gen, rendered):
    """A status in the table with no entry under "Status meanings" is a word the reader has to
    guess at -- and `partial` is the one that most needs saying, since it is drawn."""
    readme = (rendered / "README.md").read_text()
    printed = {ln.split("|")[2].strip() for ln in readme.splitlines() if ln.startswith("| `P")}
    for status in printed:
        assert f"**{status}**" in readme, f"README prints status {status!r} and never defines it"


def test_only_renders_one_figure(gen, rendered, tmp_path):
    """`--only` must not be a no-op that quietly renders everything."""
    _needs(gen, "P11_scale_ladder")
    out = tmp_path / "one"
    assert gen.main(["--out", str(out), "--only", "P11"]) == 0
    assert (out / "P11_scale_ladder" / "P11_scale_ladder.pdf").exists()
    assert not (out / "P10_programme_map").exists()


def test_unknown_figure_id_is_an_error_not_a_silent_skip(gen, tmp_path):
    assert gen.main(["--out", str(tmp_path / "none"), "--only", "P99"]) == 2


def test_grid_reader_agrees_with_the_grid_files(gen):
    """The stdlib grid reader is the one piece of parsing this script owns. If it silently
    returned {} the dev/test bars would be absent rather than wrong, so pin its output."""
    p = gen.GRIDS / "tier1_trained_qa_base.yaml"
    if not p.exists():
        pytest.skip(f"grid absent: {p}")
    g = gen.grid_fields(p)
    assert g["name"] == "tier1_trained_qa_base"
    assert g["split"] == "test"
    assert isinstance(g["n_tasks"], int) and g["n_tasks"] > 0
    assert g["suites"] == ["musique", "strategyqa", "wiki2"]
    assert "inquirer_trained" in g["arms"]
    # `notes:` and `description:` are block scalars; their prose must not leak into a field.
    assert all(not isinstance(v, str) or len(v) < 80 for v in g.values())


def test_wilson_interval_brackets_the_point_and_stays_in_range(gen):
    for k, n in ((0, 10), (10, 10), (1, 3), (1492, 2292)):
        p, lo, hi = gen.wilson(k, n)
        assert 0.0 <= lo <= p <= hi <= 1.0, (k, n, p, lo, hi)
    assert gen.wilson(0, 0)[0] != gen.wilson(0, 0)[0]  # nan, not a divide-by-zero


def test_schedule_is_a_longest_path_not_a_sum(gen):
    """P10's critical path must follow dependencies. A diamond whose two branches are 5 and
    50 finishes at 50, not at 55."""
    groups = [
        {"id": "a", "wall_h": 1, "gpu_h": 0, "after": []},
        {"id": "b", "wall_h": 5, "gpu_h": 0, "after": ["a"]},
        {"id": "c", "wall_h": 50, "gpu_h": 0, "after": ["a"]},
        {"id": "d", "wall_h": 2, "gpu_h": 0, "after": ["b", "c"]},
    ]
    start, finish, crit = gen._schedule(groups)
    assert finish["d"] == 53
    assert crit == ["a", "c", "d"]
    assert start["b"] == start["c"] == 1


def test_schedule_refuses_a_cycle_and_an_unknown_dependency(gen):
    with pytest.raises(gen.FigureRefused):
        gen._schedule(
            [
                {"id": "a", "wall_h": 1, "gpu_h": 0, "after": ["b"]},
                {"id": "b", "wall_h": 1, "gpu_h": 0, "after": ["a"]},
            ]
        )
    with pytest.raises(gen.FigureRefused):
        gen._schedule([{"id": "a", "wall_h": 1, "gpu_h": 0, "after": ["nope"]}])


# =========================================================================== P7 unit tests

# Synthetic gate verdicts, in exactly the schema `_p7_read_verdict` reads: `criteria.
# evidence_coverage.{value,ci_lo,ci_hi,n,cap8_value}`, `criteria.stop_2x2.baseline.
# {p_stop_given_done,n_states}`, `selection.{scorer_hash,baseline_model_id}`. None of these
# tests touch the real `artifacts/gate/` tree -- they build one fresh under `tmp_path` -- so
# they exercise the selection/absence/param-mapping helpers on any checkout, unlike the
# corpus-based DRAWN/HOOKS tests above which skip when the real gate verdicts are absent.


def _p7_verdict(
    value: float,
    ci_lo: float,
    ci_hi: float,
    n: int,
    *,
    p_stop: float | None = 0.05,
    n_states: int = 100,
    scorer_hash: str = "test-scorer-hash",
    baseline_model_id: str = "qwen3-9b-base",
    cap8_value: float | None = None,
    coverage_rule: str | None = "matched_cost",
) -> dict:
    return {
        "coverage_rule": coverage_rule,
        "criteria": {
            "evidence_coverage": {
                "value": value,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n": n,
                "cap8_value": cap8_value if cap8_value is not None else value,
            },
            "stop_2x2": {"baseline": {"p_stop_given_done": p_stop, "n_states": n_states}},
        },
        "selection": {"scorer_hash": scorer_hash, "baseline_model_id": baseline_model_id},
    }


def _write_p7_verdict(gate_root: Path, gate_dir: str, model_id: str, suite: str, doc: dict) -> None:
    d = gate_root / gate_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{model_id}.{suite}.json").write_text(json.dumps(doc))


def _reconcile_all(gen, monkeypatch, gate_root: Path) -> None:
    """Attest every synthetic verdict under `gate_root` as reconciled against the BCa
    unit-order fix, so a test about seed-group LOGIC exercises that logic instead of tripping
    the interval guard. A test about the guard itself must not call this.
    """
    monkeypatch.setattr(
        gen,
        "BCA_ORDER_RECONCILED",
        {gen.sha256_file(f): "synthetic fixture" for f in gate_root.rglob("*.json")},
    )


def test_p7_collects_one_point_per_declared_size_and_arm(gen, tmp_path):
    """`_p7_collect` must return exactly one row per (locator, suite) that has a verdict on
    disk -- not fewer (a silently dropped size) and not more (a size invented that was never
    declared in the locator tuple)."""
    gate_root = tmp_path / "gate"
    registry = {
        "fake-sft-small": {"base_model": "Qwen/Qwen3-3B"},
        "fake-sft-big": {"base_model": "Qwen/Qwen3-9B"},
        "fake-dpo-big": {"base_model": "Qwen/Qwen3-9B"},
    }
    locators = (
        ("sft", "Qwen3-3B", "small-t1", "fake-sft-small"),
        ("sft", "Qwen3-9B", "big-t1", "fake-sft-big"),
        ("dpo", "Qwen3-9B", "big-t1", "fake-dpo-big"),
    )
    suites = ("musique", "strategyqa")
    for arm, label, gate_dir, model_id in locators:
        for suite in suites:
            _write_p7_verdict(
                gate_root,
                gate_dir,
                model_id,
                suite,
                _p7_verdict(0.10, 0.05, 0.15, 200, baseline_model_id=f"{label.lower()}-base"),
            )

    points, absent = gen._p7_collect(locators, suites, registry, gate_root)

    assert absent == []
    assert len(points) == len(locators) * len(suites) == 6
    got = {(p["label"], p["arm"], p["suite"]) for p in points}
    want = {(label, arm, suite) for arm, label, _, _ in locators for suite in suites}
    assert got == want, "every declared (size, arm) x suite must appear exactly once"
    by_label = {p["label"]: p["params_b"] for p in points}
    assert by_label == {"Qwen3-3B": 3.0, "Qwen3-9B": 9.0}


def test_p7_missing_verdict_is_a_stated_absence_not_dropped_or_fabricated(gen, tmp_path):
    """A size/arm whose verdict file is not on disk must show up in `absent`, naming the exact
    file it looked for, and must NOT appear in `points` under any guise -- neither silently
    dropped nor filled in from a neighbouring size."""
    gate_root = tmp_path / "gate"
    registry = {
        "fake-sft-small": {"base_model": "Qwen/Qwen3-3B"},
        "fake-sft-big": {"base_model": "Qwen/Qwen3-9B"},
    }
    locators = (
        ("sft", "Qwen3-3B", "small-t1", "fake-sft-small"),
        ("sft", "Qwen3-9B", "big-t1", "fake-sft-big"),
    )
    suites = ("musique", "strategyqa")
    # The 3B strategyqa verdict is deliberately never written; every other file exists.
    _write_p7_verdict(
        gate_root,
        "small-t1",
        "fake-sft-small",
        "musique",
        _p7_verdict(0.1, 0.05, 0.15, 200, baseline_model_id="qwen3-3b-base"),
    )
    for suite in suites:
        _write_p7_verdict(
            gate_root, "big-t1", "fake-sft-big", suite, _p7_verdict(0.2, 0.15, 0.25, 300)
        )

    points, absent = gen._p7_collect(locators, suites, registry, gate_root)

    assert len(absent) == 1
    miss = absent[0]
    assert (miss["label"], miss["arm"], miss["suite"]) == ("Qwen3-3B", "sft", "strategyqa")
    assert "small-t1" in miss["reason"] and "fake-sft-small.strategyqa.json" in miss["reason"], (
        f"reason does not name the missing file: {miss['reason']!r}"
    )
    assert not any(p["label"] == "Qwen3-3B" and p["suite"] == "strategyqa" for p in points), (
        "the missing point must not appear in points under any guise"
    )
    assert len(points) == 3, "the three verdicts that DO exist must still be collected"


def test_p7_params_b_is_read_from_the_model_id_via_the_registry_not_hardcoded(gen):
    """`_p7_params_b` must look up whatever string the registry (or the verdict itself) hands
    it for THIS `model_id` and parse that -- proven with sizes that match no real Qwen3
    checkpoint, which a hardcoded per-model table could not get right by accident."""
    # Registry-sourced, with a size that exists nowhere in the real programme.
    assert (
        gen._p7_params_b(
            "some-arbitrary-id", {}, {"some-arbitrary-id": {"base_model": "Qwen/Qwen3-137B"}}
        )
        == 137.0
    )
    # A different arbitrary id/size pair must read back its OWN registry row, not the one above.
    assert (
        gen._p7_params_b(
            "another-id",
            {},
            {
                "another-id": {"base_model": "Qwen/Qwen3-0.3B"},
                "some-arbitrary-id": {"base_model": "Qwen/Qwen3-137B"},
            },
        )
        == 0.3
    )
    # A verdict's OWN base_model field takes precedence over the registry, per this figure's spec.
    assert (
        gen._p7_params_b(
            "some-arbitrary-id",
            {"base_model": "Qwen/Qwen3-0.5B"},
            {"some-arbitrary-id": {"base_model": "Qwen/Qwen3-999B"}},
        )
        == 0.5
    )
    # Refuses rather than inventing a number when neither the verdict nor the registry names one.
    with pytest.raises(gen.FigureRefused):
        gen._p7_params_b("unknown-id", {}, {})
    # Refuses rather than guessing when the named base model does not parse as a size.
    with pytest.raises(gen.FigureRefused):
        gen._p7_params_b("bad-id", {"base_model": "not-a-size-string"}, {})


def test_p7_seed_groups_computes_seed_mean_with_spread_when_replicates_resolve(gen, tmp_path):
    """With every declared seed resolved, `_p7_collect_seed_groups` must draw a SEED MEAN:
    `value` is the mean of the resolved seeds' own values, `value_min`/`value_max` their
    range, `is_seed_mean` True, and `err_lo`/`err_hi` (what the drawing code actually plots
    as a whisker) the seed-range span -- NOT the headline's own bootstrap CI, which is a
    different, narrower thing this house rule says not to substitute for seed spread.
    """
    gate_root = tmp_path / "gate"
    registry = {"fake-1p7b-s0": {"base_model": "Qwen/Qwen3-1.7B"}}
    groups = (("Qwen3-1.7B", "grp-t1", ("fake-1p7b-s0", "fake-1p7b-s1", "fake-1p7b-s2")),)
    suites = ("musique",)
    seed_values = {"fake-1p7b-s0": 0.10, "fake-1p7b-s1": 0.14, "fake-1p7b-s2": 0.09}
    for model_id, value in seed_values.items():
        _write_p7_verdict(
            gate_root,
            "grp-t1",
            model_id,
            "musique",
            _p7_verdict(
                value, value - 0.05, value + 0.05, 200, baseline_model_id="qwen3-1.7b-base"
            ),
        )

    points, absent = gen._p7_collect_seed_groups(groups, suites, registry, gate_root)

    assert absent == []
    assert len(points) == 1
    p = points[0]
    assert p["n_seeds"] == 3
    assert p["is_seed_mean"] is True
    assert p["value"] == pytest.approx(sum(seed_values.values()) / 3)
    assert p["value_min"] == pytest.approx(0.09)
    assert p["value_max"] == pytest.approx(0.14)
    # The whisker plotted is the seed range, not the headline's own (much narrower) CI.
    assert p["err_lo"] == pytest.approx(p["value"] - 0.09)
    assert p["err_hi"] == pytest.approx(0.14 - p["value"])
    assert set(p["seed_model_ids"]) == set(seed_values)
    assert sorted(p["seed_values"]) == pytest.approx(sorted(seed_values.values()))
    assert len(p["seed_paths"]) == 3
    # ci_lo/ci_hi stay the HEADLINE's own bootstrap CI, never averaged or widened across seeds.
    assert p["ci_lo"] == pytest.approx(0.10 - 0.05)
    assert p["ci_hi"] == pytest.approx(0.10 + 0.05)


def test_p7_seed_groups_single_seed_is_not_drawn_as_a_mean(gen, tmp_path, monkeypatch):
    """When only the headline resolves (its declared extra seeds are not on disk yet), the
    point must come back exactly as `_p7_collect` always returned it: `n_seeds == 1`,
    `is_seed_mean` False, `value_min == value_max == value`, and the plotted whisker
    (`err_lo`/`err_hi`) equal to the headline's own bootstrap CI -- a single-seed point must
    never be indistinguishable from a seed mean by its OWN numbers, only by the caller's
    marker style.
    """
    gate_root = tmp_path / "gate"
    registry = {"fake-4b-headline": {"base_model": "Qwen/Qwen3-4B"}}
    groups = (
        ("Qwen3-4B", "grp-t1", ("fake-4b-headline", "fake-4b-headline-s1", "fake-4b-headline-s2")),
    )
    suites = ("musique",)
    _write_p7_verdict(
        gate_root,
        "grp-t1",
        "fake-4b-headline",
        "musique",
        _p7_verdict(0.02, -0.03, 0.07, 132, baseline_model_id="qwen3-4b-base"),
    )
    # The two extra seed ids are declared but deliberately never written -- "training now".

    _reconcile_all(gen, monkeypatch, gate_root)
    points, absent = gen._p7_collect_seed_groups(groups, suites, registry, gate_root)

    assert absent == [], "an unresolved EXTRA seed must not be recorded as an absence"
    assert len(points) == 1
    p = points[0]
    assert p["n_seeds"] == 1
    assert p["is_seed_mean"] is False
    assert p["value"] == p["value_min"] == p["value_max"] == pytest.approx(0.02)
    assert p["err_lo"] == pytest.approx(0.02 - (-0.03))
    assert p["err_hi"] == pytest.approx(0.07 - 0.02)
    assert p["seed_model_ids"] == ["fake-4b-headline"]


def test_p7_seed_groups_missing_headline_is_an_absence_missing_extra_seed_is_silent(
    gen, tmp_path, monkeypatch
):
    """The headline id (`model_ids[0]`) is REQUIRED -- its absence is recorded exactly like
    `_p7_collect`'s. An extra seed id (`model_ids[1:]`) with no verdict file yet is NOT an
    absence: it is silently skipped, because an unfinished training run was never promised by
    a date. This silence is the exact mechanism that lets a new seed be picked up the moment
    its gate lands with no line of the caller's declarations touched.
    """
    gate_root = tmp_path / "gate"
    registry = {
        "fake-headline-a": {"base_model": "Qwen/Qwen3-3B"},
        "fake-headline-b": {"base_model": "Qwen/Qwen3-9B"},
    }
    groups = (
        ("Qwen3-3B", "grp-t1", ("fake-headline-a", "fake-headline-a-s1")),
        ("Qwen3-9B", "grp-t1", ("fake-headline-b", "fake-headline-b-s1")),
    )
    suites = ("musique",)
    # Group A's headline is never written; group B's headline is written, its extra seed is not.
    _write_p7_verdict(
        gate_root,
        "grp-t1",
        "fake-headline-b",
        "musique",
        _p7_verdict(0.05, 0.0, 0.1, 200, baseline_model_id="qwen3-9b-base"),
    )

    _reconcile_all(gen, monkeypatch, gate_root)
    points, absent = gen._p7_collect_seed_groups(groups, suites, registry, gate_root)

    assert len(absent) == 1
    assert (absent[0]["label"], absent[0]["suite"]) == ("Qwen3-3B", "musique")
    assert "fake-headline-a.musique.json" in absent[0]["reason"]
    assert len(points) == 1
    assert points[0]["label"] == "Qwen3-9B"
    assert points[0]["n_seeds"] == 1, "the never-written extra seed must be silently absent"


def test_p7_read_verdict_refuses_the_real_pre_matched_cost_8b_verdict(gen):
    """The concrete, on-disk hazard this guard exists for: `artifacts/gate/8b1/qwen3-8b-sft-
    headline.musique.json` is a REAL file from before the matched-cost migration. It carries
    no top-level `coverage_rule` at all, under scorer_hash `12910e78bb46c73...`, and its
    `criteria.evidence_coverage.value` is -0.0511 -- which is the FIXED-BUDGET (cap-8) delta,
    not a matched-cost delta, because the migration had not split the two fields apart yet.

    `P7_SFT_SEEDS` does not currently declare this directory for the 8B group (it declares
    `8b1-rescored-t20`, the migrated copy, instead), so this is not today's plotted point --
    but it is real data reachable through this figure's own functions given nothing more
    exotic than a `gate_dir` string, which is the bar that matters: prove the guard fires on
    the actual file, not on a fixture standing in for it.
    """
    real_path = gen.P7_GATE_ROOT / "8b1" / "qwen3-8b-sft-headline.musique.json"
    if not real_path.exists():
        pytest.skip(f"{real_path} not found in this checkout")
    doc = json.loads(real_path.read_text())
    assert doc.get("coverage_rule") is None, "fixture assumption changed -- re-check the real file"
    assert doc["criteria"]["evidence_coverage"]["value"] < 0, "expected to still be the cap8 delta"

    with pytest.raises(gen.CoverageRuleMissing, match="coverage_rule=None"):
        gen._p7_read_verdict(gen.P7_GATE_ROOT, "8b1", "qwen3-8b-sft-headline", "musique")


def test_p7_seed_groups_treats_a_real_pre_matched_cost_headline_as_an_absence(gen):
    """`_p7_collect_seed_groups` must convert `CoverageRuleMissing` into a stated absence, not
    let it propagate and refuse the whole figure: a dropped point that says why, in the same
    shape as any other declared absence. Exercised against the same real `8b1` file as the
    test above, through the actual collection function, with a `groups` declaration that
    (unlike today's `P7_SFT_SEEDS`) points the 8B group at it -- proving the conversion on the
    real artifact, not only on the leaf read.
    """
    real_path = gen.P7_GATE_ROOT / "8b1" / "qwen3-8b-sft-headline.musique.json"
    if not real_path.exists():
        pytest.skip(f"{real_path} not found in this checkout")
    registry = {"qwen3-8b-sft-headline": {"base_model": "Qwen/Qwen3-8B"}}
    groups = (("Qwen3-8B", "8b1", ("qwen3-8b-sft-headline",)),)

    points, absent = gen._p7_collect_seed_groups(groups, ("musique",), registry, gen.P7_GATE_ROOT)

    assert points == []
    assert len(absent) == 1
    assert absent[0]["label"] == "Qwen3-8B"
    assert "coverage_rule" in absent[0]["reason"]


def test_p7_read_verdict_refuses_when_coverage_rule_is_not_matched_cost(gen, tmp_path):
    """`coverage_rule` absent (pre-migration) and `coverage_rule="cap8"` (explicit, post-
    migration but not this figure's metric) must both refuse identically: either way, `value`
    is the fixed-budget delta this figure does not plot. Only `"matched_cost"` is accepted.
    """
    gate_root = tmp_path / "gate"
    _write_p7_verdict(
        gate_root,
        "grp",
        "absent-rule",
        "musique",
        _p7_verdict(0.05, 0.0, 0.1, 100, coverage_rule=None),
    )
    _write_p7_verdict(
        gate_root,
        "grp",
        "cap8-rule",
        "musique",
        _p7_verdict(0.05, 0.0, 0.1, 100, coverage_rule="cap8"),
    )
    _write_p7_verdict(
        gate_root,
        "grp",
        "matched-rule",
        "musique",
        _p7_verdict(0.05, 0.0, 0.1, 100, coverage_rule="matched_cost"),
    )

    with pytest.raises(gen.CoverageRuleMissing, match="coverage_rule=None"):
        gen._p7_read_verdict(gate_root, "grp", "absent-rule", "musique")
    with pytest.raises(gen.CoverageRuleMissing, match="coverage_rule='cap8'"):
        gen._p7_read_verdict(gate_root, "grp", "cap8-rule", "musique")
    assert gen._p7_read_verdict(gate_root, "grp", "matched-rule", "musique") is not None


def test_p7_seed_groups_pre_matched_cost_extra_seed_is_silent_not_an_absence(
    gen, tmp_path, monkeypatch
):
    """A pre-matched-cost EXTRA seed is silently skipped, exactly like a missing extra-seed
    file -- it does not create an absence entry and it does not refuse the group; only the
    headline id's absence (missing file or pre-matched-cost) is ever stated.
    """
    gate_root = tmp_path / "gate"
    registry = {"fake-headline": {"base_model": "Qwen/Qwen3-8B"}}
    groups = (("Qwen3-8B", "grp-t1", ("fake-headline", "fake-headline-s1")),)
    _write_p7_verdict(
        gate_root,
        "grp-t1",
        "fake-headline",
        "musique",
        _p7_verdict(0.10, 0.05, 0.15, 200, baseline_model_id="qwen3-8b-base"),
    )
    _write_p7_verdict(
        gate_root,
        "grp-t1",
        "fake-headline-s1",
        "musique",
        _p7_verdict(0.11, 0.06, 0.16, 200, baseline_model_id="qwen3-8b-base", coverage_rule=None),
    )

    _reconcile_all(gen, monkeypatch, gate_root)
    points, absent = gen._p7_collect_seed_groups(groups, ("musique",), registry, gate_root)

    assert absent == []
    assert len(points) == 1
    assert points[0]["n_seeds"] == 1, "the pre-matched-cost extra seed must not be folded in"


def test_p7_collect_treats_a_pre_matched_cost_verdict_as_an_absence(gen, tmp_path):
    """`_p7_collect` (used for the DPO points) goes through the same `_p7_read_verdict`, so it
    must convert `CoverageRuleMissing` into an absence exactly like `_p7_collect_seed_groups`
    does -- the guard is fixed once, at the one place both paths read a verdict, not per path.
    """
    gate_root = tmp_path / "gate"
    registry = {"fake-dpo": {"base_model": "Qwen/Qwen3-8B"}}
    locators = (("dpo", "Qwen3-8B", "grp-t1", "fake-dpo"),)
    _write_p7_verdict(
        gate_root,
        "grp-t1",
        "fake-dpo",
        "musique",
        _p7_verdict(0.10, 0.05, 0.15, 200, baseline_model_id="qwen3-8b-base", coverage_rule="cap8"),
    )

    points, absent = gen._p7_collect(locators, ("musique",), registry, gate_root)

    assert points == []
    assert len(absent) == 1
    assert "coverage_rule" in absent[0]["reason"]


def test_p7_seed_groups_refuses_on_params_b_mismatch_across_seeds(gen, tmp_path):
    """A resolved extra seed whose OWN `baseline_model_id` implies a different base-model size
    than the headline's must refuse, not silently plot a seed mean across two different base
    models.
    """
    gate_root = tmp_path / "gate"
    registry = {"fake-headline": {"base_model": "Qwen/Qwen3-8B"}}
    groups = (("Qwen3-8B", "grp-t1", ("fake-headline", "fake-headline-s1")),)
    suites = ("musique",)
    _write_p7_verdict(
        gate_root,
        "grp-t1",
        "fake-headline",
        "musique",
        _p7_verdict(0.10, 0.05, 0.15, 200, baseline_model_id="qwen3-8b-base"),
    )
    _write_p7_verdict(
        gate_root,
        "grp-t1",
        "fake-headline-s1",
        "musique",
        # Same registry row (8B), but this seed's OWN verdict claims a 3B baseline.
        _p7_verdict(0.11, 0.06, 0.16, 200, baseline_model_id="qwen3-3b-base"),
    )

    with pytest.raises(gen.FigureRefused):
        gen._p7_collect_seed_groups(groups, suites, registry, gate_root)


def test_p7_seed_groups_refuses_on_scorer_hash_mismatch_across_seeds(gen, tmp_path):
    """Two resolved seeds under different `scorer_hash` values must refuse to fold into one
    seed mean. This is the exact failure mode found in the real corpus: `qwen3-8b-sft-
    headline-s1`/`-s2` exist on disk under a DIFFERENT scorer_hash than the headline's, in a
    different directory, and must never be silently averaged in just because an id with that
    name resolved somewhere.
    """
    gate_root = tmp_path / "gate"
    registry = {"fake-headline": {"base_model": "Qwen/Qwen3-8B"}}
    groups = (("Qwen3-8B", "grp-t1", ("fake-headline", "fake-headline-s1")),)
    suites = ("musique",)
    _write_p7_verdict(
        gate_root,
        "grp-t1",
        "fake-headline",
        "musique",
        _p7_verdict(0.10, 0.05, 0.15, 200, scorer_hash="hash-a", baseline_model_id="qwen3-8b-base"),
    )
    _write_p7_verdict(
        gate_root,
        "grp-t1",
        "fake-headline-s1",
        "musique",
        _p7_verdict(0.11, 0.06, 0.16, 200, scorer_hash="hash-b", baseline_model_id="qwen3-8b-base"),
    )

    with pytest.raises(gen.FigureRefused, match="scorer_hash"):
        gen._p7_collect_seed_groups(groups, suites, registry, gate_root)


def test_p7_seed_groups_refuses_on_baseline_cell_mismatch_across_seeds(gen, tmp_path):
    """`baseline_n_states`/`baseline_p_stop_given_done` are asserted elsewhere in this module
    to be a FIXED property of (base model, suite), identical across every checkpoint trained
    from that base. This is where that claim is actually verified: two seeds disagreeing on
    the baseline cell must refuse rather than pass the assertion through unchecked.
    """
    gate_root = tmp_path / "gate"
    registry = {"fake-headline": {"base_model": "Qwen/Qwen3-8B"}}
    groups = (("Qwen3-8B", "grp-t1", ("fake-headline", "fake-headline-s1")),)
    suites = ("musique",)
    _write_p7_verdict(
        gate_root,
        "grp-t1",
        "fake-headline",
        "musique",
        _p7_verdict(0.10, 0.05, 0.15, 200, p_stop=0.05, baseline_model_id="qwen3-8b-base"),
    )
    _write_p7_verdict(
        gate_root,
        "grp-t1",
        "fake-headline-s1",
        "musique",
        # Same base model, same suite -- but a different measured baseline P(STOP|done).
        _p7_verdict(0.11, 0.06, 0.16, 200, p_stop=0.09, baseline_model_id="qwen3-8b-base"),
    )

    with pytest.raises(gen.FigureRefused, match="baseline cell"):
        gen._p7_collect_seed_groups(groups, suites, registry, gate_root)


def test_p7_seed_groups_refuses_on_task_count_mismatch_across_seeds(gen, tmp_path):
    """Seeds disagreeing on `n`, the matched-cost eval task count, must refuse rather than
    average across two differently-sized evals as though they were the same measurement.
    """
    gate_root = tmp_path / "gate"
    registry = {"fake-headline": {"base_model": "Qwen/Qwen3-8B"}}
    groups = (("Qwen3-8B", "grp-t1", ("fake-headline", "fake-headline-s1")),)
    suites = ("musique",)
    _write_p7_verdict(
        gate_root,
        "grp-t1",
        "fake-headline",
        "musique",
        _p7_verdict(0.10, 0.05, 0.15, 200, baseline_model_id="qwen3-8b-base"),
    )
    _write_p7_verdict(
        gate_root,
        "grp-t1",
        "fake-headline-s1",
        "musique",
        _p7_verdict(0.11, 0.06, 0.16, 199, baseline_model_id="qwen3-8b-base"),
    )

    with pytest.raises(gen.FigureRefused, match="task count"):
        gen._p7_collect_seed_groups(groups, suites, registry, gate_root)


def _dev(deviation: float, spread_half_range: float | None) -> dict:
    """A minimal `_p7_middle_size_deviation`-shaped dict carrying only the two fields
    `_dip_ratio` and `_p7_sign_flip_synthesis` actually read; the rest of that function's
    real return shape (labels, interpolated value, provenance flags) is `_dip_read`'s concern,
    not this pair's.
    """
    return {
        "lbl_lo": "lo",
        "lbl_mid": "mid",
        "lbl_hi": "hi",
        "v_mid": 0.0,
        "interp": 0.0,
        "deviation": deviation,
        "spread_label": "lo",
        "spread_half_range": spread_half_range,
        "is_direct": False,
        "multiple_candidates": False,
    }


def test_dip_ratio_is_none_without_a_deviation_or_a_spread(gen):
    """No deviation (not three sizes plotted) and no spread (nothing in the chain has a seed
    mean yet) must both read as "no ratio to compute", not a crash or a fabricated zero.
    """
    assert gen._dip_ratio(None) is None
    assert gen._dip_ratio(_dev(0.05, None)) is None


def test_dip_ratio_divides_deviation_by_half_range_and_guards_zero(gen):
    """The one formula `_dip_read` and `_p7_sign_flip_synthesis` must share: abs(deviation)
    over spread_half_range, with a zero half-range read as infinite rather than raising
    ZeroDivisionError.
    """
    assert gen._dip_ratio(_dev(-0.08, 0.02)) == pytest.approx(4.0)
    assert gen._dip_ratio(_dev(0.03, 0.02)) == pytest.approx(1.5)
    assert gen._dip_ratio(_dev(0.03, 0.0)) == float("inf")


def test_sign_flip_synthesis_needs_exactly_two_suites(gen):
    """The sign-flip comparison is only defined for a PAIR of suites sharing the same three
    checkpoints; one suite or three must say so rather than guess at a pairing.
    """
    one = gen._p7_sign_flip_synthesis({"musique": _dev(-0.07, 0.01)})
    assert "not two" in one
    three = gen._p7_sign_flip_synthesis(
        {
            "musique": _dev(-0.07, 0.01),
            "strategyqa": _dev(0.02, 0.01),
            "frames": _dev(0.0, 0.01),
        }
    )
    assert "not two" in three


def test_sign_flip_synthesis_defers_without_a_ratio_for_every_suite(gen):
    """If either suite has no deviation (not three sizes) or no spread yet, there is no joint
    read to make -- the guard must fire before any sign or ratio comparison is attempted.
    """
    txt = gen._p7_sign_flip_synthesis({"musique": None, "strategyqa": _dev(0.02, 0.01)})
    assert "no joint sign read" in txt
    txt2 = gen._p7_sign_flip_synthesis(
        {"musique": _dev(-0.07, None), "strategyqa": _dev(0.02, 0.01)}
    )
    assert "no joint sign read" in txt2


def test_sign_flip_synthesis_is_the_strong_claim_when_both_exceed_and_signs_disagree(gen):
    """The coordinator's own numbers: musique 0.0663 below trend at 7.8x its judged spread,
    strategyqa 0.0202 above trend at 5.3x -- both clear their noise floor, in opposite
    directions. This is the one case that earns the strong "sign flip is the finding" text.
    """
    devs = {
        "musique": _dev(-0.0663, 0.0663 / 7.8),
        "strategyqa": _dev(0.0202, 0.0202 / 5.3),
    }
    txt = gen._p7_sign_flip_synthesis(devs)
    assert "sign flip is the finding" in txt
    assert "7.8x" in txt
    assert "5.3x" in txt
    assert "below" in txt and "above" in txt
    assert "property of" in txt and "(size, suite) PAIR" in txt


def test_sign_flip_synthesis_is_the_weaker_claim_when_both_exceed_and_agree_in_sign(gen):
    """Two suites clearing their own noise floor in the SAME direction is not a sign flip, and
    must not be worded as one -- it does not by itself rule out a real effect, but it is not
    the strong joint claim either.
    """
    devs = {"musique": _dev(-0.05, 0.01), "strategyqa": _dev(-0.03, 0.01)}
    txt = gen._p7_sign_flip_synthesis(devs)
    assert "sign flip is the finding" not in txt
    assert "agree in direction" in txt


def test_sign_flip_synthesis_is_the_weakest_claim_when_either_ratio_fails_to_clear_noise(gen):
    """When at least one suite's deviation does not even clear its own noise floor, neither a
    sign-flip claim nor a same-direction claim is honest -- the text must say so plainly.
    """
    devs = {"musique": _dev(-0.005, 0.01), "strategyqa": _dev(0.02, 0.01)}
    txt = gen._p7_sign_flip_synthesis(devs)
    assert "sign flip is the finding" not in txt
    assert "agree in direction" not in txt
    assert "do not jointly support" in txt


# =========================================================================== P12

# A SYNTHETIC LOG, COPIED FROM THE REAL ONE'S SHAPE. `transformers` prints its log dict onto
# the SAME line as the tqdm progress bar (the carriage returns flatten when stdout is a file),
# the values are QUOTED STRINGS, and the dict itself carries no step -- the step is only in the
# progress token that precedes it. All three are why the parser is not `json.loads` and why it
# is pinned here rather than trusted. This fixture is four log lines of `rung1.766502.out`
# with the numbers changed.
SYNTH_LOG = (
    "  0%|          | 9/5480 [02:04<20:46:01, 13.67s/it]"
    "  0%|          | 10/5480 [02:17<20:30:42, 13.50s/it]                       "
    "{'loss': '7.316', 'grad_norm': '22.9', 'learning_rate': '5.455e-06', 'epoch': '0.00365'}\n"
    "  0%|          | 10/5480 [02:17<20:30:42, 13.50s/it]"
    "  0%|          | 20/5480 [04:34<20:50:16, 13.74s/it]                       "
    "{'loss': '5.268', 'grad_norm': '9.398', 'learning_rate': '1.152e-05', 'epoch': '0.0073'}\n"
    " 50%|#####     | 2740/5480 [10:00<10:00:00, 13.10s/it]                      "
    "{'loss': '0.1602', 'grad_norm': '0.4335', 'learning_rate': '5.0e-05', 'epoch': '1.0'}\n"
    " 99%|#########9| 5480/5480 [20:00<00:00, 13.10s/it]                         "
    "{'loss': '0.0900', 'grad_norm': '0.2109', 'learning_rate': '0.0', 'epoch': '2.0'}\n"
)


def _tierA(step_label: str, nll: float, p_stop: float, p_ask: float, acc: float) -> dict:
    """The shape `pi train eval-offline` writes (src/pi_run/cmd_train.py:cmd_train_eval_offline)."""
    return {
        "checkpoint": f"/tmp/pinq-ckpt-eval/checkpoint-{step_label}",
        "tokenizer": "Qwen/Qwen3-8B",
        "dev_sft": "data/rl/dev/sft.dev.sample.jsonl",
        "dev_pairs": "data/rl/dev/pairs.dev.sample.jsonl",
        "n_sft_rows": 1000,
        "n_pairs": 1485,
        "chat_template": True,
        "enable_thinking": False,
        "sft_nll": {"nll_per_token": nll, "nll_per_example": nll * 12, "n": 1000.0},
        "stop_confusion": {
            "p_stop_given_done": p_stop,
            "p_ask_given_not_done": p_ask,
            "n_done": 650.0,
            "n_not_done": 350.0,
            "n_skipped_no_reference": 0.0,
            "n_skipped_no_label": 0.0,
            "n_skipped_unparseable": 0.0,
        },
        "pair_accuracy": {
            "acc": acc,
            "acc_by_kind": {"ask_ask": acc, "ask_stop": 0.9, "ask_stop_synth": 0.95},
            "acc_by_decided_by": {"gain": acc},
            "mean_margin": 1.5,
            "n": 1485,
        },
    }


@pytest.fixture
def p12_dir(gen, tmp_path, monkeypatch):
    """A stand-in for `artifacts/hpc/rung1-8b/`, which `scripts/hpc/pull_rung1.sh` fills."""
    d = tmp_path / "ccc"
    d.mkdir()
    monkeypatch.setattr(gen, "P12_DIR", d)
    return d


def test_p12_parses_step_and_loss_out_of_the_progress_bar_line(gen, p12_dir):
    """The step lives in the tqdm token, not in the dict, and every value is a quoted string."""
    log = p12_dir / "rung1.766502.out"
    log.write_text(SYNTH_LOG)
    pts = gen.parse_training_log(log)
    assert [p["step"] for p in pts] == [10, 20, 2740, 5480]
    assert [p["loss"] for p in pts] == [7.316, 5.268, 0.1602, 0.09]
    assert pts[0]["learning_rate"] == 5.455e-06
    assert pts[-1]["epoch"] == 2.0
    assert pts[0]["total"] == 5480


def test_p12_draws_both_panels_when_the_log_and_an_eval_exist(gen, p12_dir, tmp_path):
    (p12_dir / "rung1.766502.out").write_text(SYNTH_LOG)
    (p12_dir / "rung1-8b.ckpt-0.tierA.json").write_text(
        json.dumps(_tierA("0", 2.5, 0.10, 0.90, 0.50))
    )
    (p12_dir / "rung1-8b.ckpt-200.tierA.json").write_text(
        json.dumps(_tierA("200", 0.4, 0.80, 0.70, 0.62))
    )
    out = tmp_path / "figs"
    assert gen.main(["--out", str(out), "--only", "P12"]) == 0
    d = out / "P12_rung1_curves"
    for name in ("P12_rung1_curves.pdf", "P12_rung1_curves.png", "provenance.json", "caption.md"):
        assert (d / name).exists() and (d / name).stat().st_size > 0, f"missing {name}"
    prov = json.loads((d / "provenance.json").read_text())
    assert prov["measured"] is True
    assert len(prov["inputs"]) == 3, "provenance must name the log and both verdicts"
    for path, digest in prov["inputs"].items():
        assert len(digest) == 64, f"{path} has no sha256"
    assert prov["extra"]["n_loss_points"] == 4
    assert prov["extra"]["dev_steps"] == [0, 200]
    assert not (d / "NOT_YET.md").exists(), "both panels drawn, nothing is awaited"


def test_p12_draws_the_loss_panel_alone_and_says_what_the_dev_panel_awaits(gen, p12_dir, tmp_path):
    """The loss curve is measured the moment the log exists; the dev panel is not. Drawing the
    half that is measured and NAMING the half that is not is the whole point of this figure
    during a run -- but the drawing must not imply the missing half is merely empty."""
    (p12_dir / "rung1.766502.out").write_text(SYNTH_LOG)
    out = tmp_path / "figs"
    assert gen.main(["--out", str(out), "--only", "P12"]) == 0
    d = out / "P12_rung1_curves"
    assert (d / "P12_rung1_curves.pdf").exists(), "the measured loss panel was not drawn"
    assert (d / "NOT_YET.md").exists(), "the absent dev panel was not declared"
    note = (d / "NOT_YET.md").read_text()
    assert "rung1-8b.ckpt-" in note, "the note does not name the file it awaits"
    prov = json.loads((d / "provenance.json").read_text())
    assert prov["extra"]["dev_steps"] == []
    assert prov["extra"]["n_loss_points"] == 4


def test_p12_draws_nothing_at_all_without_the_log(gen, p12_dir, tmp_path):
    out = tmp_path / "figs"
    assert gen.main(["--out", str(out), "--only", "P12"]) == 0
    d = out / "P12_rung1_curves"
    assert (d / "NOT_YET.md").exists()
    for forbidden in ("P12_rung1_curves.pdf", "P12_rung1_curves.png", "provenance.json"):
        assert not (d / forbidden).exists(), f"{forbidden} drawn from no data at all"


def test_p12_places_the_final_verdict_at_the_last_logged_step(gen, p12_dir, tmp_path):
    """`ckpt-final` names no step. It is the END of the run, and the only record of which step
    that was is the log -- so a final verdict with no log to place it against is dropped, not
    guessed onto the x axis."""
    (p12_dir / "rung1.766502.out").write_text(SYNTH_LOG)
    (p12_dir / "rung1-8b.ckpt-final.tierA.json").write_text(
        json.dumps(_tierA("final", 0.3, 0.85, 0.75, 0.66))
    )
    out = tmp_path / "figs"
    assert gen.main(["--out", str(out), "--only", "P12"]) == 0
    prov = json.loads((out / "P12_rung1_curves" / "provenance.json").read_text())
    assert prov["extra"]["dev_steps"] == [5480], "the final verdict was not placed at 5,480"


def test_p12_picks_the_highest_job_id_not_the_alphabetical_last(gen, p12_dir, tmp_path):
    """Two logs in the pull directory and the wrong one wins lexicographically: `rung1.9999.out`
    sorts after `rung1.10000.out`. That would draw one run's loss beside another run's dev
    verdicts, and nothing in the figure would look wrong."""
    (p12_dir / "rung1.9999.out").write_text(SYNTH_LOG.replace("'7.316'", "'9.999'"))
    (p12_dir / "rung1.10000.out").write_text(SYNTH_LOG)
    out = tmp_path / "figs"
    assert gen.main(["--out", str(out), "--only", "P12"]) == 0
    prov = json.loads((out / "P12_rung1_curves" / "provenance.json").read_text())
    assert prov["extra"]["log_file"].endswith("rung1.10000.out"), "picked the older job's log"
    assert prov["extra"]["n_logs_present"] == 2
    assert prov["extra"]["first_loss"] == 7.316


# =========================================================================== P12 tag parsing
#
# `read_ckpt_evals` derived the step by SLICING A LITERAL run-name prefix:
#
#     tag = p.name[len("rung1-8b.ckpt-") : -len(".tierA.json")]
#
# i.e. from a fixed index 14. That is correct only while every filename begins with exactly
# `rung1-8b.`, which today's caller guarantees by globbing `rung1-8b.ckpt-*.tierA.json` inside
# `artifacts/hpc/rung1-8b`. The defect is therefore LATENT rather than active -- and it is armed:
# MEASURED 2026-09-17, at index 14 the runs already on disk slice to
#
#     rung1-8b-headline.ckpt-1788      -> 'ine.ckpt-1788'
#     rung1-gptoss20b-headline.ckpt-0  -> 'b-headline.ckpt-0'
#     rung1-granite8b-headline-s0      -> 'b-headline-s0.ckpt-200'
#     rung1-32b-headline.ckpt-final    -> 'line.ckpt-final'
#
# none of which is `final` or a digit, so each hits the `continue` and vanishes WITHOUT A WORD.
# The scale ladder is gaining a third size and a second family, whose run names are exactly
# these shapes, so the first person to widen the glob gets a silently short figure. A tag is a
# property of the FILE; it must be read out of the name, and a name that cannot be parsed is a
# refusal, not a skip -- the same rule as the base resolution in eval_checkpoints.sh.


def _p12_names(d, *names):
    for n in names:
        (d / n).write_text(json.dumps(_tierA("x", 0.4, 0.8, 0.7, 0.6)))
    return sorted(d.glob("*.tierA.json"))


def test_read_ckpt_evals_derives_the_tag_from_the_file_not_a_hardcoded_run_prefix(gen, tmp_path):
    """Every run name shape now on disk, each at a step only the filename knows."""
    paths = _p12_names(
        tmp_path,
        "rung1-8b.ckpt-400.tierA.json",
        "rung1-8b-headline.ckpt-1788.tierA.json",
        "rung1-gptoss20b-headline.ckpt-0.tierA.json",
        "rung1-granite8b-headline-s0.ckpt-200.tierA.json",
        "rung1-32b-headline.ckpt-3600.tierA.json",
    )
    rows = gen.read_ckpt_evals(paths, final_step=None)
    assert [r["step"] for r in rows] == [0, 200, 400, 1788, 3600], (
        f"a run name the literal prefix does not match was dropped: {[r['tag'] for r in rows]}"
    )
    assert [r["tag"] for r in rows] == ["0", "200", "400", "1788", "3600"]


def test_read_ckpt_evals_places_final_by_the_log_for_any_run_name(gen, tmp_path):
    """`final` names no step of its own, so it is placed at the last step the log recorded --
    unchanged behaviour, now reachable for a run whose name is not `rung1-8b`."""
    paths = _p12_names(tmp_path, "rung1-granite8b-headline-s0.ckpt-final.tierA.json")
    rows = gen.read_ckpt_evals(paths, final_step=5480)
    assert [(r["tag"], r["step"]) for r in rows] == [("final", 5480)]


def test_read_ckpt_evals_still_drops_final_when_no_log_gives_it_a_position(gen, tmp_path):
    """Regression guard on a DELIBERATE skip, documented in the function's own docstring:
    without a log there is no measured x for `final`, and inventing one would put the single
    unmeasured point on the curve. That stays a skip; only an UNPARSEABLE NAME becomes a
    refusal."""
    paths = _p12_names(tmp_path, "rung1-8b.ckpt-final.tierA.json")
    assert gen.read_ckpt_evals(paths, final_step=None) == []


def test_read_ckpt_evals_refuses_a_name_it_cannot_parse_rather_than_dropping_it(gen, tmp_path):
    """The silent `continue` is what let a whole family go unplotted with no complaint. A file
    that reached this function is a file someone's glob selected, so a name it cannot read is a
    disagreement between the glob and the parser, and the honest response is to say so."""
    paths = _p12_names(tmp_path, "rung1-8b.ckpt-.tierA.json")
    with pytest.raises(gen.FigureRefused, match="ckpt"):
        gen.read_ckpt_evals(paths, final_step=None)

    _p12_names(tmp_path, "nonsense.json")
    with pytest.raises(gen.FigureRefused):
        gen.read_ckpt_evals(sorted(tmp_path.glob("nonsense.json")), final_step=None)


# ====================================================== the BCa unit-order refusal (P7)
#
# `4b7b24b` canonicalised the order `cluster_bootstrap` (and its contract-3 replica
# `pinq_train.gate.bca_ci`) resamples its units in. The RNG draws INDICES, so before that
# commit the caller's list order was an input to the endpoints. Every gate verdict on disk
# predates it, and `_p7_read_verdict` reads `ci_lo`/`ci_hi` out of one AS-IS -- so P7's
# whisker on a hollow (single-seed) point, its DPO error bars and its caption's `[lo, hi]`
# would all typeset an endpoint computed under an order no record states.
#
# There is no field in a verdict that dates it (measured: no timestamp, no code version, no
# git sha, no graph_version anywhere in any of the 176 stored gate verdicts), so the guard
# keys on the verdict's own CONTENT DIGEST against `BCA_ORDER_RECONCILED` -- the audit record
# of which verdict bytes have actually been reconciled against the post-fix estimator.


def test_p7_refuses_a_bootstrap_endpoint_from_an_unreconciled_verdict(gen, tmp_path):
    """The interval, not the point. A verdict whose bytes are not in `BCA_ORDER_RECONCILED`
    may still contribute its (order-invariant) `value`, but the moment P7 would plot or typeset
    that verdict's `ci_lo`/`ci_hi` it must refuse, naming the verdict file AND the endpoint.
    """
    gate_root = tmp_path / "gate"
    _write_p7_verdict(
        gate_root, "grp", "unreconciled", "musique", _p7_verdict(0.10, 0.05, 0.15, 200)
    )
    row = gen._p7_read_verdict(gate_root, "grp", "unreconciled", "musique")
    assert row is not None and row["value"] == 0.10, "the POINT estimate is order-invariant"

    for endpoint in ("ci_lo", "ci_hi"):
        with pytest.raises(gen.PreSortInterval) as exc:
            gen._p7_plotted_endpoint(row, endpoint=endpoint)
        msg = str(exc.value)
        assert "unreconciled.musique.json" in msg, "the refusal must name the verdict"
        assert endpoint in msg, "the refusal must name the endpoint"
        assert gen.BCA_UNIT_ORDER_FIX in msg, "the refusal must name the commit it keys on"
    assert issubclass(gen.PreSortInterval, gen.FigureRefused)
    assert not issubclass(gen.PreSortInterval, gen.CoverageRuleMissing), (
        "P7's three call sites catch CoverageRuleMissing and turn it into a stated absence; "
        "a pre-sort interval must REFUSE instead of being absorbed the same way"
    )


def test_p7_plots_a_bootstrap_endpoint_once_its_verdict_digest_is_reconciled(gen, tmp_path):
    """The predicate must be able to take the other value: a verdict whose digest IS in the
    reconciliation record passes through untouched. A guard that can only ever refuse is a
    hard stop dressed up as a check.
    """
    gate_root = tmp_path / "gate"
    _write_p7_verdict(gate_root, "grp", "reconciled", "musique", _p7_verdict(0.10, 0.05, 0.15, 200))
    row = gen._p7_read_verdict(gate_root, "grp", "reconciled", "musique")
    assert row is not None
    digest = gen.sha256_file(row["path"])
    assert row["sha256"] == digest, "the row must carry the digest the guard keys on"

    reconciled = {digest: "recomputed in this test, post-4b7b24b"}
    assert gen._p7_plotted_endpoint(row, endpoint="ci_lo", reconciled=reconciled) == 0.05
    assert gen._p7_plotted_endpoint(row, endpoint="ci_hi", reconciled=reconciled) == 0.15


def test_p7_reads_no_bootstrap_endpoint_except_through_the_guard(gen):
    """Every call site, not just the reader. FOUR places turn a stored endpoint into output --
    the hollow point's whisker, the DPO error bars, the caption's `[lo, hi]` and the provenance
    row -- and the coverage-rule refusal exists because guarding only some of them is how one
    gets missed. The provenance row is the one that armed the real asset.

    Asserted by the exact guarded text at each site, not by a pattern that a pre-existing line
    could satisfy, plus the absence of any bare subscript read below the guard's definition.
    """
    src = GEN.read_text()
    for site, expected in {
        "hollow whisker lo": '(headline_v["value"] - _p7_plotted_endpoint(headline_v, endpoint="ci_lo"))',
        "hollow whisker hi": '(_p7_plotted_endpoint(headline_v, endpoint="ci_hi") - headline_v["value"])',
        "dpo error bar lo": '[p["value"] - _p7_plotted_endpoint(p, endpoint="ci_lo") for p in dpo]',
        "dpo error bar hi": '[_p7_plotted_endpoint(p, endpoint="ci_hi") - p["value"] for p in dpo]',
        "caption row": "_fmt(_p7_plotted_endpoint(p, endpoint='ci_lo'))",
        "provenance row": "d[endpoint] = _p7_plotted_endpoint(p, endpoint=endpoint)",
    }.items():
        assert expected in src, f"{site} does not go through the guard"

    lines = src.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("def _p7_plotted_endpoint"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("def p8_frontier_trained"))
    bare = [
        (i + 1, ln.strip())
        for i, ln in enumerate(lines[start:end], start=start)
        if ('["ci_lo"]' in ln or '["ci_hi"]' in ln or "['ci_lo']" in ln or "['ci_hi']" in ln)
        and "_p7_plotted_endpoint" not in ln
    ]
    assert bare == [], f"bootstrap endpoint read without the guard at {bare}"


def test_p7_refuses_the_real_pre_sort_gate_corpus(gen, tmp_path):
    """Replayed against real history, not a fixture. Every verdict in `artifacts/gate/`
    predates `4b7b24b` and none has been reconciled, so P7 against the real corpus must refuse
    -- and it must be THIS refusal, naming a real verdict path, not a coincidental one.
    """
    if not (gen.CHECKPOINTS.exists() and gen.P7_GATE_ROOT.exists()):
        pytest.skip("real gate corpus absent")
    # SCOPED TO VERDICTS. The record may hold ASSET digests (P13's regenerated provenance is
    # one), but no gate VERDICT has been re-derived, so none of the ten P7 reads can resolve.
    verdicts = [
        p
        for p in gen.P7_GATE_ROOT.rglob("*.json")
        if gen.sha256_file(p) in gen.BCA_ORDER_RECONCILED
    ]
    assert verdicts == [], (
        f"{len(verdicts)} verdict digest(s) are recorded as reconciled; an entry without a "
        "recompute behind it is how a pre-sort endpoint gets published"
    )
    with pytest.raises(gen.PreSortInterval) as exc:
        gen.p7_scaling_by_model_size(tmp_path / "out")
    assert "artifacts/gate/" in str(exc.value)


# ============================================ the same refusal, at the level of a BUILT ASSET
#
# P7's guard fires while P7 is being drawn. It cannot fire for `F1_frontier`, `F2` or `F3`,
# which are drawn by `pi render` (`src/pi_run/render.py`) and never pass through this file --
# and F1 is the one that reaches the reader. F1 was found armed and unnamed: 192 of its own
# bootstrap band endpoints (`auc_lo`/`auc_hi`, `lo_pointwise`/`hi_pointwise`, and the sup-t
# simultaneous band), a built `.pdf`/`.png`, and `generated_at_utc` 2026-08-30, eighteen days
# before `4b7b24b`.
#
# `code_version` CANNOT be the predicate here, and reading it as one is the error this comment
# exists to prevent: `pi_eval.report.Aggregation.code_versions` is
# `SELECT DISTINCT code_version FROM runs`, i.e. the code that produced the RUNS, not the
# revision that computed the bands. An asset built today from runs recorded last month lists
# last month's commits. So the asset audit keys on the SAME content digest P7's guard keys on,
# applied to the asset's own `provenance_digest`.


def test_asset_audit_refuses_a_built_asset_whose_bootstrap_endpoints_are_unreconciled(
    gen, tmp_path
):
    """A built figure asset carrying interval endpoints must be refused unless its
    `provenance_digest` is reconciled -- naming the asset AND the endpoint fields, the same way
    the drawing guard names the verdict and the endpoint.
    """
    d = tmp_path / "F9_fake"
    d.mkdir()
    (d / "F9_fake.pdf").write_bytes(b"%PDF-1.4")
    (d / "provenance.json").write_text(
        json.dumps(
            {
                "provenance_digest": "deadbeef",
                "generated_at_utc": "2026-08-30T20:58:36+00:00",
                "code_version": ["032bd0691bbc3dcd3e18fdb34168c176581e3e05"],
                "extra": {"arms": {"trained": {"auc_lo": 0.43, "auc_hi": 0.78}}},
            }
        )
    )
    with pytest.raises(gen.PreSortInterval) as exc:
        gen.audit_figure_assets(tmp_path)
    msg = str(exc.value)
    assert "F9_fake" in msg, "the refusal must name the asset"
    assert "auc_lo" in msg and "auc_hi" in msg, "the refusal must name the endpoint fields"
    assert "code_version" not in msg, (
        "code_version is the code that produced the RUNS, not the estimator; a refusal that "
        "cites it teaches the reader to date an asset by the wrong field"
    )


def test_asset_audit_passes_an_asset_with_no_interval_endpoints_and_one_reconciled(gen, tmp_path):
    """Two ways to be clean, and the audit must distinguish them from each other and from a
    refusal: an asset that carries NO bootstrap endpoint at all (F2/F3 are really like this),
    and one whose digest has been reconciled.
    """
    bare = tmp_path / "F8_bare"
    bare.mkdir()
    (bare / "F8_bare.pdf").write_bytes(b"%PDF-1.4")
    (bare / "provenance.json").write_text(
        json.dumps({"provenance_digest": "cafe", "extra": {"depths": [0, 1, 2]}})
    )
    assert gen.audit_figure_assets(tmp_path) == {"F8_bare": "no interval endpoint"}

    ok = tmp_path / "F7_ok"
    ok.mkdir()
    (ok / "F7_ok.pdf").write_bytes(b"%PDF-1.4")
    (ok / "provenance.json").write_text(
        json.dumps({"provenance_digest": "f00d", "extra": {"arms": {"a": {"auc_lo": 0.1}}}})
    )
    got = gen.audit_figure_assets(tmp_path, reconciled={"f00d": "recomputed post-4b7b24b"})
    assert got["F7_ok"] == "1 endpoint, reconciled"


def test_asset_audit_finds_the_real_f1_frontier_and_p13_and_p7(gen):
    """Replayed against the real tree, not a fixture. F1_frontier is the figure that reaches
    the reader and it must be caught; F2/F3 must come back clean rather than be swept in with
    it, because a guard that refuses everything cannot tell anyone which asset to fix.
    """
    figs = gen.REPO / "figures"
    if not (figs / "F1_frontier" / "provenance.json").exists():
        pytest.skip("figure assets absent")
    with pytest.raises(gen.PreSortInterval) as exc:
        gen.audit_figure_assets(figs)
    assert "F1_frontier" in str(exc.value)

    clean = gen.audit_figure_assets(figs, only=("F2_coverage_at_depth", "F3_phi_vs_null"))
    assert clean == {
        "F2_coverage_at_depth": "no interval endpoint",
        "F3_phi_vs_null": "no interval endpoint",
    }
