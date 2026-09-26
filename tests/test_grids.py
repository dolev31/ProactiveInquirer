"""Sweep grids: the unit the preregistration freezes."""

import pytest

from pi_run.grids import load, load_all, total_units


def test_every_shipped_grid_loads_and_prices_itself():
    grids = load_all()
    assert {g.name for g in grids} >= {
        "tier1_pilot",
        "tier1_confirmatory",
        "tier2_confirmatory",
        "tier1_oracle",
        "tier3_drgym",
        "pare_transfer",
    }
    for g in grids:
        assert g.n_units > 0, f"{g.name} prices to zero units"
        assert g.source_sha256, f"{g.name} has no content hash"


def test_the_grid_hash_is_content_addressed(tmp_path):
    """The hash goes into the run manifest, so a silently widened grid shows up as a changed
    run identity instead of as extra rows nobody notices."""
    p = tmp_path / "g.yaml"
    p.write_text("name: g\nsuites: [synth]\narms: [a]\nn_tasks: 2\n")
    a = load(p).source_sha256
    p.write_text("name: g\nsuites: [synth]\narms: [a, b]\nn_tasks: 2\n")
    assert load(p).source_sha256 != a


def test_unit_count_is_the_full_cross_product(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("name: g\nsuites: [a, b]\narms: [x, y, z]\nseeds: [0, 1]\nn_tasks: 10\n")
    assert load(p).n_units == 2 * 3 * 2 * 10


def test_a_grid_missing_a_required_key_is_refused(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: g\narms: [a]\n")
    with pytest.raises(ValueError, match="suites"):
        load(p)


def test_the_pilot_is_marked_burned():
    """Data used to choose a threshold cannot also be used to test it."""
    pilot = next(g for g in load_all() if g.name == "tier1_pilot")
    assert pilot.pilot is True


def test_every_kill_switch_arm_is_in_the_pilot():
    """The whole point of the pilot is to find out early; an absent kill switch defeats it."""
    pilot = next(g for g in load_all() if g.name == "tier1_pilot")
    assert {"verbosity", "self_inquire", "inquirer_noevidence", "parallel_replay"} <= set(
        pilot.arms
    )


def test_oracle_arms_are_isolated_from_the_confirmatory_grid():
    """gold_evidence / oracle_vreq read gold; they must never share a grid with a primary
    endpoint, because the aggregator's exclusion is the last line of defence, not the first."""
    conf = next(g for g in load_all() if g.name == "tier1_confirmatory")
    oracle = next(g for g in load_all() if g.name == "tier1_oracle")
    assert not ({"gold_evidence", "oracle_vreq"} & set(conf.arms))
    assert set(oracle.arms) == {"gold_evidence", "oracle_vreq"}
    assert oracle.exploratory


def test_tau2_grid_uses_every_task_and_three_seeds():
    """n=97 is fixed by the benchmark, so seeds are the only lever on variance here."""
    g = next(x for x in load_all() if x.name == "tier2_confirmatory")
    assert g.n_tasks == 97 and len(g.seeds) == 3


def test_total_units_is_reported_before_launch():
    assert total_units(load_all()) > 10_000


def test_retrieval_budget_is_per_suite_where_it_has_to_be():
    """k is NOT a free parameter: it decides whether the phenomenon is measurable at all.

    As k grows the full question's top-k absorbs more gold evidence and leaves less for a
    sub-question to add. Measured with `scripts/probe_retrieval.py` at n=100, zero tokens
    (% of tasks holding gold evidence a sub-question finds and the full question misses):

        musique     k2 63%  k3 67%  k5 67%  k8 62%   robust everywhere
        strategyqa  k2 42%  k3 29%  k5 21%  k8 17%   only k=2 clears the 30% bar
        wiki2       k2 71%  k3 62%  k5 36%  k8 21%   k<=5

    A single grid-wide k=5 would therefore have made strategyqa unmeasurable while looking
    like an ordinary null.
    """
    g = load("conf/grids/tier1_confirmatory.yaml")
    assert g.k_for("strategyqa") == 2, "strategyqa loses its headroom above k=2"
    assert g.k_for("wiki2") <= 5
    assert g.k_for("musique") == 5
    assert g.k_for("a_suite_with_no_override") == g.k, "must fall back to the grid-wide k"


def test_every_multi_suite_grid_states_its_per_suite_k():
    """A grid spanning suites with different retrieval sensitivity must say so, or the choice
    is being made by a default nobody looked at."""
    for grid in load_all():
        if len(grid.suites) > 1 and not grid.exploratory:
            assert grid.k_by_suite, f"{grid.name} spans {list(grid.suites)} on a single k={grid.k}"


# --------------------------------------------------------------------------- grid coverage
#
# A grid that omits an arm a preregistered endpoint needs does NOT fail. It produces a table
# with one row missing, or -- worse, for a pooled endpoint -- a test computed over a subset of
# the suites it claims to pool, wearing the pooled label. Both look like results.

# Confirmatory grids and the suites each one covers. `tier1_pilot` is deliberately absent: its
# ids are BURNED (pilot: true) and mechanically excluded from confirmatory analysis, so it owes
# the pooled endpoints no rows. Its own obligation -- carrying every kill switch, since Gate 3
# runs `pi report killswitch` on it -- is checked separately below.
CONFIRMATORY_GRIDS = {
    "tier1_confirmatory": {"musique", "strategyqa", "wiki2"},
    "tier2_confirmatory": {"tau2"},
    "tier3_drgym": {"drgym"},
}

# Metrics that are not computable on a suite, with the reason. DRGym's corpus is a hosted
# open-web index behind an API key, so `gold_ev_uids` is empty by construction (0 of 16,156
# required nodes carry one) and resolution there is judged rather than matched by uid. A grid
# cannot owe rows for a metric its suite cannot produce.
NOT_COMPUTABLE = {
    ("drgym", "evidence_coverage"): "gold_ev_uids is empty by construction; resolution is judged",
    ("drgym", "rnr_resolve"): "same: no uid-matched resolution on a hosted index",
    ("drgym", "cad_ge2"): "same: depth is not uid-derived on this suite",
}


def test_every_confirmatory_grid_carries_the_arms_its_endpoints_need():
    """Including the POOLED ones. `tier2_confirmatory` omitted `parallel_replay`, and secondary
    S5 is `inquirer_prompted vs parallel_replay` on `evidence_coverage` with suite_id="*" --
    so the pooled test would have been computed over musique/strategyqa/wiki2 only, and
    labelled as pooled across every confirmatory suite."""
    from pi_eval.prereg import PRIMARY, SECONDARY
    from pi_run import grids

    endpoints = list(PRIMARY) + list(SECONDARY)
    problems: list[str] = []
    for name, suites in sorted(CONFIRMATORY_GRIDS.items()):
        arms = set(grids.load(f"conf/grids/{name}.yaml").arms)
        for e in endpoints:
            targets = suites if e.suite_id == "*" else ({e.suite_id} & suites)
            for suite in sorted(targets):
                if (suite, e.metric) in NOT_COMPUTABLE:
                    continue
                for arm in e.contrast:
                    if arm not in arms:
                        problems.append(
                            f"{name} ({suite}) needs {arm!r} for "
                            f"{e.metric} {e.contrast} [suite_id={e.suite_id!r}]"
                        )
    assert not problems, "\n".join(sorted(set(problems)))


def test_the_gate_3_grid_carries_every_preregistered_kill_switch():
    """Gate 3 is `pi run --sweep conf/grids/tier1_pilot.yaml` then `pi report killswitch`,
    which prints one row per preregistered kill switch. An omitted arm yields a table with a
    missing row rather than an error -- and the missing row would be a decision that was
    written down in advance and then never made."""
    from pi_eval.prereg import KILL_SWITCHES, default_stage1
    from pi_run import grids

    pilot = grids.load("conf/grids/tier1_pilot.yaml")
    arms = set(pilot.arms)
    assert set(KILL_SWITCHES) <= arms, sorted(set(KILL_SWITCHES) - arms)
    assert default_stage1().kill_switches.keys() <= arms
    # And the treatment they are all compared against.
    from pi_eval.report import KILLSWITCH_TREATMENT

    assert KILLSWITCH_TREATMENT in arms
    assert pilot.pilot is True, (
        "kill-switch day burns its ids: a threshold chosen on data cannot also be tested on it"
    )


def test_a_grid_never_names_an_arm_the_table_does_not_have():
    """An unknown arm id fails at unit construction, per task, after the sweep has started."""
    import pathlib

    from pi_run import grids
    from pinq_expt import arms as arm_table

    known = set(arm_table.arm_ids())
    for p in sorted(pathlib.Path("conf/grids").glob("*.yaml")):
        unknown = set(grids.load(str(p)).arms) - known
        assert not unknown, f"{p.name}: {sorted(unknown)}"


def _musique_selection(grid_name):
    import json
    import pathlib

    from pi_run.grids import load
    from pinq.splitting import split_of
    from pinq_adapters.musique.suite import MusiqueSuite

    hits = sorted(pathlib.Path("data/corpora/musique").glob("*/tasks.jsonl"))
    if not hits:
        import pytest

        pytest.skip("musique corpus not built")
    ids = [json.loads(line)["id"] for line in hits[0].open()]
    suite = MusiqueSuite.__new__(MusiqueSuite)
    g = load(f"conf/grids/{grid_name}.yaml")
    return set(g.select(ids, lambda t: split_of("musique", t, suite.template_id(t))))


def test_the_trained_grid_evaluates_only_on_held_out_tasks():
    """Selection was `list(suite.task_ids())[: n]` with `split` consulted nowhere in cli.py, so
    tier1_trained took the first 97 musique ids -- 55 of them `split == "train"`. The headline
    "trained beats prompted" column rested on tasks the arm may have trained on, and no
    eval-side consumer reads the `split` column, so it could not be repaired afterwards."""
    from pinq.splitting import split_of
    from pinq_adapters.musique.suite import MusiqueSuite

    suite = MusiqueSuite.__new__(MusiqueSuite)
    sel = _musique_selection("tier1_trained")
    assert sel, "the grid must select something"
    bad = [t for t in sel if split_of("musique", t, suite.template_id(t)) != "test"]
    assert not bad, f"{len(bad)} of {len(sel)} trained-grid tasks are not held out: {bad[:3]}"


def test_the_pilot_is_disjoint_from_the_confirmatory_slice():
    """A pilot chooses thresholds by looking at its results, which BURNS those tasks. While the
    pilot was a strict prefix of confirmatory, every burned task was also a confirmatory task."""
    pilot = _musique_selection("tier1_pilot")
    conf = _musique_selection("tier1_confirmatory")
    assert pilot and conf
    assert not (pilot & conf), f"{len(pilot & conf)} pilot tasks are also confirmatory tasks"


def test_the_oracle_shares_the_confirmatory_slice_on_purpose():
    """NOT a defect, and asserted so nobody 'fixes' it. A ceiling is only meaningful measured on
    the tasks it is a ceiling FOR: Gate 2's rule is
    `F1(gold_evidence) - F1(drafter_only) >= 0.10` on the confirmatory tasks."""
    assert _musique_selection("tier1_oracle") == _musique_selection("tier1_confirmatory")


def test_an_unknown_grid_key_is_refused(tmp_path):
    """`load` built the Grid from an allowlist and ignored everything else, so a grid author
    writing `split: test` got no error and no effect. That is how tier1_trained came to
    evaluate on training tasks."""
    import pytest

    from pi_run.grids import load

    p = tmp_path / "g.yaml"
    p.write_text("name: g\nsuites: [musique]\narms: [drafter_only]\nsplt: test\n")
    with pytest.raises(ValueError, match="unknown grid key"):
        load(p)


# --------------------------------------------------------------------------- trained arms
#
# `test_the_trained_grid_evaluates_only_on_held_out_tasks` above checks ONE grid, by name, on
# ONE suite, because that is the grid whose defect prompted it. That is a check of an instance,
# not of the rule, and a second trained grid added tomorrow inherits none of it. The rule:
#
#   a grid carrying a trained arm must declare `split: test`,
#   unless it is `exploratory: true`, in which case it must declare `split: dev`.
#
# The exemption is not a loophole. Checkpoint selection HAS to run the trained arm on dev -- it
# is the only split it may look at before the test split is opened -- so a rule that simply
# forbade a trained arm off `test` would forbid the dev gate itself and be deleted the first
# time someone needed it. `exploratory: true` is what keeps those rows out of every reported
# table, so the exemption is tied to the flag that makes it safe rather than to a grid name.

TRAINED_GRID_RULE = (
    "a trained arm on a non-test split is only legal in an exploratory dev grid "
    "(checkpoint selection); anything else can reach a reported table"
)


def test_a_trained_grid_that_does_not_declare_a_split_is_refused(tmp_path):
    """The predicate itself, on a synthetic grid: `tier1_trained` was exactly this file, and
    it evaluated the trained arm on 55 training tasks."""
    from pi_eval.report import TRAINED_ARMS
    from pi_run.grids import load, trained_grid_violation

    p = tmp_path / "g.yaml"
    p.write_text("name: g\nsuites: [musique]\narms: [inquirer_trained]\nn_tasks: 10\n")
    why = trained_grid_violation(load(p), TRAINED_ARMS)
    assert why and "split" in why


def test_a_trained_grid_on_dev_must_be_exploratory(tmp_path):
    """dev without `exploratory` is the dangerous case: the rows look like any other rows, and
    the split they came from is the split the checkpoint was CHOSEN on."""
    from pi_eval.report import TRAINED_ARMS
    from pi_run.grids import load, trained_grid_violation

    p = tmp_path / "g.yaml"
    p.write_text("name: g\nsuites: [musique]\narms: [inquirer_trained]\nn_tasks: 10\nsplit: dev\n")
    assert trained_grid_violation(load(p), TRAINED_ARMS)

    p.write_text(
        "name: g\nsuites: [musique]\narms: [inquirer_trained]\nn_tasks: 10\n"
        "split: dev\nexploratory: true\n"
    )
    assert trained_grid_violation(load(p), TRAINED_ARMS) is None


def test_a_grid_with_no_trained_arm_is_not_subject_to_the_rule(tmp_path):
    """The guard against over-widening: `inquirer_prompted` on dev IS the dev baseline."""
    from pi_eval.report import TRAINED_ARMS
    from pi_run.grids import load, trained_grid_violation

    p = tmp_path / "g.yaml"
    p.write_text("name: g\nsuites: [musique]\narms: [inquirer_prompted]\nn_tasks: 10\nsplit: dev\n")
    assert trained_grid_violation(load(p), TRAINED_ARMS) is None


def test_every_shipped_grid_with_a_trained_arm_is_held_out_or_exploratory_dev():
    """The sweep. Green today with one trained grid; it is here so the SECOND one inherits the
    check instead of relying on someone remembering the first one's story."""
    from pi_eval.report import TRAINED_ARMS
    from pi_run.grids import trained_grid_violation

    bad = {}
    for g in load_all():
        why = trained_grid_violation(g, TRAINED_ARMS)
        if why:
            bad[g.name] = why
    assert not bad, f"{TRAINED_GRID_RULE}: {bad}"


# --------------------------------------------------------------------------- track D grids

DEV_SELECT = ("dev_select_musique", "dev_select_strategyqa", "dev_select_wiki2")

# The dev baselines these must pair against, and the task count each one ran. Selection is
# `split: dev` + head-N in corpus order with no offset, so replicating the MECHANISM makes
# containment structural: every task the checkpoint runs has a baseline twin by construction,
# rather than because two lists were transcribed to match.
# `load_all` globs conf/grids/*.yaml only, so the baselines are named by FILE: they live in
# conf/grids/reroll/ and are deliberately not swept as top-level grids.
DEV_BASELINE_OF = {
    "dev_select_musique": ("conf/grids/reroll/dev_musique.yaml", 132),
    "dev_select_strategyqa": ("conf/grids/reroll/dev_strategyqa.yaml", 333),
    "dev_select_wiki2": ("conf/grids/reroll/dev_wiki2.yaml", 200),
}

TRACK_D_GRIDS = (
    *DEV_SELECT,
    "tier1_trained_qa_teacher",
    "tier1_trained_qa_base",
    "tier1_trained_killswitch",
    "frontier_trained_musique_cap4",
    "frontier_trained_musique_cap8",
    "frontier_trained_musique_cap12",
    "frontier_trained_musique_cap16",
)


def _named(name):
    return next(g for g in load_all() if g.name == name)


def test_every_track_d_grid_loads_and_prices_itself():
    names = {g.name for g in load_all()}
    missing = sorted(set(TRACK_D_GRIDS) - names)
    assert not missing, missing
    for n in TRACK_D_GRIDS:
        g = _named(n)
        assert g.n_units > 0, f"{n} prices to zero units"
        assert g.source_sha256


def test_the_dev_select_grids_are_dev_and_exploratory():
    """Dev rows must never be admissible as a measurement: `exploratory` is what keeps them out
    of every reported table, and it is the ONLY thing that does."""
    for n in DEV_SELECT:
        g = _named(n)
        assert g.split == "dev", n
        assert g.exploratory is True, n
        assert g.arms == ("inquirer_trained",), n
        assert g.seeds == (0,), n


def test_each_dev_select_grid_reproduces_its_baseline_task_selection():
    """PAIRING IS THE WHOLE POINT of the dev gate, and it is paired on task id. If the two
    grids select tasks differently, the deltas are computed over whatever the two happened to
    share. So these must match the baseline's mechanism exactly: same split, same n, no
    task_offset, no explicit ids.

    One grid per suite, not one grid spanning three, because `Grid` has a single `n_tasks` for
    all its suites and `cli.py` hands explicit `task_ids` to every suite unfiltered -- so a
    three-suite grid cannot express 132/333/200 at all.
    """
    for n in DEV_SELECT:
        base_path, n_tasks = DEV_BASELINE_OF[n]
        g, base = _named(n), load(base_path)
        assert len(g.suites) == 1, f"{n} must be single-suite; see this test's docstring"
        assert g.suites == base.suites, n
        assert g.n_tasks == n_tasks == base.n_tasks, n
        assert g.task_ids == () and g.task_offset == 0, n
        assert g.split == base.split == "dev", n
        assert 0 in base.seeds, f"{base_path} never ran seed 0, so {n} cannot pair against it"


def test_the_dev_gate_runs_at_the_baselines_retrieval_budget():
    """A paired delta against a baseline run at a different k is a RETRIEVAL-BUDGET contrast as
    much as a training contrast -- the confound `tier1_trained`'s own notes name. The dev
    baselines ran k=5 on all three suites, so these do too, even though the test-split grids
    use the per-suite k that tier1_confirmatory established."""
    for n in DEV_SELECT:
        g, base = _named(n), load(DEV_BASELINE_OF[n][0])
        suite = g.suites[0]
        assert g.k_for(suite) == base.k_for(suite) == 5, n


def test_the_two_qa_grids_are_identical_except_for_their_name():
    """The two-pin protocol: the ONLY difference between the teacher and base runs is
    `PI_MODEL_INQUIRER`, which is not a grid field. Two files with distinct `grid_name` are how
    the two resulting tables stay tellable apart afterwards; if their bodies drifted, the
    difference would no longer be only the pin."""
    from dataclasses import fields

    a, b = _named("tier1_trained_qa_teacher"), _named("tier1_trained_qa_base")
    ignore = {"name", "description", "notes", "source_sha256"}
    for f in fields(a):
        if f.name in ignore:
            continue
        assert getattr(a, f.name) == getattr(b, f.name), f"{f.name} differs between the two pins"


def test_the_test_split_trained_grids_share_tier1_confirmatory_k():
    """Or trained-vs-prompted is also a comparison of retrieval budget, and strategyqa becomes
    unmeasurable at k=5 (its sub-question headroom is 23% at k5 against 45% at k2)."""
    conf = _named("tier1_confirmatory")
    for n in ("tier1_trained_qa_teacher", "tier1_trained_qa_base", "tier1_trained_killswitch"):
        g = _named(n)
        for suite in g.suites:
            assert g.k_for(suite) == conf.k_for(suite), f"{n}/{suite}"
        assert g.split == "test" and g.n_tasks == 200 and g.seeds == (0, 1), n


def test_the_frontier_grids_differ_only_in_their_budget_cap():
    """The frontier is a curve in SPEND. If anything else moved across the four files the
    curve would be four unrelated points."""
    caps = {}
    for cap in (4, 8, 12, 16):
        g = _named(f"frontier_trained_musique_cap{cap}")
        assert g.budget_cap == cap
        caps[cap] = (g.suites, g.arms, g.seeds, g.n_tasks, g.split, g.max_turns, g.k_by_suite)
    assert len(set(map(str, caps.values()))) == 1, caps


def test_the_killswitch_grid_carries_the_switches_and_not_the_treatment():
    """It exists to ask whether the win is question CONTENT. Putting `inquirer_trained` in it
    would make it a second copy of the QA grid at twice the price."""
    g = _named("tier1_trained_killswitch")
    assert set(g.arms) == {
        "verbosity",
        "compute_matched",
        "self_inquire",
        "inquirer_noevidence",
        "inquirer_depth1",
        "random_q",
    }
    assert g.suites == ("musique", "strategyqa")
