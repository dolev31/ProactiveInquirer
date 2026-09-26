"""Frontier and human-ceiling metrics against cases whose answers are known by hand."""

import math

import pytest

from pi_eval.gold import GoldGraph, GoldNode
from pi_eval.matcher.base import MatchRecord
from pi_eval.metrics.frontier import (
    auc_log2,
    frontier,
    log2_grid,
    marginal_gain_per_doubling,
    step_at_spend,
)
from pi_eval.metrics.human import (
    anticipated_discovery_rate,
    discoverability_ceiling,
    user_burden,
)

# ------------------------------------------------------------------ frontier


def test_step_at_spend_does_not_interpolate():
    """At spend x the system has only produced its last COMPLETED checkpoint; smoothing
    between checkpoints would invent quality that never existed."""
    spend, q = [1, 2, 4, 8], [0.1, 0.3, 0.6, 0.7]
    assert step_at_spend(spend, q, 3) == 0.3
    assert step_at_spend(spend, q, 4) == 0.6
    assert math.isnan(step_at_spend(spend, q, 0.5))


def test_drafter_only_has_a_well_defined_x_axis():
    """The whole reason the axis is spend and not turns: a zero-question arm still has a
    curve, so a paired Delta-AUC against it is paired against something."""
    per_task = {f"t{i}": ([0.0], [0.4]) for i in range(20)}
    c = frontier(per_task, log2_grid(1, 16, 5), n_boot=200, seed=0)
    assert all(v == pytest.approx(0.4) for v in c.mean)
    assert c.auc == pytest.approx(0.4)


def test_auc_is_normalized_by_axis_length():
    grid = (1.0, 2.0, 4.0, 8.0)
    assert auc_log2(grid, [0.5] * 4) == pytest.approx(0.5)
    assert auc_log2(grid, [0.0, 0.0, 1.0, 1.0]) == pytest.approx(0.5)


def test_simultaneous_band_contains_the_pointwise_band():
    """Only the simultaneous band licenses 'that dip is noise', and it must be wider."""
    import random

    rng = random.Random(4)
    per_task = {
        f"t{i}": ([1, 2, 4, 8, 16], sorted(rng.uniform(0, 1) for _ in range(5))) for i in range(40)
    }
    c = frontier(per_task, (1, 2, 4, 8, 16), n_boot=800, seed=1)
    for i in range(len(c.grid)):
        assert c.lo_simultaneous[i] <= c.lo_pointwise[i] + 1e-9
        assert c.hi_simultaneous[i] >= c.hi_pointwise[i] - 1e-9


def test_auc_ci_brackets_the_point_estimate():
    per_task = {f"t{i}": ([1, 2, 4, 8, 16], [0.1, 0.2, 0.3, 0.4, 0.5]) for i in range(30)}
    c = frontier(per_task, (1, 2, 4, 8, 16), n_boot=500, seed=2)
    assert c.auc_lo <= c.auc <= c.auc_hi
    assert c.n_tasks == 30


def test_marginal_gain_per_doubling_is_reported():
    per_task = {f"t{i}": ([1, 2, 4], [0.0, 0.5, 0.5]) for i in range(10)}
    c = frontier(per_task, (1, 2, 4), n_boot=100, seed=0)
    gains = marginal_gain_per_doubling(c)
    assert gains[0] == pytest.approx(0.5)
    assert gains[1] == pytest.approx(0.0), "a saturated frontier must show zero marginal gain"


def test_empty_input_is_nan_not_a_crash():
    c = frontier({}, (1, 2, 4), n_boot=10)
    assert c.n_tasks == 0 and math.isnan(c.auc)


# ------------------------------------------- the band depended on the order its units arrived in
#
# `frontier` resamples unit INDICES (`units[rng.randrange(len(units))]`), so the ORDER of the
# `units` list is an input to every endpoint it returns. It built that list as
# `[groups[k] for k in sorted(groups)]` -- ordered by cluster KEY, which is a LABEL, and not by
# the values the estimator is about. Relabelling the keys holds the unit multiset identical and
# still moves the answer. Sorting keys is not canonicalising values.
#
# Same defect and same fix as `pi_eval.stats.inference.cluster_bootstrap` and
# `pinq_train.gate.bca_ci` (commit 4b7b24b). `frontier` does not go through either: it carries
# its own resampler because it recomputes the WHOLE curve per replicate, and so was missed.
#
# MEASURED at 20 clusters / 60 tasks, n_boot=2000, seed=0, before the fix: the AUC POINT moved by
# exactly 0.0 -- a mean over clusters is order-invariant by construction -- while auc_lo moved
# +0.000846576, auc_hi +0.000285956, and all 20 pointwise and simultaneous band endpoints moved,
# by up to 0.003937826. A stable centre beside a wandering interval is this defect's signature,
# and the reason it survived review. These are the bands under the headline figure, which carries
# 352 interval endpoints.


def _clustered_case(n_clusters=20, per_cluster=3, seed=11):
    """20 clusters of 3 -- the shape the defect was first measured on. Returns the grid, the
    ladders, and the cluster membership separately, so a test can relabel keys or reorder
    membership WITHOUT touching a single value."""
    import random

    rng = random.Random(seed)
    grid = (1.0, 2.0, 4.0, 8.0, 16.0)
    per_task, members = {}, {}
    for ci in range(n_clusters):
        shared = rng.gauss(0.0, 0.6)  # a per-template offset, so clusters are not exchangeable
        ids = []
        for j in range(per_cluster):
            tid = f"t{ci:02d}_{j}"
            per_task[tid] = (
                list(grid),
                [min(1.0, max(0.0, 0.35 + 0.09 * shared + rng.gauss(0, 0.25))) for _ in grid],
            )
            ids.append(tid)
        members[ci] = ids
    return grid, per_task, members


def _keys_in_order(members, order):
    """Label the cluster sitting at position p with `k{p}`, so `sorted(groups)` yields `order`.

    Membership is untouched and no value is touched, so the multiset of resampling units is
    IDENTICAL across any two calls and the only thing that differs is the order.
    """
    return {tid: f"k{p:02d}" for p, ci in enumerate(order) for tid in members[ci]}


def _unit_multiset(per_task, members, order):
    """The bag of units, as bags of value-vectors -- the only thing the estimand may depend on."""
    return sorted(tuple(sorted(tuple(per_task[t][1]) for t in members[ci])) for ci in order)


def test_frontier_endpoints_do_not_depend_on_the_order_the_units_arrive_in():
    """A RELABELLING OF THE CLUSTER KEYS MUST NOT MOVE ONE ENDPOINT.

    This is the live path: `report.frontier_curves` always passes a cluster map, built as
    `f"{suite_id}/{cluster or task}"`, and the units were then ordered by that string. The
    bootstrap draws indices into the list, so the same 20 units in a different order read as
    different units for the same RNG stream.

    Asserted on the WHOLE `Curve` and EXACTLY. A tolerance would swallow precisely the
    one-resample-step differences that are the defect. `n_tasks` and `mean` are asserted
    separately first, so a failure says which half moved: they must NOT move, and if they ever
    do, the sort has changed a value rather than an order.
    """
    import random

    grid, per_task, members = _clustered_case()
    ident = list(range(len(members)))
    base = frontier(per_task, grid, n_boot=2000, seed=0, clusters=_keys_in_order(members, ident))
    for perm_seed in (7, 13, 29):
        order = list(ident)
        random.Random(perm_seed).shuffle(order)
        assert _unit_multiset(per_task, members, order) == _unit_multiset(
            per_task, members, ident
        ), "the case is invalid: relabelling changed the unit multiset, not just the order"
        got = frontier(per_task, grid, n_boot=2000, seed=0, clusters=_keys_in_order(members, order))
        assert got.n_tasks == base.n_tasks, f"n moved under a relabelling (perm {perm_seed})"
        assert got.mean == base.mean, f"the point curve moved under a relabelling ({perm_seed})"
        assert got.auc == base.auc, f"the AUC point moved under a relabelling ({perm_seed})"
        assert got == base, (
            f"perm {perm_seed}: the interval or the bands moved under a pure relabelling.\n"
            f"  auc_lo {base.auc_lo!r} -> {got.auc_lo!r}\n"
            f"  auc_hi {base.auc_hi!r} -> {got.auc_hi!r}\n"
            f"  lo_pointwise {base.lo_pointwise!r} -> {got.lo_pointwise!r}\n"
            f"  lo_simultaneous {base.lo_simultaneous!r} -> {got.lo_simultaneous!r}"
        )


def test_frontier_endpoints_do_not_depend_on_the_order_of_an_unclustered_task_list():
    """With no cluster map every task is its own unit, ordered by TASK ID -- so the defect
    reaches every suite that mints no template_id, which is seven of the ten. Relabelling the
    task ids reorders the units while holding the ladders, and therefore the units, identical."""
    import random

    grid = (1.0, 2.0, 4.0, 8.0)
    rng = random.Random(3)
    ladders = [[min(1.0, max(0.0, rng.gauss(0.45, 0.3))) for _ in grid] for _ in range(40)]
    ident = list(range(len(ladders)))
    base = frontier(
        {f"t{p:02d}": (list(grid), ladders[i]) for p, i in enumerate(ident)},
        grid,
        n_boot=1500,
        seed=0,
    )
    for perm_seed in (5, 17):
        order = list(ident)
        random.Random(perm_seed).shuffle(order)
        got = frontier(
            {f"t{p:02d}": (list(grid), ladders[i]) for p, i in enumerate(order)},
            grid,
            n_boot=1500,
            seed=0,
        )
        assert got == base, f"unclustered relabelling {perm_seed} moved an endpoint"


def test_frontier_endpoints_do_not_depend_on_the_order_within_a_unit():
    """THE SAME ARGUMENT ONE LEVEL DOWN, which `cluster_bootstrap`'s docstring warns about:
    canonicalising only the outer list "would leave the same defect reachable one level down".

    A unit here is VECTOR-VALUED -- a list of tasks, each a curve over the grid -- so the inner
    order is a real degree of freedom rather than a hypothetical one. It is reordered by reversing
    each cluster's membership and relabelling the ids so `sorted(tasks)` reproduces the reversal.
    """
    import random

    grid, per_task, members = _clustered_case(seed=19)
    ident = list(range(len(members)))
    base = frontier(per_task, grid, n_boot=2000, seed=0, clusters=_keys_in_order(members, ident))
    for reorder in (lambda xs: list(reversed(xs)), lambda xs: xs[1:] + xs[:1]):
        per2, mem2 = {}, {}
        for ci, ids in members.items():
            new = []
            for j, tid in enumerate(reorder(ids)):
                nid = f"t{ci:02d}_{j}"
                per2[nid] = per_task[tid]
                new.append(nid)
            mem2[ci] = new
        got = frontier(per2, grid, n_boot=2000, seed=0, clusters=_keys_in_order(mem2, ident))
        assert got == base, "the order of tasks WITHIN a unit moved an endpoint"
    del random


def test_the_order_invariance_harness_can_detect_a_change_that_must_move_the_bands():
    """POSITIVE CONTROL. An invariance test that cannot detect a change is indistinguishable
    from a clean pass, so a null result from the three tests above only carries information if
    this one shows the harness has teeth.

    Two controls, one per level:

    * OUTER -- replace one whole unit's ladders. The unit multiset genuinely differs, so every
      endpoint may move and at least one must.
    * INNER -- change ONE value at ONE grid index inside ONE unit, leaving the unit count, the
      cluster keys and every other value alone. This is the control the within-unit test needs:
      it proves the harness reads unit CONTENTS at the inner level, so "reordering the inside of
      a unit changes nothing" is a measurement and not a blind spot.
    """
    grid, per_task, members = _clustered_case(seed=23)
    ident = list(range(len(members)))
    keys = _keys_in_order(members, ident)
    base = frontier(per_task, grid, n_boot=2000, seed=0, clusters=keys)

    outer = dict(per_task)
    for tid in members[0]:
        outer[tid] = (list(grid), [1.0] * len(grid))
    moved = frontier(outer, grid, n_boot=2000, seed=0, clusters=keys)
    assert moved.auc != base.auc, "control: replacing a whole unit did not move the AUC"
    assert moved.auc_lo != base.auc_lo and moved.auc_hi != base.auc_hi
    assert moved.lo_pointwise != base.lo_pointwise
    assert moved.lo_simultaneous != base.lo_simultaneous
    assert moved.hi_simultaneous != base.hi_simultaneous, "control: the sup-t band is inert"

    inner = dict(per_task)
    tid = members[0][1]
    spend, q = per_task[tid]
    bumped = list(q)
    bumped[2] = 0.0 if bumped[2] > 0.5 else 1.0
    inner[tid] = (list(spend), bumped)
    moved = frontier(inner, grid, n_boot=2000, seed=0, clusters=keys)
    assert moved.n_tasks == base.n_tasks, "control: the inner change must not change n"
    assert moved.mean[2] != base.mean[2], "control: one value inside a unit did not move the mean"
    assert (moved.lo_pointwise[2], moved.hi_pointwise[2]) != (
        base.lo_pointwise[2],
        base.hi_pointwise[2],
    ), "control: one value inside a unit did not move the pointwise band -- harness is blind"
    assert (moved.lo_simultaneous[2], moved.hi_simultaneous[2]) != (
        base.lo_simultaneous[2],
        base.hi_simultaneous[2],
    ), "control: one value inside a unit did not move the sup-t band -- harness is blind"
    assert moved.auc != base.auc, "control: one value inside a unit did not move the AUC"


# ------------------------------------------------------------------ human ceiling


def _node(nid, disc, partition="required", human_asked=None, useful=None):
    return GoldNode(
        gold_suite="s",
        gold_task_key="t",
        gold_node_id=nid,
        gold_text=nid,
        gold_partition=partition,
        gold_discoverability=disc,
        gold_human_asked=human_asked,
        gold_usefulness_rating=useful,
    )


def _graph(nodes):
    return GoldGraph(gold_suite="s", gold_task_key="t", gold_nodes=tuple(nodes))


def _rec(nid, kind="resolve"):
    return MatchRecord("r", "s", "t", nid, kind, 0, "m", "rule", 1.0, 1.0, "v1")


def test_private_share_is_the_ceiling_on_any_inquirer():
    """The framing's most important number: nobody can read the user's mind."""
    g = _graph([_node("a", "kb"), _node("b", "kb"), _node("c", "user_private")])
    ceil = discoverability_ceiling(g)
    assert ceil.private_share == pytest.approx(1 / 3)
    assert ceil.attainable == pytest.approx(2 / 3)


def test_adr_counts_only_needs_the_human_never_asked():
    g = _graph(
        [
            _node("a", "kb", human_asked=True),
            _node("b", "kb", human_asked=False, useful=4.0),
            _node("c", "kb", human_asked=False, useful=5.0),
        ]
    )
    out = anticipated_discovery_rate([_rec("a"), _rec("b"), _rec("c")], g)
    assert out["n_universe"] == 3 and out["n_found"] == 3
    assert out["n_beyond_human"] == 2
    assert out["adr"] == pytest.approx(2 / 3)
    assert out["mean_usefulness_beyond_human"] == pytest.approx(4.5)
    assert out["n_rated"] == 2, "both beyond-human needs carry a rating"


def test_n_rated_counts_only_beyond_human_needs_that_carry_a_rating():
    """The denominator behind `mean_usefulness_beyond_human` is not `n_beyond_human`: a
    usefulness-rating campaign can annotate `gold_human_asked` for a node long before (or
    without ever) rating it, so the mean must say how many ratings it actually rests on."""
    g = _graph(
        [
            _node("a", "kb", human_asked=False, useful=4.0),
            _node("b", "kb", human_asked=False, useful=None),
            _node("c", "kb", human_asked=False, useful=2.0),
        ]
    )
    out = anticipated_discovery_rate([_rec("a"), _rec("b"), _rec("c")], g)
    assert out["n_beyond_human"] == 3, "all three are found and never asked"
    assert out["n_rated"] == 2, "b has no usefulness_rating yet"
    assert out["mean_usefulness_beyond_human"] == pytest.approx(3.0)


def test_adr_excludes_user_private_needs_from_the_denominator():
    """Crediting or penalising a system for a fact only the user could supply measures
    nothing about inquiry."""
    g = _graph([_node("a", "kb", human_asked=False), _node("p", "user_private", human_asked=False)])
    out = anticipated_discovery_rate([_rec("a"), _rec("p")], g)
    assert out["n_universe"] == 1
    assert out["adr"] == pytest.approx(1.0)


def test_adr_is_zero_when_nothing_was_resolved():
    g = _graph([_node("a", "kb", human_asked=False)])
    out = anticipated_discovery_rate([_rec("a", "ask")], g)
    assert out["adr"] == 0.0, "asking is not finding"


def test_user_burden_is_zero_in_the_confirmatory_configuration():
    """The loop rejects target='user', so a non-zero value here is a bug alarm."""
    assert user_burden({}, 10) == 0.0
    assert user_burden({"rejected_user_asks": 3.0}, 12) == pytest.approx(0.25)


# --------------------------------------------------------------------------- ADR is withheld
#
# `gold_human_asked` is a HUMAN annotation and is None on every node this repository has built.
# The obvious implementation filtered `None is False` to nothing and divided by the full
# universe, returning a hard 0.0 -- not NaN, not an error, but a clean publishable zero reading
# "the system anticipated nothing the user asked for", on every task, whatever the policy did.
# Under the one metric whose purpose is to support "exceeds what the user thought to ask".


def test_an_unannotated_graph_yields_nan_and_not_zero():
    """The whole point. Every node in this repository has gold_human_asked=None."""
    import math

    g = _graph([_node("a", "kb"), _node("b", "kb"), _node("c", "kb")])
    out = anticipated_discovery_rate([_rec("a"), _rec("b")], g)
    assert math.isnan(out["adr"]), "an absent annotation must not become a 0.0"
    assert out["n_annotated"] == 0
    assert out["n_universe"] == 3
    assert out["annotation_coverage"] == 0.0
    assert out["n_rated"] == 0, "the base absent-case dict must also carry n_rated"


def test_the_scorer_writes_no_adr_row_when_nothing_is_annotated():
    """NaN in a parquet column still averages into a table in some readers. No row at all is
    the only unambiguous way to say a measurement was not taken."""
    import inspect

    from pi_eval import score

    src = inspect.getsource(score.score_run)
    assert 'if adr["n_annotated"]:' in src
    assert 'if adr["n_universe"]:' not in src


def test_adr_is_computed_over_the_annotated_subset_only():
    """Nodes nobody annotated are not evidence of anything and belong in neither half. Keeping
    them in the denominator would bias ADR downward by exactly the annotation rate."""
    g = _graph(
        [
            _node("a", "kb", human_asked=False),  # found, human never asked -> counts
            _node("b", "kb", human_asked=True),  # found, human DID ask -> does not count
            _node("c", "kb", human_asked=False),  # not found -> does not count
            _node("d", "kb"),  # unannotated -> in neither half
            _node("e", "kb"),  # unannotated
        ]
    )
    out = anticipated_discovery_rate([_rec("a"), _rec("b"), _rec("d")], g)
    assert out["n_universe"] == 5
    assert out["n_annotated"] == 3
    assert out["n_found"] == 2, "'d' is found but unannotated, so it is not counted"
    assert out["n_beyond_human"] == 1
    assert out["adr"] == pytest.approx(1 / 3), "the denominator is the ANNOTATED universe"
    assert out["annotation_coverage"] == pytest.approx(3 / 5)


def test_a_partly_annotated_graph_is_not_penalised_for_the_unannotated_part():
    """Dividing by the full universe would report a system that anticipated EVERY annotated
    need as scoring the annotation rate rather than 1.0."""
    g = _graph([_node("a", "kb", human_asked=False)] + [_node(f"u{i}", "kb") for i in range(9)])
    out = anticipated_discovery_rate([_rec("a")], g)
    assert out["adr"] == 1.0
    assert out["annotation_coverage"] == pytest.approx(0.1)


# --------------------------------------------------------------------------- adr_usefulness
#
# `mean_usefulness_beyond_human` was computed by `anticipated_discovery_rate` (human.py) but
# never emitted by `score.score_run`. A human-annotation campaign is about to populate
# `gold_usefulness_rating` for the first time, so the row now has something to carry.


def _score_run(graph, records):
    """One scored run over a real graph and real match records, exercising the actual
    row-emitting entry point `score.score_run` -- not just the metric-table declaration.
    Mirrors the `_score` helper in test_anticipation_instrument.py."""
    from pi_eval.score import score_run

    run = {"run_id": "r", "suite_id": "s", "task_id": "t", "arm_id": "a"}
    rows = score_run(
        run,
        graph=graph,
        turns=[],
        evidence=[],
        env_calls=[],
        ledger=[],
        records=records,
        answer=None,
    )
    return {r["metric_name"]: r for r in rows}


def test_adr_usefulness_is_emitted_when_ratings_exist():
    g = _graph(
        [
            _node("a", "kb", human_asked=False, useful=4.0),
            _node("b", "kb", human_asked=False, useful=5.0),
        ]
    )
    by = _score_run(g, [_rec("a"), _rec("b")])
    assert by["adr_usefulness"]["value"] == pytest.approx(4.5)
    assert by["adr_usefulness"]["n"] == 2


def test_no_adr_usefulness_row_when_annotated_but_unrated():
    """`gold_human_asked` can be annotated (so `adr` itself is written) long before any
    `gold_usefulness_rating` exists. `mean_usefulness_beyond_human` is then NaN, and a NaN
    must produce no row -- the same discipline the rest of this file exists to enforce."""
    g = _graph(
        [
            _node("a", "kb", human_asked=False),
            _node("b", "kb", human_asked=False),
        ]
    )
    by = _score_run(g, [_rec("a"), _rec("b")])
    assert "adr" in by, "sanity: the annotated-but-unrated case still writes adr itself"
    assert "adr_usefulness" not in by


def test_usefulness_of_anticipated_needs_is_the_reference_level():
    """`mean_usefulness_beyond_human` alone cannot support "exceeds what the user thought to
    ask": 4.5 out of 5 is only a claim against what the SAME annotator rated the needs they
    did anticipate. Both sides condition on `found`, so the contrast varies anticipation and
    holds discovery constant."""
    g = _graph(
        [
            _node("a", "kb", human_asked=True, useful=2.0),
            _node("b", "kb", human_asked=False, useful=4.0),
            _node("c", "kb", human_asked=False, useful=5.0),
        ]
    )
    out = anticipated_discovery_rate([_rec("a"), _rec("b"), _rec("c")], g)
    assert out["mean_usefulness_beyond_human"] == pytest.approx(4.5)
    assert out["n_rated"] == 2
    assert out["mean_usefulness_anticipated"] == pytest.approx(2.0)
    assert out["n_rated_anticipated"] == 1


def test_anticipated_usefulness_is_absent_when_no_anticipated_need_is_rated():
    """Absent, not zero, and not silently borrowed from the other group."""
    g = _graph(
        [_node("a", "kb", human_asked=True), _node("b", "kb", human_asked=False, useful=4.0)]
    )
    out = anticipated_discovery_rate([_rec("a"), _rec("b")], g)
    assert math.isnan(out["mean_usefulness_anticipated"])
    assert out["n_rated_anticipated"] == 0
    assert anticipated_discovery_rate([], _graph([]))["n_rated_anticipated"] == 0


def test_adr_usefulness_anticipated_is_emitted_beside_its_contrast():
    g = _graph(
        [
            _node("a", "kb", human_asked=True, useful=2.0),
            _node("b", "kb", human_asked=False, useful=5.0),
        ]
    )
    by = _score_run(g, [_rec("a"), _rec("b")])
    assert by["adr_usefulness"]["value"] == pytest.approx(5.0)
    assert by["adr_usefulness_anticipated"]["value"] == pytest.approx(2.0)
    assert by["adr_usefulness_anticipated"]["n"] == 1


def test_no_anticipated_row_when_only_the_beyond_side_is_rated():
    g = _graph(
        [
            _node("a", "kb", human_asked=True),
            _node("b", "kb", human_asked=False, useful=5.0),
        ]
    )
    by = _score_run(g, [_rec("a"), _rec("b")])
    assert "adr_usefulness" in by
    assert "adr_usefulness_anticipated" not in by
