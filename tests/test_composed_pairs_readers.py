"""scripts/composed_pairs/{predict,analyze}.py: the pieces that decide a number, each made to fire.

* the additivity prediction is node-count weighted and a constituent with no solo run is
  FLAGGED, never imputed;
* `pooled_levels` is `table1_by_seed.pooled_symmetric` returning levels: its mean difference
  must equal that function's delta on the same inputs, or the lock against Table 3 means
  nothing about the composed reading;
* the composed contrast clusters on the PAIR (both orders one cluster);
* the graph loader refuses an empty dict or one missing a task -- the silent `{}` that
  `load_graphs("synth")` returns on another suite is exactly what analyze_shape.py would read
  as "no coverage";
* the interference DiD is the mean over clusters of (composed delta - predicted delta).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "composed_pairs"))
sys.path.insert(0, str(REPO / "scripts" / "seed_identity"))

import analyze  # noqa: E402
import predict  # noqa: E402

from pi_eval.matcher.base import MatchRecord  # noqa: E402
from pi_eval.stats.inference import paired_difference  # noqa: E402


def _side(task: str, a: str, b: str, na: int, nb: int, order: str = "ab") -> dict:
    return {
        "task_id": task,
        "pair_id": task.rsplit("_", 1)[0],
        "order": order,
        "a_id": a,
        "b_id": b,
        "n_nodes": {"a": na, "b": nb},
    }


def test_the_prediction_is_node_count_weighted():
    assert predict.weighted(0.5, 1.0, 2, 4) == pytest.approx((2 * 0.5 + 4 * 1.0) / 6)


def test_a_constituent_without_a_solo_run_is_flagged_not_imputed():
    side = [
        _side("x2_000000000001_ab", "A", "B", 2, 3),
        _side("x2_000000000002_ab", "C", "D", 2, 2),
    ]
    levels = {
        arm: {
            basis: {m: {"A": 0.2, "B": 0.6, "C": 0.4} for m in predict.METRICS}
            for basis in predict.BASES
        }
        for arm in predict.ARMS
    }
    out = predict.predict(side, levels)
    t1 = out["tasks"]["x2_000000000001_ab"]
    assert t1["pred"]["recipe"]["symmetric"]["evidence_coverage"] == pytest.approx(
        (2 * 0.2 + 3 * 0.6) / 5
    )
    t2 = out["tasks"]["x2_000000000002_ab"]
    assert t2["pred"] is None
    assert "D" in t2["missing_solo"]
    assert out["summary"]["n_constituents"] == 4
    assert out["summary"]["n_constituents_with_solo"] == 3
    assert out["summary"]["constituents_without_solo"] == ["D"]
    assert out["summary"]["n_tasks_predicted"] == 1


def _lad(rid: str, task: str, n_asks: int, hits: list[int]):
    """A RunLadder whose records resolve one node per listed turn."""
    recs = tuple(
        MatchRecord(
            run_id=rid,
            suite_id="musique",
            task_id=task,
            node_id=f"n{i}",
            match_kind="resolve",
            matched_turn_idx=t,
            matcher_id="t",
            matcher_family="t",
            matcher_score=1.0,
            threshold=1.0,
            graph_version="v1",
        )
        for i, t in enumerate(hits)
    )
    return predict.rc.ladder.RunLadder(rid, task, "musique", n_asks, recs)


def _toy(records, graph):
    return float(len(records))


def _arms():
    s1 = {"s1a0": _lad("s1a0", "t1", 3, [0, 1, 2]), "s1a1": _lad("s1a1", "t1", 2, [0, 5])}
    s1 |= {"s1b0": _lad("s1b0", "t2", 4, [0, 1]), "s1b1": _lad("s1b1", "t2", 1, [0])}
    s2 = {"s2a0": _lad("s2a0", "t1", 5, [0, 1, 2, 3]), "s2b1": _lad("s2b1", "t2", 2, [1])}
    base = {"ba0": _lad("ba0", "t1", 2, [1]), "ba1": _lad("ba1", "t1", 4, [0, 3])}
    base |= {"bb0": _lad("bb0", "t2", 3, [0, 2]), "bb1": _lad("bb1", "t2", 6, [0, 1, 2])}
    seeds = {r: int(r[-1]) for arm in (s1, s2, base) for r in arm}
    graphs = {"t1": object(), "t2": object()}
    return s1, s2, base, seeds, graphs


def test_pooled_levels_is_pooled_symmetric_returning_levels():
    import table1_by_seed

    s1, s2, base, seeds, graphs = _arms()
    with predict.rc.sweep.registered_metrics({"toy_len": _toy}):
        per_a, per_b, n_pairs = predict.pooled_levels("toy_len", [s1, s2], base, graphs, seeds)
        want = table1_by_seed.pooled_symmetric("toy_len", [s1, s2], base, graphs, seeds)
    got = paired_difference(per_a, per_b, n_boot=1000, seed=0)
    assert got.point == pytest.approx(want["delta"], abs=1e-12)
    assert n_pairs == want["n_pairs"]
    # Hand count, t1: s1 seed0 k=min(3,2)=2 -> 2 vs base 1 (turn 1 < 2); s1 seed1 k=min(2,4)=2
    # -> 1 (turn 5 dropped) vs base 1 (turn 0); s2 seed0 k=min(5,2)=2 -> 2 vs 1.
    assert per_a["t1"] == pytest.approx((2 + 1 + 2) / 3)
    assert per_b["t1"] == pytest.approx((1 + 1 + 1) / 3)


def test_own_stop_levels_read_every_run_at_its_own_k():
    s1, s2, base, _, graphs = _arms()
    with predict.rc.sweep.registered_metrics({"toy_len": _toy}):
        lv = predict.own_stop_levels("toy_len", [s1, s2], graphs)
    # t1: s1a0 3 (turns 0,1,2 < 3), s1a1 1 (turn 5 >= 2), s2a0 4 -> mean 8/3
    assert lv["t1"] == pytest.approx(8 / 3)


def test_the_composed_contrast_clusters_on_the_pair():
    per_a = {"x2_aaaaaaaaaaaa_ab": 1.0, "x2_aaaaaaaaaaaa_ba": 0.0, "x2_bbbbbbbbbbbb_ab": 0.5}
    per_b = {k: 0.0 for k in per_a}
    clusters = analyze.pair_clusters(per_a)
    assert clusters["x2_aaaaaaaaaaaa_ab"] == clusters["x2_aaaaaaaaaaaa_ba"] == "x2_aaaaaaaaaaaa"
    est = paired_difference(per_a, per_b, clusters=clusters, n_boot=1000, seed=0)
    # unweighted mean over CLUSTERS: pair a -> 0.5, pair b -> 0.5
    assert est.point == pytest.approx(0.5)
    assert "2 clusters over 3 tasks" in est.note


def test_a_task_id_that_is_not_composed_is_refused_as_a_cluster():
    with pytest.raises(analyze.Refusal):
        analyze.pair_clusters({"2hop__1_2": 0.0})


def test_the_graph_loader_refuses_empty_and_partial(monkeypatch):
    monkeypatch.setattr(analyze, "load_graphs", lambda suite, v: {})
    with pytest.raises(analyze.Refusal, match="no graphs"):
        analyze.graphs_for("musique_x2", "v1", ["x2_aaaaaaaaaaaa_ab"])
    monkeypatch.setattr(analyze, "load_graphs", lambda suite, v: {"x2_aaaaaaaaaaaa_ab": object()})
    with pytest.raises(analyze.Refusal, match="no gold graph"):
        analyze.graphs_for("musique_x2", "v1", ["x2_aaaaaaaaaaaa_ab", "x2_aaaaaaaaaaaa_ba"])
    got = analyze.graphs_for("musique_x2", "v1", ["x2_aaaaaaaaaaaa_ab"])
    assert set(got) == {"x2_aaaaaaaaaaaa_ab"}


def test_the_did_is_composed_delta_minus_predicted_delta_over_clusters():
    per_a = {"x2_aaaaaaaaaaaa_ab": 0.8, "x2_aaaaaaaaaaaa_ba": 0.6, "x2_bbbbbbbbbbbb_ab": 0.5}
    per_b = {"x2_aaaaaaaaaaaa_ab": 0.4, "x2_aaaaaaaaaaaa_ba": 0.4, "x2_bbbbbbbbbbbb_ab": 0.5}
    pred_r = {"x2_aaaaaaaaaaaa_ab": 0.5, "x2_aaaaaaaaaaaa_ba": 0.5}
    pred_b = {"x2_aaaaaaaaaaaa_ab": 0.4, "x2_aaaaaaaaaaaa_ba": 0.4}
    cell = analyze.did_cell(per_a, per_b, pred_r, pred_b, resamples=((1000, 0),))
    # only pair a has a prediction: composed deltas 0.4, 0.2 -> 0.3; predicted 0.1 -> DiD 0.2
    r = cell["readings"][0]
    assert r["delta"] == pytest.approx(0.2)
    assert cell["n_tasks"] == 2 and cell["n_clusters"] == 1
    assert cell["n_tasks_without_prediction"] == 1


def test_one_cluster_cannot_decide_a_cell():
    """Measured on the 2-task smoke: one pair, a constant composed-minus-predicted difference,
    and every bootstrap resample identical -- a zero-width interval at -0.125 that "excludes 0"
    and read DECIDED. A cell over fewer than two clusters, or with a zero-width 50k interval,
    is not a decision."""
    per_a = {"x2_aaaaaaaaaaaa_ab": 0.8, "x2_aaaaaaaaaaaa_ba": 0.8}
    per_b = {"x2_aaaaaaaaaaaa_ab": 0.4, "x2_aaaaaaaaaaaa_ba": 0.4}
    pred_r = {"x2_aaaaaaaaaaaa_ab": 0.5, "x2_aaaaaaaaaaaa_ba": 0.5}
    pred_b = {"x2_aaaaaaaaaaaa_ab": 0.4, "x2_aaaaaaaaaaaa_ba": 0.4}
    cell = analyze.did_cell(
        per_a, per_b, pred_r, pred_b, resamples=((1000, 0), (50000, 101), (50000, 202))
    )
    assert cell["n_clusters"] == 1
    assert cell["verdict"] != "DECIDED"


def test_a_zero_width_interval_cannot_decide():
    r = [
        {"n_boot": 50000, "seed": s, "delta": 0.3, "ci_lo": 0.3, "ci_hi": 0.3}
        for s in (101, 202, 303)
    ]
    assert analyze.verdict(r, n_clusters=10)["verdict"] != "DECIDED"
    ok = [dict(x, ci_lo=0.1, ci_hi=0.5) for x in r]
    assert analyze.verdict(ok, n_clusters=10)["verdict"] == "DECIDED"
