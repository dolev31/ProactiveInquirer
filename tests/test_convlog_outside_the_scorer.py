"""The convlog metrics stay out of the confirmatory scorer, on purpose.

`scorer_hash = h("scorer", metric_defs, graph_hash, matcher_hash, judge_pins)`. Registering a
convlog metric in `pi_eval.score.METRICS` would change `metric_defs`, which changes
`scorer_hash`, which orphans the provenance of every score row already computed -- for a metric
that cannot be computed on any of those runs anyway, because they are not conversations.

The convlog rates are read by their own report. They are not endpoints, they are not
preregistered, and nothing in `tables/` may print them until Gate B has been read.
"""

from pi_eval.score import METRICS


def test_no_convlog_metric_is_registered_in_the_scorer():
    names = {m.name for m in METRICS}
    for forbidden in (
        "wasted_ask_rate",
        "necessary_ask_precision",
        "anticipation_miss_rate",
        "over_action_rate",
        "followups_per_task",
    ):
        assert forbidden not in names


def test_convlog_is_not_a_preregistered_endpoint():
    """A suite whose labels do not exist yet cannot carry a confirmatory claim."""
    from pi_eval.prereg import CALIBRATION, PRIMARY, SECONDARY

    for e in (*PRIMARY, *SECONDARY, *CALIBRATION):
        assert getattr(e, "suite", getattr(e, "suite_id", "")) != "convlog"
