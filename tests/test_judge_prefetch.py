"""Parallel judge dispatch that cannot change a single published number.

WHY. `judge_runs` grades serially -- plain `for r in runs:` -- and measured on this repo the
judge sustains ~7 live calls a minute. sigma_J at its preregistered 200-pair minimum is
3,000 live calls in the paraphrase pass, i.e. SEVEN HOURS, and drgym/P3 scores through the
same path, so the primary endpoint inherits it.

THE DESIGN, AND WHY IT IS THIS ONE. Threads only run the pure `_grade` call (the network).
Every mutation of shared state happens afterwards on the main thread, in a deterministic
order. That matters because `_Grader.get` mutates three things:

  self._cache[key] = ...          dict assignment, atomic under the GIL -- safe either way
  result.n_calls_failed += 1      read-modify-write -- NOT atomic, silently undercounts
  result.unjudged.append(...)     atomic, but ARRIVAL ORDER is nondeterministic

The second would under-report parse failures, and the third would make `unjudged` -- which
is reported -- differ between identical runs. Both are exactly the class of bug that turns
into a wrong published number, so the parallel section is kept free of them entirely.
"""

from __future__ import annotations


class _Rec:
    """A grader stub: records call order, returns a value derived from the key."""

    def __init__(self, fail_on=()):
        self.calls: list[tuple] = []
        self.fail_on = set(fail_on)

    def __call__(self, key):
        self.calls.append(key)
        if key in self.fail_on:
            raise ValueError(f"unparseable: {key}")
        return f"graded:{key}"


def test_prefetch_calls_each_key_exactly_once() -> None:
    from pi_eval.judges.harness import prefetch_gradings

    rec = _Rec()
    keys = [("r1", "kpr", "ab"), ("r2", "kpr", "ab"), ("r1", "quality", "ab")]
    out = prefetch_gradings(keys, rec, workers=4)
    assert sorted(rec.calls) == sorted(keys)
    assert len(rec.calls) == len(keys), "a key was graded twice; that double-bills"
    assert out[("r2", "kpr", "ab")] == ("graded:('r2', 'kpr', 'ab')", None)


def test_results_are_returned_in_a_deterministic_mapping() -> None:
    """Threads finish in arbitrary order; the mapping must not depend on that."""
    from pi_eval.judges.harness import prefetch_gradings

    keys = [(f"r{i}", "kpr", "ab") for i in range(24)]
    a = prefetch_gradings(keys, _Rec(), workers=8)
    b = prefetch_gradings(keys, _Rec(), workers=8)
    assert list(a) == list(b)
    assert list(a) == keys, "order must follow the INPUT, not completion"


def test_an_error_is_captured_not_raised() -> None:
    """A parse failure is an unjudged item, and the caller records it serially."""
    from pi_eval.judges.harness import prefetch_gradings

    bad = ("r2", "kpr", "ab")
    out = prefetch_gradings([("r1", "kpr", "ab"), bad], _Rec(fail_on=[bad]), workers=4)
    value, err = out[bad]
    assert value is None and isinstance(err, ValueError)
    assert out[("r1", "kpr", "ab")][1] is None, "one failure must not poison the others"


def test_every_error_is_captured_even_when_many_fail() -> None:
    """The count must be exact: `n_calls_failed` is reported."""
    from pi_eval.judges.harness import prefetch_gradings

    keys = [(f"r{i}", "kpr", "ab") for i in range(20)]
    out = prefetch_gradings(keys, _Rec(fail_on=keys[::2]), workers=8)
    assert sum(1 for v, e in out.values() if e is not None) == 10


def test_workers_of_one_is_the_serial_path() -> None:
    from pi_eval.judges.harness import prefetch_gradings

    rec = _Rec()
    keys = [(f"r{i}", "kpr", "ab") for i in range(5)]
    prefetch_gradings(keys, rec, workers=1)
    assert rec.calls == keys, "with one worker the order is the input order"


def test_an_empty_key_list_does_no_work() -> None:
    from pi_eval.judges.harness import prefetch_gradings

    rec = _Rec()
    assert prefetch_gradings([], rec, workers=8) == {}
    assert rec.calls == []


def test_duplicate_keys_are_graded_once() -> None:
    """A run appears in several pairs; grading it twice would double-bill the provider."""
    from pi_eval.judges.harness import prefetch_gradings

    rec = _Rec()
    k = ("r1", "kpr", "ab")
    out = prefetch_gradings([k, k, k], rec, workers=4)
    assert rec.calls == [k]
    assert len(out) == 1


# ------------------------------------------------- warming the grader, budget still enforced


class _Run:
    """Stands in for RunInput: warm_grader only needs `.run_id`."""

    def __init__(self, run_id):
        self.run_id = run_id


class _Result:
    """Stands in for JudgeResult: the two fields warm_grader is allowed to mutate."""

    def __init__(self):
        self.n_calls_failed = 0
        self.unjudged: list[str] = []


class _FakeGrader:
    """Mirrors the REAL _Grader's surface: _cache, check_budget, _grade(run, family, order),
    and mutations landing on `_result` rather than on the grader itself.

    The first version of this fake took `_grade(key)` and carried the counters directly, and
    every test passed against a shape the production class does not have. That is the same
    defect class as the manifest_to_dict and Grid(...) allowlists found elsewhere here, so
    the fake is now written against the real signature."""

    def __init__(self, *, fail_on=(), budget_after=None):
        self._cache: dict = {}
        self._result = _Result()
        self.checks = 0
        self.graded: list = []
        self._fail_on = set(fail_on)
        self._budget_after = budget_after

    def check_budget(self):
        self.checks += 1
        if self._budget_after is not None and self.checks > self._budget_after:
            raise RuntimeError("judge spend cap reached")

    def _grade(self, run, family, order):
        key = (run.run_id, family, order)
        self.graded.append(key)
        if key in self._fail_on:
            from pi_eval.judges._llm import JudgeParseError

            raise JudgeParseError(f"bad verdict for {key}")
        return (f"j:{key}",)


def test_warm_populates_the_memo_so_get_is_a_hit() -> None:
    from pi_eval.judges.harness import warm_grader

    g = _FakeGrader()
    items = [(_Run("r1"), "kpr", "ab"), (_Run("r2"), "kpr", "ab")]
    warm_grader(g, items, workers=4, batch=8)
    assert set(g._cache) == {("r1", "kpr", "ab"), ("r2", "kpr", "ab")}
    assert g._cache[("r1", "kpr", "ab")] == ("j:('r1', 'kpr', 'ab')",)


def test_the_budget_is_still_checked() -> None:
    """Prefetched items skip `get`'s own check, so warm must do it or the cap is bypassed."""
    from pi_eval.judges.harness import warm_grader

    g = _FakeGrader()
    warm_grader(g, [(_Run(f"r{i}"), "kpr", "ab") for i in range(9)], workers=4, batch=4)
    assert g.checks >= 3, f"only {g.checks} budget checks for 9 items in batches of 4"


def test_the_cap_stops_the_warm_and_the_overshoot_is_bounded_by_one_batch() -> None:
    """Same contract as the sweep's cap: a stop-loss, not a ceiling."""
    import pytest

    from pi_eval.judges.harness import warm_grader

    g = _FakeGrader(budget_after=1)
    items = [(_Run(f"r{i}"), "kpr", "ab") for i in range(40)]
    with pytest.raises(RuntimeError, match="cap"):
        warm_grader(g, items, workers=4, batch=4)
    assert len(g.graded) <= 8, f"graded {len(g.graded)} after the cap; overshoot unbounded"


def test_a_parse_failure_is_recorded_serially_and_counted_exactly() -> None:
    """The count and the order are reported, so neither may come from a thread."""
    from pi_eval.judges.harness import warm_grader

    items = [(_Run(f"r{i}"), "kpr", "ab") for i in range(6)]
    keys = [(r.run_id, f, o) for r, f, o in items]
    g = _FakeGrader(fail_on=[keys[1], keys[4]])
    warm_grader(g, items, workers=4, batch=8)
    assert g._result.n_calls_failed == 2
    assert g._cache[keys[1]] == (), "an unparseable verdict caches empty, never a grade"
    assert [u.split("/")[0] for u in g._result.unjudged] == ["r1", "r4"], g._result.unjudged


def test_warm_is_idempotent_and_skips_what_is_already_cached() -> None:
    from pi_eval.judges.harness import warm_grader

    g = _FakeGrader()
    items = [(_Run("r1"), "kpr", "ab")]
    warm_grader(g, items, workers=2, batch=4)
    warm_grader(g, items, workers=2, batch=4)
    assert len(g.graded) == 1, "a second warm re-billed an already-cached key"
