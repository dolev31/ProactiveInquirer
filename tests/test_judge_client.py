"""The `PI_JUDGE_CLIENT` hook, and the fact that it now has an implementation.

`PI_JUDGE_CLIENT` appeared in the tree exactly once: as the NAME of an environment variable.
Nothing implemented it. So `pi score` took the documented "no client, and here is why" branch
on every single invocation, `judgments.parquet` stayed empty, and `kpr_incremental` -- a
PRIMARY endpoint -- emitted zero rows. The failure was honest and completely invisible to
anyone reading a table, because a metric with no rows is a metric that is simply not there.
"""

from __future__ import annotations

import pytest

from pi_eval.score import judge_client_from_env


def _env(**kw):
    base = {
        "PI_MODEL_JUDGE": "openai/test-judge",
        "LITELLM_BASE_URL": "https://example.invalid",
        "PI_CACHE_ROOT": "/tmp/does-not-matter-for-construction",
    }
    base.update(kw)
    return base


def test_the_hook_now_resolves_to_a_real_client():
    c, model, family, reason = judge_client_from_env(
        _env(PI_JUDGE_CLIENT="pi_run.judge_client:judge_replay")
    )
    assert c is not None, "this returned None on every invocation until judge_client existed"
    assert reason == "env"
    assert model == "openai/test-judge"
    assert hasattr(c, "complete")


def test_an_unset_hook_still_refuses_rather_than_judging_with_a_default():
    """The refusal is the correct behaviour and must survive: a fabricated 0.0 for a judged
    metric would average into a table, whereas a missing row is visibly missing."""
    c, _model, _family, reason = judge_client_from_env(_env())
    assert c is None and "PI_JUDGE_CLIENT" in reason


def test_a_hook_without_a_model_pin_refuses(monkeypatch):
    """Two arms judged by different models and then compared is the failure this refusal
    exists to make impossible."""
    env = _env(PI_JUDGE_CLIENT="pi_run.judge_client:judge_replay")
    env.pop("PI_MODEL_JUDGE")
    c, _m, _f, reason = judge_client_from_env(env)
    assert c is None and "refused" in reason


def test_the_replay_judge_raises_on_a_miss_instead_of_dispatching(tmp_path):
    """A judged metric recomputed from the cache is reproducible by anyone holding the cache.
    One recomputed by re-querying a model is a NEW measurement wearing an old scorer_hash --
    and in CI it is also an invoice."""
    from pi_run.cache import CacheMiss
    from pi_run.judge_client import judge_replay

    c = judge_replay(_env(PI_CACHE_ROOT=str(tmp_path)), cache_root=str(tmp_path))
    with pytest.raises(CacheMiss):
        c.complete(
            role="judge",
            messages=[{"role": "user", "content": "not cached"}],
            seed=0,
            max_tokens=8,
        )


def test_a_cached_judgment_is_replayed_billed_at_zero_and_still_carries_its_pin(tmp_path):
    """Re-scoring under a new scorer_hash is meant to be a groupby, never a re-roll. A judge
    that re-billed on every `pi score` would make re-scoring cost money, and re-scoring would
    therefore stop happening."""
    from pi_run.cache import DiskCache
    from pi_run.judge_client import judge_replay

    env = _env(PI_CACHE_ROOT=str(tmp_path))
    c = judge_replay(env, cache_root=str(tmp_path))
    msgs = [{"role": "user", "content": "grade this"}]
    sha = c.request_payload(role="judge", messages=msgs, seed=0, max_tokens=8)
    from pinq.ids import request_sha

    DiskCache(tmp_path).put(
        request_sha(sha),
        {
            "model": "openai/test-judge",
            "provider": "litellm_proxy",
            "request_sha": request_sha(sha),
            "response_sha": "x",
            "text": '{"label": "Supported"}',
            "tok_prompt": 10,
            "tok_completion": 5,
            "usd": 0.002,
            "http_status": 200,
        },
    )
    text, tel = c.complete(role="judge", messages=msgs, seed=0, max_tokens=8)
    assert text == '{"label": "Supported"}'
    assert tel.cache_hit is True

    spend = c.spend()
    assert spend["n_calls"] == 1 and spend["cache_hits"] == 1
    # As-if is charged (so two runs stay comparable however warm the cache was); billed is not.
    assert spend["usd"] == pytest.approx(0.002)
    assert spend["usd_billed"] == 0.0
    # A replayed judgment still has to name the instrument, or scorer_hash does not carry it.
    assert c.pin().model_id == "openai/test-judge"


def test_the_judge_is_pinned_to_temperature_zero():
    """sigma_J is estimated from test-retest of identical and paraphrased pairs. A judge
    sampling at t>0 would inflate the identical-pair variance and widen every band built on
    it -- the instrument's noise would be read as the system's."""
    from pi_run.judge_client import JUDGE_TEMPERATURE, judge_replay

    assert JUDGE_TEMPERATURE == 0.0
    c = judge_replay(_env())
    payload = c.request_payload(
        role="judge", messages=[{"role": "user", "content": "x"}], seed=0, max_tokens=8
    )
    assert payload["temperature"] == 0.0


def test_the_hook_creates_no_import_edge_into_pi_eval():
    """`pi_eval` depends on `pinq` and nothing else first-party -- one of the four contracts.
    The judge client lives in `pi_run` and is resolved BY STRING at call time, so it exists
    without either direction gaining a static import."""
    import ast
    import pathlib

    src = pathlib.Path("src/pi_eval/score.py").read_text()
    tree = ast.parse(src)
    imported = {
        n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    } | {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "pi_run" not in imported
    assert "PI_JUDGE_CLIENT" in src, "the hook is still resolved by name, not by import"


def test_pi_score_reports_what_the_judge_spent(tmp_path, monkeypatch):
    """`--spend-cap` is checked inside `run_sweep` and covers ROLLOUTS only. Judging runs over
    every eligible run with a report and gold, for every preregistered arm pair, and produces
    `kpr_incremental` -- a primary endpoint. It was a second provider bill that nothing
    measured. A number nobody can see is a number nobody can cap."""
    import inspect

    from pi_eval import score as score_mod

    src = inspect.getsource(score_mod.score)
    assert "judge_spend=" in src, "the judge's cost never reaches the score report"

    assert "judge_spend" in score_mod.ScoreResult.__dataclass_fields__
    assert "judge_spend" in inspect.getsource(score_mod.ScoreResult.as_dict), (
        "the field exists but as_dict does not emit it, so `pi score` still prints nothing"
    )

    # And an unknown spend is UNKNOWN, never zero: a judge bill reported as $0.00 is what lets
    # an unbounded cost go unnoticed.
    assert '{"unknown": 1.0}' in src


def test_the_production_judge_reports_a_billed_and_an_as_if_total(tmp_path):
    """Their difference is exactly the cache, so printing both is also the cache's savings
    report -- and re-scoring under a new scorer_hash is supposed to be free."""
    from pi_run.judge_client import judge_replay

    c = judge_replay(_env(PI_CACHE_ROOT=str(tmp_path)), cache_root=str(tmp_path))
    spend = c.spend()
    assert set(spend) >= {"usd", "usd_billed", "n_calls", "cache_hits", "tok_total"}
    assert spend["usd_billed"] <= spend["usd"]


# --------------------------------------------------------------------------- the second bill
#
# `--spend-cap` is checked in `run_sweep` and covers ROLLOUTS. Judging is a second provider
# bill: every eligible run with a report and gold, times every criterion, times both
# presentation orders, for every preregistered arm pair -- and it produces kpr_incremental, a
# primary endpoint. Nothing bounded it.


def test_the_judge_cap_has_the_same_precedence_rule_as_the_sweeps(monkeypatch):
    """An explicit argument wins and the env var is the default -- the same rule as
    --concurrency and --spend-cap, and for the same reason: a variable exported once for an
    afternoon must not outrank an instruction typed for this invocation."""
    from pi_eval.judges.harness import judge_spend_cap

    monkeypatch.setenv("PI_JUDGE_SPEND_CAP_USD", "5")
    assert judge_spend_cap(80.0) == 80.0
    assert judge_spend_cap() == 5.0
    assert judge_spend_cap(0) is None, "non-positive means uncapped, not stop immediately"
    monkeypatch.delenv("PI_JUDGE_SPEND_CAP_USD", raising=False)
    assert judge_spend_cap() is None


def test_exceeding_the_judge_cap_writes_nothing_rather_than_half_a_parquet():
    """Raising rather than truncating is the point. `score` writes its rows only after judging
    returns, so an abort leaves nothing partial -- and a half-judged parquet is WORSE than an
    unjudged one, because some arms would carry judge-derived rows and others would not, and
    the comparison between them would silently be over different subsets."""
    from pi_eval.judges.harness import JudgeBudgetExceeded, JudgeResult, _Grader

    class Spender:
        def __init__(self, billed):
            self._billed = billed

        def spend(self):
            return {"usd_billed": self._billed}

    g = _Grader(
        Spender(0.5),
        judge_model="m",
        judge_family="openai",
        seed=0,
        result=JudgeResult(),
        cap_usd=1.0,
    )
    g.check_budget()  # under the cap: no raise

    over = _Grader(
        Spender(1.5),
        judge_model="m",
        judge_family="openai",
        seed=0,
        result=JudgeResult(),
        cap_usd=1.0,
    )
    with pytest.raises(JudgeBudgetExceeded, match="Nothing was written"):
        over.check_budget()


def test_a_client_that_cannot_report_its_spend_is_not_treated_as_free():
    """A judge bill silently reported as $0.00 is exactly what lets an unbounded cost go
    unnoticed. A client with no spend() is uncappable, and that is a fact about the client
    rather than a licence to spend."""
    from pi_eval.judges.harness import JudgeResult, _Grader

    class Mute:
        pass

    g = _Grader(
        Mute(), judge_model="m", judge_family="openai", seed=0, result=JudgeResult(), cap_usd=1.0
    )
    g.check_budget()  # cannot check; must not pretend it spent zero and must not crash

    import inspect

    from pi_eval import score as score_mod

    assert '{"unknown": 1.0}' in inspect.getsource(score_mod.score)


def test_scoring_refuses_a_sigma_j_measured_on_a_different_judge(tmp_path):
    """sigma_J is the judge's OWN noise floor and report.py suppresses effects smaller than it.
    A floor from one model applied to another is not conservative in a knowable direction: a
    quieter judge's floor lets a noisier judge's effects through, and a noisier judge's floor
    suppresses a quieter judge's real ones."""
    import json as _json

    from pi_eval.prereg import sigma_j_judge

    (tmp_path / "prereg").mkdir()
    (tmp_path / "prereg" / "sigma_j.provenance.json").write_text(
        _json.dumps({"judge_model": "openai/judge-a", "seed": 0, "retest_seed": 1})
    )
    assert sigma_j_judge(tmp_path) == "openai/judge-a"
    assert sigma_j_judge(tmp_path / "nowhere-else") == "openai/judge-a"

    import inspect

    from pi_eval import score as score_mod

    src = inspect.getsource(score_mod.score)
    assert "sigma_J was measured on judge" in src
    assert "measured_on != judge_model" in src


def test_an_absent_sigma_j_provenance_does_not_block_scoring():
    """Before sigma_J is measured at all there is no sidecar, the floor is +inf and every judged
    effect is flagged. Refusing to score in that state would make the flag unreachable."""
    from pi_eval.prereg import sigma_j_judge

    assert sigma_j_judge("/nonexistent-path-xyz") == ""
