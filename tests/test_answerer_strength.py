"""Lane L1.10 tooling (`scripts/answerer_strength/lib.py`).

Hermetic: every fixture here is synthetic and built under `tmp_path` or in-memory, never
`artifacts/testsplit_qa` or `runs/` -- both are untracked (`git status ??`) and a worktree does
not carry untracked files (see the repo's own `worktree-agents-autoclean-and-pythonpath`
lesson), so a test that opened either would pass only on the machine that built it.

CONTRIBUTING.md rule 2: `test_build_view_and_evidence_raises_on_a_tampered_subset_hash` and
`test_build_view_and_evidence_raises_on_a_missing_cited_uid` exist because the whole lane's
claim -- "the reconstructed prompt is byte-identical to the recorded run's" -- is worth nothing
if a silent mismatch could pass through unnoticed; both are written to FAIL if the assertion in
`build_view_and_evidence` is ever weakened to a warning.
"""

from __future__ import annotations

import json
import math

import duckdb
import pytest
from scripts.answerer_strength import lib

# --------------------------------------------------------------------------- env


def test_load_env_file_parses_key_value_and_strips_quotes(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "LITELLM_BASE_URL=http://127.0.0.1:4000",
                'PI_MODEL_ANSWERER="openai/aws/gpt-oss-120b"',
                "PI_GOLD_ROOT='/some/path'",
                "EMPTY=",
            ]
        )
    )
    env = lib.load_env_file(p)
    assert env["LITELLM_BASE_URL"] == "http://127.0.0.1:4000"
    assert env["PI_MODEL_ANSWERER"] == "openai/aws/gpt-oss-120b"
    assert env["PI_GOLD_ROOT"] == "/some/path"
    assert env["EMPTY"] == ""
    assert "# a comment" not in env


# --------------------------------------------------------------------------- population


def test_read_population_reads_run_id_files(tmp_path):
    for arm in lib.ARMS:
        for suite in lib.SUITES:
            (tmp_path / f"run_ids.{arm}.{suite}.txt").write_text("a\nb\n\nc\n")
    pop = lib.read_population(tmp_path)
    assert set(pop) == set(lib.ARMS)
    for arm in lib.ARMS:
        assert set(pop[arm]) == set(lib.SUITES)
        assert pop[arm]["musique"] == ["a", "b", "c"]


# --------------------------------------------------------------------------- run record


def _write_run(tmp_path, run_id, *, n_turns, seed=0, word_cap=30, corpus_hash="ch1"):
    d = tmp_path / "runs" / run_id
    d.mkdir(parents=True)
    manifest = {
        "suite_id": "musique",
        "task_id": "t1",
        "seed": seed,
        "corpus_dir": "abc123",
        "corpus_hash": corpus_hash,
        "word_cap": word_cap,
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    turns = []
    for i in range(n_turns):
        turns.append(
            {
                "turn_idx": i,
                "draft_text": f"draft at turn {i}",
                "draft_sha": f"sha{i}",
            }
        )
    (d / "turns.jsonl").write_text("\n".join(json.dumps(t) for t in turns))
    outcome = {
        "answer": {
            "text": "recorded answer text",
            "evidence_hash": "subset123",
            "cited_unit_ids": ["u1", "u2"],
        },
        "subset_hash": "subset123",
    }
    (d / "outcome.json").write_text(json.dumps(outcome))
    (d / "evidence.jsonl").write_text("")
    return d


def test_load_run_record_reads_last_turn_draft_text(tmp_path):
    _write_run(tmp_path, "r1", n_turns=3)
    rec = lib.load_run_record(tmp_path / "runs", "r1", arm="trained")
    assert rec.draft_text == "draft at turn 2"
    assert rec.n_turns == 3
    assert rec.cited_uids == ("u1", "u2")
    assert rec.evidence_hash == "subset123"
    assert rec.recorded_answer_text == "recorded answer text"
    assert rec.seed == 0
    assert rec.suite_id == "musique"
    assert rec.arm == "trained"


def test_load_run_record_zero_turns_has_empty_draft_text(tmp_path):
    _write_run(tmp_path, "r0", n_turns=0)
    rec = lib.load_run_record(tmp_path / "runs", "r0", arm="prompted")
    assert rec.draft_text == ""
    assert rec.n_turns == 0


# --------------------------------------------------------------------------- suite / evidence


def _write_corpus(tmp_path, *, suite="musique", chash="abc123", task_id="t1"):
    root = tmp_path / "corpora" / suite / chash
    root.mkdir(parents=True)
    rec = {
        "id": task_id,
        "question": "Who wrote the thing?",
        "paragraphs": [
            {"idx": 0, "title": "Doc A", "text": "Alice wrote the thing in 1990."},
            {"idx": 1, "title": "Doc B", "text": "Bob is unrelated distractor text."},
        ],
    }
    (root / "tasks.jsonl").write_text(json.dumps(rec) + "\n")
    return tmp_path / "corpora"


def _units_for(tmp_path, *, suite="musique", chash="abc123", task_id="t1"):
    from pinq_adapters.musique.suite import MusiqueSuite

    corpora_root = _write_corpus(tmp_path, suite=suite, chash=chash, task_id=task_id)
    s = MusiqueSuite(corpora_root / suite / chash)
    return s, {u.uid: u for u in s.units(task_id)}, corpora_root


def test_build_view_and_evidence_reconstructs_byte_identical_prompt_inputs(tmp_path):
    s, by_uid, corpora_root = _units_for(tmp_path)
    uid0 = s.units("t1")[0].uid
    _write_run(tmp_path, "rgood", n_turns=1, corpus_hash=s.corpus_hash)
    # patch the recorded evidence to reference the real single unit, so the reconstruction
    # both finds the uid and reproduces the recorded subset_hash
    d = tmp_path / "runs" / "rgood"
    outcome = json.loads((d / "outcome.json").read_text())
    outcome["answer"]["cited_unit_ids"] = [uid0]
    from pinq.types import Evidence

    real_hash = Evidence.of((by_uid[uid0],)).subset_hash
    outcome["answer"]["evidence_hash"] = real_hash
    outcome["subset_hash"] = real_hash
    (d / "outcome.json").write_text(json.dumps(outcome))

    rec = lib.load_run_record(tmp_path / "runs", "rgood", arm="trained")
    view, ev = lib.build_view_and_evidence(rec, corpora_root=corpora_root)
    assert view.question == "Who wrote the thing?"
    assert len(ev.units) == 1
    assert ev.units[0].text == "Alice wrote the thing in 1990."
    assert ev.subset_hash == real_hash


def test_build_view_and_evidence_raises_on_a_tampered_subset_hash(tmp_path):
    s, by_uid, corpora_root = _units_for(tmp_path)
    uid0 = s.units("t1")[0].uid
    _write_run(tmp_path, "rbad", n_turns=1, corpus_hash=s.corpus_hash)
    d = tmp_path / "runs" / "rbad"
    outcome = json.loads((d / "outcome.json").read_text())
    outcome["answer"]["cited_unit_ids"] = [uid0]
    outcome["answer"]["evidence_hash"] = "this-does-not-match-anything"
    (d / "outcome.json").write_text(json.dumps(outcome))

    rec = lib.load_run_record(tmp_path / "runs", "rbad", arm="trained")
    with pytest.raises(lib.EvidenceReconstructionError, match="subset_hash"):
        lib.build_view_and_evidence(rec, corpora_root=corpora_root)


def test_build_view_and_evidence_raises_on_a_missing_cited_uid(tmp_path):
    s, by_uid, corpora_root = _units_for(tmp_path)
    _write_run(tmp_path, "rmiss", n_turns=1, corpus_hash=s.corpus_hash)
    d = tmp_path / "runs" / "rmiss"
    outcome = json.loads((d / "outcome.json").read_text())
    outcome["answer"]["cited_unit_ids"] = ["not-a-real-uid"]
    (d / "outcome.json").write_text(json.dumps(outcome))

    rec = lib.load_run_record(tmp_path / "runs", "rmiss", arm="trained")
    with pytest.raises(lib.EvidenceReconstructionError, match="absent from the suite"):
        lib.build_view_and_evidence(rec, corpora_root=corpora_root)


# --------------------------------------------------------------------------- TryPrimaryThenFallback


class _FakeInner:
    def __init__(self, *, cache=None, dispatched_tag):
        self._cache = cache or {}
        self.dispatched_tag = dispatched_tag
        self.n_dispatch = 0

    def key_for(self, **kw):
        return kw["messages"][0]["content"]

    def complete(self, **kw):
        self.n_dispatch += 1
        return f"{self.dispatched_tag}:{kw['messages'][0]['content']}", None

    def pin(self, role):
        return {"role": role, "tag": self.dispatched_tag}


def test_try_primary_then_fallback_uses_primary_on_cache_hit():
    primary = _FakeInner(cache={"hello": {"text": "cached"}}, dispatched_tag="primary")
    fallback = _FakeInner(dispatched_tag="fallback")
    w = lib.TryPrimaryThenFallback(primary, fallback)
    text, _ = w.complete(messages=[{"content": "hello"}])
    assert text == "primary:hello"
    assert primary.n_dispatch == 1
    assert fallback.n_dispatch == 0
    assert w.used_fallback is False


def test_try_primary_then_fallback_uses_fallback_on_cache_miss():
    primary = _FakeInner(cache={}, dispatched_tag="primary")
    fallback = _FakeInner(dispatched_tag="fallback")
    w = lib.TryPrimaryThenFallback(primary, fallback)
    text, _ = w.complete(messages=[{"content": "new request"}])
    assert text == "fallback:new request"
    assert primary.n_dispatch == 0
    assert fallback.n_dispatch == 1
    assert w.used_fallback is True


def test_try_primary_then_fallback_with_no_fallback_always_dispatches_primary():
    primary = _FakeInner(cache={}, dispatched_tag="primary")
    w = lib.TryPrimaryThenFallback(primary, None)
    text, _ = w.complete(messages=[{"content": "anything"}])
    assert text == "primary:anything"
    assert primary.n_dispatch == 1
    assert w.used_fallback is False


# --------------------------------------------------------------------------- scoring


class _StubGraph:
    def __init__(self, answer, aliases=()):
        self._answer = answer
        self.gold_aliases = aliases

    @property
    def answer(self):
        return self._answer


def test_score_answer_calls_the_repos_own_metric_functions():
    g = _StubGraph("Paris", aliases=("Ville Lumiere",))
    out = lib.score_answer("The city is Paris, France.", g)
    assert out["contains_answer"] == 1.0
    assert 0.0 < out["answer_token_f1"] <= 1.0
    assert 0.0 < out["answer_token_recall"] <= 1.0

    out_wrong = lib.score_answer("The city is Berlin.", g)
    assert out_wrong["contains_answer"] == 0.0
    assert out_wrong["answer_token_recall"] == 0.0


# --------------------------------------------------------------------------- bca_two_sample_mean_delta


def test_bca_two_sample_point_estimate_is_the_mean_difference():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [0.0, 0.0, 0.0, 0.0]
    out = lib.bca_two_sample_mean_delta(a, b, n_boot=200, seed=0)
    assert out["point"] == pytest.approx(2.5)
    assert out["n_a"] == 4
    assert out["n_b"] == 4
    assert out["lo"] <= out["point"] <= out["hi"]


def test_bca_two_sample_is_order_independent():
    a = [0.1, 0.4, 0.9, 0.2, 0.5]
    b = [0.6, 0.3, 0.8, 0.0]
    forward = lib.bca_two_sample_mean_delta(a, b, n_boot=500, seed=0)
    backward = lib.bca_two_sample_mean_delta(
        list(reversed(a)), list(reversed(b)), n_boot=500, seed=0
    )
    assert forward["point"] == backward["point"]
    assert forward["lo"] == backward["lo"]
    assert forward["hi"] == backward["hi"]


def test_bca_two_sample_positive_control_excludes_zero():
    a = [1.0] * 30
    b = [0.0] * 30
    out = lib.bca_two_sample_mean_delta(a, b, n_boot=2000, seed=0)
    assert out["lo"] > 0


def test_bca_two_sample_deterministic_null_includes_zero():
    a = [0.5] * 30
    b = [0.5] * 30
    out = lib.bca_two_sample_mean_delta(a, b, n_boot=2000, seed=0)
    assert out["point"] == 0.0
    assert out["lo"] <= 0.0 <= out["hi"]


def test_bca_two_sample_empty_group_returns_nan():
    out = lib.bca_two_sample_mean_delta([], [1.0], n_boot=100, seed=0)
    assert math.isnan(out["point"])


# --------------------------------------------------------------------------- with_stability


def test_with_stability_skips_50k_when_far_from_zero():
    calls = []

    def compute(n_boot, seed):
        calls.append((n_boot, seed))
        return (0.5, 0.4, 0.6)

    out = lib.with_stability(compute, seed=0)
    assert out["verdict"] == "excludes_zero"
    assert out["stability_checked"] is False
    assert calls == [(lib.DEFAULT_N_BOOT, 0)]


def test_with_stability_confirms_a_stable_near_zero_bound():
    def compute(n_boot, seed):
        return (0.01, -0.005, 0.5)  # lo within NEAR_ZERO of 0, sign stable every time

    out = lib.with_stability(compute, seed=0)
    assert out["stability_checked"] is True
    assert out["stable"] is True
    assert out["verdict"] == "includes_zero"
    assert out["n_boot_reported"] == lib.STABILITY_N_BOOT


def test_with_stability_flags_an_unstable_bound_as_undecided():
    signs = iter([-0.005, +0.003, -0.002])  # flips sign across the three stability seeds

    def compute(n_boot, seed):
        if n_boot == lib.DEFAULT_N_BOOT:
            return (0.01, -0.005, 0.5)
        return (0.01, next(signs), 0.5)

    out = lib.with_stability(compute, seed=0)
    assert out["stable"] is False
    assert out["verdict"] == "undecided"


def test_with_stability_reports_absent_on_nan():
    def compute(n_boot, seed):
        return (float("nan"), float("nan"), float("nan"))

    out = lib.with_stability(compute, seed=0)
    assert out["verdict"] == "absent"


# --------------------------------------------------------------------------- paired_contrast / one_sample_level


def test_paired_contrast_matches_shared_keys_only():
    trained = {"a|0": 1.0, "a|1": 1.0, "b|0": 0.5}
    prompted = {"a|0": 0.0, "a|1": 0.0, "c|0": 0.9}
    out = lib.paired_contrast(trained, prompted, seed=0)
    assert out["n"] == 2  # only a|0, a|1 shared
    assert out["point"] == pytest.approx(1.0)


def test_one_sample_level_basic():
    out = lib.one_sample_level([1.0, 1.0, 1.0, 1.0], seed=0)
    assert out["point"] == pytest.approx(1.0)
    assert out["n"] == 4


def test_one_sample_level_empty_is_absent():
    out = lib.one_sample_level([], seed=0)
    assert out["verdict"] == "absent"
    assert out["n"] == 0


# --------------------------------------------------------------------------- matched_cost_coverage_by_task


SCORER = "testscorer"

_FIXTURE_RUNS = {
    # run_id: (n_asks, stop_reason, ladder{k: coverage}, terminal evidence_coverage)
    "t-A-s0": (2, "policy_stop", {0: 0.0, 1: 0.5, 2: 1.0}, 1.0),
    "t-A-s1": (3, "policy_stop", {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.0}, 1.0),
    "t-B-s0": (1, "policy_stop", {0: 0.0, 1: 0.0}, 0.0),
    "p-A-s0": (2, "policy_stop", {0: 0.0, 1: 0.25, 2: 0.5}, 0.5),
    "p-A-s1": (2, "policy_stop", {0: 0.0, 1: 0.25, 2: 0.5}, 0.5),
    "p-B-s0": (1, "policy_stop", {0: 0.0, 1: 0.0}, 0.0),
}


def _synthetic_store():
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE scores (run_id VARCHAR, metric_name VARCHAR, scorer_hash VARCHAR, value DOUBLE)"
    )
    con.execute("CREATE TABLE runs (run_id VARCHAR, n_asks INTEGER, stop_reason VARCHAR)")
    score_rows = []
    run_rows = []
    for run_id, (n_asks, stop_reason, ladder, coverage) in _FIXTURE_RUNS.items():
        run_rows.append((run_id, n_asks, stop_reason))
        for k, q in ladder.items():
            score_rows.append((run_id, f"frontier_q#{k}", SCORER, q))
        score_rows.append((run_id, "evidence_coverage", SCORER, coverage))
    con.executemany("INSERT INTO scores VALUES (?,?,?,?)", score_rows)
    con.executemany("INSERT INTO runs VALUES (?,?,?)", run_rows)
    return con


def _fixture_run_dicts(run_ids):
    out = []
    for rid in run_ids:
        task = rid.split("-")[1]
        seed = int(rid.split("-s")[-1])
        out.append({"run_id": rid, "task_id": task, "seed": seed})
    return out


def test_matched_cost_coverage_by_task_matches_gate_matched_cost():
    from pinq_train.gate import _matched_cost

    con = _synthetic_store()
    ck_runs = _fixture_run_dicts(["t-A-s0", "t-A-s1", "t-B-s0"])
    ba_runs = _fixture_run_dicts(["p-A-s0", "p-A-s1", "p-B-s0"])
    for r in [*ck_runs, *ba_runs]:
        r["suite_id"] = "musique"

    by_task = lib.matched_cost_coverage_by_task(
        con, ck_runs=ck_runs, ba_runs=ba_runs, scorer_hash=SCORER
    )

    from pinq_train.gate import _by_key

    ck_keys = _by_key(ck_runs)
    ba_keys = _by_key(ba_runs)
    official = _matched_cost(
        con,
        ckpt=ck_runs,
        base=ba_runs,
        ck_keys=ck_keys,
        ba_keys=ba_keys,
        scorer_hash=SCORER,
        seed=0,
        n_resamples=200,
    )
    assert official["pooled"]["n_tasks"] == len(by_task) == 2
    pooled_mean = sum(by_task.values()) / len(by_task)
    assert pooled_mean == pytest.approx(official["pooled"]["delta"])
    assert by_task[("musique", "A")] == pytest.approx(0.5)
    assert by_task[("musique", "B")] == pytest.approx(0.0)


def test_high_low_split_partitions_on_sign():
    delta = {
        ("musique", "A"): 0.5,
        ("musique", "B"): 0.0,
        ("musique", "C"): -0.1,
        ("strategyqa", "D"): 0.9,
    }
    high, low = lib.high_low_split(delta, suite="musique")
    assert high == {"A"}
    assert low == {"B", "C"}
