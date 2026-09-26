"""Pure helpers and refusals of `scripts/seed_identity/frames_recipe.py`, on SYNTHETIC inputs only.

What each test guards:
- the record parsers read the FRAMES block, the FRAMES estimator settings and the FRAMES contrast
  tables, not the musique or strategyqa rows printed beside them (a parser that took the first
  `teacher` line or the first `n_boot=` would lock the reader against the wrong suite and pass);
- the estimator is READ from the record, and a record that states two different settings is
  refused rather than resolved by whichever match comes first;
- the recipe unit averages training seeds 1 and 2 WITHIN the task and refuses when the two seeds
  cover different tasks or disagree on a task's cluster, rather than silently dropping units;
- a contrast refuses two unit sets that differ (`paired_difference` would silently intersect);
- the decision rule needs EVERY 50k interval to exclude zero on the SAME side;
- the population refusals of the new-roots mode: each fires on the one defect it names, and none
  fires on a clean synthetic cell (so a refusal that can never pass is caught too).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import duckdb
import pytest

from pinq.ids import h, semantic_hash

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "scripts" / "seed_identity" / "frames_recipe.py"
CODE_FULL = "a61c4f4be3e9" + "0" * 28
FROZEN = "openai/aws/gpt-oss-120b"
S1 = "qwen3-8b-dpo-stacked-notdone-both-s1"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("seed_identity_frames_recipe", GEN)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------- record parsers

BUDGET_BLOCK = """\
```
--- musique  (metric evidence_coverage, n_boot=1000)
  teacher       n= 400 +0.0650 [+0.0500, +0.0831] p=0.0001  EXCLUDES 0
  trained       n= 400 +0.0048 [+0.0018, +0.0115] p=0.0317  EXCLUDES 0

--- FRAMES  (metric answer_correct, n_boot=10000)
  teacher       n= 824 +0.0170 [-0.0024, +0.0364]  covers 0
  base8b        n= 824 +0.0061 [-0.0061, +0.0158]  covers 0
  trained       n= 824 +0.0012 [+0.0000, +0.0036]  covers 0
  drafter_only  n= 824 +0.0000 [+0.0000, +0.0000]  covers 0
```

--- strategyqa  (metric evidence_coverage, n_boot=1000)
  teacher       n= 400 +0.0717 [+0.0555, +0.0961] p=0.0001  EXCLUDES 0
"""


def test_budget_block_reads_frames_rows_only(mod):
    got = mod.parse_budget_response(BUDGET_BLOCK)
    assert set(got) == {"teacher", "base8b", "trained", "drafter_only"}
    assert got["teacher"] == (824, 0.0170, -0.0024, 0.0364)
    assert got["trained"] == (824, 0.0012, 0.0, 0.0036)


def test_budget_block_refuses_a_record_without_frames(mod):
    with pytest.raises(ValueError):
        mod.parse_budget_response(BUDGET_BLOCK.replace("FRAMES", "OTHER"))


ESTIMATOR = (
    BUDGET_BLOCK
    + """
Levels and contrasts then come from `cluster_bootstrap(n_boot=10000, seed=0)` and
`paired_difference(n_boot=10000, n_perm=10000, seed=0)` over
`COALESCE(NULLIF(template_id,
''), task_id)`, under `pi_eval.report.ELIGIBLE` unmodified.
"""
)


def test_estimator_is_read_from_the_record(mod):
    est = mod.parse_estimator(ESTIMATOR)
    # the FRAMES block's n_boot (10000), not musique's or strategyqa's (1000)
    assert est == {
        "level_n_boot": 10000,
        "level_seed": 0,
        "contrast_n_boot": 10000,
        "contrast_n_perm": 10000,
        "contrast_seed": 0,
        "cluster": "COALESCE(NULLIF(template_id,''),task_id)",
    }


def test_estimator_follows_the_record_not_a_default(mod):
    text = ESTIMATOR.replace(
        "cluster_bootstrap(n_boot=10000, seed=0)", "cluster_bootstrap(n_boot=2000, seed=7)"
    )
    est = mod.parse_estimator(text)
    assert (est["level_n_boot"], est["level_seed"]) == (2000, 7)


@pytest.mark.parametrize(
    "mutate",
    [
        # two different contrast settings in one record
        lambda t: t + "\nalso `paired_difference(n_boot=1000, n_perm=1000, seed=0)`\n",
        # the FRAMES block disagrees with the stated contrast resample count
        lambda t: t.replace(
            "(metric answer_correct, n_boot=10000)", "(metric answer_correct, n_boot=1000)"
        ),
        # no level estimator stated at all
        lambda t: t.replace("cluster_bootstrap(n_boot=10000, seed=0)", "cluster_bootstrap"),
        # no clustering stated
        lambda t: t.replace("COALESCE(NULLIF(template_id,\n''), task_id)", "task_id"),
    ],
)
def test_estimator_refuses_an_ambiguous_or_silent_record(mod, mutate):
    with pytest.raises(ValueError):
        mod.parse_estimator(mutate(ESTIMATOR))


def test_the_script_clusters_on_the_expression_the_record_states(mod):
    assert mod.normalise_cluster(mod.CLUSTER_SQL) == mod.parse_estimator(ESTIMATOR)["cluster"]


LEVELS = """\
| arm | cap | n | clusters | mean n_asks | mean retrieval_calls | answer_correct [BCa 95%] | ceiling-hit |
|---|---|---|---|---|---|---|---|
| teacher | 4 | 824 | 824 | 3.2609 | 3.2609 | 0.2609 [0.2306, 0.2913] | 461/824 (55.95%) |
| trained | 24 | 824 | 812 | 2.3944 | 2.3944 | 0.2257 [0.1966, 0.2549] | 8/824 (0.97%) |

| arm | `answer_correct` at cap 4 | at cap 24 | rise | mean asks 4 -> 24 |
|---|---|---|---|---|
| `teacher` (120B prompted) | 0.2609 | 0.2779 | **+0.0170** | 3.26 -> 11.72 |
"""


def test_level_extras_read_clusters_and_ceiling_hits(mod):
    assert mod.parse_level_extras(LEVELS) == {("teacher", 4): (824, 461), ("trained", 24): (812, 8)}


CONTRASTS = """\
#### `trained` minus `teacher`

| cap | n | clusters | delta answer_correct | BCa 95% | zero |
|---|---|---|---|---|---|
| 4 | 824 | 824 | -0.0364 | [-0.0680, -0.0073] | excludes 0 |
| 8 | 824 | 824 | -0.0461 | [-0.0789, -0.0158] | excludes 0 |

#### `trained` minus `base8b`

| cap | n | clusters | delta answer_correct | BCa 95% | zero |
|---|---|---|---|---|---|
| 8 | 824 | 823 | +0.0267 | [-0.0049, +0.0570] | covers 0 |

### What the trained arm does not spend, in asks

| contrast | cap | delta n_asks | BCa 95% | zero |
|---|---|---|---|---|
| `trained` - `teacher` | 4 | -1.1141 | [-1.1942, -1.0340] | excludes 0 |
"""


def test_contrast_tables_keyed_by_pair_and_cap(mod):
    got = mod.parse_contrast_tables(CONTRASTS, "answer_correct")
    assert set(got) == {("trained", "teacher"), ("trained", "base8b")}
    assert got[("trained", "teacher")][4] == (824, 824, -0.0364, -0.0680, -0.0073)
    assert got[("trained", "base8b")] == {8: (824, 823, 0.0267, -0.0049, 0.0570)}


def test_contrast_tables_ignore_other_metrics(mod):
    # the n_asks table has no `delta answer_correct` column and must not be read as one
    assert mod.parse_contrast_tables(CONTRASTS, "n_asks") == {}


# --------------------------------------------------------------------------- estimators


def _u(cl, y, asks, rc, hit):
    return {"cl": cl, "y": y, "asks": asks, "rc": rc, "hit": hit}


def test_pool_seeds_averages_within_the_task(mod):
    a = {"t1|0": _u("t1", 1.0, 2, 2.0, 0), "t2|0": _u("t2", 0.0, 4, 4.0, 1)}
    b = {"t1|0": _u("t1", 0.0, 4, 4.0, 1), "t2|0": _u("t2", 0.0, 2, 2.0, 1)}
    got = mod.pool_seeds(a, b)
    assert list(got) == ["t1|0", "t2|0"]
    assert got["t1|0"] == {"cl": "t1", "y": 0.5, "asks": 3.0, "rc": 3.0, "hit": 0.5}
    assert got["t2|0"] == {"cl": "t2", "y": 0.0, "asks": 3.0, "rc": 3.0, "hit": 1.0}


def test_pool_seeds_refuses_uncovered_units(mod):
    a = {"t1|0": _u("t1", 1.0, 2, 2.0, 0), "t2|0": _u("t2", 0.0, 4, 4.0, 1)}
    b = {"t1|0": _u("t1", 0.0, 4, 4.0, 1)}
    with pytest.raises(SystemExit):
        mod.pool_seeds(a, b)


def test_pool_seeds_refuses_a_cluster_disagreement(mod):
    a = {"t1|0": _u("t1", 1.0, 2, 2.0, 0)}
    b = {"t1|0": _u("other", 0.0, 4, 4.0, 1)}
    with pytest.raises(SystemExit):
        mod.pool_seeds(a, b)


EST = {
    "level_n_boot": 200,
    "level_seed": 0,
    "contrast_n_boot": 200,
    "contrast_n_perm": 200,
    "contrast_seed": 0,
}


def test_a_contrast_refuses_unpaired_unit_sets(mod):
    a = {"t1|0": _u("t1", 1.0, 2, 2.0, 0), "t2|0": _u("t2", 0.0, 4, 4.0, 1)}
    b = {"t1|0": _u("t1", 0.0, 4, 4.0, 1), "t3|0": _u("t3", 0.0, 2, 2.0, 1)}
    with pytest.raises(SystemExit):
        mod.run_contrasts({"x": (a, b, False)}, EST, workers=1)


def test_a_contrast_is_paired_over_every_unit(mod):
    a = {f"t{i}|0": _u(f"t{i}", float(i % 2), 1, 1.0, 0) for i in range(6)}
    b = {f"t{i}|0": _u(f"t{i}", 0.0, 1, 1.0, 0) for i in range(6)}
    got = mod.run_contrasts({"x": (a, b, False)}, EST, workers=1)["x"]
    assert got["n"] == 6 and got["n_clusters"] == 6
    assert got["delta"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("cis", "want"),
    [
        ([(0.01, 0.2), (0.02, 0.3), (0.001, 0.1)], True),
        ([(-0.2, -0.01), (-0.3, -0.02), (-0.1, -0.001)], True),
        ([(0.01, 0.2), (-0.001, 0.3), (0.001, 0.1)], False),
        ([(0.01, 0.2), (-0.2, -0.01), (0.001, 0.1)], False),
        ([(0.0, 0.2), (0.01, 0.2), (0.01, 0.2)], False),
        ([], False),
    ],
)
def test_decided_needs_every_interval_on_one_side(mod, cis, want):
    assert mod.decided(cis) is want


# --------------------------------------------------------------------------- population refusals
#
# A synthetic run root: one directory per run, each with manifest.json, status.json and
# calls.jsonl, in the shapes a real FRAMES run writes (field names checked against a published
# run under runs/). N_CELL is patched to 2 so a cell is two tasks.


def _manifest(
    task, *, cap=8, arm="inquirer_trained", model=S1, code=CODE_FULL, dirty=False, base="u1"
):
    pins = {
        "drafter": {"model_id": FROZEN, "base_url_sha": base, "adapter_sha": None},
        "answerer": {"model_id": FROZEN, "base_url_sha": base, "adapter_sha": None},
    }
    if model is not None:
        pins["inquirer"] = {"model_id": model, "base_url_sha": base, "adapter_sha": "ad"}
    m = {
        "suite_id": "frames",
        "arm_id": arm,
        "seed": 0,
        "task_id": task,
        "corpus_hash": "c",
        "model_pin_hash": "p",
        "prompt_hashes": {"inquirer": "q"},
        "budget_cap": cap,
        "max_turns": 16,
        "k": 5,
        "questions_hash": None,
        "pilot_flag": False,
        "exploratory": False,
        "prompt_variant_id": None,
        "code_version": code,
        "upstream_pins": {},
        "counterfactual_kind": "none",
        "prefix_k": None,
        "dirty": dirty,
        "grid_name": f"frames_recipe_cap{cap}",
        "grid_sha256": "g",
        "canary_nonces_loaded": 17844,
        "concurrency": 8,
        "pins": pins,
    }
    sem = semantic_hash(m)
    m["semantic_hash"] = sem
    m["run_id"] = ("dev-" if dirty else "") + h("runid", sem, "none", "None")[:32]
    return m


def _write(root, m, *, status="ok", calls=None, name=None):
    d = root / (name or m["run_id"])
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(m))
    (d / "status.json").write_text(json.dumps({"status": status, "started_at": 0.0}))
    if calls is None:
        model = (m["pins"].get("inquirer") or {}).get("model_id")
        calls = ([("inquirer", model)] if model else []) + [
            ("drafter", FROZEN),
            ("answerer", FROZEN),
        ]
    (d / "calls.jsonl").write_text(
        "".join(json.dumps({"actor": a, "model": x}) + "\n" for a, x in calls)
    )
    return d


@pytest.fixture
def small(mod, monkeypatch):
    monkeypatch.setattr(mod, "N_CELL", 2)
    return mod


def _problems(mod, root, label="s1"):
    return mod.population(label, root=root)[8]["problems"]


def test_a_clean_cell_raises_no_refusal(small, tmp_path):
    for t in ("frames_0000", "frames_0001"):
        _write(tmp_path, _manifest(t))
    cell = small.population("s1", root=tmp_path)[8]
    assert cell["problems"] == []
    assert cell["n_ok_distinct_units"] == 2
    # the stream's other caps have no directories, and each of those is refused on its own
    assert small.population("s1", root=tmp_path)[4]["problems"] == [
        "no run directories at this cap"
    ]


def test_a_short_cell_is_refused(small, tmp_path):
    _write(tmp_path, _manifest("frames_0000"))
    assert any(p.startswith("ok=1") for p in _problems(small, tmp_path))


def test_an_errored_run_beside_the_ok_ones_is_refused(small, tmp_path):
    for t in ("frames_0000", "frames_0001"):
        _write(tmp_path, _manifest(t))
    _write(tmp_path, _manifest("frames_0002"), status="error")
    assert any(p.startswith("non-ok run dirs: 1") for p in _problems(small, tmp_path))


def test_a_second_code_version_is_refused(small, tmp_path):
    _write(tmp_path, _manifest("frames_0000"))
    _write(tmp_path, _manifest("frames_0001", code="107ef2221a55" + "0" * 28))
    probs = _problems(small, tmp_path)
    assert any(p.startswith("code_versions") and "107ef2221a55" in p for p in probs)


def test_a_dirty_manifest_is_refused(small, tmp_path):
    _write(tmp_path, _manifest("frames_0000"))
    _write(tmp_path, _manifest("frames_0001", dirty=True))
    probs = _problems(small, tmp_path)
    assert "dirty=1" in probs and "dev=1" in probs


def test_a_dev_directory_is_refused(small, tmp_path):
    _write(tmp_path, _manifest("frames_0000"))
    m = _manifest("frames_0001")
    _write(tmp_path, m, name="dev-" + m["run_id"])
    probs = _problems(small, tmp_path)
    assert "dev=1" in probs


def test_a_run_id_that_does_not_reproduce_is_refused(small, tmp_path):
    _write(tmp_path, _manifest("frames_0000"))
    m = _manifest("frames_0001")
    m["k"] = 6  # a semantic field moved after the id was minted
    _write(tmp_path, m)
    assert "id_not_reproduced=1" in _problems(small, tmp_path)


def test_a_wrong_inquirer_model_on_one_call_is_refused(small, tmp_path):
    _write(tmp_path, _manifest("frames_0000"))
    m = _manifest("frames_0001")
    _write(
        tmp_path, m, calls=[("inquirer", S1), ("inquirer", "qwen3-8b-base"), ("drafter", FROZEN)]
    )
    assert "wrong_inquirer_model=1" in _problems(small, tmp_path)


def test_a_questioner_run_with_no_inquirer_call_is_refused(small, tmp_path):
    _write(tmp_path, _manifest("frames_0000"))
    _write(tmp_path, _manifest("frames_0001"), calls=[("drafter", FROZEN)])
    assert "no_inquirer_call=1" in _problems(small, tmp_path)


def test_drafter_only_with_an_inquirer_call_is_refused(small, tmp_path):
    for t in ("frames_0000", "frames_0001"):
        m = _manifest(t, arm="drafter_only", model=None)
        calls = [("drafter", FROZEN)] + ([("inquirer", FROZEN)] if t.endswith("1") else [])
        _write(tmp_path, m, calls=calls)
    assert "wrong_inquirer_model=1" in _problems(small, tmp_path, "drafter")


def test_drafter_only_clean_raises_no_refusal(small, tmp_path):
    for t in ("frames_0000", "frames_0001"):
        _write(tmp_path, _manifest(t, arm="drafter_only", model=None))
    assert _problems(small, tmp_path, "drafter") == []


def test_a_frozen_role_on_another_model_is_refused(small, tmp_path):
    _write(tmp_path, _manifest("frames_0000"))
    m = _manifest("frames_0001")
    _write(tmp_path, m, calls=[("inquirer", S1), ("drafter", "qwen3-8b-base")])
    assert "wrong_frozen_call_model=1" in _problems(small, tmp_path)


def test_a_missing_manifest_is_refused_not_a_crash(small, tmp_path):
    for t in ("frames_0000", "frames_0001"):
        _write(tmp_path, _manifest(t))
    (tmp_path / "stray").mkdir()
    pop = small.population("s1", root=tmp_path)
    assert "no_manifest=1" in pop["unreadable"]["problems"]


def test_two_ok_runs_on_one_unit_are_refused(small, tmp_path):
    _write(tmp_path, _manifest("frames_0000"))
    _write(tmp_path, _manifest("frames_0001"))
    m = _manifest("frames_0001")
    m["concurrency"] = 4  # not a semantic field: same unit, same id -> use another dir name
    _write(tmp_path, m, name="0" * 32)
    probs = _problems(small, tmp_path)
    assert "duplicate_units=1" in probs


def _cell(units, base="u1"):
    return {"problems": [], "base_url_sha": [base], "ok_units": {u: "rid" + u for u in units}}


def test_across_cells_clean(mod):
    pub = {"t1|0", "t2|0"}
    pop = {"s1": {8: _cell(pub)}, "s2": {8: _cell(pub)}}
    assert mod.across_cells(pop, pub)["problems"] == []


def test_across_cells_refuses_two_base_urls(mod):
    pub = {"t1|0", "t2|0"}
    pop = {"s1": {8: _cell(pub)}, "s2": {8: _cell(pub, base="u2")}}
    assert any("base_url_sha" in p for p in mod.across_cells(pop, pub)["problems"])


def test_across_cells_refuses_unequal_unit_sets(mod):
    pub = {"t1|0", "t2|0"}
    pop = {"s1": {8: _cell(pub)}, "s2": {8: _cell({"t1|0", "t3|0"})}}
    probs = mod.across_cells(pop, pub)["problems"]
    assert any("unit sets" in p for p in probs)
    assert any("published" in p for p in probs)


def test_store_cell_must_be_the_population(small):
    units = {"t1|0": {"run_id": "a"}, "t2|0": {"run_id": "b"}}
    assert small.store_problems(units, {"a", "b"}) == []
    assert small.store_problems(units, {"a", "c"}) != []
    assert small.store_problems({"t1|0": {"run_id": "a"}}, {"a"}) != []


def test_stores_must_share_one_scorer_and_graph_version(mod):
    ok = {"s1/cap8": {"scorer_hash": ["x"], "graph_version": ["v1"]}}
    assert mod.store_identity_problems(ok) == []
    two = dict(ok, **{"s2/cap8": {"scorer_hash": ["y"], "graph_version": ["v1"]}})
    assert any("scorer_hash" in p for p in mod.store_identity_problems(two))


# --------------------------------------------------------------------------- the scored store


def _store(path, rows, calls, scorer="sh"):
    """A synthetic scored store in the three parquet tables `store_units` reads.

    A row is (run_id, task, arm_id, dirty[, budget_cap[, answer_correct]])."""
    path.mkdir()
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE r (run_id VARCHAR, task_id VARCHAR, seed INTEGER, template_id VARCHAR, "
        "arm_id VARCHAR, suite_id VARCHAR, budget_cap INTEGER, n_asks INTEGER, "
        "retrieval_calls DOUBLE, stop_reason VARCHAR, status VARCHAR, gold_exposed BOOLEAN, "
        "is_dev_run BOOLEAN, dirty BOOLEAN, pilot_flag BOOLEAN, canary_hit BOOLEAN, "
        "exploratory BOOLEAN, firewall_ok BOOLEAN, counterfactual_kind VARCHAR, split VARCHAR, "
        "reconciled_tokens BOOLEAN, reconciled_docs BOOLEAN)"
    )
    con.execute("CREATE TABLE y (run_id VARCHAR, value DOUBLE)")
    for rid, task, arm, dirty, *rest in rows:
        cap, value = (list(rest) + [8, 1.0][len(rest) :])[:2]
        con.execute(
            "INSERT INTO r VALUES (?, ?, 0, NULL, ?, 'frames', ?, 2, 2.0, 'stop', 'ok', FALSE, "
            "FALSE, ?, FALSE, FALSE, FALSE, TRUE, 'none', 'test', TRUE, TRUE)",
            [rid, task, arm, cap, dirty],
        )
        con.execute("INSERT INTO y VALUES (?, ?)", [rid, value])
    con.execute(
        "CREATE TABLE s AS SELECT run_id, 'answer_correct' AS metric_name, value, "
        "? AS scorer_hash, 'v1' AS graph_version FROM y",
        [scorer],
    )
    con.execute("CREATE TABLE c (run_id VARCHAR, actor VARCHAR, model VARCHAR)")
    for rid, actor, model in calls:
        con.execute("INSERT INTO c VALUES (?, ?, ?)", [rid, actor, model])
    for t, name in (("r", "runs"), ("s", "scores"), ("c", "calls")):
        con.execute(f"COPY {t} TO '{path / name}.parquet' (FORMAT parquet)")


def test_store_units_reads_only_eligible_runs(mod, tmp_path):
    rows = [("a", "t1", "inquirer_trained", False), ("b", "t2", "inquirer_trained", True)]
    _store(tmp_path / "st", rows, [("a", "inquirer", S1), ("b", "inquirer", S1)])
    units, hashes, graphs = mod.store_units("s1", 8, store=tmp_path / "st")
    assert set(units) == {"t1|0"} and units["t1|0"]["run_id"] == "a"
    assert (hashes, graphs) == (["sh"], ["v1"])


def test_store_units_refuses_a_wrong_inquirer_model(mod, tmp_path):
    rows = [("a", "t1", "inquirer_trained", False)]
    _store(tmp_path / "st", rows, [("a", "inquirer", S1), ("a", "inquirer", "qwen3-8b-base")])
    with pytest.raises(SystemExit):
        mod.store_units("s1", 8, store=tmp_path / "st")


def test_store_units_refuses_drafter_only_with_an_inquirer_call(mod, tmp_path):
    rows = [("a", "t1", "drafter_only", False)]
    _store(tmp_path / "st", rows, [("a", "inquirer", FROZEN)])
    with pytest.raises(SystemExit):
        mod.store_units("drafter", 8, store=tmp_path / "st")


def test_store_units_refuses_two_eligible_runs_on_one_unit(mod, tmp_path):
    rows = [("a", "t1", "inquirer_trained", False), ("b", "t1", "inquirer_trained", False)]
    _store(tmp_path / "st", rows, [("a", "inquirer", S1), ("b", "inquirer", S1)])
    with pytest.raises(SystemExit):
        mod.store_units("s1", 8, store=tmp_path / "st")


# --------------------------------------------------------------------------- the reading, end to end
#
# Five synthetic stores (one per root label) of three tasks each, every cap the label runs, and
# synthetic published cells. The bootstrap counts are patched down; what is tested is that the
# reading runs from population to contrasts, pools training seeds within the task, pairs each
# contrast over all units, and refuses stores that are not the population.

Y = {
    "teacher": [1.0, 1.0, 0.0],
    "base": [0.0, 1.0, 0.0],
    "s1": [1.0, 1.0, 1.0],
    "s2": [0.0, 0.0, 0.0],
    "drafter": [0.0, 0.0, 0.0],
}
TASKS = ["frames_0000", "frames_0001", "frames_0002"]


def _reading(mod, tmp_path, monkeypatch, scorer_of=lambda label: "sh"):
    monkeypatch.setattr(mod, "OUT", tmp_path)
    monkeypatch.setattr(mod, "N_CELL", 3)
    monkeypatch.setattr(mod, "READ_N_BOOT", 100)
    monkeypatch.setattr(mod, "DECIDE_N_BOOT", 100)
    pop: dict = {}
    for label, (arm, model, caps) in mod.ARMS.items():
        rows, calls = [], []
        for cap in caps:
            ok = pop.setdefault(label, {}).setdefault(cap, {"ok_units": {}})["ok_units"]
            for t, y in zip(TASKS, Y[label], strict=True):
                rid = f"{label}-{cap}-{t}"
                rows.append((rid, t, arm, False, cap, y))
                calls += [(rid, "inquirer", model)] if model else []
                calls.append((rid, "drafter", FROZEN))
                ok[f"{t}|0"] = rid
        _store(tmp_path / f"scores_parquet_{label}", rows, calls, scorer=scorer_of(label))
    pub = {
        (arm, cap): {
            f"{t}|0": {
                "run_id": f"p-{arm}-{t}",
                "cl": t,
                "asks": 1.0,
                "rc": 1.0,
                "hit": 0,
                "y": 1.0,
            }
            for t in TASKS
        }
        for arm in ("trained", "teacher", "base8b")
        for cap in mod.CAPS
    }
    return pop, pub


def test_the_reading_runs_end_to_end_on_synthetic_stores(mod, tmp_path, monkeypatch):
    pop, pub = _reading(mod, tmp_path, monkeypatch)
    out = mod.read(pop, EST, 1, pub, ["sh"])
    assert out["scorer_hash"] == ["sh"] and out["scorer_hash_equals_published"] is True
    rec = out["levels"]["recipe"]
    assert set(rec) == set(mod.CAPS)
    # seeds averaged WITHIN the task: s1 is all 1, s2 all 0, so every task is 0.5
    assert all(v["y"] == 0.5 and v["n"] == 3 for v in rec.values())
    assert out["levels"]["drafter_only"][8]["y"] == 0.0
    ct = out["contrasts"]
    full = [f"recipe-{c}/cap{k}" for c in ("base", "teacher") for k in mod.CAPS]
    full += ["recipe-drafter_only/cap8"] + [
        f"budget/{a}" for a in ("recipe", "teacher", "base", "s1", "s2")
    ]
    for name in full:
        assert name in ct and ct[name]["n"] == 3
        assert len(ct[name]["ci_50k"]) == 3 and isinstance(ct[name]["decided"], bool)
    assert ct["recipe-drafter_only/cap8"]["delta"] == pytest.approx(0.5)
    assert ct["recipe-teacher/cap8"]["delta"] == pytest.approx(0.5 - 2 / 3)
    assert ct["budget/recipe"]["delta"] == 0.0
    assert out["comparator_replay_vs_published"]["teacher/cap8"]["same_answer_correct"] == 2


def test_the_reading_refuses_two_scorer_hashes(mod, tmp_path, monkeypatch):
    pop, pub = _reading(
        mod, tmp_path, monkeypatch, lambda label: "other" if label == "s2" else "sh"
    )
    with pytest.raises(SystemExit):
        mod.read(pop, EST, 1, pub, ["sh"])


def test_the_reading_refuses_a_store_that_is_not_the_population(mod, tmp_path, monkeypatch):
    pop, pub = _reading(mod, tmp_path, monkeypatch)
    pop["base"][12]["ok_units"]["frames_0002|0"] = "a-run-the-store-does-not-hold"
    with pytest.raises(SystemExit):
        mod.read(pop, EST, 1, pub, ["sh"])
