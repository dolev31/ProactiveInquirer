"""`scripts/seed_identity/frames_structured.py` (the FRAMES panel of Figure 4), on SYNTHETIC inputs only.

What each test guards:
- the comparator specs are the (arm_id, inquirer model) pairs the HPC campaign passes, on the
  committed grid `frames_structured_cap8` (a label that reads the wrong model is a pooled pin);
- THE LOCK: the recipe, base and teacher cap-8 levels and the recipe-base and recipe-teacher
  contrasts are rebuilt by the same code path and compared with the record frames_recipe.py
  wrote, to 1e-9. The record here IS what `frames_recipe.read` writes on synthetic stores (not a
  hand-made dict), so the lock holding is not circular; a perturbed field, a record whose own
  lock failed, and a store that is not the recorded one are each refused;
- POPULATION of a new root: a clean root passes (so a refusal that can never pass is caught), and
  the wrong arm_id, the wrong inquirer model, the recipe's grid instead of the structured one, a
  missing unit, another unit set and a second base_url_sha are each refused; a root that is not
  on disk is refused with a message naming it; frames_recipe's own default grid check is
  unchanged by the new parameter (extended, not weakened);
- the recipe is training seeds 1 and 2 averaged WITHIN the task (keyed by task, not by row order),
  and every contrast is paired over the recipe's units;
- the reading refuses stores that are not the population or that disagree on scorer_hash.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import duckdb
import pytest
import yaml

from pinq.ids import h, semantic_hash

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "scripts" / "seed_identity" / "frames_structured.py"
CODE_FULL = "a61c4f4be3e9" + "0" * 28
FROZEN = "openai/aws/gpt-oss-120b"
OPUS = "openai/aws/claude-opus-5"
BASE8B = "qwen3-8b-base"
S1 = "qwen3-8b-dpo-stacked-notdone-both-s1"
S2 = "qwen3-8b-dpo-stacked-notdone-both-s2"
TASKS = ["frames_0000", "frames_0001", "frames_0002"]
NEW = ("par2_8b", "par2_120b", "par2_opus", "opus")

EST = {
    "level_n_boot": 100,
    "level_seed": 0,
    "contrast_n_boot": 100,
    "contrast_n_perm": 100,
    "contrast_seed": 0,
    "cluster": "COALESCE(NULLIF(template_id,''),task_id)",
}


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("seed_identity_frames_structured", GEN)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def fast(mod, monkeypatch, tmp_path):
    """Resample counts down, every root and store under tmp_path."""
    monkeypatch.setattr(mod.fr, "OUT", tmp_path)
    monkeypatch.setattr(mod.fr, "READ_N_BOOT", 100)
    monkeypatch.setattr(mod.fr, "DECIDE_N_BOOT", 100)
    monkeypatch.setattr(mod.fr, "N_CELL", len(TASKS))
    return mod


# --------------------------------------------------------------------------- the specs


def test_comparator_specs_are_the_campaigns_models(mod):
    assert mod.GRID == "frames_structured_cap8" and mod.CAP == 8
    assert {k: mod.spec(k)[:2] for k in NEW} == {
        "par2_8b": ("par2_rag", BASE8B),
        "par2_120b": ("par2_rag", FROZEN),
        "par2_opus": ("par2_rag", OPUS),
        "opus": ("inquirer_prompted", OPUS),
    }
    for k in NEW:
        assert mod.spec(k)[2:] == ((8,), "frames_structured_cap8")
    # the existing roots keep frames_recipe's own spec and its default grid
    assert mod.spec("base") == (*mod.fr.ARMS["base"], None)
    assert set(mod.COMPARATORS) == {"base", "teacher", *NEW}


def test_the_committed_grid_is_the_one_the_specs_name(mod):
    g = yaml.safe_load((REPO / "conf/grids/frames_structured_cap8.yaml").read_text())
    assert g["name"] == mod.GRID and g["budget_cap"] == mod.CAP
    assert set(g["arms"]) == {mod.spec(k)[0] for k in NEW}
    assert g["suites"] == ["frames"] and g["n_tasks"] == 824 and g["seeds"] == [0]


# --------------------------------------------------------------------------- synthetic stores


def _store(path, rows, calls, scorer="sh", graph="v1"):
    """A scored store in the three parquet tables `store_units` reads.

    A row is (run_id, task, arm_id, budget_cap, answer_correct, n_asks)."""
    path.mkdir(parents=True)
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
    for rid, task, arm, cap, value, asks in rows:
        stop = "budget" if asks >= cap else "policy_stop"
        con.execute(
            "INSERT INTO r VALUES (?, ?, 0, NULL, ?, 'frames', ?, ?, ?, ?, 'ok', FALSE, "
            "FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, 'none', 'test', TRUE, TRUE)",
            [rid, task, arm, cap, asks, float(asks), stop],
        )
        con.execute("INSERT INTO y VALUES (?, ?)", [rid, value])
    con.execute(
        "CREATE TABLE s AS SELECT run_id, 'answer_correct' AS metric_name, value, "
        "? AS scorer_hash, ? AS graph_version FROM y",
        [scorer, graph],
    )
    con.execute("CREATE TABLE c (run_id VARCHAR, actor VARCHAR, model VARCHAR)")
    for rid, actor, model in calls:
        con.execute("INSERT INTO c VALUES (?, ?, ?)", [rid, actor, model])
    for t, name in (("r", "runs"), ("s", "scores"), ("c", "calls")):
        con.execute(f"COPY {t} TO '{path / name}.parquet' (FORMAT parquet)")


# answer_correct per task, and n_asks per task, for every label the reading touches
Y = {
    "teacher": [1.0, 1.0, 0.0],
    "base": [0.0, 1.0, 0.0],
    "s1": [1.0, 0.0, 1.0],
    "s2": [0.0, 0.0, 1.0],
    "drafter": [0.0, 0.0, 0.0],
    "par2_8b": [0.0, 0.0, 0.0],
    "par2_120b": [1.0, 0.0, 0.0],
    "par2_opus": [1.0, 1.0, 1.0],
    "opus": [0.0, 1.0, 1.0],
}
ASKS = {"s1": [2, 8, 4], "s2": [6, 2, 4]}


def _write_store(root, label, arm, model, caps, *, scorer="sh", reverse=False, graph="v1"):
    rows, calls, ok = [], [], {}
    for cap in caps:
        ok[cap] = {}
        for i, (t, y) in enumerate(zip(TASKS, Y[label], strict=True)):
            rid = f"{label}-{cap}-{t}"
            rows.append((rid, t, arm, cap, y, ASKS.get(label, [3, 3, 3])[i]))
            calls += [(rid, "inquirer", model)] if model else []
            calls.append((rid, "drafter", FROZEN))
            ok[cap][f"{t}|0"] = rid
    if reverse:  # row order must not decide which seed-2 value meets which seed-1 value
        rows, calls = rows[::-1], calls[::-1]
    _store(root / f"scores_parquet_{label}", rows, calls, scorer=scorer, graph=graph)
    return ok


def _record(mod, tmp_path):
    """What frames_recipe.py writes, produced by frames_recipe.read itself on synthetic stores."""
    fr = mod.fr
    pop: dict = {}
    for label, (arm, model, caps) in fr.ARMS.items():
        ok = _write_store(tmp_path, label, arm, model, caps, reverse=label == "s2")
        pop[label] = {cap: {"ok_units": ok[cap]} for cap in caps}
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
        for cap in fr.CAPS
    }
    reading = fr.read(pop, EST, 1, pub, ["sh"])
    ok_row = {"ok": True}
    rec = {
        "code_version": fr.CODE,
        "metric": fr.METRIC,
        "lock": {
            "estimator": EST,
            "levels": {"x": ok_row},
            "contrasts": {"x": ok_row},
            "budget_response": {"x": ok_row},
        },
        "population": {
            lab: {str(c): {"run_id_sha16": fr._sha16(v["ok_units"].values())} for c, v in p.items()}
            for lab, p in pop.items()
        },
        "population_failures": [],
        "across_cells": {"published_unit_set_sha16": fr._sha16(f"{t}|0" for t in TASKS)},
        "reading": reading,
    }
    return json.loads(json.dumps(rec, default=str)), pop


def _new_stores(root, *, scorer_of=lambda label: "sh"):
    pop = {}
    for label in NEW:
        arm = "par2_rag" if label.startswith("par2") else "inquirer_prompted"
        model = {"par2_8b": BASE8B, "par2_120b": FROZEN}.get(label, OPUS)
        ok = _write_store(root, label, arm, model, (8,), scorer=scorer_of(label))
        pop[label] = {8: {"ok_units": ok[8], "problems": []}}
    return pop


# --------------------------------------------------------------------------- the lock


def test_the_lock_reproduces_frames_recipes_own_record(fast, tmp_path):
    rec, _ = _record(fast, tmp_path)
    lk, cells = fast.lock(EST, rec, workers=1)
    assert lk["failures"] == []
    assert set(lk["levels"]) == {"recipe", "s1", "s2", "base", "teacher"}
    assert set(lk["contrasts"]) == {"recipe-base/cap8", "recipe-teacher/cap8"}
    assert all(v["ok"] for sec in ("levels", "contrasts") for v in lk[sec].values())
    assert set(cells) == {"s1", "s2", "base", "teacher"}


@pytest.mark.parametrize(
    "path",
    [
        ("levels", "recipe", "8", "y"),
        ("levels", "base", "8", "ci_lo"),
        ("levels", "teacher", "8", "mean_n_asks"),
        ("levels", "recipe", "8", "mean_retrieval_calls"),
        ("contrasts", "recipe-base/cap8", "delta"),
        ("contrasts", "recipe-teacher/cap8", "ci_hi"),
        ("contrasts", "recipe-base/cap8", "ci_50k", 2, 1),
        ("contrasts", "recipe-teacher/cap8", "ci_1k", 0),
    ],
)
def test_the_lock_refuses_a_perturbed_published_value(fast, tmp_path, path):
    rec, _ = _record(fast, tmp_path)
    node = rec["reading"]
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] += 1e-8  # ten times the tolerance
    lk, _ = fast.lock(EST, rec, workers=1)
    assert lk["failures"], f"a perturbation at {path} was not refused"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("levels", "recipe", "8", "n"), 4),
        (("levels", "base", "8", "n_clusters"), 2),
        (("contrasts", "recipe-base/cap8", "decided"), None),  # flipped below
    ],
)
def test_the_lock_refuses_a_changed_count_or_verdict(fast, tmp_path, path, value):
    rec, _ = _record(fast, tmp_path)
    node = rec["reading"]
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = (not node[path[-1]]) if value is None else value
    lk, _ = fast.lock(EST, rec, workers=1)
    assert lk["failures"]


def test_the_lock_tolerance_is_1e_9_not_zero(fast, tmp_path):
    rec, _ = _record(fast, tmp_path)
    rec["reading"]["levels"]["recipe"]["8"]["y"] += 1e-11
    lk, _ = fast.lock(EST, rec, workers=1)
    assert lk["failures"] == []
    assert fast.LOCK_TOL == 1e-9


def test_the_lock_refuses_a_record_missing_a_field(fast, tmp_path):
    rec, _ = _record(fast, tmp_path)
    del rec["reading"]["contrasts"]["recipe-teacher/cap8"]["ci_50k"]
    lk, _ = fast.lock(EST, rec, workers=1)
    assert lk["failures"]


def test_the_lock_refuses_a_record_whose_own_lock_failed(fast, tmp_path):
    rec, _ = _record(fast, tmp_path)
    rec["lock"]["contrasts"]["x"] = {"ok": False}
    lk, _ = fast.lock(EST, rec, workers=1)
    assert any("record" in f for f in lk["failures"])


def test_the_lock_refuses_a_record_with_an_empty_lock(fast, tmp_path):
    rec, _ = _record(fast, tmp_path)
    rec["lock"] = {"levels": {}, "contrasts": {}, "budget_response": {}}
    lk, _ = fast.lock(EST, rec, workers=1)
    assert lk["failures"]


def test_the_lock_refuses_a_store_that_is_not_the_recorded_one(fast, tmp_path):
    rec, _ = _record(fast, tmp_path)
    rec["reading"]["store_check"]["base/cap8"]["run_id_sha16"] = "0" * 16
    lk, _ = fast.lock(EST, rec, workers=1)
    assert any("base" in f for f in lk["failures"])


def test_the_lock_refuses_a_recipe_unit_set_the_record_did_not_certify(fast, tmp_path):
    rec, _ = _record(fast, tmp_path)
    rec["across_cells"]["published_unit_set_sha16"] = "f" * 16
    lk, _ = fast.lock(EST, rec, workers=1)
    assert lk["failures"]


def test_the_lock_refuses_a_missing_store(fast, tmp_path):
    rec, _ = _record(fast, tmp_path)
    (tmp_path / "scores_parquet_teacher" / "runs.parquet").unlink()
    with pytest.raises(SystemExit, match="scores_parquet_teacher"):
        fast.lock(EST, rec, workers=1)


# --------------------------------------------------------------------------- the recipe's pooling


def test_the_recipe_is_pooled_within_the_task(fast, tmp_path):
    rec, _ = _record(fast, tmp_path)  # s2's store is written in REVERSED row order
    _, cells = fast.lock(EST, rec, workers=1)
    got = fast.levels_and_contrasts(cells, EST, 1, ("base",))
    pooled = got["cells"]["recipe"]
    # seeds averaged per task: s1 = (1, 0, 1), s2 = (0, 0, 1) -> (0.5, 0, 1)
    assert {k: v["y"] for k, v in pooled.items()} == {
        "frames_0000|0": 0.5,
        "frames_0001|0": 0.0,
        "frames_0002|0": 1.0,
    }
    # asks s1 = (2, 8, 4), s2 = (6, 2, 4) -> 4, 5, 4 per task; ceiling hit only s1 on task 1
    assert {k: v["asks"] for k, v in pooled.items()} == {
        "frames_0000|0": 4.0,
        "frames_0001|0": 5.0,
        "frames_0002|0": 4.0,
    }
    lv = got["levels"]["recipe"]
    assert (lv["n"], lv["n_clusters"]) == (3, 3)  # tasks, not 6 seed-runs
    assert lv["ceiling_hit"] == 0.5
    ct = got["contrasts"]["recipe-base/cap8"]
    assert ct["n"] == 3 and ct["delta"] == pytest.approx((0.5 - 0 + 0 - 1 + 1 - 0) / 3)


# --------------------------------------------------------------------------- population refusals
#
# Synthetic run roots, one directory per run with manifest.json, status.json, calls.jsonl in the
# shapes a real FRAMES run writes (as tests/test_seed_identity_frames_recipe.py builds them).


def _manifest(task, *, arm, model, grid, cap=8, base="u1", code=CODE_FULL):
    pins = {
        "drafter": {"model_id": FROZEN, "base_url_sha": base, "adapter_sha": None},
        "answerer": {"model_id": FROZEN, "base_url_sha": base, "adapter_sha": None},
        "inquirer": {"model_id": model, "base_url_sha": base, "adapter_sha": None},
    }
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
        "dirty": False,
        "grid_name": grid,
        "grid_sha256": "g",
        "canary_nonces_loaded": 17844,
        "concurrency": 6,
        "pins": pins,
    }
    sem = semantic_hash(m)
    m["semantic_hash"] = sem
    m["run_id"] = h("runid", sem, "none", "None")[:32]
    return m


def _write_run(root, m, *, calls=None):
    d = root / m["run_id"]
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(m))
    (d / "status.json").write_text(json.dumps({"status": "ok", "started_at": 0.0}))
    if calls is None:
        calls = [("inquirer", m["pins"]["inquirer"]["model_id"]), ("drafter", FROZEN)]
    (d / "calls.jsonl").write_text(
        "".join(json.dumps({"actor": a, "model": x}) + "\n" for a, x in calls)
    )


def _root(tmp_path, label, tasks=TASKS, **kw):
    arm, model, _, grid = kw.pop("spec", None) or _spec(label)
    root = tmp_path / f"runs_{label}"
    root.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        m = _manifest(
            t,
            arm=kw.get("arm", arm),
            model=kw.get("model", model),
            grid=kw.get("grid", grid or "frames_recipe_cap8"),
            base=kw.get("base", "u1"),
        )
        _write_run(root, m, calls=kw.get("calls"))
    return root


def _spec(label):
    return {
        "par2_8b": ("par2_rag", BASE8B, (8,), "frames_structured_cap8"),
        "par2_120b": ("par2_rag", FROZEN, (8,), "frames_structured_cap8"),
        "par2_opus": ("par2_rag", OPUS, (8,), "frames_structured_cap8"),
        "opus": ("inquirer_prompted", OPUS, (8,), "frames_structured_cap8"),
        "teacher": ("inquirer_prompted", FROZEN, (8,), None),
        "base": ("inquirer_prompted", BASE8B, (8,), None),
        "s1": ("inquirer_trained", S1, (8,), None),
        "s2": ("inquirer_trained", S2, (8,), None),
    }[label]


def _all_roots(tmp_path, **override):
    """Every root the reading needs, clean; `override[label]` = kwargs for that one root."""
    for label in (*NEW, "teacher", "base", "s1", "s2"):
        _root(tmp_path, label, **override.get(label, {}))


def _recipe_units():
    return {f"{t}|0" for t in TASKS}


def _gate(mod, tmp_path, record=None):
    """The gate over the roots on disk. By default against a record that certified exactly the
    existing roots' cap-8 run ids now on disk (what frames_recipe.json holds for the real roots)."""
    pop = mod.populations()
    if record is None:
        record = {
            "population": {
                lab: {"8": {"run_id_sha16": mod.fr._sha16(pop[lab][8]["ok_units"].values())}}
                for lab in ("teacher", "base", "s1", "s2")
            }
        }
    return mod.population_gate(pop, _recipe_units(), record)


def test_a_clean_set_of_roots_raises_no_refusal(fast, tmp_path):
    _all_roots(tmp_path)
    gate = _gate(fast, tmp_path)
    assert gate["failures"] == []
    assert gate["across"]["base_url_sha"] == ["u1"]
    for label in NEW:
        assert gate["population"][label]["8"]["n_ok_distinct_units"] == len(TASKS)


@pytest.mark.parametrize("label", NEW)
def test_a_new_root_with_the_wrong_arm_id_is_refused(fast, tmp_path, label):
    wrong = "inquirer_prompted" if label.startswith("par2") else "par2_rag"
    _all_roots(tmp_path, **{label: {"arm": wrong}})
    fails = _gate(fast, tmp_path)["failures"]
    assert any(label in f and "arms" in f for f in fails), fails


@pytest.mark.parametrize(
    ("label", "wrong"),
    [("par2_8b", FROZEN), ("par2_120b", BASE8B), ("par2_opus", FROZEN), ("opus", BASE8B)],
)
def test_a_new_root_with_the_wrong_inquirer_model_is_refused(fast, tmp_path, label, wrong):
    _all_roots(tmp_path, **{label: {"calls": [("inquirer", wrong), ("drafter", FROZEN)]}})
    fails = _gate(fast, tmp_path)["failures"]
    assert any(label in f and "wrong_inquirer_model=3" in f for f in fails), fails


def test_a_new_root_on_the_recipe_grid_is_refused(fast, tmp_path):
    _all_roots(tmp_path, par2_opus={"grid": "frames_recipe_cap8"})
    fails = _gate(fast, tmp_path)["failures"]
    assert any("par2_opus" in f and "grids" in f for f in fails), fails


def test_frames_recipes_default_grid_check_is_unchanged(fast, tmp_path):
    fr = fast.fr
    ok = _root(tmp_path, "s1")
    assert fr.population("s1", root=ok)[8]["problems"] == []
    moved = _root(tmp_path / "x", "s1", grid="frames_structured_cap8")
    assert any(p.startswith("grids") for p in fr.population("s1", root=moved)[8]["problems"])


def test_a_new_root_missing_a_unit_is_refused(fast, tmp_path):
    _all_roots(tmp_path, par2_120b={"tasks": TASKS[:2]})
    fails = _gate(fast, tmp_path)["failures"]
    assert any("par2_120b" in f and "ok=2" in f for f in fails), fails
    assert any("unit set" in f for f in fails), fails


def test_a_new_root_on_another_unit_set_is_refused(fast, tmp_path):
    _all_roots(tmp_path, opus={"tasks": [*TASKS[:2], "frames_0999"]})
    fails = _gate(fast, tmp_path)["failures"]
    assert not any(f.startswith("opus/") for f in fails)  # the cell itself is well-formed
    assert any("unit set" in f for f in fails), fails


def test_a_second_base_url_sha_is_refused(fast, tmp_path):
    _all_roots(tmp_path, par2_8b={"base": "u2"})
    fails = _gate(fast, tmp_path)["failures"]
    assert any("base_url_sha" in f for f in fails), fails


def test_an_existing_root_that_moved_since_the_record_is_refused(fast, tmp_path):
    _all_roots(tmp_path)
    rec = {"population": {"base": {"8": {"run_id_sha16": "0" * 16}}}}
    fails = _gate(fast, tmp_path, rec)["failures"]
    assert any("base" in f and "record" in f for f in fails), fails


@pytest.mark.parametrize("label", NEW)
def test_a_missing_root_is_refused_with_a_clear_message(fast, tmp_path, label):
    _all_roots(tmp_path)
    root = tmp_path / f"runs_{label}"
    for d in root.iterdir():
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    root.rmdir()
    with pytest.raises(SystemExit, match=f"runs_{label}"):
        fast.populations()


# --------------------------------------------------------------------------- the reading


def _reading_setup(mod, tmp_path, scorer_of=lambda label: "sh"):
    _, pop_old = _record(mod, tmp_path)
    pop = {
        lab: {8: {"ok_units": pop_old[lab][8]["ok_units"]}}
        for lab in ("s1", "s2", "base", "teacher")
    }
    pop.update(_new_stores(tmp_path, scorer_of=scorer_of))
    return pop


def test_the_reading_runs_end_to_end(fast, tmp_path):
    pop = _reading_setup(fast, tmp_path)
    rd = fast.read(pop, EST, 1)
    assert set(rd["levels"]) == {"recipe", "s1", "s2", "base", "teacher", *NEW}
    for v in rd["levels"].values():
        assert v["n"] == 3 and "mean_n_asks" in v and "mean_retrieval_calls" in v
    want = {f"recipe-{c}/cap8" for c in ("base", "teacher", *NEW)}
    assert set(rd["contrasts"]) == want
    for name in want:
        c = rd["contrasts"][name]
        assert c["n"] == 3 and len(c["ci_50k"]) == 3 and isinstance(c["decided"], bool)
        assert "ci_1k" in c and "zero_verdict_flips_at_1k" in c
    # recipe per task (0.5, 0, 1); par2_opus is all 1 -> delta (-0.5 - 1 + 0) / 3
    assert rd["contrasts"]["recipe-par2_opus/cap8"]["delta"] == pytest.approx(-1.5 / 3)
    assert rd["scorer_hash"] == ["sh"] and rd["graph_version"] == ["v1"]
    assert set(rd["run_id_sha256"]) == {"recipe", "s1", "s2", "base", "teacher", *NEW}
    assert all(len(v) == 64 for v in rd["run_id_sha256"].values())


def test_the_reading_refuses_a_second_scorer_hash(fast, tmp_path):
    pop = _reading_setup(fast, tmp_path, lambda label: "other" if label == "opus" else "sh")
    with pytest.raises(SystemExit, match="scorer_hash"):
        fast.read(pop, EST, 1)


def test_the_reading_refuses_a_store_that_is_not_the_population(fast, tmp_path):
    pop = _reading_setup(fast, tmp_path)
    pop["par2_8b"][8]["ok_units"] = copy.deepcopy(pop["par2_8b"][8]["ok_units"])
    pop["par2_8b"][8]["ok_units"]["frames_0002|0"] = "a-run-the-store-does-not-hold"
    with pytest.raises(SystemExit):
        fast.read(pop, EST, 1)


def test_the_reading_refuses_a_missing_store(fast, tmp_path):
    pop = _reading_setup(fast, tmp_path)
    for f in (tmp_path / "scores_parquet_par2_opus").iterdir():
        f.unlink()
    (tmp_path / "scores_parquet_par2_opus").rmdir()
    with pytest.raises(SystemExit, match="scores_parquet_par2_opus"):
        fast.read(pop, EST, 1)
