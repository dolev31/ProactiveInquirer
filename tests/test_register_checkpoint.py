"""Registering a checkpoint must be a COMMAND, not a careful hand-edit of two JSON files.

WHAT IT REPLACES. `conf/checkpoints.json`'s first row was typed by hand, and so was the $0
row in `scripts/price_tables/2026-09.json` that has to accompany it -- `PriceTable.rates` is
strict, so a served name with no price row raises `LLMConfigError` and the sweep refuses to
start. Two files, five shas, one of them the sha256 of a 350 MB file on another machine.
There are eleven more rung-1 runs and six rung-2 runs coming; done by hand seventeen more
times, a wrong sha is not a possibility but a schedule.

THE TWO RULES THE SCRIPT ENFORCES, both from `src/pinq_adapters/llm/checkpoints.py`:

  * `adapter_sha` is inside `ModelPin.key` -> `model_pin_hash` -> `semantic_hash` ->
    `run_id`. Re-pointing an existing name at different weights renames every finished run
    that used it, so a changed sha under an existing name is REFUSED (`--force` is the
    escape, and it is loud).
  * a partial row is worse than none. All five fields or nothing.

AND THE ONE IT RECORDS RATHER THAN ENFORCES. `training_manifest_sha` should be the sha of
`rung{1,2}.manifest.json`, which only exists once the trainer finishes. A checkpoint served
from a still-running job has no manifest, and waiting is not free -- so the trainer's own
`trainer_state.json` stands in and the entry is marked `"provisional": true`. That flag is
the difference between "this row is final" and "this row must be rewritten when the job
ends", which the first hand-written row could only say in a prose comment.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "hpc" / "register_checkpoint.py"

PRICE_ROW = {"cached_input": 0.0, "input": 0.0, "output": 0.0, "reasoning": None}


def _mod():
    if not SCRIPT.exists():
        pytest.fail(f"{SCRIPT} does not exist")
    spec = importlib.util.spec_from_file_location("register_checkpoint", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def world(tmp_path):
    """A registry, a price table, an adapter and a finished run, all fake but shaped right."""
    reg = tmp_path / "checkpoints.json"
    reg.write_text(json.dumps({"_schema": "notes", "_why_it_is_empty": "notes"}, indent=2) + "\n")

    price = tmp_path / "price.json"
    price.write_text(
        json.dumps(
            {
                "_why_local_deployments": "GPU-hour amortisation",
                "fallback": {"cached_input": 0.0, "input": 0.0, "output": 0.0, "reasoning": None},
                "models": {
                    "aws/gpt-oss-120b": {
                        "cached_input": None,
                        "input": 0.15,
                        "output": 0.75,
                        "reasoning": None,
                    }
                },
                "unit": "usd_per_1m_tokens",
                "version": "2026-09",
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )

    adapter = tmp_path / "rung1-8b-headline"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"the-headline-weights")
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 32}))

    run = tmp_path / "run"
    run.mkdir()
    (run / "rung1.manifest.json").write_text(json.dumps({"config": {"seed": 0}}, indent=2))
    (run / "trainer_state.json").write_text(json.dumps({"global_step": 2600}))

    ds = tmp_path / "sft.manifest.json"
    ds.write_text(json.dumps({"train_id_set_hash": "c8" + "b" * 62, "split": "train"}, indent=2))

    return {
        "registry": reg,
        "price": price,
        "adapter": adapter,
        "run": run,
        "dataset": ds,
        # Neither exists on disk. That is the point: every test in this file except the ones in
        # the "finished-run history" section below must be UNAFFECTED by the history check, so
        # the default here has to be a path `_history_offenders` finds absent and skips, exactly
        # like a stranger's checkout with no rented box and no calls.parquet.
        "calls_parquet": tmp_path / "calls.parquet",
        "runs_dir": tmp_path / "runs",
        "weights_sha": hashlib.sha256(b"the-headline-weights").hexdigest(),
        "manifest_sha": hashlib.sha256((run / "rung1.manifest.json").read_bytes()).hexdigest(),
        "state_sha": hashlib.sha256((run / "trainer_state.json").read_bytes()).hexdigest(),
    }


def _argv(w, **over):
    a = [
        "--name",
        "qwen3-8b-sft-headline",
        "--adapter",
        str(w["adapter"]),
        "--base",
        "Qwen/Qwen3-8B",
        "--rung",
        "1",
        "--run-dir",
        str(w["run"]),
        "--dataset-manifest",
        str(w["dataset"]),
        "--registry",
        str(w["registry"]),
        "--price-table",
        str(w["price"]),
        "--calls-parquet",
        str(w["calls_parquet"]),
        "--runs-dir",
        str(w["runs_dir"]),
    ]
    for k, v in over.items():
        flag = "--" + k.replace("_", "-")
        if v is None:
            i = a.index(flag)
            del a[i : i + 2]
        elif v is True:
            a.append(flag)
        else:
            i = a.index(flag) if flag in a else None
            if i is None:
                a += [flag, str(v)]
            else:
                a[i + 1] = str(v)
    return a


def _entry(w, name="qwen3-8b-sft-headline"):
    return json.loads(w["registry"].read_text())[name]


# --------------------------------------------------------------------------- a fresh row


def test_a_fresh_registration_writes_all_five_fields(world):
    rc = _mod()
    assert rc.main(_argv(world)) == 0
    e = _entry(world)
    assert e["adapter_sha"] == world["weights_sha"]
    assert e["training_manifest_sha"] == world["manifest_sha"]
    assert e["base_model"] == "Qwen/Qwen3-8B"
    assert e["rung"] == 1
    assert e["dataset_sha"] == "c8" + "b" * 62
    assert "provisional" not in e, "a real rung1.manifest.json was there; nothing is provisional"


def test_the_dataset_sha_is_the_manifests_train_id_set_hash(world):
    """Not the sha of the manifest file: `train_id_set_hash` is the id set the adapter was
    fitted on, and that is what makes two runs comparable."""
    rc = _mod()
    rc.main(_argv(world))
    assert (
        _entry(world)["dataset_sha"]
        == json.loads(world["dataset"].read_text())["train_id_set_hash"]
    )


def test_the_written_registry_loads_through_the_real_loader(world, monkeypatch):
    """The only check that matters: `checkpoints.registry()` must accept what was written,
    with all five REQUIRED_FIELDS, or the row is provenance in appearance only."""
    rc = _mod()
    rc.main(_argv(world))
    from pinq_adapters.llm import checkpoints

    monkeypatch.setenv("PI_CHECKPOINTS", str(world["registry"]))
    checkpoints.reset_cache()
    got = checkpoints.registry()
    assert set(got) == {"qwen3-8b-sft-headline"}
    assert not [f for f in checkpoints.REQUIRED_FIELDS if f not in got["qwen3-8b-sft-headline"]]


def test_the_price_row_is_added_and_is_zero(world):
    """A served name with no price row raises LLMConfigError before the sweep starts."""
    rc = _mod()
    rc.main(_argv(world))
    models = json.loads(world["price"].read_text())["models"]
    assert models["qwen3-8b-sft-headline"] == PRICE_ROW


def test_existing_rates_are_never_touched(world):
    """`_why_2026_09` forbids editing a table a finished sweep was priced under. Adding a name
    that has never run re-prices nothing; changing a rate re-prices everything."""
    rc = _mod()
    before = json.loads(world["price"].read_text())["models"]["aws/gpt-oss-120b"]
    rc.main(_argv(world))
    after = json.loads(world["price"].read_text())["models"]["aws/gpt-oss-120b"]
    assert before == after


# --------------------------------------------------------------------------- idempotence


def test_running_it_twice_changes_no_byte(world):
    """It will be run again -- by a `for` loop over seventeen runs, by a retry after an ssh
    drop. A second run must be a no-op, not a second row or a reformatted file."""
    rc = _mod()
    rc.main(_argv(world))
    reg1, price1 = world["registry"].read_text(), world["price"].read_text()
    assert rc.main(_argv(world)) == 0
    assert world["registry"].read_text() == reg1
    assert world["price"].read_text() == price1


def test_the_price_row_is_added_once(world):
    rc = _mod()
    rc.main(_argv(world))
    rc.main(_argv(world))
    raw = world["price"].read_text()
    assert raw.count('"qwen3-8b-sft-headline"') == 1


def test_the_json_formatting_of_both_files_is_preserved(world):
    """A registration whose diff is "the whole file" cannot be reviewed. checkpoints.json is
    indent=2 in insertion order; the price table is indent=1 with sorted keys."""
    rc = _mod()
    rc.main(_argv(world))
    reg_raw = world["registry"].read_text()
    assert reg_raw == json.dumps(json.loads(reg_raw), indent=2) + "\n"
    price_raw = world["price"].read_text()
    assert price_raw == json.dumps(json.loads(price_raw), indent=1, sort_keys=True) + "\n"


def test_the_documentation_keys_of_the_registry_survive(world):
    rc = _mod()
    rc.main(_argv(world))
    raw = json.loads(world["registry"].read_text())
    assert raw["_schema"] == "notes" and raw["_why_it_is_empty"] == "notes"


# --------------------------------------------------------------------------- the refusal


def test_a_changed_adapter_sha_under_an_existing_name_is_refused(world, capsys):
    """THE REASON THIS SCRIPT EXISTS. adapter_sha is inside run_id: re-pointing a name at new
    weights renames every finished run that used it, retroactively, in a repo whose tables
    cite run ids."""
    rc = _mod()
    rc.main(_argv(world))
    (world["adapter"] / "adapter_model.safetensors").write_bytes(b"RETRAINED")
    assert rc.main(_argv(world)) != 0
    err = capsys.readouterr().err
    assert "run_id" in err or "renames" in err
    assert _entry(world)["adapter_sha"] == world["weights_sha"], "the file must be unchanged"


def test_force_is_the_escape_and_it_overwrites(world):
    rc = _mod()
    rc.main(_argv(world))
    (world["adapter"] / "adapter_model.safetensors").write_bytes(b"RETRAINED")
    assert rc.main(_argv(world, force=True)) == 0
    assert _entry(world)["adapter_sha"] == hashlib.sha256(b"RETRAINED").hexdigest()


# ------------------------------------------------- the SAME refusal, arriving the OTHER way
#
# `test_a_changed_adapter_sha_under_an_existing_name_is_refused` above catches a RE-POINT: the
# registry FILE already names a sha and this call would change it. It cannot catch the mirror
# case -- a name registered for the FIRST time (no row in the registry file at all) that
# ALREADY has finished runs on disk, recorded when the name was unregistered and so carrying
# `adapter_sha: None`. `old = registry.get(name)` is None either way, so nothing above fires,
# and `granite33-8b-sft-headline-s0/s1/s2` are exactly this: registered (commit 60bd644,
# 2026-09-18) months into their own run history, which
# `tests/test_checkpoint_registry.py::test_no_registry_row_RENAMES_A_RUN_THAT_HAS_ALREADY_
# FINISHED` measures at 5,611 finished runs (3,600 of them `split=test`, all of them
# `adapter_sha: None` on disk) that a recomputed `run_id` would no longer match. This section
# closes that gap by reading the same evidence the guard test reads -- `calls.parquet` and
# `runs/<run_id>/manifest.json` -- before any file is written, not after a paper artifact has
# already cited the runs.


def _write_calls_parquet(path, rows: list[tuple[str, str]]) -> None:
    """`rows` is `(run_id, model)`. A real parquet, written by the same engine that reads it."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE calls (run_id VARCHAR, model VARCHAR)")
    con.executemany("INSERT INTO calls VALUES (?, ?)", rows)
    con.execute("COPY calls TO '" + str(path).replace("'", "''") + "' (FORMAT PARQUET)")


def _write_finished_run(runs_dir: Path, run_id: str, model_id: str, adapter_sha, role="inquirer"):
    d = runs_dir / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "pins": {
                    role: {
                        "role": role,
                        "model_id": model_id,
                        "adapter_sha": adapter_sha,
                    }
                },
            }
        )
    )


def test_a_first_time_registration_that_disagrees_with_finished_run_history_is_refused(
    world, capsys
):
    """THE GAP `60bd644` FELL THROUGH. No row in the registry FILE (`old is None`), so the
    existing re-point check is silent -- but 5 finished runs already used this bare name and
    recorded `adapter_sha: None`, exactly the shape of the real granite33 incident. Registering
    it now, for the first time, with a real sha would still rename every one of them."""
    name = "qwen3-8b-sft-headline"
    _write_calls_parquet(world["calls_parquet"], [("r" + str(i) * 31, name) for i in range(5)])
    for i in range(5):
        _write_finished_run(world["runs_dir"], "r" + str(i) * 31, name, None)

    rc = _mod()
    assert rc.main(_argv(world)) != 0
    err = capsys.readouterr().err
    assert "run_id" in err or "renames" in err
    assert name not in json.loads(world["registry"].read_text()), "nothing is written on a refusal"


def test_force_overrides_the_history_refusal_too(world):
    """The same escape as the re-point case, for the same reason: a human said these runs may
    be renamed."""
    name = "qwen3-8b-sft-headline"
    _write_calls_parquet(world["calls_parquet"], [("r" + "1" * 31, name)])
    _write_finished_run(world["runs_dir"], "r" + "1" * 31, name, None)

    rc = _mod()
    assert rc.main(_argv(world, force=True)) == 0
    assert _entry(world)["adapter_sha"] == world["weights_sha"]


def test_a_first_time_registration_with_AGREEING_history_is_not_refused(world):
    """The check is about disagreement, not about history existing at all -- a checkpoint
    correctly registered before its first use also has `calls.parquet` rows once it has run,
    and re-registering the same weights (idempotence, see above) must stay a no-op."""
    name = "qwen3-8b-sft-headline"
    _write_calls_parquet(world["calls_parquet"], [("r" + "2" * 31, name)])
    _write_finished_run(world["runs_dir"], "r" + "2" * 31, name, world["weights_sha"])

    rc = _mod()
    assert rc.main(_argv(world)) == 0
    assert _entry(world)["adapter_sha"] == world["weights_sha"]


def test_the_history_check_is_a_noop_when_calls_parquet_is_absent(world):
    """A stranger's checkout, and every test above this section: no rented box, no calls, no
    history to disagree with. This is the default `world` already exercises; stated once,
    explicitly, so a future change to the default cannot silently stop meaning this."""
    assert not world["calls_parquet"].exists()
    rc = _mod()
    assert rc.main(_argv(world)) == 0


def test_a_dataset_manifest_without_a_train_id_set_hash_is_refused(world, tmp_path):
    bad = tmp_path / "bad.manifest.json"
    bad.write_text(json.dumps({"split": "train"}))
    rc = _mod()
    assert rc.main(_argv(world, dataset_manifest=bad)) != 0
    assert "qwen3-8b-sft-headline" not in json.loads(world["registry"].read_text())


def test_a_rung_outside_1_2_3_is_refused(world):
    rc = _mod()
    assert rc.main(_argv(world, rung=4)) != 0


def test_a_non_hex_adapter_sha_is_refused(world):
    rc = _mod()
    argv = _argv(world, adapter=None)
    argv += ["--adapter-sha", "not-a-sha"]
    assert rc.main(argv) != 0


def test_nothing_is_written_when_any_input_is_bad(world):
    """Two files must move together or not at all: a registry row with no price row is a
    sweep that refuses to start, and a price row with no registry row is an unprovenanced
    deployment."""
    rc = _mod()
    reg1, price1 = world["registry"].read_text(), world["price"].read_text()
    rc.main(_argv(world, rung=9))
    assert world["registry"].read_text() == reg1
    assert world["price"].read_text() == price1


# --------------------------------------------------------------------------- provisional


def test_the_manifest_sha_is_provisional_when_only_trainer_state_exists(world):
    """The case the first hand-written row was in: job 766502 had not finished, so the row
    named the trainer's own state file instead -- and said so only in a prose comment."""
    rc = _mod()
    (world["run"] / "rung1.manifest.json").unlink()
    assert rc.main(_argv(world)) == 0
    e = _entry(world)
    assert e["training_manifest_sha"] == world["state_sha"]
    assert e["provisional"] is True
    assert e["provisional_source"] == "trainer_state.json"


def test_the_provisional_flag_is_CLEARED_when_the_manifest_finally_appears(world):
    """The second half of the same story. Re-registering the SAME weights against a finished
    run must replace the sha and remove the flag -- and must not be mistaken for the refused
    case above, because the ADAPTER did not change."""
    rc = _mod()
    (world["run"] / "rung1.manifest.json").unlink()
    rc.main(_argv(world))
    assert _entry(world)["provisional"] is True

    (world["run"] / "rung1.manifest.json").write_text(json.dumps({"config": {"seed": 0}}, indent=2))
    assert rc.main(_argv(world)) == 0
    e = _entry(world)
    assert e["training_manifest_sha"] == world["manifest_sha"]
    assert "provisional" not in e and "provisional_source" not in e


def test_a_rung_2_run_is_read_from_rung2_manifest_json(world, tmp_path):
    """WHY THIS GREW AN `--init-from`, AND WHY THAT IS NOT A WEAKENING. Its subject is
    `training_manifest_sha`, and every assertion about that is unchanged. What changed is that
    a rung-2 row registered with no lineage is now REFUSED -- see the `--init-from` section --
    so the fixture, not the claim, had to become legal. It was registering a row that stamps a
    strict subset of the ids its adapter saw, which is the defect, not the baseline."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    _register_sft(rc, world, ids)
    (world["run"] / "rung1.manifest.json").unlink()
    (world["run"] / "rung2.manifest.json").write_text(json.dumps({"dpo": True}))
    argv = _argv(
        world,
        rung=2,
        name="qwen3-8b-dpo-pooled-control",
        dataset_manifest=ids["pairs_manifest"],
    ) + ["--init-from", "qwen3-8b-sft-headline"]
    assert rc.main(argv) == 0
    e = _entry(world, "qwen3-8b-dpo-pooled-control")
    assert e["rung"] == 2
    assert (
        e["training_manifest_sha"]
        == hashlib.sha256((world["run"] / "rung2.manifest.json").read_bytes()).hexdigest()
    )
    assert "provisional" not in e


def test_a_run_dir_with_neither_manifest_nor_trainer_state_is_refused(world, tmp_path):
    empty = tmp_path / "empty-run"
    empty.mkdir()
    rc = _mod()
    assert rc.main(_argv(world, run_dir=empty)) != 0


# ------------------------------------------------------- the adapter is on another machine


def test_the_sha_may_be_supplied_by_hex_when_the_adapter_is_not_local(world):
    """The adapter is 350 MB on GPFS and this runs on the Mac. `ssh box sha256sum ...` is the
    measurement; this flag is how it gets in without moving the file."""
    rc = _mod()
    remote = "a" * 64
    argv = _argv(world, adapter=None) + ["--adapter-sha", remote]
    assert rc.main(argv) == 0
    assert _entry(world)["adapter_sha"] == remote


def test_a_supplied_sha_that_disagrees_with_a_local_adapter_is_refused(world, capsys):
    """If both are given, they are a cross-check, not a preference. A disagreement means the
    file that was hashed on the cluster is not the file that is here."""
    rc = _mod()
    argv = _argv(world) + ["--adapter-sha", "b" * 64]
    assert rc.main(argv) != 0
    assert "b" * 64 in capsys.readouterr().err


def test_a_supplied_sha_that_agrees_with_a_local_adapter_is_accepted(world):
    rc = _mod()
    argv = _argv(world) + ["--adapter-sha", world["weights_sha"].upper()]
    assert rc.main(argv) == 0
    assert _entry(world)["adapter_sha"] == world["weights_sha"]


def test_neither_an_adapter_nor_a_sha_is_refused(world):
    rc = _mod()
    assert rc.main(_argv(world, adapter=None)) != 0


def test_an_adapter_dir_without_weights_is_refused(world, tmp_path):
    d = tmp_path / "no-weights"
    d.mkdir()
    rc = _mod()
    assert rc.main(_argv(world, adapter=d)) != 0


# --------------------------------------------------------------------------- the price guard


def test_a_name_that_already_has_a_NONZERO_price_is_refused(world, capsys):
    """Serving local weights under a name the table prices as paid would bill a sweep that
    cost nothing -- or, read the other way, would make a paid model look free."""
    price = json.loads(world["price"].read_text())
    price["models"]["qwen3-8b-sft-headline"] = {
        "cached_input": 0.2,
        "input": 0.2,
        "output": 0.2,
        "reasoning": None,
    }
    world["price"].write_text(json.dumps(price, indent=1, sort_keys=True) + "\n")
    rc = _mod()
    assert rc.main(_argv(world)) != 0
    assert "price" in capsys.readouterr().err.lower()


# ------------------------------------------------------------------------------- --backfill
#
# `train_id_set_hash` became a REQUIRED field after six rows already existed. It is the same
# value `dataset_sha` has always held, under the name every other artifact uses for it -- the
# export manifest, the id-set sidecar, `RunManifest`, the `runs.parquet` column and the
# contamination check that compares them. `--backfill` is the one-time migration, and it is a
# command rather than a hand-edit for the same reason the rest of this script is.


def _rows_without_the_new_field(w, n=2):
    """A registry in the state the file was in before the field existed."""
    reg = json.loads(w["registry"].read_text())
    for i in range(n):
        reg[f"qwen3-8b-sft-{i}"] = {
            "adapter_sha": str(i) * 64,
            "training_manifest_sha": "b" * 64,
            "base_model": "Qwen/Qwen3-8B",
            "rung": 1,
            "dataset_sha": "c" * 64,
        }
    w["registry"].write_text(json.dumps(reg, indent=2) + "\n")
    return reg


def test_backfill_copies_dataset_sha_into_every_row_that_lacks_the_hash(world):
    _rows_without_the_new_field(world)
    rc = _mod()
    assert rc.main(["--backfill", "--registry", str(world["registry"])]) == 0
    reg = json.loads(world["registry"].read_text())
    for name in ("qwen3-8b-sft-0", "qwen3-8b-sft-1"):
        assert reg[name]["train_id_set_hash"] == reg[name]["dataset_sha"] == "c" * 64


def test_the_backfilled_registry_loads_through_the_real_loader(world, monkeypatch):
    """The only check that matters: `registry()` now REFUSES a row without the field, so a
    migration that does not satisfy the loader has migrated nothing."""
    _rows_without_the_new_field(world)
    rc = _mod()
    rc.main(["--backfill", "--registry", str(world["registry"])])

    from pinq_adapters.llm import checkpoints

    monkeypatch.setenv("PI_CHECKPOINTS", str(world["registry"]))
    checkpoints.reset_cache()
    assert set(checkpoints.registry()) == {"qwen3-8b-sft-0", "qwen3-8b-sft-1"}


def test_backfill_REFUSES_when_a_row_already_disagrees(world, capsys):
    """The one thing a migration must never do: decide which of two disagreeing values is the
    real one. They are one quantity under two names, so a difference means a row is wrong and a
    human has to say which half."""
    reg = _rows_without_the_new_field(world)
    reg["qwen3-8b-sft-0"]["train_id_set_hash"] = "d" * 64
    world["registry"].write_text(json.dumps(reg, indent=2) + "\n")
    before = world["registry"].read_text()

    rc = _mod()
    assert rc.main(["--backfill", "--registry", str(world["registry"])]) != 0
    assert "dataset_sha" in capsys.readouterr().err
    assert world["registry"].read_text() == before, "nothing is written on a refusal"


def test_backfill_twice_changes_no_byte(world):
    _rows_without_the_new_field(world)
    rc = _mod()
    rc.main(["--backfill", "--registry", str(world["registry"])])
    once = world["registry"].read_text()
    assert rc.main(["--backfill", "--registry", str(world["registry"])]) == 0
    assert world["registry"].read_text() == once


def test_backfill_leaves_the_documentation_keys_alone(world):
    _rows_without_the_new_field(world)
    rc = _mod()
    rc.main(["--backfill", "--registry", str(world["registry"])])
    raw = json.loads(world["registry"].read_text())
    assert raw["_schema"] == "notes" and "train_id_set_hash" not in raw["_schema"]


def test_a_fresh_registration_writes_the_hash_beside_the_dataset_sha(world):
    """The migration is for the six rows that predate the field; every row written from now on
    carries it because the script writes both names from one measurement."""
    rc = _mod()
    assert rc.main(_argv(world)) == 0
    e = _entry(world)
    assert e["train_id_set_hash"] == e["dataset_sha"] == "c8" + "b" * 62


# ------------------------------------------------------------------------------ --init-from
#
# LINEAGE. A rung-1 row's `dataset_sha` and `train_id_set_hash` are one quantity because a
# rung-1 adapter was fitted on exactly one file. A rung-2 adapter was not: DPO is INITIALISED
# from an SFT adapter (`DPOConfig.adapter`/`reference="adapter"`, which `validate()` refuses to
# leave empty), so the weights saw the SFT file's tasks AND the pairs file's. A row that names
# only its own export names a strict subset of what the adapter was fitted on.
#
# MEASURED 2026-09-15 on the shipped headline export: the SFT set holds 2,525 ids, the pairs
# set 1,027, and SFT - pairs is 1,499. So a rung-2 row registered the 0cbcaa7 way stamps runs
# with 1,027 of the 2,526 tasks its weights saw, and the id-set canary is blind to the other
# 1,499 -- what a leak through a served NAME could reach unseen, which is the case layer 4 of
# the firewall exists for. (The canary also covers a task exported and since re-bucketed, which
# the split predicate structurally cannot see; that is a statement about what the predicates
# can express, not a count -- 0 of the 2,525 SFT ids are test or dev today.)
#
# `train_id_set_hash` becomes the UNION and `dataset_sha` keeps its meaning (this row's own
# export). `init_from` names the row the union came from, so the reader can re-derive it; and
# because an init row's own `train_id_set_hash` is already a union when IT is rung >= 2, a
# stacked arm recurses for free.


def _ids_world(
    tmp_path,
    sft=("musique/a", "musique/b", "musique/c"),
    pairs=("musique/c", "musique/d"),
):
    """An export directory shaped like `data/rl/headline/`: two manifests, two sidecars."""
    from pinq_train.split import id_set_hash

    d = tmp_path / "rl"
    d.mkdir(exist_ok=True)
    out = {}
    for tag, ids in (("sft", sft), ("pairs", pairs)):
        h = id_set_hash(ids)
        (d / f"train_ids.{h[:16]}.txt").write_text("".join(f"{i}\n" for i in sorted(set(ids))))
        (d / f"{tag}.manifest.json").write_text(
            json.dumps({"train_id_set_hash": h, "split": "train"}, indent=2)
        )
        out[tag] = h
        out[f"{tag}_manifest"] = d / f"{tag}.manifest.json"
        out[f"{tag}_ids"] = set(ids)
    out["dir"] = d
    return out


def _register_sft(rc, world, ids, name="qwen3-8b-sft-headline"):
    """The rung-1 ancestor, registered the way it always was."""
    assert rc.main(_argv(world, name=name, dataset_manifest=ids["sft_manifest"])) == 0
    return name


def test_a_rung2_row_with_init_from_carries_the_UNION_and_the_sidecar_proves_it(world, tmp_path):
    """THE FIX. The row's `train_id_set_hash` is the union of what it was fitted on, the union
    sidecar beside the manifest is its preimage, and `dataset_sha` still names THIS export."""
    from pinq_train.split import id_set_hash

    rc = _mod()
    ids = _ids_world(tmp_path)
    _register_sft(rc, world, ids)
    (world["run"] / "rung2.manifest.json").write_text(json.dumps({"dpo": True}))

    argv = _argv(
        world,
        name="qwen3-8b-dpo-headline-control",
        rung=2,
        dataset_manifest=ids["pairs_manifest"],
        adapter_sha="9" * 64,
        adapter=None,
    ) + ["--init-from", "qwen3-8b-sft-headline"]
    assert rc.main(argv) == 0

    e = _entry(world, "qwen3-8b-dpo-headline-control")
    union = ids["sft_ids"] | ids["pairs_ids"]
    assert len(union) == 4, "the fixture must have sets that actually differ, or this is vacuous"
    assert e["dataset_sha"] == ids["pairs"], "dataset_sha still names THIS export, unchanged"
    assert e["train_id_set_hash"] == id_set_hash(union)
    assert e["train_id_set_hash"] != e["dataset_sha"], "the union is not the pairs set"
    assert e["init_from"] == "qwen3-8b-sft-headline"

    sidecar = ids["dir"] / f"train_ids.{e['train_id_set_hash'][:16]}.txt"
    got = [ln for ln in sidecar.read_text().splitlines() if ln]
    assert set(got) == union
    assert got == sorted(union), "sorted, like every other sidecar"
    assert id_set_hash(got) == e["train_id_set_hash"], "the preimage must reproduce the hash"


def test_a_rung2_registration_WITHOUT_init_from_is_refused(world, tmp_path, capsys):
    """A rung-2 row that names only its own export is the gap this flag closes, so it may not
    be written by accident."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    (world["run"] / "rung2.manifest.json").write_text(json.dumps({"dpo": True}))
    argv = _argv(world, name="qwen3-8b-dpo-x", rung=2, dataset_manifest=ids["pairs_manifest"])
    assert rc.main(argv) != 0
    assert "--init-from" in capsys.readouterr().err
    assert "qwen3-8b-dpo-x" not in json.loads(world["registry"].read_text())


def test_a_rung1_registration_WITH_init_from_is_refused(world, tmp_path, capsys):
    """A rung-1 adapter is fitted on one file and initialised from a raw base. An `init_from`
    here would put a union on a row whose two names are genuinely one quantity."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    _register_sft(rc, world, ids, name="qwen3-8b-sft-headline")
    argv = _argv(world, name="qwen3-8b-sft-other", dataset_manifest=ids["pairs_manifest"]) + [
        "--init-from",
        "qwen3-8b-sft-headline",
    ]
    assert rc.main(argv) != 0
    assert "rung 1" in capsys.readouterr().err.lower()


def test_an_init_row_on_a_DIFFERENT_base_model_is_refused(world, tmp_path, capsys):
    """A LoRA is applied to a base. An init row on another base names weights this adapter
    cannot have started from, so the union would be arithmetic over unrelated training."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    _register_sft(rc, world, ids)
    (world["run"] / "rung2.manifest.json").write_text(json.dumps({"dpo": True}))
    argv = _argv(
        world,
        name="qwen3-4b-dpo-x",
        rung=2,
        base="Qwen/Qwen3-4B",
        dataset_manifest=ids["pairs_manifest"],
        adapter_sha="8" * 64,
        adapter=None,
    ) + ["--init-from", "qwen3-8b-sft-headline"]
    assert rc.main(argv) != 0
    assert "base_model" in capsys.readouterr().err


def test_an_init_row_of_a_HIGHER_rung_is_refused(world, tmp_path, capsys):
    """Lineage does not run backwards: a DPO pass cannot have started from a GRPO. This is all
    the rung has to guarantee -- it used to demand a STRICTLY LOWER rung, which would have made
    the planned DPO-on-DPO arm unregisterable; termination comes from the chain walk instead."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    _register_sft(rc, world, ids)
    (world["run"] / "rung2.manifest.json").write_text(json.dumps({"dpo": True}))
    assert (
        rc.main(
            _argv(
                world,
                name="qwen3-8b-grpo-a",
                rung=3,
                dataset_manifest=ids["pairs_manifest"],
                adapter_sha="7" * 64,
                adapter=None,
            )
            + ["--init-from", "qwen3-8b-sft-headline"]
        )
        == 0
    )
    argv = _argv(
        world,
        name="qwen3-8b-dpo-b",
        rung=2,
        dataset_manifest=ids["pairs_manifest"],
        adapter_sha="6" * 64,
        adapter=None,
    ) + ["--init-from", "qwen3-8b-grpo-a"]
    assert rc.main(argv) != 0
    assert "rung" in capsys.readouterr().err


def test_a_registration_that_would_CLOSE_A_CYCLE_is_refused(world, tmp_path, capsys):
    """Re-registering a name as the child of its own descendant. The rung comparison cannot see
    this now that it is not strict, so the chain walk is what refuses it."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    _register_sft(rc, world, ids)
    (world["run"] / "rung2.manifest.json").write_text(json.dumps({"dpo": True}))
    for name, sha in (("qwen3-8b-dpo-a", "7" * 64), ("qwen3-8b-dpo-b", "6" * 64)):
        init = "qwen3-8b-sft-headline" if name.endswith("-a") else "qwen3-8b-dpo-a"
        assert (
            rc.main(
                _argv(
                    world,
                    name=name,
                    rung=2,
                    dataset_manifest=ids["pairs_manifest"],
                    adapter_sha=sha,
                    adapter=None,
                )
                + ["--init-from", init]
            )
            == 0
        )
    # now point -a at -b, which already descends from -a.
    argv = _argv(
        world,
        name="qwen3-8b-dpo-a",
        rung=2,
        dataset_manifest=ids["pairs_manifest"],
        adapter_sha="7" * 64,
        adapter=None,
    ) + ["--init-from", "qwen3-8b-dpo-b"]
    assert rc.main(argv) != 0
    assert "cycle" in capsys.readouterr().err


def test_a_row_that_would_init_from_ITSELF_is_refused(world, tmp_path, capsys):
    rc = _mod()
    ids = _ids_world(tmp_path)
    _register_sft(rc, world, ids)
    (world["run"] / "rung2.manifest.json").write_text(json.dumps({"dpo": True}))
    argv = _argv(
        world,
        name="qwen3-8b-dpo-a",
        rung=2,
        dataset_manifest=ids["pairs_manifest"],
        adapter_sha="7" * 64,
        adapter=None,
    ) + ["--init-from", "qwen3-8b-dpo-a"]
    assert rc.main(argv) != 0
    assert "cycle" in capsys.readouterr().err


def test_an_unregistered_init_from_is_refused(world, tmp_path, capsys):
    rc = _mod()
    ids = _ids_world(tmp_path)
    (world["run"] / "rung2.manifest.json").write_text(json.dumps({"dpo": True}))
    argv = _argv(world, name="qwen3-8b-dpo-x", rung=2, dataset_manifest=ids["pairs_manifest"]) + [
        "--init-from",
        "nobody",
    ]
    assert rc.main(argv) != 0
    assert "nobody" in capsys.readouterr().err


def test_a_STACKED_rung2_row_unions_all_three_sets(world, tmp_path):
    """THE PLANNED DPO-ON-DPO ARM: a second DPO pass initialised from
    `qwen3-8b-dpo-headline-control`, which is rung 2 initialised from rung 2.

    And the reason the recursion is free: the init row's `train_id_set_hash` is ALREADY a union
    when that row is itself a lineage row, so one union per registration suffices however deep
    the chain goes."""
    from pinq_train.split import id_set_hash

    rc = _mod()
    ids = _ids_world(tmp_path)
    third = {"musique/e", "musique/f"}
    h3 = id_set_hash(third)
    (ids["dir"] / f"train_ids.{h3[:16]}.txt").write_text("".join(f"{i}\n" for i in sorted(third)))
    (ids["dir"] / "notdone.manifest.json").write_text(
        json.dumps({"train_id_set_hash": h3, "split": "train"}, indent=2)
    )
    _register_sft(rc, world, ids)
    (world["run"] / "rung2.manifest.json").write_text(json.dumps({"dpo": True}))
    assert (
        rc.main(
            _argv(
                world,
                name="qwen3-8b-dpo-headline-control",
                rung=2,
                dataset_manifest=ids["pairs_manifest"],
                adapter_sha="5" * 64,
                adapter=None,
            )
            + ["--init-from", "qwen3-8b-sft-headline"]
        )
        == 0
    )
    assert (
        rc.main(
            _argv(
                world,
                name="qwen3-8b-dpo-headline-notdone",
                rung=2,
                dataset_manifest=ids["dir"] / "notdone.manifest.json",
                adapter_sha="4" * 64,
                adapter=None,
            )
            + ["--init-from", "qwen3-8b-dpo-headline-control"]
        )
        == 0
    )
    e = _entry(world, "qwen3-8b-dpo-headline-notdone")
    all_three = ids["sft_ids"] | ids["pairs_ids"] | third
    assert len(all_three) == 6
    assert e["train_id_set_hash"] == id_set_hash(all_three)
    assert e["dataset_sha"] == h3
    sidecar = ids["dir"] / f"train_ids.{e['train_id_set_hash'][:16]}.txt"
    assert set(sidecar.read_text().split()) == all_three


def test_the_union_sidecar_is_idempotent_but_a_DIFFERENT_file_is_refused(world, tmp_path, capsys):
    """A sidecar is named by the hash it is the preimage of, so a file of that name holding
    other ids is a proof of the wrong thing. Identical content is a re-run and is fine."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    _register_sft(rc, world, ids)
    (world["run"] / "rung2.manifest.json").write_text(json.dumps({"dpo": True}))
    argv = _argv(
        world,
        name="qwen3-8b-dpo-headline-control",
        rung=2,
        dataset_manifest=ids["pairs_manifest"],
        adapter_sha="3" * 64,
        adapter=None,
    ) + ["--init-from", "qwen3-8b-sft-headline"]
    assert rc.main(argv) == 0
    e = _entry(world, "qwen3-8b-dpo-headline-control")
    sidecar = ids["dir"] / f"train_ids.{e['train_id_set_hash'][:16]}.txt"
    before = sidecar.read_text()
    assert rc.main(argv) == 0, "a re-run over an identical sidecar is a no-op"
    assert sidecar.read_text() == before

    sidecar.write_text("musique/tampered\n")
    assert rc.main(argv) != 0
    assert "tampered" in sidecar.read_text(), "a refusal writes nothing"
    assert str(sidecar.name) in capsys.readouterr().err


def test_an_UNCHANGED_union_sidecar_is_not_REWRITTEN(world, tmp_path, capsys):
    """Identical bytes must mean no write at all, not a write that happens to be harmless.

    WHY THE MTIME AND NOT THE BYTES. The test beside this one already pins the content, and
    the content was never at risk: `union_sidecar` refuses a differing file before it writes,
    so a re-run could only ever rewrite the same bytes. What it did do was rewrite them --
    `dest.write_text(body)` ran unconditionally -- and the cost of that is not in this repo's
    bytes but in its FRESHNESS SIGNALS. Sidecars live under `data/rl/`, which is shared, and
    an unconditional rewrite bumps the mtime of a training artefact every time anyone
    re-registers or re-runs a migration. `find -newermt` over `data/rl/` then reports a file
    that nothing changed, and this project has twice killed a healthy job on exactly that
    class of reading. A migration that is a no-op must look like one from the outside too.

    THE MTIME IS SET, NOT SAMPLED, so this cannot pass by accident. Reading the clock twice
    and comparing would be a test of the filesystem's timestamp granularity: two writes inside
    one tick produce equal mtimes and the assertion would hold with the bug still in place.
    Stamping a distinctive past value and demanding it survive is decidable either way.
    """
    rc = _mod()
    ids = _ids_world(tmp_path)
    _register_sft(rc, world, ids)
    (world["run"] / "rung2.manifest.json").write_text(json.dumps({"dpo": True}))
    argv = _argv(
        world,
        name="qwen3-8b-dpo-headline-control",
        rung=2,
        dataset_manifest=ids["pairs_manifest"],
        adapter_sha="3" * 64,
        adapter=None,
    ) + ["--init-from", "qwen3-8b-sft-headline"]

    assert rc.main(argv) == 0
    e = _entry(world, "qwen3-8b-dpo-headline-control")
    sidecar = ids["dir"] / f"train_ids.{e['train_id_set_hash'][:16]}.txt"
    union = ids["sft_ids"] | ids["pairs_ids"]
    body = sidecar.read_text()
    assert set(body.split()) == union

    # A value no clock will hand back: 2001-09-09T01:46:40Z, to the nanosecond.
    STAMP = 1_000_000_000_123_456_789
    os.utime(sidecar, ns=(STAMP, STAMP))
    capsys.readouterr()

    assert rc.main(argv) == 0, "a re-run over an identical sidecar is a no-op, not a failure"
    assert sidecar.stat().st_mtime_ns == STAMP, (
        "the sidecar was REWRITTEN with its own bytes. Identical content must skip the write "
        "entirely: these files sit in the shared data/rl/ tree and a bumped mtime is a false "
        "freshness signal to every `find -newermt` that looks there."
    )
    assert sidecar.read_text() == body, "and the content is still the union"
    assert "unchanged" in capsys.readouterr().out, (
        "the operator has to be able to see WHICH case ran -- written or unchanged"
    )

    # A MISSING file is still written: the skip is about identical bytes, not about never
    # writing. Without this the fix could be "never write" and the test would not notice.
    sidecar.unlink()
    assert rc.main(argv) == 0
    assert sidecar.is_file() and sidecar.read_text() == body
    assert "written" in capsys.readouterr().out

    # And a DIFFERING file is still a refusal, unchanged by any of the above.
    sidecar.write_text("musique/tampered\n")
    assert rc.main(argv) != 0
    assert "tampered" in sidecar.read_text(), "a refusal writes nothing"


def test_a_lineage_registry_loads_through_the_real_loader(world, tmp_path, monkeypatch):
    """The only check that matters: `registry()` must ACCEPT a rung-2 row whose two names
    differ, now that they are two quantities rather than one."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    _register_sft(rc, world, ids)
    (world["run"] / "rung2.manifest.json").write_text(json.dumps({"dpo": True}))
    rc.main(
        _argv(
            world,
            name="qwen3-8b-dpo-headline-control",
            rung=2,
            dataset_manifest=ids["pairs_manifest"],
            adapter_sha="2" * 64,
            adapter=None,
        )
        + ["--init-from", "qwen3-8b-sft-headline"]
    )
    from pinq_adapters.llm import checkpoints

    monkeypatch.setenv("PI_CHECKPOINTS", str(world["registry"]))
    checkpoints.reset_cache()
    got = checkpoints.registry()
    assert set(got) == {"qwen3-8b-sft-headline", "qwen3-8b-dpo-headline-control"}
    row = got["qwen3-8b-dpo-headline-control"]
    assert row["train_id_set_hash"] != row["dataset_sha"]
    assert (
        checkpoints.train_id_set_hash_of("qwen3-8b-dpo-headline-control")
        == (row["train_id_set_hash"])
    ), "the stamp is the UNION, so the reader resolves the set the weights actually saw"


# --------------------------------------------------------------- --backfill and lineage rows


def test_backfill_REFUSES_a_rung2_row_that_names_no_init_from(world, capsys):
    """The migration may not quietly copy `dataset_sha` onto a rung-2 row: that writes the
    1,027-of-2,526 stamp this whole section exists to stop."""
    reg = json.loads(world["registry"].read_text())
    reg["qwen3-8b-dpo-headline-control"] = {
        "adapter_sha": "1" * 64,
        "training_manifest_sha": "b" * 64,
        "base_model": "Qwen/Qwen3-8B",
        "rung": 2,
        "dataset_sha": "c" * 64,
    }
    world["registry"].write_text(json.dumps(reg, indent=2) + "\n")
    before = world["registry"].read_text()

    rc = _mod()
    assert rc.main(["--backfill", "--registry", str(world["registry"])]) != 0
    err = capsys.readouterr().err
    assert "--init-from" in err and "qwen3-8b-dpo-headline-control" in err
    assert world["registry"].read_text() == before, "nothing is written on a refusal"


def test_backfill_init_from_performs_the_union_for_ONE_existing_row(world, tmp_path):
    """The per-row migration, for a rung-2 row that was registered before `--init-from` existed.
    It is the same union by the same function; only the row already exists."""
    from pinq_train.split import id_set_hash

    rc = _mod()
    ids = _ids_world(tmp_path)
    reg = json.loads(world["registry"].read_text())
    reg["qwen3-8b-sft-headline"] = {
        "adapter_sha": "a" * 64,
        "training_manifest_sha": "b" * 64,
        "base_model": "Qwen/Qwen3-8B",
        "rung": 1,
        "dataset_sha": ids["sft"],
    }
    reg["qwen3-8b-dpo-headline-control"] = {
        "adapter_sha": "1" * 64,
        "training_manifest_sha": "b" * 64,
        "base_model": "Qwen/Qwen3-8B",
        "rung": 2,
        "dataset_sha": ids["pairs"],
    }
    world["registry"].write_text(json.dumps(reg, indent=2) + "\n")

    assert (
        rc.main(
            [
                "--backfill-init-from",
                "qwen3-8b-sft-headline",
                "--name",
                "qwen3-8b-dpo-headline-control",
                "--dataset-manifest",
                str(ids["pairs_manifest"]),
                "--registry",
                str(world["registry"]),
            ]
        )
        == 0
    )
    e = _entry(world, "qwen3-8b-dpo-headline-control")
    union = ids["sft_ids"] | ids["pairs_ids"]
    assert e["dataset_sha"] == ids["pairs"], "its own export is unchanged"
    assert e["train_id_set_hash"] == id_set_hash(union)
    assert e["init_from"] == "qwen3-8b-sft-headline"
    sidecar = ids["dir"] / f"train_ids.{e['train_id_set_hash'][:16]}.txt"
    assert set(sidecar.read_text().split()) == union
    # And the plain migration now passes over it rather than refusing.
    assert rc.main(["--backfill", "--registry", str(world["registry"])]) == 0
    assert _entry(world, "qwen3-8b-dpo-headline-control")["train_id_set_hash"] == id_set_hash(union)


def test_backfill_init_from_REFUSES_a_manifest_that_is_not_the_rows_export(world, tmp_path, capsys):
    """The cross-check that makes `--dataset-manifest` more than a path: the manifest handed in
    must be the export the row's `dataset_sha` already names, or the union is over another
    dataset and the row would claim a lineage it does not have."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    reg = json.loads(world["registry"].read_text())
    reg["qwen3-8b-sft-headline"] = {
        "adapter_sha": "a" * 64,
        "training_manifest_sha": "b" * 64,
        "base_model": "Qwen/Qwen3-8B",
        "rung": 1,
        "dataset_sha": ids["sft"],
    }
    reg["qwen3-8b-dpo-headline-control"] = {
        "adapter_sha": "1" * 64,
        "training_manifest_sha": "b" * 64,
        "base_model": "Qwen/Qwen3-8B",
        "rung": 2,
        "dataset_sha": ids["pairs"],
    }
    world["registry"].write_text(json.dumps(reg, indent=2) + "\n")
    before = world["registry"].read_text()
    assert (
        rc.main(
            [
                "--backfill-init-from",
                "qwen3-8b-sft-headline",
                "--name",
                "qwen3-8b-dpo-headline-control",
                "--dataset-manifest",
                str(ids["sft_manifest"]),  # the WRONG export
                "--registry",
                str(world["registry"]),
            ]
        )
        != 0
    )
    assert "dataset_sha" in capsys.readouterr().err
    assert world["registry"].read_text() == before


def test_a_sidecar_whose_contents_do_not_reproduce_its_name_is_refused(world, tmp_path, capsys):
    """The union is only as good as the sets it is taken over. A sidecar is named by the hash
    it is the preimage of; one that does not reproduce it proves nothing."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    _register_sft(rc, world, ids)
    (ids["dir"] / f"train_ids.{ids['sft'][:16]}.txt").write_text("musique/z\n")
    (world["run"] / "rung2.manifest.json").write_text(json.dumps({"dpo": True}))
    argv = _argv(
        world,
        name="qwen3-8b-dpo-x",
        rung=2,
        dataset_manifest=ids["pairs_manifest"],
        adapter_sha="9" * 64,
        adapter=None,
    ) + ["--init-from", "qwen3-8b-sft-headline"]
    assert rc.main(argv) != 0
    err = capsys.readouterr().err
    assert "train_id_set_hash" in err or "reproduce" in err


# ------------------------------------------------------------------- --init-from base
#
# THE ABLATION THAT HAS NO ANCESTOR. `pi train rung2 --reference base` (the dpo_from_base arm)
# trains a fresh LoRA on the UNTRAINED base: no rung-1 adapter is loaded, merged or recorded,
# and `rung2.manifest.json` says so (`reference_policy.reference == "base"`, `adapter` null).
# Such a row is rung 2 and has no init checkpoint, so "rung >= 2 requires a registered
# --init-from" would make a legitimate arm unregisterable -- and the honest answer for it is
# the one rung 1 already gives: the weights saw exactly this export, so the two id-set names
# are one quantity again. The literal `base` records THAT rather than leaving the field off,
# because "no ancestor" and "nobody said" are different facts and only one of them is checked.
#
# AND IT IS CROSS-CHECKED AGAINST THE TRAINER'S OWN MANIFEST where one exists, because this is
# the single field on the row that a reader cannot re-derive from the weights.


def _rung2_manifest(run: Path, reference: str) -> None:
    run.joinpath("rung2.manifest.json").write_text(
        json.dumps(
            {
                "config": {"reference": reference},
                "reference_policy": {"reference": reference},
                "pairs": {"n_pairs": 17_430},
            }
        )
    )


def test_a_rung2_row_on_the_BARE_BASE_takes_init_from_base_and_unions_nothing(world, tmp_path):
    """No ancestor, so no union: `train_id_set_hash` is this export's own set, exactly as at
    rung 1, and the row says `base` rather than saying nothing."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    (world["run"] / "rung1.manifest.json").unlink()
    _rung2_manifest(world["run"], "base")
    argv = _argv(
        world,
        name="qwen3-8b-dpo-from-base",
        rung=2,
        dataset_manifest=ids["pairs_manifest"],
        adapter_sha="a" * 64,
        adapter=None,
    ) + ["--init-from", "base"]
    assert rc.main(argv) == 0
    e = _entry(world, "qwen3-8b-dpo-from-base")
    assert e["init_from"] == "base"
    assert e["train_id_set_hash"] == e["dataset_sha"] == ids["pairs"]
    union = ids["sft_ids"] | ids["pairs_ids"]
    from pinq_train.split import id_set_hash

    assert not (ids["dir"] / f"train_ids.{id_set_hash(union)[:16]}.txt").exists(), (
        "nothing was initialised from, so no union sidecar may be written"
    )


def test_init_from_base_is_REFUSED_when_the_trainers_manifest_says_an_adapter(
    world, tmp_path, capsys
):
    """The cross-check. `init_from` is the one field on the row nobody can re-derive from the
    weights, and a run that loaded rung 1's adapter saw rung 1's ids whatever the row says."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    (world["run"] / "rung1.manifest.json").unlink()
    _rung2_manifest(world["run"], "adapter")
    argv = _argv(
        world,
        name="qwen3-8b-dpo-x",
        rung=2,
        dataset_manifest=ids["pairs_manifest"],
        adapter_sha="a" * 64,
        adapter=None,
    ) + ["--init-from", "base"]
    assert rc.main(argv) != 0
    err = capsys.readouterr().err
    assert "reference" in err and "adapter" in err
    assert "qwen3-8b-dpo-x" not in json.loads(world["registry"].read_text())


def test_init_from_A_ROW_is_REFUSED_when_the_trainers_manifest_says_the_bare_base(
    world, tmp_path, capsys
):
    """The other direction, and the one that would make a union out of nothing: a row claiming
    a lineage its own training run says it did not have."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    _register_sft(rc, world, ids)
    (world["run"] / "rung1.manifest.json").unlink()
    _rung2_manifest(world["run"], "base")
    argv = _argv(
        world,
        name="qwen3-8b-dpo-y",
        rung=2,
        dataset_manifest=ids["pairs_manifest"],
        adapter_sha="a" * 64,
        adapter=None,
    ) + ["--init-from", "qwen3-8b-sft-headline"]
    assert rc.main(argv) != 0
    err = capsys.readouterr().err
    assert "reference" in err and "base" in err


def test_a_run_dir_with_no_rung2_manifest_registers_without_the_cross_check(world, tmp_path):
    """A checkpoint served out of a still-running job has no manifest (that is what
    `provisional` is for), and refusing it would make the flag unusable exactly when it is
    needed. The cross-check fires when there is something to check against."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    (world["run"] / "rung1.manifest.json").unlink()
    argv = _argv(
        world,
        name="qwen3-8b-dpo-provisional",
        rung=2,
        dataset_manifest=ids["pairs_manifest"],
        adapter_sha="a" * 64,
        adapter=None,
    ) + ["--init-from", "base"]
    assert rc.main(argv) == 0
    assert _entry(world, "qwen3-8b-dpo-provisional")["provisional"] is True


def test_a_row_may_not_be_NAMED_base(world, tmp_path, capsys):
    """`base` is the literal that means "no ancestor" in `init_from`. A deployment wearing that
    name would make the field mean two things at once, and neither reading is checkable."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    assert rc.main(_argv(world, name="base", dataset_manifest=ids["sft_manifest"])) != 0
    assert "base" in capsys.readouterr().err


def test_backfill_init_from_base_records_the_literal_and_unions_nothing(world, tmp_path):
    """The per-row migration for a dpo_from_base row: the field gets filled in, and filling it
    in must not invent a lineage. `--backfill` then passes over the row instead of refusing."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    reg = json.loads(world["registry"].read_text())
    reg["qwen3-8b-dpo-from-base"] = {
        "adapter_sha": "1" * 64,
        "training_manifest_sha": "b" * 64,
        "base_model": "Qwen/Qwen3-8B",
        "rung": 2,
        "dataset_sha": ids["pairs"],
    }
    world["registry"].write_text(json.dumps(reg, indent=2) + "\n")
    assert (
        rc.main(
            [
                "--backfill-init-from",
                "base",
                "--name",
                "qwen3-8b-dpo-from-base",
                "--dataset-manifest",
                str(ids["pairs_manifest"]),
                "--registry",
                str(world["registry"]),
            ]
        )
        == 0
    )
    e = _entry(world, "qwen3-8b-dpo-from-base")
    assert e["init_from"] == "base"
    assert e["train_id_set_hash"] == e["dataset_sha"] == ids["pairs"]
    assert rc.main(["--backfill", "--registry", str(world["registry"])]) == 0


def test_the_union_lands_beside_the_manifest_AS_NAMED_not_beside_a_symlinks_target(world, tmp_path):
    """A worktree reads one export without copying it by symlinking the manifest. Dereferencing
    that link would write the sidecar into the OTHER checkout, where this caller's reader does
    not look and whose files it was told not to touch."""
    rc = _mod()
    ids = _ids_world(tmp_path)
    linked = tmp_path / "linked"
    linked.mkdir()
    for f in ids["dir"].iterdir():
        (linked / f.name).symlink_to(f)

    _register_sft(rc, world, ids, name="qwen3-8b-sft-headline")
    (world["run"] / "rung2.manifest.json").write_text(json.dumps({"dpo": True}))
    argv = _argv(
        world,
        name="qwen3-8b-dpo-headline-control",
        rung=2,
        dataset_manifest=linked / "pairs.manifest.json",
        adapter_sha="a" * 64,
        adapter=None,
    ) + ["--init-from", "qwen3-8b-sft-headline"]
    assert rc.main(argv) == 0

    name = (
        f"train_ids.{_entry(world, 'qwen3-8b-dpo-headline-control')['train_id_set_hash'][:16]}.txt"
    )
    assert (linked / name).is_file() and not (linked / name).is_symlink()
    assert not (ids["dir"] / name).exists(), "the linked-to directory must be untouched"
