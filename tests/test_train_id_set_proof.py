"""Layer 3, made true: the id set a checkpoint saw, written down and read back.

`pinq_train.split`'s docstring has always called the manifest's `train_id_set_hash` layer 3 of
five, "so a later run can prove which ids a checkpoint saw". It could not. A sha256 proves
membership only to someone who HOLDS the set, and the set was never written anywhere: the
exporter hashed a list of ids and dropped the list on the floor. Nothing stamped the hash onto
a run either, and `pi_eval.report.train_split_violations` selected the column and ignored it.

This file covers the three halves that were missing, and `tests/test_train_contamination.py`
covers the fourth (the reader):

  1. the exporter writes `train_ids.<hash16>.txt` beside the manifest -- the SET, so the hash
     can be checked against something rather than believed;
  2. `conf/checkpoints.json` carries the hash on every row, so a served deployment names the
     id set it was fitted on;
  3. `pi_run.manifest.build_manifest` stamps it on the run when the Inquirer's served model id
     is one the registry lists -- and stamps None when it is not, which is the compatibility
     half: an unregistered id must produce exactly the manifest it always did.

RUN IDENTITY IS NOT TOUCHED, and that is the constraint that shapes all of this.
`train_id_set_hash` is NOT in `RunManifest.semantic_hash`, so stamping it renames nothing. The
moment it entered identity, every finished run made against a registered checkpoint would be
renamed retroactively -- the same failure `adapter_sha`'s opt-in registry exists to avoid.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pinq_train.export.dataset import export_sft, write_jsonl
from pinq_train.split import id_set_hash, split_of

REPO = Path(__file__).resolve().parents[1]


def _train_task(start: int = 0, suite: str = "musique") -> str:
    for i in range(start, start + 2000):
        if split_of(suite, f"t{i}") == "train":
            return f"t{i}"
    raise AssertionError("no train bucket found")


T1, T2 = _train_task(), _train_task(start=800)


def _row(*, task: str, turn: int = 1, run: str = "r1", action: str = '{"q": "a"}') -> dict:
    return {
        "suite_id": "musique",
        "task_id": task,
        "run_id": run,
        "turn_idx": turn,
        "state_text": "TASK\nEVIDENCE RETRIEVED SO FAR\n- d1: x\n",
        "action_json": action,
        "value": 1.0,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "code_version": "c" * 40,
        "pins_sha": "p1",
        "done_before": False,
    }


# --------------------------------------------------------------- 1. the exporter's sidecar


def test_the_sidecar_holds_exactly_the_ids_the_manifest_hash_names(tmp_path) -> None:
    """The whole point. A hash is a proof of membership only to someone holding the set, and
    until now the set existed for the length of one function call."""
    items, man = export_sft(
        [_row(task=T1), _row(task=T2, run="r2", action='{"q": "b"}')], margin_threshold=-1.0
    )
    path = write_jsonl(tmp_path / "sft.jsonl", items, man)

    assert man.train_ids_file, "the manifest must name the file, or a reader cannot find it"
    sidecar = path.parent / man.train_ids_file
    raw = sidecar.read_text()
    ids = raw.splitlines()

    assert id_set_hash(ids) == man.train_id_set_hash
    assert ids == sorted(set(ids)) == [f"musique/{t}" for t in sorted({T1, T2})]
    assert raw.endswith("\n"), "newline-terminated, so `wc -l` and `sort -c` are usable"
    assert man.train_ids_file == f"train_ids.{man.train_id_set_hash[:16]}.txt"


def test_the_sidecar_is_order_independent(tmp_path) -> None:
    """`id_set_hash` is a SET hash; the file beside it must be the same set, not an accident of
    iteration order. Two exports of the same rows in opposite order are one dataset."""
    rows = [_row(task=T1), _row(task=T2, run="r2", action='{"q": "b"}')]
    fwd_items, fwd_man = export_sft(rows, margin_threshold=-1.0)
    rev_items, rev_man = export_sft(rows[::-1], margin_threshold=-1.0)
    a = write_jsonl(tmp_path / "a" / "sft.jsonl", fwd_items, fwd_man)
    b = write_jsonl(tmp_path / "b" / "sft.jsonl", rev_items, rev_man)
    fa = next(a.parent.glob("train_ids.*.txt"))
    fb = next(b.parent.glob("train_ids.*.txt"))
    assert fa.name == fb.name
    assert fa.read_text() == fb.read_text()


def test_a_manifest_hash_the_items_do_not_reproduce_is_refused(tmp_path) -> None:
    """The sidecar is derived from the ITEMS being written, so it can disagree with a manifest
    handed in from somewhere else. A file whose name says one id set and whose contents are
    another is worse than no file: the reader would believe it."""
    items, man = export_sft([_row(task=T1)], margin_threshold=-1.0)
    man.train_id_set_hash = "0" * 64
    with pytest.raises(ValueError, match="train_id_set_hash"):
        write_jsonl(tmp_path / "sft.jsonl", items, man)


# --------------------------------------------------------------- 2. the checkpoint registry


def _entry(**over):
    """A registry row. The two names for the id set MIRROR each other unless a test sets them
    apart on purpose -- the loader refuses a row where they disagree, so a fixture that let
    them drift would be testing that refusal by accident everywhere."""
    base = {
        "adapter_sha": "a" * 64,
        "training_manifest_sha": "b" * 64,
        "base_model": "Qwen/Qwen3-8B",
        "rung": 1,
        "dataset_sha": "c" * 64,
    }
    base.update(over)
    base.setdefault("train_id_set_hash", base["dataset_sha"])
    return base


def _registry(monkeypatch, tmp_path, obj):
    from pinq_adapters.llm import checkpoints

    p = tmp_path / "checkpoints.json"
    p.write_text(json.dumps(obj, indent=1))
    monkeypatch.setenv("PI_CHECKPOINTS", str(p))
    checkpoints.reset_cache()
    return checkpoints, p


def test_a_row_without_a_train_id_set_hash_is_refused(monkeypatch, tmp_path) -> None:
    """The field the stamp is read from. A row that omits it is a deployment that cannot say
    which tasks it saw, which is the entire thing this layer is for."""
    bad = _entry()
    del bad["train_id_set_hash"]
    c, _ = _registry(monkeypatch, tmp_path, {"qwen3-8b-sft": bad})
    with pytest.raises(ValueError, match="train_id_set_hash"):
        c.registry()


def test_a_row_whose_two_names_for_one_hash_disagree_is_refused(monkeypatch, tmp_path) -> None:
    """`dataset_sha` and `train_id_set_hash` are the SAME quantity under two names -- the
    export manifest's `train_id_set_hash`. Two fields that must agree will eventually not, and
    a silent disagreement here means the run is stamped with an id set the checkpoint was not
    fitted on."""
    c, _ = _registry(monkeypatch, tmp_path, {"qwen3-8b-sft": _entry(train_id_set_hash="d" * 64)})
    with pytest.raises(ValueError, match="dataset_sha"):
        c.registry()


def test_every_shipped_row_carries_the_hash_and_it_agrees(monkeypatch) -> None:
    """The migration actually ran. Six rows existed before this field did.

    WHY THIS NOW ASKS THE RUNG. It read `train_id_set_hash == dataset_sha` for EVERY row, and
    that belief was true only of a registry with no rung-2 rows in it. A DPO adapter is
    initialised from an SFT adapter, so it was fitted on the SFT file's ids AND its own -- the
    two names are one quantity at rung 1 and two at rung >= 2, where `train_id_set_hash` is the
    union and `init_from` names the row it came from. Asserting equality everywhere would
    forbid the correct rung-2 row, so the test encoded a wrong belief rather than the code
    being wrong. The rung-1 half is UNCHANGED and is the half that has ever had rows.
    """
    from pinq_adapters.llm import checkpoints

    monkeypatch.delenv("PI_CHECKPOINTS", raising=False)
    checkpoints.reset_cache()
    reg = checkpoints.registry()
    assert reg, "the shipped registry has rows; an empty one would make this vacuous"
    for name, entry in reg.items():
        if int(entry["rung"]) == 1:
            assert entry["train_id_set_hash"] == entry["dataset_sha"], name
            assert "init_from" not in entry, name
        else:
            assert entry.get("init_from"), f"{name}: a rung-{entry['rung']} row names no init"
            assert entry["init_from"] in reg, f"{name}: init_from is not a registered row"


def test_train_id_set_hash_of_is_none_for_an_unregistered_id(monkeypatch, tmp_path) -> None:
    c, _ = _registry(monkeypatch, tmp_path, {"qwen3-8b-sft": _entry()})
    assert c.train_id_set_hash_of("openai/aws/gpt-oss-120b") is None
    assert c.train_id_set_hash_of("qwen3-8b-sft") == "c" * 64


# --------------------------------------------------------------- 3. the stamp on the run


def _manifest(monkeypatch, model_id: str):
    pytest.importorskip("tenacity")
    from pi_run.manifest import build_manifest
    from pinq.budget import BudgetLedger
    from pinq_adapters.llm.litellm_client import MeteredClient, PriceTable

    monkeypatch.setenv("PI_MODEL_INQUIRER", model_id)
    pt = PriceTable.load(REPO / "scripts" / "price_tables" / "2026-09.json")
    client = MeteredClient(BudgetLedger(cap=4), price_table=pt, env=dict(os.environ))
    return build_manifest(
        suite_id="musique",
        task_id="t1",
        arm_id="inquirer_trained",
        policy_id="trained",
        seed=0,
        corpus_hash="ch",
        budget_cap=8,
        max_turns=16,
        word_cap=120,
        code_version="v" * 40,
        dirty=False,
        pins={"inquirer": client.pin("inquirer")},
    )


def test_a_run_whose_inquirer_is_a_registered_checkpoint_carries_the_hash(
    monkeypatch, tmp_path
) -> None:
    """The stamp. Without it the reader has nothing to look up, and `train_id_set_hash` stays
    a column that every run fills with NULL."""
    _registry(monkeypatch, tmp_path, {"qwen3-8b-sft": _entry(dataset_sha="e" * 64)})
    m = _manifest(monkeypatch, "qwen3-8b-sft")
    assert m.train_id_set_hash == "e" * 64

    from pi_run.manifest import manifest_to_dict

    assert manifest_to_dict(m)["train_id_set_hash"] == "e" * 64


def test_an_unregistered_inquirer_carries_none(monkeypatch, tmp_path) -> None:
    """Every model id in use today is unregistered, and a None here is what keeps their
    manifests identical to the ones already on disk."""
    _registry(monkeypatch, tmp_path, {"qwen3-8b-sft": _entry()})
    assert _manifest(monkeypatch, "openai/aws/gpt-oss-120b").train_id_set_hash is None


def test_the_stamp_does_not_move_the_run_id(monkeypatch, tmp_path) -> None:
    """THE CONSTRAINT. `train_id_set_hash` is provenance, not identity: it is a property of the
    WEIGHTS the pin already names (`adapter_sha` is in identity and covers it), so putting it in
    `semantic_hash` would add nothing and would rename every run made against a registered
    checkpoint the moment the registry gained a field."""
    _registry(monkeypatch, tmp_path, {"qwen3-8b-sft": _entry(dataset_sha="e" * 64)})
    stamped = _manifest(monkeypatch, "qwen3-8b-sft")
    bare = stamped.__class__(
        **{
            **{f: getattr(stamped, f) for f in stamped.__dataclass_fields__},
            "train_id_set_hash": None,
        }
    )
    assert stamped.train_id_set_hash != bare.train_id_set_hash
    assert stamped.run_id == bare.run_id
    assert stamped.semantic_hash == bare.semantic_hash


def test_an_absent_registry_stamps_none_rather_than_raising(monkeypatch, tmp_path) -> None:
    """A stranger's checkout has no rented box and no adapters; it must still run."""
    from pinq_adapters.llm import checkpoints

    monkeypatch.setenv("PI_CHECKPOINTS", str(tmp_path / "nope.json"))
    checkpoints.reset_cache()
    assert _manifest(monkeypatch, "openai/aws/gpt-oss-120b").train_id_set_hash is None


# ------------------------------------------------- 4. the reader, for a row with an ANCESTOR
#
# THE GAP THIS CLOSES, measured 2026-09-15 on the shipped headline export. `qwen3-8b-dpo-
# headline-control` is initialised from `qwen3-8b-sft-headline`, so its weights were fitted on
# 2,526 tasks: the pairs export's 1,027 and the SFT export's 2,525. Its row named only the
# pairs export, so every run it stamped carried a hash resolving to 1,027 of them and the
# id-set canary was blind to the other 1,499.
#
# WHAT THE SPLIT PREDICATE DOES NOT COVER, which is the whole reason the canary exists. It is a
# statement about `split_of` TODAY; the case it structurally cannot catch is a task that was
# exported into a training file and has since been re-bucketed -- afterwards `split_of` answers
# `test` and is RIGHT, and the weights still saw the task. That is a structural argument, and
# the fixture below is SYNTHETIC on purpose: measured 2026-09-15, 0 of the 2,525 SFT ids and 0
# of the 1,027 pairs ids are test or dev today, so there is no real-data example to point at.
# (An earlier version of this comment claimed 94 from `split_of(suite, task_id)` with no
# template id; `bucket()` hashes `template_id or task_id` and musique carries template ids, so
# that call is the wrong one for its 453 ids. The reader keys on the parquet's `template_id`
# and was never affected.)
#
# The 1,499 SFT-only tasks stay the motivation regardless: they are what a leak through a
# served NAME could reach with nothing to catch it.


def _held_out_task(suite: str = "musique") -> str:
    """A task id `split_of` calls `test` today -- a task that may legitimately be evaluated on,
    and that a training export from before a re-bucket could still contain."""
    for i in range(20_000):
        if split_of(suite, f"h{i}") == "test":
            return f"h{i}"
    raise AssertionError("no test bucket found")


class _Agg:
    """Just enough Agg for `train_split_violations`: it is one SQL call over `runs`."""

    parquet_dir = None

    def __init__(self, rows):
        self._rows = rows

    def sql(self, _q):
        return list(self._rows)


def _lineage_world(tmp_path):
    """An SFT export holding a since-re-bucketed task, a pairs export that does not, and the
    two registry rows, written by `scripts/hpc/register_checkpoint.py` itself."""
    import importlib.util

    from pinq_train.split import id_set_hash

    script = REPO / "scripts" / "hpc" / "register_checkpoint.py"
    spec = importlib.util.spec_from_file_location("register_checkpoint", script)
    rc = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(rc)

    victim = f"musique/{_held_out_task()}"
    sft_ids = sorted({victim, f"musique/{T1}"})
    pairs_ids = sorted({f"musique/{T2}"})
    assert victim not in pairs_ids, "the gap is a task the row's OWN export does not hold"

    d = tmp_path / "rl"
    d.mkdir()
    hashes = {}
    for tag, ids in (("sft", sft_ids), ("pairs", pairs_ids)):
        h = id_set_hash(ids)
        hashes[tag] = h
        (d / f"train_ids.{h[:16]}.txt").write_text("".join(f"{i}\n" for i in ids))
        (d / f"{tag}.manifest.json").write_text(
            json.dumps({"train_id_set_hash": h, "split": "train"})
        )

    run = tmp_path / "run"
    run.mkdir()
    (run / "rung1.manifest.json").write_text(json.dumps({"config": {"seed": 0}}))

    reg = tmp_path / "checkpoints.json"
    reg.write_text(json.dumps({"_schema": "notes"}, indent=2) + "\n")
    price = tmp_path / "price.json"
    price.write_text(json.dumps({"models": {}, "unit": "usd_per_1m_tokens"}, indent=1) + "\n")

    common = [
        "--base",
        "Qwen/Qwen3-8B",
        "--run-dir",
        str(run),
        "--registry",
        str(reg),
        "--price-table",
        str(price),
        # Neither path exists. Without this, register_checkpoint.py's finished-run-history
        # check falls back to the REAL repo's calls.parquet/runs -- and "qwen3-8b-sft" and
        # "qwen3-8b-dpo-control" are real, heavily-used registered names there, so the real
        # history would refuse this fixture's registration. See
        # tests/test_register_checkpoint.py's `world` fixture for the same isolation.
        "--calls-parquet",
        str(tmp_path / "calls.parquet"),
        "--runs-dir",
        str(tmp_path / "runs"),
    ]
    assert (
        rc.main(
            [
                "--name",
                "qwen3-8b-sft",
                "--adapter-sha",
                "a" * 64,
                "--rung",
                "1",
                "--dataset-manifest",
                str(d / "sft.manifest.json"),
                *common,
            ]
        )
        == 0
    )
    assert (
        rc.main(
            [
                "--name",
                "qwen3-8b-dpo-control",
                "--adapter-sha",
                "b" * 64,
                "--rung",
                "2",
                "--dataset-manifest",
                str(d / "pairs.manifest.json"),
                "--init-from",
                "qwen3-8b-sft",
                *common,
            ]
        )
        == 0
    )

    return {
        "ids_dir": d,
        "registry": reg,
        "victim": victim,
        **hashes,
        "union": id_set_hash(set(sft_ids) | set(pairs_ids)),
    }


def _trained_run(victim: str, stamp: str) -> dict:
    suite, task = victim.split("/", 1)
    return {
        "run_id": "r1",
        "arm_id": "inquirer_trained",
        "suite_id": suite,
        "task_id": task,
        "template_id": None,
        "split": "test",
        "train_id_set_hash": stamp,
    }


def test_the_reader_flags_a_task_the_dpo_row_saw_only_through_its_SFT_ANCESTOR(
    monkeypatch, tmp_path
) -> None:
    """THE TEST THE GAP IS DEFINED BY, end to end: registration -> stamp -> reader.

    The task is in the SFT export the DPO adapter was initialised from and in no pairs export;
    `split_of` calls it `test` today and the run is stamped `test`, so the split predicate says
    nothing. Only the id set can answer "did these weights see it?", and it can only answer if
    the stamp resolves to the UNION rather than to the pairs export alone.
    """
    from pi_eval.report import train_split_violations
    from pinq_adapters.llm import checkpoints

    w = _lineage_world(tmp_path)
    monkeypatch.setenv("PI_CHECKPOINTS", str(w["registry"]))
    checkpoints.reset_cache()

    stamp = checkpoints.train_id_set_hash_of("qwen3-8b-dpo-control")
    assert stamp == w["union"], "the row must stamp the union, not its own export"
    assert stamp != w["pairs"]

    out = train_split_violations(
        _Agg([_trained_run(w["victim"], stamp)]), ["r1"], ids_dir=w["ids_dir"]
    )
    assert out, f"{w['victim']} was fitted on and the check reported clean"
    assert "train_id_set" in out[0]["reason"]
    assert "split" not in out[0]["reason"], (
        "the split predicate must NOT be what fired -- this is the case it cannot see"
    )


def test_the_same_run_stamped_with_the_PAIRS_EXPORT_ALONE_is_reported_clean(
    monkeypatch, tmp_path
) -> None:
    """The gap, stated as a permanent property of the reader rather than as a story. A stamp
    resolves to EXACTLY its own set -- that is what "found by the hash, no index in between"
    means -- so a rung-2 row that names only its own export cannot flag anything its ancestor
    contributed. This is what every rung-2 row stamped before `init_from` existed."""
    from pi_eval.report import train_split_violations

    w = _lineage_world(tmp_path)
    out = train_split_violations(
        _Agg([_trained_run(w["victim"], w["pairs"])]), ["r1"], ids_dir=w["ids_dir"]
    )
    assert out == [], "the pairs-only set does not contain it; the reader is right and blind"


def test_the_union_sidecar_the_registration_wrote_is_a_real_preimage(tmp_path) -> None:
    """The reader finds the set BY THE HASH, so the union is only usable if the registration
    left a file of exactly that name holding exactly those ids."""
    from pinq_train.split import id_set_hash

    w = _lineage_world(tmp_path)
    sidecar = w["ids_dir"] / f"train_ids.{w['union'][:16]}.txt"
    ids = [ln for ln in sidecar.read_text().splitlines() if ln]
    assert id_set_hash(ids) == w["union"]
    assert w["victim"] in ids
    assert ids == sorted(set(ids))
