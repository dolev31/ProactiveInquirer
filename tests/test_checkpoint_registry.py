"""A checkpoint must be NAMEABLE from the run that used it.

THE PROBLEM. `ModelPin.adapter_sha` has existed, defaulted to None, and been populated by
nothing. So two runs of `inquirer_trained` against two different LoRA adapters served under
the same model id produce the same `model_pin_hash`, the same `semantic_hash` and the same
`run_id` -- and `--resume` would skip the second as already done. The manifest could name the
deployment but not the weights.

THE CONSTRAINT THAT SHAPES THE FIX. `adapter_sha` is inside `ModelPin.key`, which is inside
`model_pin_hash`, which is inside `semantic_hash`, which is inside `run_id`. Populating it for
a model id that has ALREADY RUN would change every one of those run ids retroactively. So the
registry is opt-in by model id and returns None for everything else, and
`test_an_unregistered_model_id_has_no_adapter_sha` is the compatibility guarantee: it is the
test that must never be relaxed, because relaxing it renames finished runs.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TABLE_2026_09 = REPO / "scripts" / "price_tables" / "2026-09.json"

# The deployments I.14 serves off the rented box, one vLLM `--lora-modules` name each.
# THIS IS THE FIRST WAVE, NOT THE WHOLE SET: `scripts/hpc/register_checkpoint.py` adds a
# registry row and a $0 price row together for every checkpoint that gets deployed, and
# `test_the_shipped_file_parses_and_documents_its_schema` checks the registry against the
# price table rather than against this list. What these seven still pin is that the names
# docs/GPU_RUNBOOK.md 7-9 TELLS you to export as PI_MODEL_INQUIRER are priced, so the
# commands in the runbook start rather than raising LLMConfigError.
LOCAL_DEPLOYMENTS = (
    "qwen3-8b-base",
    "qwen3-8b-sft",
    "qwen3-8b-dpo-control",
    "qwen3-8b-dpo-rater",
    "qwen3-8b-dpo-reaches",
    "qwen3-4b-base",
    "qwen3-4b-sft",
)

# SIX FROM 2026-09-15. `train_id_set_hash` is `dataset_sha` under the name every other
# artifact uses for that value -- the export manifest, the id-set sidecar, `RunManifest`, the
# `runs.parquet` column, and the contamination check that compares them. The loader refuses a
# row whose two names disagree; see `tests/test_train_id_set_proof.py` for that half.
FIELDS = (
    "adapter_sha",
    "training_manifest_sha",
    "base_model",
    "rung",
    "dataset_sha",
    "train_id_set_hash",
)


def _entry(**over):
    base = {
        "adapter_sha": "a" * 64,
        "training_manifest_sha": "b" * 64,
        "base_model": "Qwen/Qwen3-8B",
        "rung": 1,
        "dataset_sha": "c" * 64,
        "train_id_set_hash": "c" * 64,
    }
    base.update(over)
    return base


def _write(tmp_path, obj, *, raw: str | None = None) -> Path:
    p = tmp_path / "checkpoints.json"
    p.write_text(raw if raw is not None else json.dumps(obj, indent=1))
    return p


def _fresh(monkeypatch, path: Path | None):
    """Point the loader at a path and clear its module cache."""
    from pinq_adapters.llm import checkpoints

    if path is None:
        monkeypatch.delenv("PI_CHECKPOINTS", raising=False)
    else:
        monkeypatch.setenv("PI_CHECKPOINTS", str(path))
    checkpoints.reset_cache()
    return checkpoints


# --------------------------------------------------------------------------- the loader


def test_no_registry_row_RENAMES_A_RUN_THAT_HAS_ALREADY_FINISHED(monkeypatch) -> None:
    """The invariant this file has always been about, measured against the runs themselves.

    TWO REWRITES, AND WHY. Until 2026-09-15 this asserted `registry() == {}`, which was the
    state of the world while no deployment existed rather than the property anyone cares
    about. It then became "no registered id appears in calls.parquet" -- which was right for
    exactly as long as the first registered deployment went unused. It FAILED on 2026-09-15
    at 610 calls: `qwen3-8b-sft` was registered AT deployment (which is what
    checkpoints.py asks for) and then, correctly, served a dev sweep. A guard that fires when
    the thing it guards is used as intended is measuring the wrong quantity.

    WHAT IS ACTUALLY FORBIDDEN is a row that changes a FINISHED run's identity. `adapter_sha`
    sits inside `ModelPin.key` -> `model_pin_hash` -> `semantic_hash` -> `run_id`, so the
    damage is done by a row that was absent (or different) when a run was recorded and is
    present (or changed) now. Every run writes its own pins into `runs/<run_id>/manifest.json`
    at the moment it runs, so the two can simply be compared -- which is strictly stronger
    than the id-never-ran test it replaces: that one could not see a sha EDITED under an
    existing name at all, and this one fails on it. Shown red before it was shown green, by
    handing it a one-character-different sha:

        AssertionError: conf/checkpoints.json would RENAME 135 finished run(s):
        qwen3-8b-sft is registered at 8f91d868... but run a85533a618e4a6a265e3c45635f0409a
        recorded 7f91d868... for role inquirer
    """
    c = _fresh(monkeypatch, None)
    _assert_no_finished_run_is_renamed(c.registry())


def _runs_using(model_id: str) -> list[str]:
    """Run ids whose call log names this model. `[]` where there is no call log."""
    calls = REPO / "scores" / "parquet" / "calls.parquet"
    if not calls.is_file():
        return []
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    rows = con.execute(
        "SELECT DISTINCT run_id FROM read_parquet(?) WHERE model = ?", [str(calls), model_id]
    ).fetchall()
    return [r[0] for r in rows]


def _assert_no_finished_run_is_renamed(reg) -> None:
    offenders: list[str] = []
    checked = unverifiable = 0
    for model_id, entry in reg.items():
        want = entry["adapter_sha"]
        for run_id in _runs_using(model_id):
            m = REPO / "runs" / run_id / "manifest.json"
            if not m.is_file():
                unverifiable += 1  # a pruned run dir is not evidence either way
                continue
            pins = json.loads(m.read_text()).get("pins") or {}
            for role, pin in pins.items():
                if pin.get("model_id") != model_id:
                    continue
                checked += 1
                if pin.get("adapter_sha") != want:
                    offenders.append(
                        f"{model_id} is registered at {want} but run {run_id} recorded "
                        f"{pin.get('adapter_sha')} for role {role}"
                    )
    assert not offenders, (
        f"conf/checkpoints.json would RENAME {len(offenders)} finished run(s):\n"
        + "\n".join(offenders[:5])
        + "\nadapter_sha is inside run_id. Serve the new weights under a NEW deployment name."
    )
    if not checked and unverifiable:
        pytest.skip(f"{unverifiable} run(s) have no manifest on disk; nothing to compare")


def test_an_absent_file_is_an_empty_registry_not_a_crash(monkeypatch, tmp_path) -> None:
    """A stranger's checkout has no rented box and no adapters; it must still run."""
    c = _fresh(monkeypatch, tmp_path / "nope.json")
    assert c.registry() == {}


def test_duplicate_keys_are_refused_at_load(monkeypatch, tmp_path) -> None:
    """JSON lets a later key win SILENTLY, so a checkpoint re-pointed by a careless paste would
    take effect with nothing to see. Two entries for one deployment is never intentional."""
    raw = (
        '{"qwen3-8b-sft": ' + json.dumps(_entry(adapter_sha="1" * 64)) + ", "
        '"qwen3-8b-sft": ' + json.dumps(_entry(adapter_sha="2" * 64)) + "}"
    )
    c = _fresh(monkeypatch, _write(tmp_path, None, raw=raw))
    with pytest.raises(ValueError, match="qwen3-8b-sft"):
        c.registry()


def test_every_entry_must_carry_all_six_fields(monkeypatch, tmp_path) -> None:
    """A partial entry is worse than none: it makes the manifest look provenanced while the
    field that would let anyone re-derive the weights is absent."""
    bad = _entry()
    del bad["dataset_sha"]
    c = _fresh(monkeypatch, _write(tmp_path, {"qwen3-8b-sft": bad}))
    with pytest.raises(ValueError, match="dataset_sha"):
        c.registry()


def test_documentation_keys_are_not_entries(monkeypatch, tmp_path) -> None:
    """`_schema` explains the file and is not a deployment."""
    c = _fresh(monkeypatch, _write(tmp_path, {"_schema": "notes", "qwen3-8b-sft": _entry()}))
    assert set(c.registry()) == {"qwen3-8b-sft"}


def test_the_shipped_file_parses_and_documents_its_schema() -> None:
    raw = json.loads((REPO / "conf" / "checkpoints.json").read_text())
    assert "_schema" in raw, "the file must say what a row means before anyone adds one"
    for f in FIELDS:
        assert f in raw["_schema"], f"_schema does not mention {f}"
    # Every non-documentation key is a DEPLOYMENT, and must carry all five fields. This was
    # `== []` while the file had no rows; see
    # `test_no_registry_row_RENAMES_A_RUN_THAT_HAS_ALREADY_FINISHED` for why "empty" was the
    # state of the world rather than the invariant, and for the check that replaced it.
    #
    # THE KEY CHECK CHANGED ON 2026-09-15, AND NOT TO MAKE A ROW FIT. It read
    # `assert name in LOCAL_DEPLOYMENTS` -- a hand-written list of seven ids, which said
    # "a row key is a served name, never an arm id" by enumerating the served names that
    # existed the week it was written. Registration is now a script over many checkpoints
    # (`scripts/hpc/register_checkpoint.py`), so the enumeration stopped being the invariant
    # and became a list to keep updating. What replaces it is the OPERATIONAL form of the same
    # sentence, and it is strictly stronger: a key must be a name `PriceTable.rates` can price
    # at $0, which an arm id (`inquirer_trained`) is not and never will be, and which the
    # seven ids in LOCAL_DEPLOYMENTS all are. It also catches a registration that added the
    # registry row and forgot the price row -- the exact pair that makes a sweep refuse to
    # start, which the enumeration could not see.
    priced = json.loads(TABLE_2026_09.read_text())["models"]
    rows = {k: v for k, v in raw.items() if not k.startswith("_")}
    for name, entry in rows.items():
        assert name in priced, (
            f"{name!r} has no row in {TABLE_2026_09.name}; a registry key is what "
            "PI_MODEL_INQUIRER is set to and what the price table is looked up by, never an "
            "arm. PriceTable.rates is strict, so this row would make the sweep refuse to start"
        )
        assert priced[name]["input"] == 0.0 and priced[name]["output"] == 0.0, (
            f"{name!r} is registered as a locally served checkpoint but is priced as paid"
        )
        missing = [f for f in FIELDS if f not in entry]
        assert not missing, f"{name!r} is missing {missing}"
        assert entry["rung"] in (1, 2, 3), f"{name!r} has rung {entry['rung']!r}"
        for f in ("adapter_sha", "training_manifest_sha", "dataset_sha", "train_id_set_hash"):
            assert re.fullmatch(r"[0-9a-f]{64}", str(entry[f])), f"{name}.{f} is not a sha256"


# --------------------------------------------------------------------------- into the pin


def _client(monkeypatch, model: str):
    pytest.importorskip("tenacity")
    from pinq.budget import BudgetLedger
    from pinq_adapters.llm.litellm_client import MeteredClient, PriceTable

    monkeypatch.setenv("PI_MODEL_INQUIRER", model)
    pt = PriceTable.load(TABLE_2026_09)
    return MeteredClient(BudgetLedger(cap=4), price_table=pt, env=dict(os.environ))


def test_an_unregistered_model_id_has_no_adapter_sha(monkeypatch, tmp_path) -> None:
    """THE COMPATIBILITY GUARANTEE. `adapter_sha` is inside `run_id`; if this ever returns a
    value for an id that is not in the registry, every finished run using that id is renamed
    and `--resume` stops recognising its own work. Do not relax this test."""
    _fresh(monkeypatch, _write(tmp_path, {"qwen3-8b-sft": _entry()}))
    pin = _client(monkeypatch, "openai/aws/gpt-oss-120b").pin("inquirer")
    assert pin.adapter_sha is None

    from pinq.types import ModelPin

    same = ModelPin(
        role="inquirer",
        model_id=pin.model_id,
        provider=pin.provider,
        base_url_sha=pin.base_url_sha,
        sampling_sha=pin.sampling_sha,
        price_table_version=pin.price_table_version,
    )
    assert pin.key == same.key, "an unregistered id must hash exactly as it did before"


def test_a_registered_id_reaches_the_run_manifest(monkeypatch, tmp_path) -> None:
    """The whole point: the sha must survive `collect_pins` -> `RunManifest` -> the JSON a
    reader actually opens."""
    sha = "d" * 64
    _fresh(monkeypatch, _write(tmp_path, {"qwen3-8b-sft": _entry(adapter_sha=sha)}))
    client = _client(monkeypatch, "qwen3-8b-sft")
    assert client.pin("inquirer").adapter_sha == sha

    from pi_run.manifest import manifest_to_dict
    from pi_run.worker import collect_pins
    from pinq.types import RunManifest

    class _Comp:
        llm_role = "inquirer"

    pins = collect_pins(client, (_Comp(),))
    m = RunManifest(
        suite_id="musique",
        task_id="t1",
        arm_id="inquirer_trained",
        policy_id="prompted",
        seed=0,
        split="test",
        corpus_hash="ch",
        budget_cap=8,
        max_turns=16,
        word_cap=120,
        pins=pins,
    )
    assert manifest_to_dict(m)["pins"]["inquirer"]["adapter_sha"] == sha


def test_a_registered_id_changes_the_pin_key(monkeypatch, tmp_path) -> None:
    """If it did not, two adapters served under one name would share a run_id and `--resume`
    would skip the second as already done -- the failure `collect_pins` was written for."""
    from pinq.types import ModelPin

    _fresh(monkeypatch, _write(tmp_path, {"qwen3-8b-sft": _entry(adapter_sha="e" * 64)}))
    pinned = _client(monkeypatch, "qwen3-8b-sft").pin("inquirer")
    unpinned = ModelPin(
        role="inquirer",
        model_id=pinned.model_id,
        provider=pinned.provider,
        base_url_sha=pinned.base_url_sha,
        sampling_sha=pinned.sampling_sha,
        price_table_version=pinned.price_table_version,
    )
    assert pinned.key != unpinned.key


def test_the_caching_client_forwards_the_adapter_sha(monkeypatch, tmp_path) -> None:
    """`CachingClient` is what the worker actually holds. It once had no `pin` at all and the
    worker wrote `"pins": {}`; a wrapper that dropped this field would do the same damage in
    miniature."""
    _fresh(monkeypatch, _write(tmp_path, {"qwen3-8b-sft": _entry(adapter_sha="f" * 64)}))
    from pi_run.cache import CachingClient

    inner = _client(monkeypatch, "qwen3-8b-sft")
    wrapped = CachingClient(inner)
    assert wrapped.pin("inquirer").adapter_sha == "f" * 64


# --------------------------------------------------------------------------- pricing


def test_the_price_table_accepts_every_local_deployment() -> None:
    """`PriceTable.rates` is STRICT: an unpriced model raises, so a sweep against the rented
    box would refuse to start. These rows exist so it starts."""
    from pinq_adapters.llm.pricing import PriceTable

    pt = PriceTable.load(TABLE_2026_09)
    for model in LOCAL_DEPLOYMENTS:
        rates = pt.rates(model)
        assert rates["input"] == 0.0 and rates["output"] == 0.0
        assert pt.usd(model, tok_prompt=10_000, tok_completion=5_000) == 0.0


def test_the_zero_rows_say_why_they_are_zero() -> None:
    """A $0 row is the silent-zero failure this table exists to prevent UNLESS it says why.
    These are locally served weights: there is no per-token price to read, and the real cost is
    GPU-hour amortisation, which a per-token table cannot express."""
    raw = json.loads(TABLE_2026_09.read_text())
    why = raw.get("_why_local_deployments", "")
    assert "GPU" in why or "gpu" in why
    for model in LOCAL_DEPLOYMENTS:
        assert model in raw["models"], model


def test_the_existing_rates_in_the_table_were_not_touched() -> None:
    """`_why_2026_09` forbids editing a table a finished sweep was priced under. Adding a model
    that has never run re-prices nothing; changing a rate re-prices everything."""
    raw = json.loads(TABLE_2026_09.read_text())
    assert raw["models"]["openai/aws/gpt-oss-120b"]["input"] > 0
    assert raw["version"] == "2026-09"


# ------------------------------------------------------- and the client can actually call them


def test_a_bare_local_deployment_id_is_routable_through_the_proxy(monkeypatch) -> None:
    """The price table and the registry are keyed on the BARE served name, so the pin must stay
    bare -- and a bare name is exactly what the litellm SDK refuses to route.

    THE BUG THIS CAUGHT, MEASURED 2026-09-15 against the real proxy. `PI_MODEL_INQUIRER` is set
    to `qwen3-8b-sft` (docs/GPU_RUNBOOK.md 7-8; it has to be, because `PriceTable.rates` and
    `conf/checkpoints.json` are both looked up by that exact string, and a `litellm_proxy/`
    prefix would miss both and raise `LLMConfigError` before the sweep started). But
    `litellm.completion(model="qwen3-8b-sft", api_base=...)` raises

        BadRequestError: LLM Provider NOT provided. You passed model=qwen3-8b-sft

    inside the SDK, before any request is sent -- litellm infers the provider from the model
    prefix and a served name has none. Every frozen role is pinned `openai/aws/gpt-oss-120b`
    and so has always had one, which is why no existing test saw this: the local-deployment
    path had never been dispatched.

    `_provider_of` already answers `litellm_proxy` whenever a base url is set. This asserts that
    `_dispatch` tells the SDK the same thing, and it asserts it through litellm's OWN resolver
    rather than by matching the kwarg, so it fails if the fix stops being sufficient.
    """
    litellm = pytest.importorskip("litellm")
    import pinq_adapters.llm.litellm_client as lc
    from pinq.budget import BudgetLedger

    seen: dict = {}

    def fake_completion(**kw):
        seen.update(kw)
        # What litellm does with these kwargs before it opens a socket. This is the call that
        # raised, so routing is tested rather than described.
        litellm.get_llm_provider(
            model=kw["model"],
            custom_llm_provider=kw.get("custom_llm_provider"),
            api_base=kw.get("api_base"),
        )

        class _M:
            content = "ok"

        class _C:
            message = _M()

        class _U:
            prompt_tokens = 1
            completion_tokens = 1

        class _R:
            choices = [_C()]
            usage = _U()

        return _R()

    monkeypatch.setattr(litellm, "completion", fake_completion)

    # The served name must be priced, and only the 2026-09 table lists the local deployments.
    # Passed explicitly: importing litellm runs load_dotenv(), so an environment lookup would read
    # a developer's .env here and the default table in a clean checkout, where this raised.
    from pinq_adapters.llm.pricing import PriceTable

    client = lc.MeteredClient(
        BudgetLedger(8),
        models={"inquirer": "qwen3-8b-sft"},
        base_url="http://127.0.0.1:4000",
        env={},
        price_table=PriceTable.load(REPO / "scripts" / "price_tables" / "2026-09.json"),
    )
    client.complete(role="inquirer", messages=[{"role": "user", "content": "hi"}], seed=0)

    assert seen["model"] == "qwen3-8b-sft", (
        "the served name must reach the wire unchanged: the proxy routes by it, and the price "
        "table and checkpoint registry are keyed on it"
    )


def test_a_prefixed_model_id_is_left_exactly_as_it_was(monkeypatch) -> None:
    """The compatibility half. Every id in use today carries a provider prefix, and the fix
    above must not touch them -- `aws/gpt-oss-120b` reaching the gateway as `litellm_proxy/`
    anything would re-route every frozen role in every finished sweep's shape."""
    litellm = pytest.importorskip("litellm")
    import pinq_adapters.llm.litellm_client as lc
    from pinq.budget import BudgetLedger

    seen: dict = {}

    def fake_completion(**kw):
        seen.update(kw)

        class _M:
            content = "ok"

        class _C:
            message = _M()

        class _U:
            prompt_tokens = 1
            completion_tokens = 1

        class _R:
            choices = [_C()]
            usage = _U()

        return _R()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    client = lc.MeteredClient(
        BudgetLedger(8),
        models={"drafter": "openai/aws/gpt-oss-120b"},
        base_url="http://127.0.0.1:4000",
        env={},
    )
    client.complete(role="drafter", messages=[{"role": "user", "content": "hi"}], seed=0)

    assert seen["model"] == "openai/aws/gpt-oss-120b"
    assert "custom_llm_provider" not in seen, (
        "a prefixed id already routes; naming a provider for it would override the prefix"
    )


# ------------------------------------------------------------------- lineage: rung >= 2 rows
#
# `dataset_sha` and `train_id_set_hash` are one quantity for a rung-1 row and TWO for a rung-2
# one. A DPO adapter is initialised from an SFT adapter, so its weights were fitted on the SFT
# file's tasks AND the pairs file's; `dataset_sha` keeps naming this row's own export and
# `train_id_set_hash` becomes the union, with `init_from` naming the row it was unioned with.
#
# MEASURED 2026-09-15 on the shipped headline export: SFT 2,525 ids, pairs 1,027, SFT - pairs
# 1,499, union 2,526. A rung-2 row registered under the old rule stamps 1,027 of the 2,526
# tasks its weights saw, and the id-set canary is blind to the other 1,499 -- which is what a
# leak through a served NAME could reach unseen, the case layer 4 of the firewall exists for.
# (The canary ALSO covers a task exported and since re-bucketed, which the split predicate
# structurally cannot see; that is an argument about what the two predicates can express, not
# a count -- 0 of the 2,525 SFT ids are test or dev today.)


def test_a_rung2_row_whose_two_hashes_DIFFER_is_accepted_when_init_from_names_a_row(
    monkeypatch, tmp_path
) -> None:
    """The rule that changed, and the only row shape that may use it."""
    c = _fresh(
        monkeypatch,
        _write(
            tmp_path,
            {
                "qwen3-8b-sft": _entry(),
                "qwen3-8b-dpo": _entry(
                    rung=2,
                    dataset_sha="d" * 64,
                    train_id_set_hash="e" * 64,
                    init_from="qwen3-8b-sft",
                ),
            },
        ),
    )
    reg = c.registry()
    assert reg["qwen3-8b-dpo"]["train_id_set_hash"] == "e" * 64
    assert c.train_id_set_hash_of("qwen3-8b-dpo") == "e" * 64, (
        "the stamp is the UNION: that is what makes the reader resolve the set the weights saw"
    )


def test_a_rung2_row_whose_two_hashes_differ_WITHOUT_init_from_is_refused(
    monkeypatch, tmp_path
) -> None:
    """Without a lineage the two names are still one quantity, so a difference is still a row
    naming an id set the adapter was not fitted on."""
    c = _fresh(
        monkeypatch,
        _write(
            tmp_path,
            {"qwen3-8b-dpo": _entry(rung=2, dataset_sha="d" * 64, train_id_set_hash="e" * 64)},
        ),
    )
    with pytest.raises(ValueError, match="init_from"):
        c.registry()


def test_a_rung2_row_with_NO_init_from_at_all_is_refused(monkeypatch, tmp_path) -> None:
    """Even when the two hashes agree. A rung-2 adapter that was fitted on exactly one file
    does not exist -- DPO is initialised from rung 1 (`DPOConfig.validate` refuses an empty
    `adapter`) -- so a row whose hashes agree is claiming a subset, not a clean single export."""
    c = _fresh(monkeypatch, _write(tmp_path, {"qwen3-8b-dpo": _entry(rung=2)}))
    with pytest.raises(ValueError, match="init_from"):
        c.registry()


def test_a_rung2_row_whose_init_from_is_UNREGISTERED_is_refused(monkeypatch, tmp_path) -> None:
    """`init_from` is how a reader re-derives the union. A dangling name makes the union
    unverifiable while the row looks provenanced."""
    c = _fresh(
        monkeypatch,
        _write(
            tmp_path,
            {
                "qwen3-8b-dpo": _entry(
                    rung=2,
                    dataset_sha="d" * 64,
                    train_id_set_hash="e" * 64,
                    init_from="a-name-nobody-registered",
                )
            },
        ),
    )
    with pytest.raises(ValueError, match="a-name-nobody-registered"):
        c.registry()


def test_a_rung2_row_MAY_init_from_another_rung2_row(monkeypatch, tmp_path) -> None:
    """DPO-ON-DPO IS STILL RUNG 2, and the rule that forbade it was wrong. A second DPO pass
    from the `qwen3-8b-dpo-headline-control` adapter is a planned arm; its ancestor is rung 2,
    so "strictly decreasing" would have made a legitimate checkpoint unregisterable. What the
    rung must guarantee is only that lineage does not run BACKWARDS -- a DPO cannot start from
    a GRPO -- and termination comes from walking the chain, not from the rung."""
    c = _fresh(
        monkeypatch,
        _write(
            tmp_path,
            {
                "sft": _entry(),
                "dpo1": _entry(rung=2, train_id_set_hash="e" * 64, init_from="sft"),
                "dpo2": _entry(rung=2, train_id_set_hash="f" * 64, init_from="dpo1"),
            },
        ),
    )
    assert set(c.registry()) == {"sft", "dpo1", "dpo2"}
    assert c.train_id_set_hash_of("dpo2") == "f" * 64


def test_a_row_initialised_from_a_HIGHER_rung_is_refused(monkeypatch, tmp_path) -> None:
    """The direction that is still nonsense: a DPO pass cannot have started from a GRPO."""
    c = _fresh(
        monkeypatch,
        _write(
            tmp_path,
            {
                "grpo": _entry(rung=3, init_from="base"),
                "dpo": _entry(rung=2, train_id_set_hash="f" * 64, init_from="grpo"),
            },
        ),
    )
    with pytest.raises(ValueError, match="rung"):
        c.registry()


def test_a_TWO_ROW_CYCLE_in_init_from_is_refused(monkeypatch, tmp_path) -> None:
    """WHAT ACTUALLY GUARANTEES TERMINATION, now that the rung does not. Two rows naming each
    other is a chain with no base case, so no finite union and no re-derivable stamp."""
    c = _fresh(
        monkeypatch,
        _write(
            tmp_path,
            {
                "a": _entry(rung=2, train_id_set_hash="e" * 64, init_from="b"),
                "b": _entry(rung=2, train_id_set_hash="f" * 64, init_from="a"),
            },
        ),
    )
    with pytest.raises(ValueError, match="cycle"):
        c.registry()


def test_a_row_that_inits_from_ITSELF_is_refused(monkeypatch, tmp_path) -> None:
    """The one-row cycle, which a pairwise rung comparison alone would wave through the moment
    the comparison stopped being strict."""
    c = _fresh(
        monkeypatch,
        _write(tmp_path, {"a": _entry(rung=2, train_id_set_hash="e" * 64, init_from="a")}),
    )
    with pytest.raises(ValueError, match="cycle"):
        c.registry()


def test_a_chain_that_reaches_base_or_a_rung1_row_terminates(monkeypatch, tmp_path) -> None:
    """Both base cases, walked end to end: a rung-1 row carries no `init_from` at all, and the
    literal `base` is where a dpo_from_base chain stops."""
    c = _fresh(
        monkeypatch,
        _write(
            tmp_path,
            {
                "sft": _entry(),
                "dpo1": _entry(rung=2, train_id_set_hash="e" * 64, init_from="sft"),
                "dpo2": _entry(rung=2, train_id_set_hash="f" * 64, init_from="dpo1"),
                "fromb": _entry(rung=2, init_from="base"),
                "dpo3": _entry(rung=2, train_id_set_hash="0" * 64, init_from="fromb"),
            },
        ),
    )
    assert set(c.registry()) == {"sft", "dpo1", "dpo2", "fromb", "dpo3"}


def test_a_rung1_row_may_not_carry_an_init_from(monkeypatch, tmp_path) -> None:
    """Rung 1 starts from the raw base. An `init_from` here would be a lineage claim with no
    weights behind it, and would license the two hashes to drift on a row where they may not."""
    c = _fresh(
        monkeypatch,
        _write(tmp_path, {"x": _entry(), "qwen3-8b-sft": _entry(init_from="x")}),
    )
    with pytest.raises(ValueError, match="init_from"):
        c.registry()


def test_a_rung1_row_whose_two_hashes_differ_is_STILL_refused(monkeypatch, tmp_path) -> None:
    """UNCHANGED BEHAVIOUR, kept deliberately. The lineage rule relaxes rung >= 2 and nothing
    else; a rung-1 adapter is fitted on exactly one file and its two names are one quantity."""
    c = _fresh(monkeypatch, _write(tmp_path, {"qwen3-8b-sft": _entry(train_id_set_hash="d" * 64)}))
    with pytest.raises(ValueError, match="dataset_sha"):
        c.registry()


def test_the_schema_documents_init_from_and_every_shipped_lineage_row_carries_it() -> None:
    """A field the loader requires and the file does not explain is a trap for the next row."""
    raw = json.loads((REPO / "conf" / "checkpoints.json").read_text())
    assert "init_from" in raw["_schema"], "_schema does not mention init_from"
    for name, entry in raw.items():
        if name.startswith("_"):
            continue
        if int(entry["rung"]) >= 2:
            assert entry.get("init_from"), f"{name!r} is rung {entry['rung']} and names no init"
            assert entry["init_from"] in raw, f"{name!r} inits from an unregistered row"
        else:
            assert "init_from" not in entry, f"{name!r} is rung 1 and may not claim a lineage"


def test_a_rung2_row_whose_init_from_is_the_literal_base_is_accepted(monkeypatch, tmp_path) -> None:
    """`pi train rung2 --reference base` trains a fresh LoRA on the untrained base, so the row
    is rung 2 with no ancestor: the weights saw exactly this export and the two id-set names
    are one quantity again, as at rung 1."""
    c = _fresh(monkeypatch, _write(tmp_path, {"qwen3-8b-dpo": _entry(rung=2, init_from="base")}))
    assert c.registry()["qwen3-8b-dpo"]["init_from"] == "base"
    assert c.train_id_set_hash_of("qwen3-8b-dpo") == "c" * 64


def test_a_rung2_row_on_the_base_whose_two_hashes_DIFFER_is_refused(monkeypatch, tmp_path) -> None:
    """There is no ancestor to union in, so a union is exactly what this row cannot have."""
    c = _fresh(
        monkeypatch,
        _write(
            tmp_path,
            {"qwen3-8b-dpo": _entry(rung=2, init_from="base", train_id_set_hash="d" * 64)},
        ),
    )
    with pytest.raises(ValueError, match="base"):
        c.registry()


def test_a_deployment_LITERALLY_NAMED_base_is_refused(monkeypatch, tmp_path) -> None:
    """Otherwise `init_from: "base"` names a row and means "no ancestor" at the same time, and
    nothing downstream can tell which was meant."""
    c = _fresh(monkeypatch, _write(tmp_path, {"base": _entry()}))
    with pytest.raises(ValueError, match="base"):
        c.registry()
