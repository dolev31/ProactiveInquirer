"""Firewall layer 4: the only layer that catches leakage through a STRING.

Layers 1-3 are structural and stop leakage through code. None of them can stop a gold answer
being interpolated into a prompt. This one can, and only if it is actually armed — which is
what these tests check.
"""

import json
from pathlib import Path

import pytest

from pi_eval.build.synth_build import build
from pi_eval.canary import (
    PREFIX,
    load,
    mint,
    register,
    registry_path,
    scan_text,
    scan_tree,
    strip,
)


def test_mint_is_deterministic_under_a_salt():
    """A build must be reproducible; without a salt the nonce must be unguessable."""
    a = mint("synth", "s0", salt="pepper")
    assert a == mint("synth", "s0", salt="pepper")
    assert a != mint("synth", "s1", salt="pepper")
    assert a != mint("synth", "s0", salt="other")
    assert a.startswith(PREFIX)


def test_a_gold_build_arms_the_registry(tmp_path):
    """The gap `pi verify firewall` reported: canaries_configured must not be 0."""
    build(n_tasks=3, n_facets=2, depth=2, root=tmp_path)
    canaries = load(tmp_path)
    assert len(canaries) == 3, "one nonce per task graph"
    assert registry_path(tmp_path).exists()


def test_every_graph_carries_its_nonce(tmp_path):
    import json

    _, gold, _ = build(n_tasks=3, n_facets=2, depth=2, root=tmp_path)
    rows = [json.loads(x) for x in gold.read_text().splitlines() if x]
    assert all(r["gold_canary"].startswith(PREFIX) for r in rows)
    assert len({r["gold_canary"] for r in rows}) == 3, "nonces must be distinct per task"


def test_the_registry_is_append_only(tmp_path):
    """A stale build's leak must still be caught after its graph was regenerated."""
    register({"PINQCANARY_AAAA000000000000"}, tmp_path)
    register({"PINQCANARY_BBBB000000000000"}, tmp_path)
    assert load(tmp_path) == {
        "PINQCANARY_AAAA000000000000",
        "PINQCANARY_BBBB000000000000",
    }


def test_scan_finds_a_leak_with_surrounding_context():
    c = mint("synth", "s0", salt="p")
    hits = scan_text(f"system prompt ... {c} ... rest", {c}, where="cache/ab/xy.json")
    assert len(hits) == 1
    assert hits[0].canary == c and hits[0].where == "cache/ab/xy.json"
    assert c in hits[0].excerpt


def test_scan_is_silent_on_clean_text():
    assert scan_text("a perfectly ordinary prompt", {mint("s", "t", salt="p")}) == []


def test_scan_tree_walks_the_cache(tmp_path):
    """It scans the CACHE, not the prompt templates: a template can look innocent while the
    value interpolated at runtime is not."""
    c = mint("synth", "s0", salt="p")
    d = tmp_path / "cache" / "ab"
    d.mkdir(parents=True)
    (d / "req.json").write_text('{"messages":[{"content":"leak %s here"}]}' % c)
    (d / "clean.json").write_text('{"messages":[{"content":"fine"}]}')
    hits = scan_tree(tmp_path / "cache", {c})
    assert len(hits) == 1 and "req.json" in hits[0].where


def test_scan_tree_is_a_noop_without_canaries(tmp_path):
    assert scan_tree(tmp_path, set()) == []


def test_strip_removes_nonces_for_release():
    """Released gold must not carry the nonces, or a published list would reveal what was
    withheld."""
    c = mint("synth", "s0", salt="p")
    out = strip(f"the answer is 42 {c} end")
    assert PREFIX not in out and "42" in out


@pytest.mark.parametrize("suite", ["synth", "musique", "tau2"])
def test_nonces_do_not_collide_across_suites(suite):
    others = {mint(s, "t0", salt="p") for s in ("synth", "musique", "tau2") if s != suite}
    assert mint(suite, "t0", salt="p") not in others


# --------------------------------------------------------------------------- every suite
#
# Layer 4 was implemented in `synth_build` and NOWHERE ELSE, so `pi verify firewall` scanned
# every cached request against a registry containing nonces for the one suite that carries no
# real corpus text. It reported "clean" on musique, strategyqa, wiki2, drgym and tau2 -- every
# suite where a leak could actually happen -- because there was nothing there to find.
#
# Minting now happens in `write_graphs`, the single choke point every real builder passes
# through, so it is a property of "gold was written" rather than of "the builder remembered".


def test_write_graphs_mints_and_registers_a_canary_for_every_suite(tmp_path, monkeypatch):
    from pi_eval.build.common import write_graphs
    from pi_eval.canary import load

    monkeypatch.setenv("PI_CANARY_SALT", "test-salt")
    rows = [
        {"gold_suite": "musique", "gold_task_key": f"t{i}", "gold_nodes": [], "gold_edges": []}
        for i in range(3)
    ]
    path = write_graphs(tmp_path, "musique", "v1", rows)

    written = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    assert len(written) == 3
    minted = {w["gold_canary"] for w in written}
    assert all(c.startswith(PREFIX) for c in minted)
    assert len(minted) == 3, "one nonce per task, not one per suite"
    assert minted <= load(tmp_path), "a canary on disk with none registered is a hole"


def test_minting_is_idempotent_so_a_builder_with_its_own_nonce_is_untouched(tmp_path):
    """`synth_build` mints its own. Re-stamping would give one graph two nonces, and the second
    would be the only one registered."""
    from pi_eval.build.common import write_graphs

    rows = [{"gold_suite": "s", "gold_task_key": "t0", "gold_canary": "PINQCANARY_KEEPME"}]
    path = write_graphs(tmp_path, "s", "v1", rows)
    assert json.loads(path.read_text().splitlines()[0])["gold_canary"] == "PINQCANARY_KEEPME"


def test_a_rebuild_under_the_same_salt_reproduces_the_same_registry(tmp_path, monkeypatch):
    """Without a salt every rebuild mints fresh nonces: the registry is append-only so nothing
    stops being scanned, but the build stops being reproducible and the registry grows without
    bound. PI_CANARY_SALT is what makes `pi data build` twice a no-op."""
    from pi_eval.build.common import write_graphs
    from pi_eval.canary import load

    monkeypatch.setenv("PI_CANARY_SALT", "fixed")
    rows = [{"gold_suite": "wiki2", "gold_task_key": "t0"}]
    write_graphs(tmp_path, "wiki2", "v1", [dict(r) for r in rows])
    first = load(tmp_path)
    write_graphs(tmp_path, "wiki2", "v1", [dict(r) for r in rows])
    assert load(tmp_path) == first

    monkeypatch.setenv("PI_CANARY_SALT", "different")
    write_graphs(tmp_path, "wiki2", "v1", [dict(r) for r in rows])
    assert load(tmp_path) > first, "a new salt mints a new nonce; the old one is still scanned"


def test_the_canary_survives_both_gold_readers(tmp_path, monkeypatch):
    """A nonce that is written and then dropped on read is a nonce no scanner can be handed.
    There are two readers -- the path-addressed one for builders and the env-guarded one
    production scoring uses -- and both have to carry it."""
    from pi_eval.build.common import read_graphs, write_graphs
    from pi_eval.gold import load_graphs

    monkeypatch.setenv("PI_CANARY_SALT", "test-salt")
    path = write_graphs(
        tmp_path,
        "musique",
        "v1",
        [
            {
                "gold_suite": "musique",
                "gold_task_key": "t0",
                "gold_answer": "x",
                "gold_nodes": [],
                "gold_edges": [],
                "gold_facets": [],
                "gold_seed_node_ids": [],
                "gold_graph_version": "v1",
            }
        ],
    )
    expected = json.loads(path.read_text().splitlines()[0])["gold_canary"]

    assert read_graphs(path)[0].gold_canary == expected
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
    assert load_graphs("musique", "v1")["t0"].gold_canary == expected


def test_no_builder_writes_gold_without_passing_through_write_graphs():
    """The choke point only holds if it IS one. A builder that wrote its own jsonl would slip
    a whole suite past layer 4 while every test here still passed."""
    import pathlib

    build_dir = pathlib.Path(__file__).resolve().parents[1] / "src" / "pi_eval" / "build"
    offenders = []
    for f in sorted(build_dir.glob("*_build.py")):
        src = f.read_text()
        if "write_graphs" in src or "from pi_eval.canary import" in src:
            continue
        if "gold" in src and "write_text" in src:
            offenders.append(f.name)
    assert not offenders, f"these builders write gold without minting a canary: {offenders}"


@pytest.mark.integration
def test_the_real_gold_tree_carries_a_canary_on_every_graph_of_every_suite():
    """The check that would have caught the original hole. Runs against whatever is built on
    this machine, and skips rather than fails when nothing is."""
    import pathlib

    from pi_eval.canary import load

    root = pathlib.Path(__file__).resolve().parents[1]
    graphs = sorted((root / "data" / "gold" / "graphs").glob("*/*.jsonl"))
    if not graphs:
        pytest.skip("no gold built on this machine")

    registry = load(root)
    missing: dict[str, int] = {}
    unregistered: dict[str, int] = {}
    for p in graphs:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            canary = json.loads(line).get("gold_canary") or ""
            if not canary:
                missing[p.parent.name] = missing.get(p.parent.name, 0) + 1
            elif canary not in registry:
                unregistered[p.parent.name] = unregistered.get(p.parent.name, 0) + 1
    assert not missing, f"gold graphs with no canary: {missing}"
    assert not unregistered, f"canaries on disk but not in the registry: {unregistered}"


# --------------------------------------------------------------------------- reachable
#
# Arming the registry is necessary and not sufficient. The nonce lived in its own sibling
# field, `gold_canary` -- and no code path that leaks gold would ever carry that field along
# with the text it leaked. The detector scanned every cached request for a token that appeared
# in no gold STRING, so it could not fire on the thing it exists to catch.


def test_the_nonce_is_embedded_in_a_string_a_leak_would_carry(tmp_path, monkeypatch):
    from pi_eval.build.common import write_graphs

    monkeypatch.setenv("PI_CANARY_SALT", "test-salt")
    path = write_graphs(
        tmp_path,
        "musique",
        "v1",
        [{"gold_suite": "musique", "gold_task_key": "t0", "gold_answer": "1960"}],
    )
    row = json.loads(path.read_text().splitlines()[0])
    assert row["gold_canary"] in row["gold_answer"], (
        "a nonce only in its own field is unreachable: leaking the ANSWER must carry it"
    )
    assert row["gold_answer"].startswith("1960")


def test_the_carrier_is_the_one_gold_string_that_never_reaches_a_model():
    """`gold_text` is handed to the judges as key points and emitted by `pi gold questions`;
    `gold_aliases` is appended to a gold_evidence question. Putting the nonce in either would
    make the firewall fire on a SANCTIONED path, and a detector that cries wolf is a detector
    that gets disabled. `gold_answer` is read at exactly two places in `pi_eval.score`, both
    local comparisons."""
    import pathlib

    src = pathlib.Path("src/pi_eval/build/common.py").read_text()
    assert 'row["gold_answer"] = f"{answer} {canary}"' in src
    assert 'row["gold_text"]' not in src
    assert 'row["gold_aliases"]' not in src


def test_the_nonce_is_stripped_before_the_answer_comparison(tmp_path, monkeypatch):
    """An extra reference token dilutes token-F1 and breaks exact match outright, so the whole
    scheme would trade a firewall for two wrong metrics."""
    from pi_eval.build.common import read_graphs, write_graphs
    from pi_eval.canary import strip
    from pi_eval.metrics import quality

    monkeypatch.setenv("PI_CANARY_SALT", "test-salt")
    path = write_graphs(
        tmp_path,
        "musique",
        "v1",
        [
            {
                "gold_suite": "musique",
                "gold_task_key": "t0",
                "gold_answer": "1960",
                "gold_nodes": [],
                "gold_edges": [],
                "gold_facets": [],
                "gold_seed_node_ids": [],
                "gold_graph_version": "v1",
            }
        ],
    )
    g = read_graphs(path)[0]
    assert quality.exact_match("1960", strip(g.gold_answer), ()) == 1.0
    assert quality.exact_match("1960", g.gold_answer, ()) == 0.0, (
        "the unstripped comparison really is wrong; this is not a hypothetical"
    )

    import pathlib
    import re

    score_src = pathlib.Path("src/pi_eval/score.py").read_text()
    assert "gold_answer = graph.answer" in score_src
    assert "quality.exact_match(answer.text, gold_answer" in score_src
    # THE PROPERTY, NOT THE SPELLING. This asserted one function NAME and broke when the
    # primary was decomposed into f1/precision/recall -- while the firewall property it
    # exists to defend was never in question. What matters is that EVERY quality comparison
    # is handed the stripped `gold_answer`, never `graph.gold_answer`, which still carries
    # the nonce. Checked by parsing the call sites rather than matching one of them.
    calls = re.findall(r"quality\.(\w+)\(\s*answer\.text,\s*([A-Za-z_.]+)", score_src)
    assert len(calls) >= 2, f"expected the answer comparisons, found {calls}"
    for fn, arg in calls:
        assert arg == "gold_answer", f"quality.{fn} reads {arg}, not the stripped answer"
    # Narrower, and the narrowness is the point: `graph.gold_answer` is legitimate as a
    # truthiness GUARD ("is there a gold answer at all?"). What must never happen is it
    # reaching a comparison. Comment lines are stripped first -- this repo has three past
    # bugs where a test matched a comment and passed while the code said otherwise.
    code = "\n".join(ln for ln in score_src.splitlines() if not ln.lstrip().startswith("#"))
    for ln in code.splitlines():
        if "quality." in ln:
            assert "graph.gold_answer" not in ln, f"nonce-bearing field in a comparison: {ln}"


@pytest.mark.integration
def test_a_leaked_gold_answer_is_caught_on_a_real_suite(tmp_path):
    """End to end, on musique rather than on the synthetic suite: an adapter interpolating the
    gold answer into a prompt lands in the cache, and `pi verify firewall` exits 1."""
    import pathlib
    import subprocess
    import sys

    from pi_eval.build.common import read_graphs
    from pi_run.cache import DiskCache

    gold = pathlib.Path("data/gold/graphs/musique/v1.jsonl")
    if not gold.exists():
        pytest.skip("musique gold not built on this machine")
    g = read_graphs(gold)[0]
    assert g.gold_canary and g.gold_canary in g.gold_answer

    DiskCache(tmp_path).put(
        "cd" + "0" * 62,
        {
            "model": "m",
            "request_sha": "cd" + "0" * 62,
            "text": f"Answer the question. Hint: the answer is {g.gold_answer}",
        },
    )
    out = subprocess.run(
        [sys.executable, "-m", "pi_run.cli", "verify", "firewall", "--cache-root", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).resolve().parents[1],
    )
    assert out.returncode == 1, out.stdout + out.stderr
    report = json.loads(out.stdout)
    assert report["canary_hits"], report
    assert report["ok"] is False
    # The report must not print the whole nonce back: it is what an attacker would strip.
    assert g.gold_canary not in out.stdout


# --------------------------------------------------------------------------- pointed the right way
#
# Arming the registry and embedding the nonce in a leakable string were both necessary and
# still not sufficient. `pi verify firewall` scans the RESPONSE CACHE, and the cache record
# carries the response text plus the request's SHA -- never the request (see
# `pi_run.cache._DETERMINISTIC_FIELDS`, whose minimality is what makes two racing writers
# produce byte-identical files). A gold answer interpolated into a PROMPT is therefore written
# nowhere the scan can see, so the scan could only ever have caught the rarer event of a model
# echoing gold back in its reply. The layer was armed and pointed the wrong way.


def test_the_cache_record_does_not_contain_the_request():
    """The fact the whole tripwire follows from. If this ever changes, the forensic scan
    becomes able to see a leaked prompt and this test should be revisited -- but storing a
    leaked prompt would also PERSIST the leak, which is its own problem."""
    from pi_run.cache import _record_from
    from pinq.types import CallTelemetry

    rec = _record_from(
        "a reply",
        CallTelemetry(
            call_id="c",
            actor="drafter",
            model="m",
            provider="p",
            request_sha="a" * 64,
            response_sha="b" * 64,
        ),
    )
    assert "request_sha" in rec and "text" in rec
    assert not any("messages" in str(v) for v in rec.values())


def test_the_tripwire_refuses_a_request_carrying_a_canary(monkeypatch, tmp_path):
    from pinq import tripwire

    canary = "PINQCANARY_ABCDEF0123456789"
    (tmp_path / "data" / "canaries").mkdir(parents=True)
    (tmp_path / "data" / "canaries" / "canaries.txt").write_text(canary + "\n")
    monkeypatch.setenv("PI_CANARY_ROOT", str(tmp_path))
    tripwire._cache.clear()

    known = tripwire.load(tmp_path)
    assert canary in known
    assert tripwire.scan(f"Answer. Hint: 1960 {canary}", known) == (canary,)
    assert tripwire.scan("Answer the question.", known) == ()

    with pytest.raises(tripwire.CanaryLeak, match="void"):
        tripwire.assert_clean(f"the answer is {canary}", where="a drafter request", canaries=known)
    tripwire.assert_clean("nothing to see", where="a drafter request", canaries=known)


def test_an_unregistered_lookalike_does_not_fire():
    """A tripwire that fires on lookalikes is one that gets disabled. A nonce the registry has
    never seen is not evidence of a leak from this build."""
    from pinq import tripwire

    known = frozenset({"PINQCANARY_1111111111111111"})
    assert tripwire.scan("PINQCANARY_2222222222222222", known) == ()
    assert tripwire.scan("PINQCANARY_1111111111111111", known) != ()


def test_the_common_case_costs_one_substring_scan():
    """It runs on every request of every unit of a 15,600-unit sweep. Walking 16,000 needles
    per call would make the firewall the reason the sweep is slow, and a slow guard is a guard
    someone turns off."""
    from pinq import tripwire

    big = frozenset(f"PINQCANARY_{i:016X}" for i in range(20_000))
    # No prefix anywhere: returns before it ever looks at the set.
    assert tripwire.scan("an ordinary prompt with no nonce in it", big) == ()

    import inspect

    src = inspect.getsource(tripwire.scan)
    assert "if PREFIX not in text:" in src
    assert "findall" in src, "candidates are extracted then looked up, never iterated over"


@pytest.mark.integration
def test_a_real_gold_answer_in_a_real_prompt_is_refused_before_dispatch():
    """End to end on musique, through the actual client, with the repository's own registry."""
    from pi_eval.build.common import read_graphs
    from pinq import tripwire
    from pinq.budget import BudgetLedger
    from pinq_adapters.llm.litellm_client import MeteredClient

    gold = Path("data/gold/graphs/musique/v1.jsonl")
    if not gold.exists():
        pytest.skip("musique gold not built on this machine")
    tripwire._cache.clear()
    g = read_graphs(gold)[0]
    assert g.gold_canary in g.gold_answer

    c = MeteredClient(BudgetLedger(cap=16), models={"drafter": "stub/echo"})
    with pytest.raises(tripwire.CanaryLeak):
        c.complete(
            role="drafter",
            seed=0,
            max_tokens=8,
            messages=[{"role": "user", "content": f"Answer this. Hint: {g.gold_answer}"}],
        )


# --------------------------------------------------------------------------- layer 4 evidence
#
# Layer 4 is the only firewall layer that catches gold leaking as a STRING, and
# `tripwire.assert_armed`'s own docstring says an unarmed layer is indistinguishable from a
# clean one because `scan` returns () either way. These three tests are about the ARTEFACT: a
# run that cannot state whether the layer could fire leaves the project's "all four layers stay
# alive" claim true of the code and unverifiable in the data.


def _layer4_manifest(**kw):
    """The smallest real manifest, built the way `pi run` builds one."""
    from pi_run.manifest import build_manifest

    base = dict(
        suite_id="musique",
        task_id="t1",
        arm_id="inquirer_prompted",
        policy_id="p",
        seed=3,
        corpus_hash="c0ffee",
        budget_cap=8,
        max_turns=16,
        word_cap=300,
        code_version="deadbeef",
        dirty=False,
    )
    base.update(kw)
    return build_manifest(**base)


def _layer4_dict(**kw):
    from pi_run.manifest import manifest_to_dict

    return manifest_to_dict(_layer4_manifest(**kw))


def _point_layer4_at(root, monkeypatch, nonces=None):
    """Point layer 4 at `root`, optionally writing `nonces` there, and re-run its check."""
    from pinq import tripwire

    if nonces is not None:
        reg = root / tripwire.REGISTRY
        reg.parent.mkdir(parents=True, exist_ok=True)
        reg.write_text("\n".join(nonces))
    monkeypatch.setenv("PI_CANARY_ROOT", str(root))
    tripwire._cache.clear()
    tripwire.reset_status()
    return tripwire


def test_the_manifest_says_whether_layer_4_was_armed_not_only_that_it_found_nothing(
    tmp_path, monkeypatch
) -> None:
    """The manifest recorded `firewall_ok`, `canary_hit` and `gold_exposed` -- and all three read
    IDENTICALLY for a run with layer 4 armed and no leak as for a run with the layer never armed
    at all. So no run on disk carried evidence the layer was live, and it could not be
    established after the fact.

    Measured 2026-09-17, which is why this is not hypothetical: from the main checkout
    `armed()` is `(True, "17844 canaries from .../data/canaries/canaries.txt")`, while from a
    pinned worktree -- the launch pattern this project adopted for every campaign, and which has
    no `data/` of its own -- it is `(False, "no canary registry at ...")` AND `assert_armed()`
    returns without raising, because it is deliberately conditioned on gold being built at that
    root. That condition is sound for a fresh clone and silent for a worktree sweep.

    The count comes from the arming check itself rather than being recomputed at write time, so
    a manifest cannot report the layer armed on a different reading than the one that guarded
    the unit.
    """
    from pinq import tripwire

    nonces = [f"{tripwire.PREFIX}_{i:016X}" for i in range(7)]
    tw = _point_layer4_at(tmp_path / "armed", monkeypatch, nonces)
    ok, why = tw.armed()
    assert ok and "7 canaries" in why
    armed = _layer4_dict()

    # Same world, except layer 4 has no registry to load: nothing leaked here either.
    tw = _point_layer4_at(tmp_path / "nowhere", monkeypatch)
    ok2, why2 = tw.armed()
    assert not ok2 and "no canary registry" in why2
    unarmed = _layer4_dict()

    # The three existing flags agree in both worlds. That agreement IS the defect.
    for flag in ("firewall_ok", "canary_hit", "gold_exposed"):
        assert armed[flag] == unarmed[flag], f"{flag} was expected to be uninformative here"

    # So the count has to be recorded, and the two runs have to be tellable apart.
    assert armed["canary_nonces_loaded"] == 7
    assert unarmed["canary_nonces_loaded"] == 0
    assert armed != unarmed


def test_an_unavailable_nonce_count_is_recorded_as_unknown_not_as_zero(monkeypatch) -> None:
    """A zero and an unknown are different facts and the paper needs them apart: 0 means the
    arming check RAN and layer 4 could not fire, which is the alarming case, while null means no
    arming check ran in this process, so the manifest declines to state it rather than defaulting
    to a number that would read as a measurement. Defaulting either way manufactures exactly the
    false confidence this field exists to remove.
    """
    from pinq import tripwire

    tripwire.reset_status()
    assert tripwire.last_status() is None
    assert _layer4_dict()["canary_nonces_loaded"] is None


def test_the_nonce_count_is_outside_run_identity(tmp_path, monkeypatch) -> None:
    """`semantic_hash` is run identity. How many needles the tripwire had loaded is a property of
    the LAUNCH ENVIRONMENT, not of the experiment: inside identity it would give one experiment
    two run_ids depending on where it was launched, renaming every run directory and making
    `--resume` re-roll finished work. Same argument that keeps `dirty`, `canary_hit` and
    `firewall_ok` out of it.
    """
    from pinq import tripwire
    from pinq.ids import SEMANTIC_FIELDS

    assert "canary_nonces_loaded" not in SEMANTIC_FIELDS

    tw = _point_layer4_at(
        tmp_path / "a", monkeypatch, [f"{tripwire.PREFIX}_{i:016X}" for i in range(3)]
    )
    tw.armed()
    armed_h = _layer4_manifest().semantic_hash

    tw = _point_layer4_at(tmp_path / "b", monkeypatch)
    tw.armed()
    assert _layer4_manifest().semantic_hash == armed_h, "the nonce count moved run identity"
