"""The gold firewall, asserted mechanically.

These tests are the reason a leak becomes a red build instead of a wrong paper.
"""

import dataclasses
from pathlib import Path

import pytest

from pi_eval.gold import GoldAccessError, GoldEdge, GoldGraph, GoldNode, gold_root
from pinq.types import State, TaskView
from pinq.view import ALLOWED, LeakageError, make_view

pytestmark = pytest.mark.firewall


def _good_kwargs(**over):
    kw = dict(
        task_id="t1",
        suite_id="synth",
        question="Who founded the company that makes the Model S?",
        instructions="Answer briefly.",
        corpus_id="synth_v1",
        corpus_hash="deadbeef",
        word_cap=60,
    )
    kw.update(over)
    return kw


def test_make_view_accepts_exactly_the_allowlist():
    v = make_view(**_good_kwargs())
    assert isinstance(v, TaskView)
    assert set(ALLOWED) == set(TaskView.__dataclass_fields__)


@pytest.mark.parametrize("gold_field", sorted(GoldNode.__dataclass_fields__))
def test_make_view_raises_on_every_gold_node_field(gold_field):
    """The strong form: not 'raises on some unknown key' but 'raises on EVERY gold field'."""
    with pytest.raises(LeakageError):
        make_view(**_good_kwargs(**{gold_field: "leaked"}))


@pytest.mark.parametrize("gold_field", sorted(GoldEdge.__dataclass_fields__))
def test_make_view_raises_on_every_gold_edge_field(gold_field):
    with pytest.raises(LeakageError):
        make_view(**_good_kwargs(**{gold_field: "leaked"}))


def test_taskview_and_gold_share_no_field_name():
    """A same-named field would let a leak typecheck. The intersection must be empty."""
    for gold_cls in (GoldNode, GoldEdge, GoldGraph):
        overlap = set(TaskView.__dataclass_fields__) & set(gold_cls.__dataclass_fields__)
        assert overlap == set(), f"{gold_cls.__name__} shares field names with TaskView: {overlap}"


def test_taskview_and_gold_share_no_base_class():
    """No common ancestor besides object, so isinstance() can never confuse them."""
    for gold_cls in (GoldNode, GoldEdge, GoldGraph):
        assert set(TaskView.__mro__) & set(gold_cls.__mro__) == {object}


def test_taskview_has_no_catchall_mapping():
    """A dict-typed field would be a hiding place for an unvetted key."""
    for f in dataclasses.fields(TaskView):
        assert "dict" not in str(f.type).lower(), f"{f.name} is a catch-all"
        assert "Any" not in str(f.type), f"{f.name} is untyped"


BANNED_ON_STATE = frozenset(
    {"budget", "cap", "remaining", "spent", "ledger", "max_turns", "steps_left", "arm_id"}
)


def _reachable_from(root) -> list[tuple[str, str]]:
    """(dotted path, field name) for every dataclass field reachable from `root`.

    Follows the ANNOTATIONS, including through tuple[...] and | None, so a banned field cannot
    hide one hop down. `typing.get_type_hints` rather than `f.type` because
    `from __future__ import annotations` makes every f.type a string.
    """
    import dataclasses
    import typing

    out: list[tuple[str, str]] = []
    seen: set[type] = set()

    def walk(t, path: str) -> None:
        if t in seen or not dataclasses.is_dataclass(t):
            return
        seen.add(t)
        hints = typing.get_type_hints(t)
        for f in dataclasses.fields(t):
            out.append((f"{path}.{f.name}", f.name))
            ann = hints.get(f.name, f.type)
            for arg in typing.get_args(ann) or (ann,):
                if dataclasses.is_dataclass(arg):
                    walk(arg, f"{path}.{f.name}")

    walk(root, root.__name__)
    return out


def test_state_has_no_budget_field():
    """A budget-aware policy breaks prefix-exchangeability and confounds STOP with the cap.

    TRANSITIVELY, WHICH THIS DID NOT DO. It checked `State` and `TaskView` while its own comment
    said "it must not be reachable transitively through the view either" -- so a banned field on
    Turn, Action, Evidence, EvidenceUnit, Draft or Usage, all of them reachable from
    `State.history` and `State.evidence`, would have passed. 51 fields are reachable from State
    and the old assertion looked at 12.
    """
    hits = [p for p, name in _reachable_from(State) if name in BANNED_ON_STATE]
    assert not hits, f"budget information reachable from State: {hits}"


def test_the_one_spend_signal_on_state_is_deliberate():
    """`State.history[*].usage` carries tok_*, usd, wall_ms and n_calls, so a policy CAN see
    what it has spent so far. Recorded rather than removed, because the invariant is about the
    CAP, not about spend:

      * CONTRIBUTING.md's reason is that a budget-aware policy "confounds STOP with the cap you
        announced". Usage says what was spent; it does not say what the cap is, so a policy
        cannot stop at a cap it cannot see.
      * Prefix-exchangeability survives: the first k turns of a long rollout carry exactly the
        usage the same k turns of a short run would, because usage is stamped per turn and
        accumulates identically. Prefix-k really is a true k-step run.

    This test exists so that reasoning is written down at the place a reader will check it, and
    so that ADDING a cap or a remaining-budget field next to `usage` fails loudly."""
    paths = {p for p, _ in _reachable_from(State)}
    assert "State.history.usage.usd" in paths, "if this moves, re-read the argument above"
    assert not {p for p in paths if p.endswith((".cap", ".remaining", ".budget"))}


def test_gold_root_raises_without_env(monkeypatch):
    """In a rollout worker PI_GOLD_ROOT is unset and this raise IS the firewall."""
    monkeypatch.delenv("PI_GOLD_ROOT", raising=False)
    with pytest.raises(GoldAccessError):
        gold_root()


def test_gold_root_returns_path_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path))
    assert gold_root() == tmp_path


def test_word_cap_must_be_positive_int():
    for bad in (0, -1, "60", 60.0, None):
        with pytest.raises(LeakageError):
            make_view(**_good_kwargs(word_cap=bad))


def test_missing_field_raises():
    kw = _good_kwargs()
    del kw["corpus_hash"]
    with pytest.raises(LeakageError):
        make_view(**kw)


# --------------------------------------------------------------------------- layer 4 fails CLOSED
#
# The tripwire resolved its registry against `Path.cwd()` and swallowed the resulting OSError
# into an empty frozenset -- which it then MEMOIZED. `scan` returns () on an empty set, so a
# worker whose cwd was not the repository had no canaries, reported clean on every request, and
# was indistinguishable from one that had checked. Measured before the fix, from /private/tmp:
# `load()` -> 0 nonces and a planted real leak -> (). After: 16,751 nonces and the leak matched.


def test_the_tripwire_arms_from_a_foreign_working_directory(tmp_path, monkeypatch):
    """The registry belongs to the REPOSITORY, not to whatever directory a worker started in."""
    from pinq import tripwire

    monkeypatch.chdir(tmp_path)
    tripwire._cache.clear()
    monkeypatch.delenv("PI_CANARY_ROOT", raising=False)
    ok, why = tripwire.armed()
    assert ok, why
    assert "canaries.txt" in why and str(tmp_path) not in why


def test_an_unfindable_registry_is_refused_and_not_treated_as_clean(tmp_path, monkeypatch):
    """Fail CLOSED. An unarmed layer 4 and a clean layer 4 return the same () from `scan`, so
    the distinction has to be drawn once at unit start or it is never drawn at all."""
    from pinq import tripwire

    (tmp_path / "data" / "gold" / "graphs").mkdir(parents=True)  # gold is BUILT
    monkeypatch.setenv("PI_CANARY_ROOT", str(tmp_path))
    tripwire._cache.clear()
    assert tripwire.armed(str(tmp_path))[0] is False
    with pytest.raises(tripwire.CanaryLeak, match="cannot fire"):
        tripwire.assert_armed(str(tmp_path))


def test_a_checkout_with_no_gold_is_still_runnable(tmp_path):
    """A guard that fires on a fresh clone with nothing to leak is a guard someone deletes."""
    from pinq import tripwire

    tripwire.assert_armed(str(tmp_path))  # no data/gold -> nothing to protect


def test_the_worker_refuses_a_unit_when_layer_4_cannot_fire(tmp_path, monkeypatch):
    """assert_firewall is the one function all four rollout entry points already call."""
    from pi_run.worker import assert_firewall
    from pinq import tripwire

    (tmp_path / "data" / "gold").mkdir(parents=True)
    monkeypatch.setenv("PI_CANARY_ROOT", str(tmp_path))
    tripwire._cache.clear()
    with pytest.raises(tripwire.CanaryLeak):
        assert_firewall({})


# --------------------------------------------- layer 4 and the pinned-worktree launch pattern
#
# Every campaign is launched from a PINNED WORKTREE, because a dirty tree stamps every run
# `dev-` and `dev-` runs are training-only. A worktree has no `data/` of its own: `.gitignore`
# excludes `data/gold/` and `data/canaries/`, so this is structural rather than an oversight.
#
# Measured 2026-09-18, from /private/tmp/gridflag-wt with that worktree's own `src` on
# PYTHONPATH and PI_CANARY_ROOT unset: `_registry_path()` -> `<worktree>/data/canaries/
# canaries.txt`, `load()` -> 0 nonces, `assert_clean` on a REAL registered nonce returned
# without raising, and `assert_armed()` returned without raising -- because `<worktree>/data/
# gold` is absent too, which the gold condition reads as "nothing built, nothing to leak".
#
# So the standard operating procedure disarmed the only layer that catches a leak through a
# STRING, and `canary_hit: False` on such a run was entailed by the launch root rather than
# measured. These two tests hold the two halves: a worktree must ARM itself from the tree that
# owns the registry, and when it cannot it must REFUSE rather than report clean.


def _worktree_probe(tmp_path, *, canaries, gold):
    """Run layer 4's own checks inside a REAL git worktree and return what they saw.

    Not a simulation. `git worktree add` writes the `.git` FILE and the `commondir` that
    resolution has to follow, `.gitignore` keeps `data/` out of the worktree exactly as in this
    repository, and the worktree gets its own copy of `src` -- which is what makes
    `Path(__file__)` resolve inside it, and is the whole mechanism.
    """
    import json
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    from pinq import tripwire

    main = tmp_path / "main"
    (main / "src").mkdir(parents=True)
    shutil.copytree(Path(tripwire.__file__).resolve().parent, main / "src" / "pinq")
    (main / "pyproject.toml").write_text("[project]\nname = 'faux'\nversion = '0'\n")
    (main / ".gitignore").write_text("data/\n")  # as in this repository
    if canaries is not None:
        reg = main / tripwire.REGISTRY
        reg.parent.mkdir(parents=True)
        reg.write_text("\n".join(canaries) + "\n")
    if gold:
        (main / "data" / "gold" / "graphs" / "musique").mkdir(parents=True)

    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(main)]
    for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-qm", "init"]):
        subprocess.run(git + args, check=True, capture_output=True)
    wt = tmp_path / "wt"
    subprocess.run(git + ["worktree", "add", "-q", "--detach", str(wt)], check=True)
    assert not (wt / "data").exists(), "the worktree was supposed to carry no data/"

    probe = (
        "import json\n"
        "from pinq import tripwire\n"
        "needle = %r\n"
        "st = tripwire.status()\n"
        "raised = None\n"
        "try:\n"
        "    tripwire.assert_armed()\n"
        "except tripwire.CanaryLeak as e:\n"
        "    raised = str(e)\n"
        "leak = None\n"
        "try:\n"
        "    tripwire.assert_clean('answer. hint: ' + needle, where='a probe request')\n"
        "except tripwire.CanaryLeak as e:\n"
        "    leak = str(e)\n"
        "print(json.dumps({'module': tripwire.__file__, 'registry': str("
        "tripwire._registry_path()), 'n': st.n_canaries, 'ok': st.ok, 'why': st.why,\n"
        "    'armed_raised': raised, 'leak_raised': leak}))\n"
    ) % ((canaries or ["PINQCANARY_0000000000000000"])[0],)

    env = {k: v for k, v in os.environ.items() if k != "PI_CANARY_ROOT"}
    env["PYTHONPATH"] = str(wt / "src")
    r = subprocess.run(
        [sys.executable, "-c", probe], cwd=wt, env=env, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["module"].startswith(str(wt)), out["module"]  # the WORKTREE's copy ran
    return out


def test_a_worktree_launch_arms_layer_4_from_the_tree_that_owns_the_registry(tmp_path):
    """The default has to be the safe one. A worktree is not a fresh clone: it is a checkout of
    a repository whose registry exists and is one `commondir` hop away, and the launch pattern
    every campaign uses must not be the one that disarms the firewall."""
    nonces = [f"PINQCANARY_{i:016X}" for i in range(7)]
    out = _worktree_probe(tmp_path, canaries=nonces, gold=True)
    assert out["n"] == 7, out  # was 0: the worktree's own absent registry
    assert out["ok"], out["why"]
    assert out["armed_raised"] is None, out["armed_raised"]
    assert out["leak_raised"] is not None, "a REAL registered nonce passed the tripwire"


def test_a_worktree_launch_with_no_registry_anywhere_refuses_instead_of_reporting_clean(
    tmp_path,
):
    """Gold built, registry nowhere: the unarmed scanner must stop the unit. Fail CLOSED means
    the missing registry cannot be absorbed as "nothing to leak" just because THIS directory
    happens to hold neither the gold nor the nonces."""
    out = _worktree_probe(tmp_path, canaries=None, gold=True)
    assert out["n"] == 0 and not out["ok"], out
    assert out["armed_raised"] is not None, (
        "layer 4 could not fire and the unit was allowed to start anyway"
    )
    assert "cannot fire" in out["armed_raised"]


def test_a_mistyped_canary_root_refuses_rather_than_reading_as_nothing_to_leak(
    tmp_path, monkeypatch
):
    """The opt-in that armed one lane is also a fail-open: PI_CANARY_ROOT pointing somewhere that
    holds neither the nonces nor the gold used to be absorbed as "nothing built, nothing to
    leak", which is the identical defect as the worktree case with a typo in place of a missing
    directory. Where the nonces are looked for and whether gold exists at all are two questions,
    and only the first is answered by that variable."""
    from pinq import tripwire

    monkeypatch.setenv("PI_CANARY_ROOT", str(tmp_path / "typo"))  # nothing here at all
    monkeypatch.chdir(tmp_path)
    tripwire._cache.clear()
    repo = Path(tripwire.__file__).resolve().parents[2]  # src/pinq/tripwire.py -> the repo
    if not (repo / "data" / "gold").is_dir():
        pytest.skip("no gold built in the repository this test runs from")
    assert tripwire.armed()[0] is False
    with pytest.raises(tripwire.CanaryLeak, match="cannot fire"):
        tripwire.assert_armed()


# ------------------------------------- layer 4 on the path the artifact actually reproduces on
#
# `tripwire.assert_clean` had exactly ONE call site, inside `MeteredClient.complete`.
# `CachingClient.complete` returns from the cache BEFORE reaching it, and `ReplayClient` never
# calls through at all -- so on the path docs/REPRODUCE.md ships as the artifact ("given the
# cache, every number regenerates with no network"), the only layer that catches a gold string
# reaching a model scanned nothing at all.


class _InnerLLM:
    def pin(self, role):
        return type(
            "P", (), {"model": "m", "provider": "p", "base_url_sha": "", "as_dict": lambda s: {}}
        )()

    def complete(self, *, role, messages, seed, max_tokens=None, **kw):
        from pinq.types import CallTelemetry

        return "innocent", CallTelemetry(
            call_id="c",
            actor=kw.get("actor", role),
            model="m",
            provider="p",
            request_sha="y",
            response_sha="x",
            tok_prompt=1,
            tok_completion=1,
            tok_reasoning=0,
            tok_cached=0,
            usd=0.0,
            wall_ms=1,
            cache_hit=False,
            http_status=200,
        )


def _real_nonce():
    from pinq import tripwire

    reg = tripwire._registry_path()
    if not reg.exists():
        pytest.skip("canary registry not built")
    return reg.read_text().split()[0]


@pytest.mark.parametrize("cls_name", ["CachingClient", "ReplayClient"])
def test_a_canary_in_a_cached_or_replayed_request_is_caught(tmp_path, cls_name):
    """A REAL registered nonce, through the real client stack."""
    from pi_run import cache as cache_mod
    from pinq import tripwire

    nonce = _real_nonce()
    cache = cache_mod.DiskCache(tmp_path)
    clean = [{"role": "user", "content": "an ordinary question"}]
    cache_mod.CachingClient(_InnerLLM(), cache).complete(role="drafter", messages=clean, seed=0)

    cli = getattr(cache_mod, cls_name)(_InnerLLM(), cache)
    # the clean request still works on the hit/replay path
    assert cli.complete(role="drafter", messages=clean, seed=0)[0] == "innocent"
    with pytest.raises(tripwire.CanaryLeak):
        cli.complete(
            role="drafter", messages=[{"role": "user", "content": f"gold is {nonce}"}], seed=0
        )


def test_every_client_that_can_dispatch_scans_the_request():
    """A structural guard: a new client wrapper that forgets the scan is the way this comes
    back. Each of the three has its own reason to be missed -- MeteredClient dispatches,
    CachingClient short-circuits it, ReplayClient never calls it."""
    import inspect

    from pi_run import cache as cache_mod
    from pinq_adapters.llm import litellm_client

    for fn in (
        cache_mod.CachingClient.complete,
        cache_mod.ReplayClient.complete,
        litellm_client.MeteredClient.complete,
    ):
        assert "assert_clean" in inspect.getsource(fn), fn.__qualname__


def test_verify_firewall_names_the_suites_layer_4_cannot_fire_on():
    """A single `canaries_configured: 16752` reads as uniform coverage and is not.

    `write_graphs` threads the nonce into `gold_answer`, chosen because it is the one gold
    string that NEVER legitimately reaches a model. tau2 and drgym have no `gold_answer`: their
    gold is entirely needs, and their only leakable string is `gold_text`, which reaches a model
    twice over -- the judges are handed gold_text key points, and `pi gold questions` emits
    gold_text to seed both ceiling arms. A nonce there would abort judging and both ceiling arms
    on their first call.

    So layer 4 cannot fire on a text leak on those two suites. Measured 2026-09-17 (per-suite row
    counts carrying the nonce): these are two of the tool-use / transfer suites -- drgym, tau2,
    tau2_airline, tau2_retail, tau2_telecom -- NOT the primary QA endpoint; frames, musique,
    strategyqa, synth and wiki2 all carry the nonce in gold_answer. That is a fact about the
    experiment, reported rather than left to be inferred."""
    from pi_run.cli import _canary_carriers

    carriers = _canary_carriers()
    if not carriers:
        pytest.skip("gold not built")
    assert carriers.get("musique") == "gold_answer"
    for blind in ("tau2", "drgym"):
        if blind in carriers:
            assert carriers[blind].startswith("NONE"), (
                f"{blind} gained a carrier; if gold_text now carries the nonce, judging and "
                "the ceiling arms must have been re-checked"
            )


# ----------------------------------------------- layer 4 asked about the wrong gold, and one
# ----------------------------------------------- process that holds gold never armed it at all
#
# TWO DEFECTS, one referent and one call site.
#
# THE REFERENT. `assert_armed` derived the tree it tested for gold from the REGISTRY's own root
# (`_registry_path(root).parents[2]`), so it asked "does gold sit beside the registry?" rather
# than "is the gold this process can actually reach covered by a loaded registry?". Nothing in it
# consulted PI_GOLD_ROOT -- the variable that names the gold in PLAY and the one `pi_eval.gold`
# resolves against. Measured at a4422c0 with the real gold path exported and a registry-less
# root: `assert_armed()` returned, `load()` was 0 needles, and `scan` of a string carrying a real
# REGISTERED nonce returned (). The path arithmetic was right and the referent was wrong.
#
# THE CALL SITE. There was exactly one, in `pi_run.worker.assert_firewall`. The scoring and
# judging process -- the one that holds gold in memory by construction, and the one that hands
# gold-derived text to a judge -- never armed the layer. Measured at a4422c0 in a process with
# PI_GOLD_ROOT set and 12,576 wiki2 graphs loaded: `assert_clean` over a real judge payload
# containing a registered nonce RETURNED CLEAN, and `last_status()` was None both before and
# after, so no artefact could show the layer had been inert.
#
# The silent pass on a fresh checkout stays: a tree with no gold built has nothing to leak, and
# a guard that fires on a clean clone is a guard someone deletes. What changes is which gold the
# question is about.


def _registered_nonce_or_skip():
    """A nonce the REAL registry has actually seen. A lookalike would prove nothing: `scan`
    deliberately ignores unregistered tokens, so a test built on one could pass on a layer that
    can never fire on anything real."""
    from pinq import tripwire

    for cand in Path(__file__).resolve().parents:
        reg = cand / tripwire.REGISTRY
        if reg.is_file():
            toks = reg.read_text().split()
            if toks:
                return cand, toks[0], len(toks)
    pytest.skip("no canary registry built in this tree")


def test_layer_4_refuses_when_the_gold_IN_PLAY_has_no_registry_anywhere(tmp_path, monkeypatch):
    """PROPERTY ONE. A process that can reach gold and has no loaded registry must refuse.

    The gold here is reachable through PI_GOLD_ROOT and lives nowhere near the registry root,
    which is the shape of every campaign this project launches: a pinned worktree with no
    `data/` of its own, scoring against the main checkout's gold.
    """
    from pinq import tripwire

    gold = tmp_path / "elsewhere" / "data" / "gold"
    (gold / "graphs").mkdir(parents=True)
    registryless = tmp_path / "pinned-worktree"
    registryless.mkdir()

    monkeypatch.setenv("PI_GOLD_ROOT", str(gold))
    monkeypatch.setenv("PI_CANARY_ROOT", str(registryless))
    tripwire._cache.clear()
    tripwire.reset_status()

    assert tripwire.armed()[0] is False, "precondition: the layer cannot fire"
    with pytest.raises(tripwire.CanaryLeak, match="cannot fire"):
        tripwire.assert_armed()


def test_layer_4_still_passes_when_no_gold_is_reachable_at_all(tmp_path):
    """PROPERTY TWO, and the reason the fix is not "make absence fatal".

    A fresh checkout has no gold built, nothing to leak, and must stay runnable. It holds at
    a4422c0 too, so this is a regression guard on the fix rather than a demonstration of the bug
    -- and it is here because property one is exactly the change that could break it.

    IT HAS TO BE ISOLATED IN A SUBPROCESS, and the first draft was not. That draft set
    PI_CANARY_ROOT to an empty tmp_path and asked in-process, which read as "no gold reachable"
    only because the referent was then a single configured root. Once the referent became every
    tree this process can name UNION the one PI_GOLD_ROOT points at -- which is the whole fix --
    that state stopped meaning "nothing built": the repository's own `data/gold` is reachable
    from any test running inside it, so refusing became the correct answer and the draft failed.
    Measured, on the merged version: `CanaryLeak: gold is reachable under <repo>/data/gold`. The
    property was right and the isolation was fake, so the isolation is now real -- a git worktree
    of a faux repository with no gold and no registry anywhere, which is what a fresh checkout
    actually is. Same harness as the worktree tests above, whose two halves this completes.
    """
    out = _worktree_probe(tmp_path, canaries=None, gold=False)
    assert out["n"] == 0 and not out["ok"], out
    assert out["armed_raised"] is None, (
        f"a tree with no gold anywhere must stay runnable: {out['armed_raised']}"
    )
    assert out["leak_raised"] is None, "and with no needles there is nothing for it to catch"


def test_layer_4_finds_the_registry_WHERE_THE_GOLD_IS(tmp_path, monkeypatch):
    """The resolution, and why the fix cannot be "export a variable".

    PI_CANARY_ROOT is set NOWHERE in this codebase and the one call site passed no argument, so
    a fix that depends on someone exporting it fails the same way for the same reason. The
    builder settles it: `pi_eval.build.common.write_graphs` calls `register(minted, root)`, which
    writes `<root>/data/canaries/canaries.txt` -- beside `<root>/data/gold`. So the registry for
    a gold tree is findable FROM that gold tree, with no configuration at all.
    """
    from pi_eval.canary import register
    from pinq import tripwire

    tree = tmp_path / "built"
    (tree / "data" / "gold" / "graphs").mkdir(parents=True)
    nonces = {f"{tripwire.PREFIX}_{i:016X}" for i in range(5)}
    register(nonces, tree)

    # cwd and the repo walk-up know nothing about `tree`; only the gold path leads to it.
    monkeypatch.setenv("PI_GOLD_ROOT", str(tree / "data" / "gold"))
    monkeypatch.delenv("PI_CANARY_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    tripwire._cache.clear()
    tripwire.reset_status()

    st = tripwire.status()
    assert st.ok, st.why
    assert st.n_canaries == 5, st
    tripwire.assert_armed()


def test_layer_4_actually_FIRES_on_a_registered_nonce_once_armed(tmp_path, monkeypatch):
    """PROPERTY THREE, the non-vacuity check.

    Properties one and two can both pass on a layer that cannot detect anything: `scan` returns
    () for "armed and clean" and for "no needles loaded" alike, which is the whole reason
    `assert_armed` exists. So plant a REAL registered nonce and require a hit. Without this, a
    clean pass is indistinguishable from a false zero.
    """
    from pi_eval.canary import register
    from pinq import tripwire

    _repo, nonce, n_real = _registered_nonce_or_skip()

    # AN ISOLATED TREE HOLDING EXACTLY ONE REAL NONCE, and the count is the discriminator.
    # Pointing PI_GOLD_ROOT at the repository's own gold would prove nothing: the repo walk-up
    # finds that registry too, so the test would pass without the gold-derived resolution ever
    # running. Written the first time, it did exactly that. Here only the gold path leads to
    # this registry, and `n_canaries == 1` fails loudly if resolution fell back to the
    # repository's own registry.
    tree = tmp_path / "gold-elsewhere"
    (tree / "data" / "gold" / "graphs").mkdir(parents=True)
    register({nonce}, tree)

    monkeypatch.setenv("PI_GOLD_ROOT", str(tree / "data" / "gold"))
    monkeypatch.delenv("PI_CANARY_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    tripwire._cache.clear()
    tripwire.reset_status()

    st = tripwire.status()
    assert st.ok, st.why
    assert st.n_canaries == 1, f"resolved the wrong registry: {st} (the repo holds {n_real})"
    assert str(tree) in st.why, st.why
    assert tripwire.scan(f"the answer is {nonce}") == (nonce,), "the scan must actually fire"
    assert tripwire.scan("nothing to see here") == ()
    with pytest.raises(tripwire.CanaryLeak, match="void"):
        tripwire.assert_clean(f"the answer is {nonce}", where="a judge request")


def test_the_scoring_process_arms_layer_4(tmp_path, monkeypatch):
    """DEFECT TWO. `pi_eval.score.score` is the process that holds gold by construction.

    It resolves `gold_root()`, loads every graph into memory and hands gold-derived text to the
    judges. It never armed layer 4, so a scoring pass whose registry was unfindable scanned
    every judge request against zero needles and recorded nothing about it. Arming there is the
    half that matters most, because that is where gold is held on purpose rather than by
    accident.
    """
    from pi_eval.build.synth_build import build
    from pi_eval.score import score
    from pi_run.compact import compact
    from pi_run.worker import UnitSpec, run_unit
    from pinq import tripwire

    corpus, _gold, _chash = build(n_tasks=4, n_facets=2, depth=2, root=tmp_path)
    runs_root = tmp_path / "runs"
    run_unit(
        UnitSpec(
            suite_id="synth",
            corpus_dir=str(corpus.parent),
            task_id="s0",
            arm_id="fake_chain",
            seed=0,
            runs_root=str(runs_root),
            cache_root=str(tmp_path / "cache"),
            code_version="t",
            dirty=False,
        )
    )
    compact(runs_root, tmp_path / "parquet", include_dev=True)

    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
    monkeypatch.delenv("PI_CANARY_ROOT", raising=False)
    tripwire._cache.clear()
    tripwire.reset_status()
    assert tripwire.last_status() is None, "precondition: nothing has armed the layer yet"

    score(
        parquet_dir=tmp_path / "parquet",
        runs_root=runs_root,
        corpora_root=tmp_path / "data" / "corpora",
    )

    st = tripwire.last_status()
    assert st is not None, "the scoring process must ARM layer 4, not merely scan with it"
    assert st.ok and st.n_canaries > 0, st
