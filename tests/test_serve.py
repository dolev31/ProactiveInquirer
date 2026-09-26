"""The HTTP seam: /rollout, /score, /retrieve, /healthz.

The refusal is the feature, so these tests are mostly about refusing. Two of them exist
because a docstring elsewhere promises they do: `pi_run.serve.score` says its duplicated
`EVAL_ONLY_SUITES` is pinned equal to `pinq_train.split`'s, and `pinq_train.client` says its
duplicated status codes are pinned against the server's. A duplication defended by a comment
is a duplication that drifts; a duplication defended by a test is a duplication.

Everything here runs without a server, a port, or `fastapi`: the handlers are plain functions
over dataclasses, which is what lets the split refusal be tested as a unit. A refusal that can
only be tested through a socket is a refusal nobody runs in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_run.serve import score as serve_score
from pi_run.serve.app import (
    FROZEN_ROLES_STATUS,
    NO_GOLD_ROOT_STATUS,
    SPLIT_REFUSED_STATUS,
    ServerConfig,
)
from pi_run.serve.retrieve import RetrieverPool, SuiteNotConfigured, handle_retrieve
from pi_run.serve.rollout import FrozenRolesWouldMove, check_base_url
from pi_run.serve.score import SplitRefused, assert_scorable, handle_score, scorer_hash
from pinq.wire import SPLIT_REFUSED_EXIT, RetrieveRequest, RolloutRequest, ScoreRequest
from pinq_train import client as train_client
from pinq_train.split import EVAL_ONLY_SUITES as TRAIN_EVAL_ONLY

# --------------------------------------------------------------------- the pinned duplications


def test_the_servers_eval_only_list_equals_the_trainers():
    """One definition, reached from both sides, checked by identity.

    THIS TEST CHANGED, AND THE BELIEF IT ENCODED WAS THE WRONG ONE. It used to pin that two
    retyped literals happened to be equal, because `score.py` may not import `pinq_train`
    (contract 4) and both files therefore carried their own copy. The premise was incomplete:
    `pinq.splitting` also defines the set, is stdlib-only, and is importable from both -- so
    the duplication was never required, only assumed. Both now delegate there.

    Identity rather than equality is the point. Two literals that match today are two
    literals that can drift tomorrow and pass an equality check for a whole release; `is`
    fails the moment someone re-types one, which is the failure this test exists to catch.
    """
    from pinq.splitting import EVAL_ONLY_SUITES as CANON

    assert serve_score.EVAL_ONLY_SUITES is CANON
    assert TRAIN_EVAL_ONLY is CANON
    # THE LITERAL MOVED, AND THE BELIEF THAT WAS WRONG WAS "there are two of these". It was
    # true when written and is not a property of the design: `userbench` is a third eval-only
    # suite, and for a stronger reason than either of the first two (upstream ships 2,651
    # TRAIN scenarios, so training on it is one config line away). The equality is kept, not
    # loosened to a subset check, because its whole job is to make ADDING or REMOVING an
    # eval-only suite a deliberate three-line edit rather than something that happens to a
    # frozenset in passing.
    #
    # Updated again for `frames`. `frames` is the fourth, and its reason is the plainest of the four: FRAMES ships a single `test` split of 824 questions and there is no train split upstream at all. Declaring it rather than relying on that absence is the point -- "upstream ships no train split" is a fact about today's dataset card, and this is a fact about the code.
    #
    # Updated again on 2026-09-15 for `drgym`, and this is the edit the paragraph above was
    # written for. The other three literals that pin this set all spell `EVAL_ONLY_SUITES`
    # and a grep for that name finds them; this one compares against `CANON` and the grep
    # missed it, so the full suite is what caught it. That is the test working exactly as
    # described -- a deliberate edit, refused until someone makes it -- and it is the reason
    # the assertion stays an equality rather than becoming `>=`.
    #
    # Updated again on 2026-09-23 for `musique_x2`, the composed MuSiQue pairs: every task joins
    # two held-out MuSiQue TEST tasks, so a training row from it would put test content into a
    # training file. The belief that was wrong is "there are five"; the equality stays.
    assert CANON == frozenset({"tau2", "pare", "userbench", "frames", "drgym", "musique_x2"})


def test_the_clients_status_codes_equal_the_servers():
    """Two processes that must not share an import must still agree on three integers."""
    assert train_client.SPLIT_REFUSED_STATUS == SPLIT_REFUSED_STATUS == 403
    assert train_client.FROZEN_ROLES_STATUS == FROZEN_ROLES_STATUS == 409
    assert train_client.NO_GOLD_ROOT_STATUS == NO_GOLD_ROOT_STATUS == 503


# --------------------------------------------------------------------------- the split refusal


def test_an_eval_only_suite_is_refused_by_name_before_any_hashing():
    """tau2 is the zero-shot transfer target. "It happened to hash into train" is not a reason
    to burn it, so the name is checked before the split is.

    Iterates the real set rather than a retyped `("tau2", "pare")`. That literal was a fourth
    copy of a list that already exists in one place, and it meant a newly added eval-only
    suite was refused by `assert_scorable` and untested here -- the guard would have been
    real and unverified, which is how a guard gets deleted by someone who cannot find its
    test. The set's CONTENTS are pinned one test above; this one pins the behaviour.
    """
    from pinq.splitting import EVAL_ONLY_SUITES

    assert EVAL_ONLY_SUITES, "an empty eval-only set would make this test vacuous"
    for suite in sorted(EVAL_ONLY_SUITES):
        with pytest.raises(SplitRefused, match="eval-only"):
            assert_scorable({"suite_id": suite, "task_id": "t", "split": "train"})


def test_a_non_train_task_is_refused():
    for split in ("dev", "test", ""):
        with pytest.raises(SplitRefused, match="not 'train'"):
            assert_scorable({"suite_id": "musique", "task_id": "t", "split": split})
    assert_scorable({"suite_id": "musique", "task_id": "t", "split": "train"})


def test_a_gold_exposed_episode_is_refused_even_inside_train():
    """A ceiling arm's questions were chosen where gold is readable; training on them is oracle
    distillation, and no ablation could recover the inflation afterwards."""
    base = {"suite_id": "musique", "task_id": "t", "split": "train"}
    with pytest.raises(SplitRefused, match="gold-exposed"):
        assert_scorable({**base, "gold_exposed": True})
    with pytest.raises(SplitRefused, match="gold-exposed"):
        assert_scorable({**base, "canary_hit": True})


def test_the_refusal_carries_its_own_exit_code():
    """3, not 1: a CI job must tell "you asked for the one thing that voids the experiment"
    apart from "something broke"."""
    assert SplitRefused.exit_code == SPLIT_REFUSED_EXIT == 3


# --------------------------------------------------------------------------- end to end


def _synth_root(tmp_path: Path) -> tuple[Path, str, list[str]]:
    """Build the closed-form suite into a scratch root and return (root, task_id, uids)."""
    from pi_eval.build.synth_build import build
    from pinq_adapters.synth.suite import SynthSuite

    corpus, *_ = build(n_tasks=4, n_facets=2, depth=2, root=tmp_path)
    suite = SynthSuite(Path(corpus).parent)
    tid = suite.task_ids()[0]
    units = suite.retriever(tid)._units
    return tmp_path, str(tid), [u.uid for u in units]


def _write_run(runs: Path, run_id: str, *, suite: str, task: str, split: str, uids: list[str]):
    d = runs / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "suite_id": suite, "task_id": task, "split": split})
    )
    (d / "turns.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "turn_idx": i,
                    "retrieved_uids": [u],
                    "new_uids": [u],
                    "question_id": f"q{i}",
                    "subset_hash_before": "",
                }
            )
            for i, u in enumerate(uids)
        )
    )
    (d / "ledger.jsonl").write_text(
        json.dumps({"currency": "retrieval_calls", "charged": 1.0, "cumulative": float(len(uids))})
    )
    (d / "status.json").write_text(
        json.dumps({"stop_reason": "policy_stop", "wall_ms": 10, "usage": {"tok_total": 100}})
    )
    return d


def test_the_potential_ladder_is_monotone_over_retrieved_evidence(tmp_path, monkeypatch):
    """Phi is coverage over RETRIEVED EvidenceUnits, so it can only go up as they accumulate.
    That is what makes the reward's shaping differences non-negative without any clipping --
    and it is the anti-gaming property: rewriting a question cannot move it."""
    root, tid, uids = _synth_root(tmp_path)
    monkeypatch.setenv("PI_GOLD_ROOT", str(root / "data" / "gold"))
    runs = tmp_path / "runs"
    _write_run(runs, "r1", suite="synth", task=tid, split="train", uids=uids[:3])

    resp = handle_score(ScoreRequest(episode_id="r1", mode="prefix_ladder"), runs_root=runs)
    assert resp.ok and resp.split == "train"
    assert len(resp.potential) == len(resp.turns) + 1
    assert list(resp.potential) == sorted(resp.potential)
    assert resp.potential[0] == 0.0
    assert resp.q_terminal == resp.potential[-1]
    assert resp.scorer_hash == scorer_hash("v1")
    assert resp.stopped and resp.stop_reason == "policy_stop"


def test_the_scorer_refuses_a_test_split_run_end_to_end(tmp_path, monkeypatch):
    root, tid, uids = _synth_root(tmp_path)
    monkeypatch.setenv("PI_GOLD_ROOT", str(root / "data" / "gold"))
    runs = tmp_path / "runs"
    _write_run(runs, "r2", suite="synth", task=tid, split="test", uids=uids[:2])
    with pytest.raises(SplitRefused):
        handle_score(ScoreRequest(episode_id="r2"), runs_root=runs)


def test_the_scorer_cli_exits_three_on_a_refusal(tmp_path, monkeypatch, capsys):
    """The guarantee has to be checkable from a shell script without standing a server up."""
    root, tid, uids = _synth_root(tmp_path)
    monkeypatch.setenv("PI_GOLD_ROOT", str(root / "data" / "gold"))
    runs = tmp_path / "runs"
    _write_run(runs, "r3", suite="synth", task=tid, split="dev", uids=uids[:1])
    rc = serve_score.main(["r3", "--runs-root", str(runs)])
    assert rc == SPLIT_REFUSED_EXIT
    assert "REFUSED" in capsys.readouterr().err

    _write_run(runs, "r4", suite="synth", task=tid, split="train", uids=uids[:1])
    assert serve_score.main(["r4", "--runs-root", str(runs)]) == 0


def test_a_traversal_attempt_is_not_a_run_id(tmp_path):
    from pi_run.serve.score import EpisodeNotFound, episode_dir

    for bad in ("../../etc", "a/b", "", ".", ".."):
        with pytest.raises(EpisodeNotFound):
            episode_dir(tmp_path, bad)


def test_stop_probe_is_refused_rather_than_synthesised(tmp_path, monkeypatch):
    """A fabricated probe would average into the stopping table indistinguishably from a real
    one, so the endpoint says it cannot answer instead of approximating."""
    root, tid, uids = _synth_root(tmp_path)
    monkeypatch.setenv("PI_GOLD_ROOT", str(root / "data" / "gold"))
    runs = tmp_path / "runs"
    _write_run(runs, "r5", suite="synth", task=tid, split="train", uids=uids[:2])
    resp = handle_score(ScoreRequest(episode_id="r5", mode="stop_probe"), runs_root=runs)
    assert not resp.supported and "forced-continue" in resp.note


# --------------------------------------------------------------------------- /rollout, /retrieve


def test_a_base_url_override_is_refused_because_it_moves_the_frozen_roles(monkeypatch):
    """There is ONE LITELLM_BASE_URL per process and MeteredClient applies it to every role, so
    overriding it re-routes the frozen Drafter and Answerer along with the policy."""
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    req = RolloutRequest(task_id="t", suite_id="synth", policy_base_url="http://elsewhere")
    with pytest.raises(FrozenRolesWouldMove, match="policy_model"):
        check_base_url(req)
    check_base_url(req, allow_base_url_override=True)  # a human decision, made at start-up
    check_base_url(RolloutRequest(task_id="t", suite_id="synth", policy_model="ckpt"))


def test_retrieve_speaks_search_r1_verbatim_plus_our_uid(tmp_path):
    """Byte-compatibility is the point: an unmodified Search-R1 client must be able to point at
    this endpoint and get the same documents our own arms see."""
    root, tid, uids = _synth_root(tmp_path)
    pool = RetrieverPool(root=root, default_suite="synth", default_task=tid)
    resp = handle_retrieve(
        RetrieveRequest(queries=["K"], topk=2, return_scores=True, suite_id="synth", task_id=tid),
        pool=pool,
    )
    assert len(resp.result) == 1
    for doc in resp.result[0]:
        assert set(doc.document) == {"id", "contents"}  # upstream's shape exactly
        assert doc.uid in uids  # additive, and joinable back to evidence.parquet


def test_retrieve_refuses_to_answer_from_an_arbitrary_corpus():
    """Silently answering from the wrong corpus produces a baseline number that looks correct."""
    with pytest.raises(SuiteNotConfigured, match="no suite_id"):
        handle_retrieve(RetrieveRequest(queries=["x"]), pool=RetrieverPool(root="."))


# --------------------------------------------------------------------------- /healthz


def test_health_reports_which_role_this_process_may_play(monkeypatch, tmp_path):
    """A /score server needs PI_GOLD_ROOT set; a /rollout server needs it unset. One field
    answers both questions, and answering them is cheaper than debugging either."""
    monkeypatch.delenv("PI_GOLD_ROOT", raising=False)
    cfg = ServerConfig(root=tmp_path)
    assert cfg.health().ok and cfg.health().gold_root_set is False
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path))
    assert cfg.health().gold_root_set is True
    assert set(cfg.health().endpoints) == {"/rollout", "/score", "/retrieve", "/healthz"}


# --------------------------------------------------------- the training server's three refusals


def test_a_client_cannot_choose_the_split_the_firewall_compares_against():
    """`handle_score` passed `req.split_assert` -- a field of the REQUEST BODY -- into the guard
    that decides whether a run may be scored for a trainer. wire.py types it Literal["train"]
    and its docstring says "the server refuses anything else", but a Literal is an annotation,
    not a runtime check, and `from_dict` deserializes whatever the body contains.

    wire.py's own comment gives the reason this is fatal: "a check the trainer performs on
    itself is a check the trainer can skip"."""
    from pi_run.serve.score import SplitRefused, assert_scorable

    held_out = {"suite_id": "musique", "task_id": "t1", "split": "test", "run_id": "r"}
    with pytest.raises(SplitRefused, match="not a value a client may ask for"):
        assert_scorable(held_out, split_assert="test")
    with pytest.raises(SplitRefused, match="not a value a client may ask for"):
        assert_scorable(held_out, split_assert="dev")
    # and the ordinary refusal still works
    with pytest.raises(SplitRefused, match="not 'train'"):
        assert_scorable(held_out, split_assert="train")


def test_a_genuine_training_episode_still_scores():
    from pi_run.serve.score import assert_scorable

    assert_scorable({"suite_id": "musique", "task_id": "t1", "split": "train", "run_id": "r"})


def test_the_request_type_still_deserializes_a_hostile_value():
    """Recorded rather than fixed at the wire: `from_dict` does not enforce Literal, so the
    refusal has to live on the server. If this ever starts raising, the server-side guard is
    still the one that matters."""
    from pinq.wire import ScoreRequest, from_dict

    r = from_dict(ScoreRequest, {"episode_id": "e1", "split_assert": "test"})
    assert r.split_assert == "test", "the type annotation is not a runtime check"


def test_gold_built_against_another_corpus_is_refused(monkeypatch):
    """Same task ids, different evidence uids: every match misses, coverage is 0.0 for every
    episode, and the trainer gets a uniform zero reward indistinguishable from a policy that
    never retrieves. pi_eval.score already refuses this; this server did not check."""
    import pi_eval.gold as gold_mod
    from pi_run.serve.score import NoGoldForTask, _gold_uids

    class _G:
        gold_corpus_hash = "a" * 16

        def required(self):
            return []

    monkeypatch.setattr(gold_mod, "load_graphs", lambda s, v: {"t1": _G()})
    # `corpus_dir`, not `corpus_hash`: the guard compares DIRECTORY against DIRECTORY. This
    # fixture originally passed `corpus_hash`, encoding the very confusion the guard had --
    # gold_corpus_hash is a 16-hex directory name and corpus_hash is a 64-hex set hash over
    # content triples, so the two can never be equal and every run was flagged mismatched.
    with pytest.raises(NoGoldForTask, match="gold was built against corpus"):
        _gold_uids(
            "musique", "t1", "v1", {"corpus_dir": "b" * 16, "suite_id": "musique", "task_id": "t1"}
        )


def test_an_unhashed_corpus_stays_scoreable(monkeypatch):
    """Empty on either side means "unknown", not "mismatched" -- a suite whose corpus is
    upstream must not become unscoreable."""
    import pi_eval.gold as gold_mod
    from pi_run.serve.score import _gold_uids

    class _G:
        gold_corpus_hash = ""

        def required(self):
            return [types.SimpleNamespace(gold_ev_uids=("u1",))]

    import types

    monkeypatch.setattr(gold_mod, "load_graphs", lambda s, v: {"t1": _G()})
    assert _gold_uids("musique", "t1", "v1", {"corpus_dir": "b" * 16}) == frozenset({"u1"})


def test_retrieve_refuses_an_unnamed_task_instead_of_picking_one():
    """It fell back to `task_ids()[0]`. Every MuSiQue and 2Wiki task has its OWN closed pool of
    ~20 paragraphs, so a query for task B answered from task A's pool returns confident,
    well-formed, entirely unrelated evidence. `resolve` already refuses a missing SUITE for
    exactly this reason and then answered from an arbitrary task inside it."""
    import types

    from pi_run.serve.retrieve import RetrieverPool, TaskNotSpecified

    pool = RetrieverPool.__new__(RetrieverPool)
    pool._suites, pool._retrievers, pool.default_task = {}, {}, None
    pool.suite = lambda sid: types.SimpleNamespace(  # noqa: ARG005
        task_ids=lambda: ["t0", "t1"], retriever=lambda t: t
    )
    with pytest.raises(TaskNotSpecified, match="no task_id"):
        pool.retriever("musique", None)
    assert pool.retriever("musique", "t1") == "t1", "a named task still resolves"


def test_a_served_rollout_carries_real_code_provenance():
    """`ServerConfig` defaults to code_version="" and dirty=True and the CLI passed neither, so
    every rollout served over HTTP had no code version and a `dev-` run_id -- which `pi agg`
    mechanically excludes from every reported table, with no flag anywhere to change it. Derived
    from git now, by the same rule `pi run` uses."""
    import inspect

    from pi_run.serve import app

    src = inspect.getsource(app.main)
    assert "git_info" in src, "the CLI must derive provenance, not accept the hardcoded default"
    assert "code_version=g.sha" in src and "dirty=g.dirty" in src

    from pi_run.manifest import git_info

    g = git_info(".")
    assert g.sha and isinstance(g.dirty, bool)
    cfg = app.ServerConfig(root=".", code_version=g.sha, dirty=g.dirty)
    assert cfg.code_version == g.sha and cfg.dirty is g.dirty


def test_the_gold_corpus_check_compares_directory_against_directory():
    """`gold_corpus_hash` is `write_corpus`'s return value and IS the corpus directory name
    (16 hex). `manifest["corpus_hash"]` is `pinq.ids.corpus_hash` -- a domain-separated set hash
    over (doc_id, title, sha256(text)) triples (64 hex). They are two different functions over
    two different inputs and can never be equal.

    I added this guard in f4f76e5 using the wrong field, which flagged EVERY run as mismatched:
    `collect_rows` returned 0 rows with 137 `no_gold` skips, zeroing rungs 0, 1 and 2 at once.
    The eval-side twin in `pi_eval.score` documents the exact mistake in a comment and compares
    `corpus_dir`."""
    import inspect

    from pi_run.serve import score as serve_score

    src = inspect.getsource(serve_score._gold_uids)
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert '.get("corpus_dir")' in code, "must compare the DIRECTORY, as the eval side does"
    assert '.get("corpus_hash")' not in code, "the 64-hex set hash can never equal a dir name"


def test_a_matching_corpus_dir_is_accepted(monkeypatch):
    """The positive case: gold whose directory name matches the run's corpus_dir scores."""
    import pi_eval.gold as gold_mod
    from pi_run.serve.score import _gold_uids

    class _G:
        gold_corpus_hash = "38f5afb69fb7ea18"

        def required(self):
            import types as _t

            return [_t.SimpleNamespace(gold_ev_uids=("u1",))]

    monkeypatch.setattr(gold_mod, "load_graphs", lambda s, v: {"t1": _G()})
    got = _gold_uids("musique", "t1", "v1", {"corpus_dir": "38f5afb69fb7ea18"})
    assert got == frozenset({"u1"})


def test_a_genuinely_different_corpus_is_still_refused(monkeypatch):
    """The guard must still do its job: gold from another corpus returns 0.0 for every episode,
    which is indistinguishable from a policy that never retrieves."""
    import pytest

    import pi_eval.gold as gold_mod
    from pi_run.serve.score import NoGoldForTask, _gold_uids

    class _G:
        gold_corpus_hash = "38f5afb69fb7ea18"

        def required(self):
            return []

    monkeypatch.setattr(gold_mod, "load_graphs", lambda s, v: {"t1": _G()})
    with pytest.raises(NoGoldForTask, match="gold was built against corpus"):
        _gold_uids("musique", "t1", "v1", {"corpus_dir": "deadbeefdeadbeef"})
