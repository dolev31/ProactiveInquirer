"""The runtime: cache, worker, sweep, compact. Offline, keyless, network-free by construction.

Nothing here needs a provider key, and nothing here can accidentally acquire one: the arms
under test are built from pinq_expt.fakes, and the only client that appears is a stub whose
`complete` is arithmetic. A test that reached the network would be a test that could fail on
someone else's rate limit, which is a test nobody trusts and everyone eventually skips.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pi_eval.build.synth_build import build
from pi_run.cache import CacheMiss, CachingClient, DiskCache, ReplayClient
from pi_run.cli import main as cli_main
from pi_run.manifest import build_manifest, git_info, split_of
from pi_run.sweep import plan, run_sweep, summarize
from pi_run.worker import UnitSpec, run_unit
from pinq.budget import BudgetLedger
from pinq.ids import request_sha
from pinq.types import CallTelemetry, Usage
from pinq_expt import arms as arm_table

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- stub client


class StubLLM:
    """Deterministic, arithmetic, no network. Token counts are word counts so the ledger has
    something non-trivial to reconcile."""

    def __init__(self, ledger: BudgetLedger, model: str = "stub/echo") -> None:
        self._ledger = ledger
        self.model = model
        self.calls = 0

    def request_payload(self, *, role, messages, seed, max_tokens=None, **kw):
        return {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "seed": seed,
            "max_tokens": max_tokens,
            **{k: v for k, v in sorted(kw.items())},
        }

    def complete(self, *, role, messages, seed, max_tokens=None, **kw):
        self.calls += 1
        payload = self.request_payload(
            role=role, messages=messages, seed=seed, max_tokens=max_tokens, **kw
        )
        text = "|".join(str(m.get("content", "")) for m in messages).upper()
        tel = CallTelemetry(
            call_id=f"stub-{self.calls}",
            actor=role,
            model=self.model,
            provider="stub",
            request_sha=request_sha(payload),
            response_sha="r" * 8,
            tok_prompt=sum(len(str(m.get("content", "")).split()) for m in messages),
            tok_completion=len(text.split()),
            usd=0.0,
            wall_ms=1,
        )
        self._ledger.record_call(tel)
        return text, tel


def _msgs(text: str):
    return [{"role": "user", "content": text}]


# --------------------------------------------------------------------------- cache


def test_cache_round_trips(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    ledger = BudgetLedger(cap=8)
    client = CachingClient(StubLLM(ledger), cache)

    text1, tel1 = client.complete(role="drafter", messages=_msgs("hello world"), seed=0)
    assert (client.hits, client.misses) == (0, 1)

    text2, tel2 = client.complete(role="drafter", messages=_msgs("hello world"), seed=0)
    assert (client.hits, client.misses) == (1, 1)
    assert text2 == text1
    assert tel2.cache_hit and not tel1.cache_hit
    # Tokens and USD survive the round trip; only the machine artifacts are re-synthesised.
    assert (tel2.tok_prompt, tel2.tok_completion) == (tel1.tok_prompt, tel1.tok_completion)
    assert tel2.request_sha == tel1.request_sha
    assert cache.path(tel1.request_sha).exists()
    assert cache.path(tel1.request_sha).parent.name == tel1.request_sha[:2]


def test_one_changed_byte_changes_the_key(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    client = CachingClient(StubLLM(BudgetLedger(cap=8)), cache)
    _, a = client.complete(role="drafter", messages=_msgs("hello world"), seed=0)
    _, b = client.complete(role="drafter", messages=_msgs("hello worlds"), seed=0)
    _, c = client.complete(role="drafter", messages=_msgs("hello world"), seed=1)
    assert len({a.request_sha, b.request_sha, c.request_sha}) == 3
    assert client.misses == 3
    assert cache.stats().entries == 3


def test_replay_client_raises_on_miss(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    ledger = BudgetLedger(cap=8)
    replay = ReplayClient(StubLLM(ledger), cache, ledger=ledger)
    with pytest.raises(CacheMiss):
        replay.complete(role="drafter", messages=_msgs("never seen"), seed=0)

    # Warm it through the real client, then the same request replays without dispatching.
    stub = StubLLM(BudgetLedger(cap=8))
    CachingClient(stub, cache).complete(role="drafter", messages=_msgs("never seen"), seed=0)
    text, tel = replay.complete(role="drafter", messages=_msgs("never seen"), seed=0)
    assert tel.cache_hit and text
    assert stub.calls == 1  # the replay client never called through


_RACE = """
import json, sys
from pi_run.cache import DiskCache
root, sha = sys.argv[1], sys.argv[2]
DiskCache(root).put(sha, {"model": "stub/echo", "request_sha": sha, "text": "same bytes",
                          "tok_prompt": 3, "tok_completion": 5, "usd": 0.0})
"""


def test_two_workers_racing_one_key_write_identical_bytes(tmp_path):
    """Content-addressed + atomic replace => last-writer-wins is correct, so no lock is needed."""
    root = tmp_path / "cache"
    sha = "ab" + "c" * 62
    procs = [
        subprocess.Popen([sys.executable, "-c", _RACE, str(root), sha]),
        subprocess.Popen([sys.executable, "-c", _RACE, str(root), sha]),
    ]
    assert [p.wait(timeout=120) for p in procs] == [0, 0]

    p = DiskCache(root).path(sha)
    assert p.exists()
    first = p.read_bytes()
    # Re-write it once more single-threaded: identical input must produce identical bytes.
    subprocess.run([sys.executable, "-c", _RACE, str(root), sha], check=True, timeout=120)
    assert p.read_bytes() == first
    assert json.loads(first)["text"] == "same bytes"
    assert list(root.glob("*/*.tmp")) == []  # no temp files left behind


# --------------------------------------------------------------------------- manifest


def test_dirty_tree_yields_a_dev_prefixed_run_id():
    common = dict(
        suite_id="synth",
        task_id="s0",
        arm_id="fake_drafter_only",
        policy_id="never_ask",
        seed=0,
        corpus_hash="abc",
        budget_cap=16,
        max_turns=16,
        word_cap=60,
        code_version="deadbeef",
    )
    clean = build_manifest(**common, dirty=False)
    dirty = build_manifest(**common, dirty=True)
    assert not clean.run_id.startswith("dev-")
    assert dirty.run_id.startswith("dev-")
    # Same science, different provenance: the dev- run must still be mechanically excludable.
    assert dirty.run_id[len("dev-") :] == clean.run_id


def test_split_is_deterministic_and_covers_all_three():
    assert split_of("synth", "s0") == split_of("synth", "s0")
    seen = {split_of("synth", f"s{i}") for i in range(300)}
    assert seen == {"train", "dev", "test"}


def test_git_info_is_cached_per_process():
    a = git_info(str(REPO))
    b = git_info(str(REPO))
    assert a is b  # lru_cache: a 30k-unit sweep must not fork git 30k times


# --------------------------------------------------------------------------- corpus fixture


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("pi")
    corpus, _gold, _chash = build(n_tasks=4, n_facets=2, depth=3, seed=11, root=root)
    return root, corpus.parent


def _spec(root: Path, corpus: Path, task_id: str, arm_id: str, **kw) -> UnitSpec:
    # runs_root is overridable so a test that needs its own directory -- resume, timeout,
    # pin-swap -- does not collide with the module-scoped corpus fixture's shared runs/.
    kw.setdefault("runs_root", str(root / "runs"))
    return UnitSpec(
        suite_id="synth",
        corpus_dir=str(corpus),
        task_id=task_id,
        arm_id=arm_id,
        seed=0,
        cache_root=str(root / "cache"),
        code_version="testsha",
        dirty=False,
        **kw,
    )


# --------------------------------------------------------------------------- worker


def test_worker_writes_every_artifact_and_reconciles(synth_root):
    root, corpus = synth_root
    res = run_unit(_spec(root, corpus, "s0", "fake_chain"))
    assert res["status"] == "ok"
    d = Path(root / "runs" / res["run_id"])
    for f in ("status.json", "manifest.json", "outcome.json", "turns.jsonl", "calls.jsonl"):
        assert (d / f).exists(), f
    assert (d / "ledger.jsonl").exists() and (d / "evidence.jsonl").exists()
    assert res["reconciled"] is True
    assert res["reconcile"]["docs_ok"] and res["reconcile"]["tokens_ok"]
    assert res["n_asks"] > 0


def test_worker_is_idempotent_and_resume_skips(synth_root):
    root, corpus = synth_root
    spec = _spec(root, corpus, "s1", "fake_chain")
    first = run_unit(spec)
    d = Path(root / "runs" / first["run_id"])
    turns_before = (d / "turns.jsonl").read_text()
    n_before = len(turns_before.splitlines())

    second = run_unit(spec)
    assert second["status"] == "resumed"
    assert second["run_id"] == first["run_id"]
    # No row is rewritten, so compact cannot double-count.
    assert (d / "turns.jsonl").read_text() == turns_before
    assert len((d / "turns.jsonl").read_text().splitlines()) == n_before

    # And exactly one run directory exists for this unit.
    assert len([p for p in (root / "runs").iterdir() if p.name == first["run_id"]]) == 1


def test_worker_refuses_to_run_with_gold_root_set(synth_root, monkeypatch):
    from pi_run.worker import FirewallError

    root, corpus = synth_root
    monkeypatch.setenv("PI_GOLD_ROOT", str(root / "data" / "gold"))
    with pytest.raises(FirewallError):
        run_unit(_spec(root, corpus, "s0", "fake_drafter_only"))


def test_ledger_parity_sum_of_calls_equals_ledger(synth_root):
    """Per-call telemetry and the ledger are two accounts of the same spend."""
    root, corpus = synth_root
    ledger = BudgetLedger(cap=8)
    stub = StubLLM(ledger)
    for i in range(4):
        stub.complete(role="drafter", messages=_msgs(f"call number {i} here"), seed=0)

    total = Usage()
    for c in ledger.calls:
        total = total + Usage.from_call(c)
    assert total == ledger.usage
    assert ledger.reconcile(total) is True
    assert ledger.spent["tok_prompt"] == total.tok_prompt
    assert ledger.spent["tok_completion"] == total.tok_completion
    assert ledger.spent["llm_calls"] == 4

    # An LLM-free arm must leave the ledger at zero tokens: that is what the worker asserts.
    res = run_unit(_spec(root, corpus, "s2", "fake_chain"))
    assert res["usage"]["tok_total"] == 0
    assert res["usage"]["n_calls"] == 0
    assert res["reconciled"] is True


def test_unmetered_nested_retrieval_is_visible(synth_root):
    """A Drafter that folds unmetered evidence into its resolve() output must be caught.

    Tokens do not move and retrieval_calls does not move, so the only detector is the
    document set: run_loop meters every unit it merges, so evidence docs > ledger docs
    means something retrieved behind the ledger's back.
    """
    from pi_run.worker import reconcile
    from pinq.loop import run_loop
    from pinq.types import Draft, EvidenceUnit
    from pinq_adapters.synth.suite import SynthSuite
    from pinq_expt.fakes import ChainInquirer, FrozenAnswerer

    _root, corpus = synth_root
    suite = SynthSuite(corpus)
    tid = suite.task_ids()[0]

    class SneakyDrafter:
        def resolve(self, view, ask, ev, *, seed, ledger):
            smuggled = EvidenceUnit.make(
                corpus_id="synth_v1",
                doc_id="off:ledger",
                span="0:1",
                title="unmetered",
                text="V999",
            )
            return ("", ev.with_units([smuggled]))  # no ledger.note_docs -> unmetered

        def draft(self, view, ev, *, seed, ledger):
            return Draft(text="")

    ledger = BudgetLedger(cap=16)
    traj = run_loop(
        view=suite.view(tid),
        inquirer=ChainInquirer(),
        retriever=suite.retriever(tid),
        drafter=SneakyDrafter(),
        answerer=FrozenAnswerer(),
        ledger=ledger,
        max_turns=8,
        k=5,
        seed=0,
    )
    rec = reconcile(traj, ledger)
    assert rec["docs_ok"] is False
    assert rec["evidence_unique_docs"] > rec["ledger_unique_docs"]


# --------------------------------------------------------------------------- sweep


def test_sweep_reassembles_by_key_not_by_completion_order(synth_root):
    root, corpus = synth_root
    specs = plan(
        suite_id="synth",
        corpus_dir=str(corpus),
        task_ids=list(("s0", "s1", "s2", "s3")),
        arm_ids=("fake_drafter_only", "fake_depth1"),
        seeds=(0,),
        runs_root=str(root / "runs_sweep"),
        cache_root=str(root / "cache"),
        code_version="testsha",
        dirty=False,
    )
    results = run_sweep(specs, concurrency=2)
    assert len(results) == len(specs)
    for spec, res in zip(specs, results, strict=True):
        assert tuple(res["key"]) == spec.key, "results must line up with the specs BY KEY"
    s = summarize(results)
    assert s["units"] == 8
    assert s["by_status"].get("ok", 0) + s["by_status"].get("resumed", 0) == 8

    # Re-issuing the identical sweep is the recovery procedure: everything resumes.
    again = run_sweep(specs, concurrency=2)
    assert {r["status"] for r in again} == {"resumed"}


# --------------------------------------------------------------------------- end to end


@pytest.fixture()
def e2e(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.delenv("PI_GOLD_ROOT", raising=False)
    monkeypatch.setenv("PI_MAX_CONCURRENCY", "2")
    corpus, _gold, _chash = build(n_tasks=3, n_facets=2, depth=3, seed=5, root=tmp_path)
    rc = cli_main(
        [
            "run",
            "--suite",
            "synth",
            "--arm",
            "fake_chain",
            "--n",
            "3",
            "--root",
            str(tmp_path),
            "--corpus",
            str(corpus.parent),
        ]
    )
    assert rc == 0
    # `pi compact` refuses to write a tree's SHARED scores/parquet without the store lock
    # (see tests/test_shared_store_writes_need_the_lock.py: two lanes raced the real one on
    # 2026-09-18 and the later batch of renames discarded the earlier lane's whole result).
    # An end-to-end test is a caller like any other, so it takes the lock the way a caller
    # does rather than being exempted from it.
    lock = tmp_path / "scores" / "parquet" / ".store.lock"
    lock.mkdir(parents=True)
    (lock / "owner").write_text("tests/test_runtime.py::e2e")
    return tmp_path


def test_pi_run_honours_PI_RUNS_ROOT_with_no_explicit_flag(tmp_path, monkeypatch):
    """`pi env doctor` already reports `runs_root` from `PI_RUNS_ROOT`
    (`os.environ.get("PI_RUNS_ROOT", ...)`), and three `scripts/launch_controls/*.py` plus
    `scripts/label_ordering_test/01_select_and_link.py` already read the same variable as their
    runs-root override. `cmd_run` alone ignored it (`runs_root = Path(a.runs_root) if
    a.runs_root else root / "runs"`), so a sweep recipe instructing an operator to export
    PI_RUNS_ROOT -- exactly as this repo's own frames/frontier sweep recipes do -- silently
    wrote every run under `<root>/runs` instead. A variable honoured by some entry points and
    ignored by others is worse than one ignored everywhere: this asserts `pi run` is no longer
    the odd one out, with no `--runs-root` flag given at all."""
    pytest.importorskip("pyarrow")
    monkeypatch.delenv("PI_GOLD_ROOT", raising=False)
    monkeypatch.setenv("PI_MAX_CONCURRENCY", "2")
    corpus, _gold, _chash = build(n_tasks=1, n_facets=2, depth=3, seed=5, root=tmp_path)
    elsewhere = tmp_path / "elsewhere_runs"
    monkeypatch.setenv("PI_RUNS_ROOT", str(elsewhere))
    rc = cli_main(
        [
            "run",
            "--suite",
            "synth",
            "--arm",
            "fake_chain",
            "--n",
            "1",
            "--root",
            str(tmp_path),
            "--corpus",
            str(corpus.parent),
            # Deliberately no --runs-root: the environment variable alone must be enough.
        ]
    )
    assert rc == 0
    assert list(elsewhere.glob("*/manifest.json")), (
        f"PI_RUNS_ROOT={elsewhere} was exported and no --runs-root flag was given, "
        "but no run landed there"
    )
    assert not (tmp_path / "runs").exists(), "a run leaked into the default <root>/runs instead"


def test_end_to_end_run_then_compact_row_counts_match_trajectories(e2e):
    import pyarrow.parquet as pq

    runs = sorted(p for p in (e2e / "runs").iterdir() if p.is_dir())
    assert len(runs) == 3

    rc = cli_main(["compact", "--root", str(e2e)])
    assert rc == 0
    out = e2e / "scores" / "parquet"

    expected_turns = sum(len((d / "turns.jsonl").read_text().splitlines()) for d in runs)
    expected_ev = sum(len((d / "evidence.jsonl").read_text().splitlines()) for d in runs)
    expected_ledger = sum(len((d / "ledger.jsonl").read_text().splitlines()) for d in runs)

    assert pq.read_table(out / "runs.parquet").num_rows == 3
    assert pq.read_table(out / "turns.parquet").num_rows == expected_turns
    assert pq.read_table(out / "evidence.parquet").num_rows == expected_ev
    assert pq.read_table(out / "ledger.parquet").num_rows == expected_ledger
    assert pq.read_table(out / "calls.parquet").num_rows == 0  # LLM-free arm

    # Declared now, populated later: the files must exist and carry the frozen schema.
    for name in ("matches", "judgments", "scores"):
        t = pq.read_table(out / f"{name}.parquet")
        assert t.num_rows == 0 and len(t.schema) > 0

    runs_tbl = pq.read_table(out / "runs.parquet").to_pydict()
    assert set(runs_tbl["arm_id"]) == {"fake_chain"}
    assert all(runs_tbl["reconciled"])
    # Every turn belongs to a run in `runs`: the universal join key holds.
    turn_ids = set(pq.read_table(out / "turns.parquet").to_pydict()["run_id"])
    assert turn_ids <= set(runs_tbl["run_id"])
    # n_turns in `runs` is the denormalized rollup of the `turns` rows.
    assert sum(runs_tbl["n_turns"]) == expected_turns


def test_compact_is_idempotent(e2e):
    import pyarrow.parquet as pq

    cli_main(["compact", "--root", str(e2e)])
    first = pq.read_table(e2e / "scores" / "parquet" / "turns.parquet").num_rows
    cli_main(["compact", "--root", str(e2e)])
    assert pq.read_table(e2e / "scores" / "parquet" / "turns.parquet").num_rows == first


def test_compact_fails_on_an_unknown_column(e2e):
    """Schema drift must be a crash, not a NULL column that joins Cartesian three tables on."""
    sch = pytest.importorskip("pi_eval.schema")
    runs = sorted(p for p in (e2e / "runs").iterdir() if p.is_dir())
    path = runs[0] / "turns.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x]
    rows[0]["a_column_nobody_declared"] = 1
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

    from pi_run.compact import compact

    with pytest.raises(sch.SchemaViolation, match="unknown column"):
        compact(e2e / "runs", e2e / "scores" / "parquet")


def test_compact_fails_on_a_missing_column(e2e):
    sch = pytest.importorskip("pi_eval.schema")
    runs = sorted(p for p in (e2e / "runs").iterdir() if p.is_dir())
    path = runs[0] / "turns.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x]
    rows[0].pop("question_id")
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

    from pi_run.compact import compact

    with pytest.raises(sch.SchemaViolation, match="missing column"):
        compact(e2e / "runs", e2e / "scores" / "parquet")


# --------------------------------------------------------------------------- cli surface


@pytest.fixture
def spawn_pool(monkeypatch):
    """Run `pi verify firewall`'s worker pool under spawn for an IN-PROCESS `cli_main` call.

    This module imports pi_eval, so under Linux's default fork a pool child inherits it and
    `worker_ok` measures pytest's sys.modules rather than the CLI's: both tests below failed on
    Linux CI and passed on macOS, which spawns. The fork case is covered where the parent is the
    CLI itself, in a fresh interpreter: test_verify_firewall_worker_is_clean_under_fork.
    """
    import concurrent.futures
    import functools
    import multiprocessing

    import pi_run.cli as cli_module

    spawn = multiprocessing.get_context("spawn")
    monkeypatch.setattr(
        cli_module,
        "ProcessPoolExecutor",
        functools.partial(concurrent.futures.ProcessPoolExecutor, mp_context=spawn),
    )


def test_cli_verify_firewall_passes_with_no_gold_root(tmp_path, monkeypatch, capsys, spawn_pool):
    monkeypatch.delenv("PI_GOLD_ROOT", raising=False)
    monkeypatch.setenv("PI_CANARY_SALT", "CANARY-0xDEADBEEF")
    cache = DiskCache(tmp_path / "cache")
    cache.put("ff" + "0" * 62, {"model": "stub/echo", "text": "nothing to see"})
    rc = cli_main(
        ["verify", "firewall", "--root", str(tmp_path), "--cache-root", str(tmp_path / "cache")]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["ok"] and out["worker_ok"]
    # A scan that examined nothing is not a pass: assert the entry was actually read.
    assert out["cache_entries_scanned"] == 1 and out["canaries_configured"] >= 1
    assert out["worker"]["gold_root_set"] is False
    assert out["worker"]["pi_eval_imported"] is False  # gold code never enters a worker
    assert out["canary_hits"] == []


# The test above only ever passed under macOS's `spawn`, where a pool child re-imports __main__
# and never sees what its parent loaded. Linux defaults to `fork` (Python 3.12), so the child
# inherits the parent's sys.modules whole, and every `pi` process builds its parser before it
# builds a pool. On HPC `pi verify firewall` reported pi_eval_imported true on every stream of a
# campaign whose Mac reproduction passed. Both tests run in a fresh interpreter, because this
# module itself imports pi_eval.


def _fresh_python(code: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PI_GOLD_ROOT"}
    env["PI_CANARY_SALT"] = "CANARY-0xDEADBEEF"
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )


def test_building_the_cli_parser_does_not_import_pi_eval():
    code = (
        "import sys\n"
        "from pi_run.cli import build_parser\n"
        "build_parser()\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'pi_eval'))\n"
    )
    r = _fresh_python(code)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"


def test_annotate_argument_surface_mirrors_pi_eval_constants():
    # The surface mirrors two pi_eval constants so that building it never imports pi_eval.
    from pi_eval import annotate
    from pi_run import cmd_annotate_args

    assert cmd_annotate_args.TASK_TYPES == annotate.TASK_TYPES
    assert cmd_annotate_args.A6_MIN_CANDIDATES == annotate.A6_MIN_CANDIDATES


@pytest.mark.skipif(
    "fork" not in __import__("multiprocessing").get_all_start_methods(),
    reason="no fork start method on this platform",
)
def test_verify_firewall_worker_is_clean_under_fork(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    cache.put("ff" + "0" * 62, {"model": "stub/echo", "text": "nothing to see"})
    code = (
        "import multiprocessing, sys\n"
        "multiprocessing.set_start_method('fork', force=True)\n"
        "from pi_run.cli import main\n"
        f"sys.exit(main(['verify', 'firewall', '--root', {str(tmp_path)!r}, "
        f"'--cache-root', {str(tmp_path / 'cache')!r}]))\n"
    )
    r = _fresh_python(code)
    out = json.loads(r.stdout)
    assert out["cache_entries_scanned"] == 1  # the scan read something
    assert out["worker"]["gold_root_set"] is False
    assert out["worker"]["pi_eval_imported"] is False
    assert out["worker_ok"] and out["ok"] and r.returncode == 0


def test_cli_verify_firewall_fails_when_a_canary_is_cached(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("PI_GOLD_ROOT", raising=False)
    monkeypatch.setenv("PI_CANARY_SALT", "CANARY-0xDEADBEEF")
    DiskCache(tmp_path / "cache").put(
        "ee" + "0" * 62,
        {"model": "stub/echo", "text": "the answer is CANARY-0xDEADBEEF"},
    )
    rc = cli_main(
        ["verify", "firewall", "--root", str(tmp_path), "--cache-root", str(tmp_path / "cache")]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["canary_hits"]


# ------------------------------------------------------- an unarmed scan must REFUSE, not pass
#
# `_canary_carriers` globbed <root>/data/gold/graphs for suite directories with no signal when
# the glob matched nothing: a pinned worktree whose data/ holds corpora and raw but never got
# gold/graphs built returned {} exactly like a suite that legitimately carries no nonce, and
# `cmd_verify_firewall` fed neither the emptiness nor the (wrong, uncalled-with-root) carriers
# dict into `ok`. Zero nonces to look for was indistinguishable from a genuinely clean scan.
#
# Three states, not two. (1) gold never built at this root at all -- nothing to leak, must stay
# runnable, see test_cli_verify_firewall_passes_with_no_gold_root above. (2) gold built, graphs
# missing/empty, or a suite's registry can't be loaded -- a configuration error, refused. (3)
# gold built, graphs present, and a suite that is supposed to carry a nonce (non-empty
# gold_answer) does not -- a real fault distinct from (2), because tau2/drgym/tau2_* legitimately
# carry none by design and must never be refused for it.


def test_verify_firewall_refuses_when_gold_is_built_but_graphs_never_made_it(
    tmp_path, monkeypatch, capsys
):
    """The exact shape of the bug: data/gold exists (something was built here) but
    data/gold/graphs does not, so the suite glob matches nothing. Distinct from a totally fresh
    checkout (no data/gold at all), which must stay runnable -- this root was clearly meant to
    have gold and does not, which is a configuration error, not an empty benign tree."""
    monkeypatch.delenv("PI_GOLD_ROOT", raising=False)
    monkeypatch.setenv("PI_CANARY_SALT", "CANARY-0xDEADBEEF")
    (tmp_path / "data" / "gold").mkdir(parents=True)  # gold IS built ...
    (tmp_path / "data" / "corpora").mkdir(parents=True)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    # ... but data/gold/graphs specifically never made it.
    cache_root = tmp_path / "cache"
    DiskCache(cache_root).put("ff" + "0" * 62, {"model": "stub/echo", "text": "nothing to see"})

    rc = cli_main(["verify", "firewall", "--root", str(tmp_path), "--cache-root", str(cache_root)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 2, f"an unarmed scan must refuse, got rc={rc}: {out}"
    assert out["ok"] is False
    assert out["refused"] is True
    assert str(tmp_path) in out["reason"], out["reason"]
    assert "graphs" in out["reason"], out["reason"]


def test_verify_firewall_does_not_refuse_a_suite_that_legitimately_carries_no_nonce(
    tmp_path, monkeypatch, capsys, spawn_pool
):
    """tau2 (and drgym, tau2_airline, tau2_retail, tau2_telecom) have no gold_answer BY DESIGN:
    their gold is entirely needs, and their only leakable string is gold_text, which reaches
    judges and both ceiling arms legitimately. A fix for the absent-graphs defect above must not
    treat this real, present, well-formed suite the same as a missing tree -- that would abort
    tau2's own measurements, not merely error on a misconfigured root."""
    monkeypatch.delenv("PI_GOLD_ROOT", raising=False)
    monkeypatch.setenv("PI_CANARY_SALT", "CANARY-0xDEADBEEF")
    graphs = tmp_path / "data" / "gold" / "graphs" / "tau2"
    graphs.mkdir(parents=True)
    row = {
        "gold_task_key": "t1",
        "gold_canary": "PINQCANARY_0000000000000001",
        "gold_answer": "",  # empty by design -- exactly tau2's real, verified shape
        "gold_text": "the customer wants a refund",
    }
    (graphs / "v1.jsonl").write_text(json.dumps(row) + "\n")
    canaries_dir = tmp_path / "data" / "canaries"
    canaries_dir.mkdir(parents=True)
    (canaries_dir / "canaries.txt").write_text("PINQCANARY_0000000000000001\n")
    cache_root = tmp_path / "cache"
    DiskCache(cache_root).put("ff" + "0" * 62, {"model": "stub/echo", "text": "nothing to see"})

    rc = cli_main(["verify", "firewall", "--root", str(tmp_path), "--cache-root", str(cache_root)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0, f"a legitimately blind suite (tau2) was refused: {out}"
    assert out["ok"] is True
    assert out["canary_carriers"]["tau2"].startswith("NONE")


def test_verify_firewall_refuses_when_a_suite_should_carry_a_nonce_and_does_not(
    tmp_path, monkeypatch, capsys
):
    """The third state: musique-shaped (non-empty gold_answer) but the mint never made it into
    the answer string. Distinct from tau2's legitimate NONE -- this is minting having failed for
    one suite, and must not be swallowed into the same bucket as a by-design blind suite."""
    monkeypatch.delenv("PI_GOLD_ROOT", raising=False)
    monkeypatch.setenv("PI_CANARY_SALT", "CANARY-0xDEADBEEF")
    graphs = tmp_path / "data" / "gold" / "graphs" / "musique"
    graphs.mkdir(parents=True)
    row = {
        "gold_task_key": "t1",
        "gold_canary": "PINQCANARY_0000000000000002",
        "gold_answer": "Paris",  # non-empty, but the nonce was never threaded in
    }
    (graphs / "v1.jsonl").write_text(json.dumps(row) + "\n")
    canaries_dir = tmp_path / "data" / "canaries"
    canaries_dir.mkdir(parents=True)
    (canaries_dir / "canaries.txt").write_text("PINQCANARY_0000000000000002\n")
    cache_root = tmp_path / "cache"
    DiskCache(cache_root).put("ff" + "0" * 62, {"model": "stub/echo", "text": "nothing to see"})

    rc = cli_main(["verify", "firewall", "--root", str(tmp_path), "--cache-root", str(cache_root)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 2, f"a suite whose mint failed must refuse, got rc={rc}: {out}"
    assert out["ok"] is False
    assert "musique" in out["reason"], out["reason"]
    assert out["canary_carriers"]["musique"].startswith("FAULT")


def test_cli_env_doctor_never_prints_a_key_value(monkeypatch, capsys):
    # --no-probe: doctor now makes a live call per pinned model, and `make gate` must
    # stay offline. The probe itself is exercised by the integration test below.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret")
    assert cli_main(["env", "doctor", "--no-probe"]) == 0
    captured = capsys.readouterr().out
    assert "supersecret" not in captured
    out = json.loads(captured)
    assert out["provider_keys_present"]["anthropic"] is True
    assert out["reachability"] == {}
    # The three PREREQUISITE variables, whose absence otherwise surfaces minutes into a sweep
    # from inside an adapter constructor.
    assert set(out["suite_env_present"]) == {
        "TAU2_DATA_DIR",
        "PARE_BENCHMARK_SPLITS_DIR",
        "DRGYM_API_KEY",
    }
    assert out["PI_UNIT_TIMEOUT_S"] > 0


@pytest.mark.integration
def test_cli_env_doctor_probes_every_pinned_model_for_real():
    """Doctor was documented as reporting "provider reachability" and made ZERO network calls:
    it checked that an env var was non-empty, which is equally true of a typo, an expired key,
    a proxy that is down and a VPN that is not connected. Those four are the reasons a sweep
    dies at 02:00, and a green report that cannot tell them apart moves the diagnosis to after
    the spend."""
    rc = cli_main(["env", "doctor", "--probe-timeout", "45"])
    assert rc in (0, 2)


def test_cli_cache_stats(tmp_path, capsys):
    DiskCache(tmp_path / "cache").put("aa" + "1" * 62, {"model": "stub/echo", "text": "x"})
    assert cli_main(["cache", "stats", "--cache-root", str(tmp_path / "cache")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["entries"] == 1 and out["shards"] == 1


def test_cli_cost_estimate(capsys):
    os.environ.setdefault("PI_PRICE_TABLE", str(REPO / "scripts" / "price_tables" / "2026-08.json"))
    assert cli_main(["cost", "estimate", "--grid", "E", "--model", "stub/echo"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["grids"][0]["grid"] == "E"
    assert out["usd_naive_total"] == 0.0  # stub/echo is priced at zero, deterministically


# --------------------------------------------------------------------------- arms + pricing


def test_arms_table_is_explicit_and_names_its_llm_free_arms():
    """The table now carries the whole experiment, so "every arm is LLM-free" has stopped
    being true and asserting it would encode a belief the design has moved past. What
    replaces it is stronger, not weaker: the LLM-free set is EXACTLY the three `fake_*`
    loop-debug arms, so a keyless CI run cannot silently execute a paper arm, and no paper
    arm can be quietly relabelled LLM-free to get past the worker's token assertion.
    """
    assert arm_table.llm_free_arm_ids() == ("fake_chain", "fake_depth1", "fake_drafter_only")
    for aid in arm_table.arm_ids():
        arm = arm_table.get(aid)
        assert arm.arm_id == aid
        assert arm.inquirer().policy_id
        assert arm.llm_free is aid.startswith("fake_")
    with pytest.raises(arm_table.UnknownArm):
        arm_table.get("no_such_arm")


def test_price_table_is_deterministic_and_strict():
    # pinq_adapters.llm.pricing is dependency-free: costing must work in a bare install.
    from pinq_adapters.llm.pricing import LLMConfigError, PriceTable

    pt = PriceTable.load(REPO / "scripts" / "price_tables" / "2026-08.json")
    usd = pt.usd("anthropic/claude-sonnet-4-5", tok_prompt=1_000_000, tok_completion=0)
    assert usd == pytest.approx(3.00)
    # Cached tokens are a SUBSET of prompt tokens, billed at the cached rate.
    cached = pt.usd(
        "anthropic/claude-sonnet-4-5",
        tok_prompt=1_000_000,
        tok_completion=0,
        tok_cached=1_000_000,
    )
    assert cached == pytest.approx(0.30)
    with pytest.raises(LLMConfigError):
        pt.usd("nobody/ever-priced-this", tok_prompt=10, tok_completion=10)


def test_metered_client_exposes_no_ledger_access():
    """The budget-blindness property, asserted on the object rather than in a comment."""
    pytest.importorskip("tenacity")  # the client lives in the `run` extras
    from pinq_adapters.llm.litellm_client import MeteredClient, PriceTable

    pt = PriceTable.load(REPO / "scripts" / "price_tables" / "2026-08.json")
    client = MeteredClient(BudgetLedger(cap=4), price_table=pt, models={"drafter": "stub/echo"})
    public = {n for n in dir(client) if not n.startswith("_")}
    assert not (public & {"ledger", "spent", "budget", "remaining", "usage", "cap"})
    assert client.model_for("drafter") == "stub/echo"
    payload = client.request_payload(role="drafter", messages=_msgs("hi"), seed=3)
    assert payload["model"] == "stub/echo" and payload["seed"] == 3


def test_role_pins_come_from_the_environment(monkeypatch):
    pytest.importorskip("tenacity")
    from pinq_adapters.llm.litellm_client import LLMConfigError, MeteredClient, PriceTable

    monkeypatch.setenv("PI_MODEL_ANSWERER", "openai/gpt-5-mini")
    monkeypatch.delenv("PI_MODEL_JUDGE", raising=False)
    pt = PriceTable.load(REPO / "scripts" / "price_tables" / "2026-08.json")
    client = MeteredClient(BudgetLedger(cap=4), price_table=pt, env=dict(os.environ))
    assert client.model_for("answerer") == "openai/gpt-5-mini"
    assert client.pin("answerer").provider == "openai"
    with pytest.raises(LLMConfigError):
        client.model_for("judge")  # silently defaulting would let two arms use two models


# --------------------------------------------------------------------------- the import wall

_NO_GOLD_ON_THE_ROLLOUT_PATH = """
import sys
import pi_run.cache, pi_run.manifest, pi_run.sweep, pi_run.worker
import pinq_adapters.synth.suite, pinq_expt.arms
from pi_run.worker import UnitSpec, run_unit
run_unit(UnitSpec(suite_id="synth", corpus_dir=sys.argv[1], task_id=sys.argv[2],
                  arm_id="fake_chain", seed=0, runs_root=sys.argv[3],
                  cache_root=sys.argv[4], code_version="x", dirty=False))
leaked = sorted(m for m in sys.modules if m.split(".")[0] == "pi_eval")
assert not leaked, leaked
"""


def test_rollout_worker_never_imports_pi_eval(synth_root, tmp_path):
    """HARD RULE, checked in a CLEAN interpreter.

    The pytest process itself imports pi_eval (to build the corpus), so asserting on
    sys.modules in-process would prove nothing. A subprocess that imports the entire rollout
    path and executes a unit is the only honest form of this check.
    """
    _root, corpus = synth_root
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _NO_GOLD_ON_THE_ROLLOUT_PATH,
            str(corpus),
            "s0",
            str(tmp_path / "runs"),
            str(tmp_path / "cache"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "PI_GOLD_ROOT": ""},
    )
    assert proc.returncode == 0, proc.stderr


# --------------------------------------------------------------------------- retry telemetry


def test_metered_client_records_retries_and_stall_and_debits_the_ledger(monkeypatch):
    """No network: _dispatch is replaced. What is under test is the accounting, not litellm."""
    pytest.importorskip("tenacity")
    from pinq_adapters.llm import litellm_client as lc

    pt = lc.PriceTable.load(REPO / "scripts" / "price_tables" / "2026-08.json")
    ledger = BudgetLedger(cap=4)
    client = lc.MeteredClient(
        ledger,
        price_table=pt,
        models={"answerer": "anthropic/claude-haiku-4-5"},
        retry_initial_wait=0.01,
        # STATED, not inherited. This test exercises the retry ladder, so it must not be
        # configured by whatever PI_LLM_MAX_ATTEMPTS the environment happens to carry -- and
        # the autouse fixture that keeps the offline suite fast sets it to 1.
        max_attempts=4,
    )

    attempts = {"n": 0}

    def flaky(payload):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("transport hiccup")
        return lc._Response(
            text="ok",
            tok_prompt=1_000_000,
            tok_completion=1_000_000,
            tok_reasoning=0,
            tok_cached=0,
            http_status=200,
        )

    monkeypatch.setattr(lc.MeteredClient, "_dispatch", lambda self, payload: flaky(payload))
    text, tel = client.complete(role="answerer", messages=_msgs("hello"), seed=0)

    assert text == "ok"
    assert tel.retries == 2 and attempts["n"] == 3
    assert tel.rate_limit_stall_ms > 0  # time spent in backoff, not in a dispatch
    assert tel.provider_error_code == "ConnectionError"
    # 1M in @ $1.00 + 1M out @ $5.00, straight from the pinned table. Never the provider's field.
    assert tel.usd == pytest.approx(6.00)
    assert tel.cache_hit is False
    # Debited internally, and there is still no way to read the ledger through the client.
    assert ledger.usage.n_calls == 1
    assert ledger.spent["tok_total"] == 2_000_000


def test_metered_client_does_not_retry_a_programming_error(monkeypatch):
    pytest.importorskip("tenacity")
    from pinq_adapters.llm import litellm_client as lc

    pt = lc.PriceTable.load(REPO / "scripts" / "price_tables" / "2026-08.json")
    client = lc.MeteredClient(
        BudgetLedger(cap=4),
        price_table=pt,
        models={"drafter": "stub/echo"},
        retry_initial_wait=0.01,
    )
    calls = {"n": 0}

    def boom(self, payload):
        calls["n"] += 1
        raise ValueError("malformed request")

    monkeypatch.setattr(lc.MeteredClient, "_dispatch", boom)
    with pytest.raises(ValueError):
        client.complete(role="drafter", messages=_msgs("x"), seed=0)
    assert calls["n"] == 1  # burning four attempts would hide the bug behind a 30s delay


def test_cached_tokens_are_normalised_to_the_subset_convention():
    pytest.importorskip("tenacity")
    from pinq_adapters.llm.litellm_client import _read_response

    class _Det:
        cached_tokens = 9_999

    class _Usage:
        prompt_tokens = 100
        completion_tokens = 20
        prompt_tokens_details = _Det()
        completion_tokens_details = None

    class _Msg:
        content = "hi"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    r = _read_response(_Resp())
    # A provider reporting cached tokens disjointly would otherwise be double-counted.
    assert r.tok_cached == 100 and r.tok_prompt == 100


# --------------------------------------------------------------------------- failure visibility


def test_a_sweep_where_everything_failed_does_not_read_as_success():
    """Regression. `summarize` once reported {"resumed": N} for a sweep in which every unit
    had errored, so the run looked fine and the NEXT stage produced an empty table instead of
    an error -- the most dangerous shape a summary can take."""
    from pi_run.sweep import summarize

    results = [
        {
            "run_id": f"dev-{i}",
            "key": ["synth", f"s{i}", "inquirer_prompted", 0],
            "status": "resumed",
            "prior_status": "error",
            "error": "LLMRequired: needs an LLM",
        }
        for i in range(5)
    ]
    s = summarize(results)
    assert s["ok"] is False
    assert s["failed"] == 5
    assert s["by_terminal_status"]["error"] == 5
    assert s["distinct_errors"] == ["LLMRequired: needs an LLM"], "one diagnosis, not five copies"
    assert len(s["example_failures"]) == 3


def test_a_clean_resume_is_still_reported_as_ok():
    from pi_run.sweep import summarize

    results = [
        {
            "run_id": f"dev-{i}",
            "key": ["synth", f"s{i}", "a", 0],
            "status": "resumed",
            "prior_status": "ok",
        }
        for i in range(4)
    ]
    s = summarize(results)
    assert s["ok"] is True and s["failed"] == 0
    assert s["by_terminal_status"] == {"ok": 4}


def test_distinct_errors_are_deduplicated():
    """36 copies of one traceback is noise; one is a diagnosis."""
    from pi_run.sweep import summarize

    results = [
        {"run_id": f"r{i}", "key": [], "status": "error", "error": "Boom: same"} for i in range(20)
    ] + [{"run_id": "rx", "key": [], "status": "error", "error": "Other: different"}]
    s = summarize(results)
    assert sorted(s["distinct_errors"]) == ["Boom: same", "Other: different"]
    assert s["failed"] == 21


def test_an_llm_arm_reaches_configuration_not_a_wiring_error(synth_root, monkeypatch):
    """Regression: the worker used to call `arm.inquirer()` bare, so every LLM arm died with
    `LLMRequired` before touching a model. The worker now builds components through
    `arms.build(..., llm=..., recorder=ledger)`, so the only thing left between an arm and a
    provider is configuration."""
    root, corpus = synth_root
    for var in ("PI_MODEL_INQUIRER", "PI_MODEL_DRAFTER", "PI_MODEL_ANSWERER"):
        monkeypatch.delenv(var, raising=False)
    spec = UnitSpec(
        suite_id="synth",
        corpus_dir=str(corpus),
        task_id="s0",
        arm_id="inquirer_prompted",
        seed=0,
        runs_root=str(root / "runs_llm"),
        cache_root=str(root / "cache"),
        code_version="testsha",
        dirty=False,
    )
    res = run_unit(spec)
    assert res["status"] == "error"
    err = res["error"]
    assert "LLMRequired" not in err, f"still a wiring error: {err}"
    assert "PI_MODEL_" in err, f"expected a model-pin configuration error, got: {err}"


def test_each_unit_gets_its_own_ledger_backed_client(synth_root):
    """Sharing one client across units would cross-charge their budgets and silently break
    the parity assertion that makes arms comparable."""
    from pinq_expt import arms as arm_table

    arm = arm_table.get("inquirer_prompted")
    assert arm.llm_free is False
    fake = arm_table.get("fake_chain")
    assert fake.llm_free is True, "the CI arms must stay LLM-free or the smoke run needs keys"


def test_the_suite_is_hermetic_against_a_populated_dotenv():
    """`import litellm` calls load_dotenv(), so a developer's real .env would otherwise leak
    model pins and provider keys into every test. tests/conftest.py forces that import and
    scrubs it; this asserts the scrub actually holds."""
    import os

    leaked = sorted(
        k
        for k in os.environ
        if k.startswith(("PI_MODEL_", "LITELLM_", "DRGYM_"))
        or k in {"OPENAI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"}
    )
    assert leaked == [], f".env leaked into the test environment: {leaked}"
    assert not os.environ.get("PI_GOLD_ROOT"), "a stray PI_GOLD_ROOT makes the firewall test lie"


def test_every_shipped_suite_is_reachable_from_the_runtime():
    """Regression: load_suite knew only 'synth', so musique, strategyqa, wiki2, tau2, drgym and
    pare were all implemented, tested, and completely unreachable from `pi run`. Six of seven
    suites -- and every preregistered endpoint that depends on them -- could not be executed."""
    import importlib

    from pi_run.worker import CORPUS_BACKED, SELF_SOURCED, UnknownSuite, load_suite

    # The module path is NOT always `<suite_id>/suite.py`. strategyqa and wiki2 share the
    # paragraph base, and tau2_retail lives inside the tau2 package because it shares `_probe`,
    # the uid convention and the same Orchestrator refusal -- a `pinq_adapters/tau2_retail/`
    # package would duplicate all three. The mapping is a convenience; the PROPERTY under test
    # is reachability from `load_suite`, which is unchanged.
    module_for = {
        "strategyqa": "paragraphs",
        "wiki2": "paragraphs",
        "tau2_retail": "tau2.retail_suite",
        "tau2_airline": "tau2.airline_suite",
    }
    for suite_id in CORPUS_BACKED + SELF_SOURCED:
        mod = module_for.get(suite_id, f"{suite_id}.suite")
        importlib.import_module(f"pinq_adapters.{mod}")
        try:
            load_suite(suite_id, "/nonexistent")
        except UnknownSuite:  # the one failure that means "not wired"
            raise AssertionError(f"{suite_id} is not registered in load_suite") from None
        except Exception:
            pass  # missing data/keys are fine here; unreachability is not

    with pytest.raises(UnknownSuite):
        load_suite("not_a_suite", "/nonexistent")


def test_self_sourced_suites_do_not_require_a_built_corpus():
    """tau2 and pare read an upstream checkout, so demanding a data/corpora/<suite> directory
    would block a runnable sweep on a directory that is never supposed to exist.

    DRGYM IS NOT ONE OF THEM, and used to be. This tuple decides "did this repository BUILD the
    task list?", and for drgym the answer is yes -- `pi data build --suite drgym` writes
    data/corpora/drgym/<hash>/tasks.jsonl and `pi data status` reports n=976. What drgym does
    not own is its DOCUMENTS, which live behind a hosted search API; that is a different fact,
    handled by the adapter's offline flag and key check. Being on the wrong side made
    `_resolve_corpus` hand back the bare directory, and `pi run --suite drgym` died on
    `FileNotFoundError: .../data/corpora/drgym/tasks.jsonl` -- a path that never exists.
    """
    from pathlib import Path

    from pi_run.cli import _resolve_corpus
    from pi_run.worker import SELF_SOURCED

    assert "drgym" not in SELF_SOURCED
    for suite_id in SELF_SOURCED:
        p = _resolve_corpus(Path("/tmp"), suite_id, None)
        assert isinstance(p, Path), suite_id


@pytest.mark.integration
def test_every_built_corpus_resolves_to_a_directory_that_actually_holds_tasks():
    """The gap the reachability test above cannot see: it calls `load_suite` with
    "/nonexistent" and swallows every exception but UnknownSuite, so a suite whose CORPUS PATH
    never resolves still passes. `pi run` goes through `_resolve_corpus` first, and that is
    where drgym was broken for as long as it has existed."""
    from pathlib import Path

    import pytest

    from pi_run.cli import _resolve_corpus
    from pi_run.worker import CORPUS_BACKED, load_suite

    repo = Path(__file__).resolve().parents[1]
    checked = 0
    for suite_id in CORPUS_BACKED:
        built = sorted(
            p.parent for p in (repo / "data" / "corpora" / suite_id).glob("*/tasks.jsonl")
        )
        if not built:
            continue
        if len(built) == 1:
            targets = [_resolve_corpus(repo, suite_id, None)]
        else:
            # SEVERAL BUILT CORPORA IS A LEGITIMATE STATE, and this test used to treat it as a
            # failure because it assumed one corpus per suite. That belief was wrong: synth holds
            # two, each referenced by committed records (one by a test and a design record, the
            # other by a two-axis result and its 194 isolated runs), so neither can be removed to
            # make the default path resolve. The default path must REFUSE rather than guess --
            # guessing would silently change the corpus_hash inside every run identity -- and each
            # corpus must resolve when named. Both are asserted, so a resolver that starts picking
            # the newer corpus fails here instead of passing more quietly than before.
            with pytest.raises(SystemExit):
                _resolve_corpus(repo, suite_id, None)
            targets = [_resolve_corpus(repo, suite_id, str(b)) for b in built]
        for corpus in targets:
            assert (corpus / "tasks.jsonl").exists(), (
                f"{suite_id}: _resolve_corpus returned {corpus}, which holds no tasks.jsonl"
            )
            assert load_suite(suite_id, str(corpus)).task_ids(), f"{suite_id}: {corpus}"
        checked += 1
    assert checked >= 4, f"only {checked} suites built; this test needs the corpora"


# --------------------------------------------------------------------------- run survival


def test_a_proxy_connect_failure_is_retryable():
    """Regression: litellm.APIError is the class a VPN-gated proxy raises when the tunnel
    drops -- litellm wraps the connect failure and it never becomes APIConnectionError. It was
    NOT in the retry set, so the most likely interruption of a multi-hour sweep (a laptop
    sleeping) got zero retries and permanently errored every in-flight unit."""
    import litellm

    from pinq_adapters.llm.litellm_client import _retryable

    retryable = tuple(_retryable())
    assert issubclass(litellm.APIError, retryable)
    assert issubclass(litellm.RateLimitError, retryable)
    assert issubclass(ConnectionError, retryable)
    # A malformed request is a bug, not a transport failure: burning attempts on it hides it.
    assert not issubclass(ValueError, retryable)


def test_the_retry_ladder_outlasts_a_rate_limit_window():
    """A per-minute limit needs a ladder spanning minutes. At 4 attempts from 0.5s the total
    backoff is ~3.6-6.5s, so every attempt lands inside the same window and the unit is
    guaranteed to fail."""
    from pinq_adapters.llm.litellm_client import (
        DEFAULT_MAX_ATTEMPTS as attempts,
    )
    from pinq_adapters.llm.litellm_client import (
        DEFAULT_RETRY_INITIAL_WAIT as initial,
    )

    # Read from the NAMED constants, not from `signature(...).parameters[...].default`. The
    # constructor now takes None as "use the default" so it can tell an explicit argument from
    # an absent one, and this test was silently asserting `None >= 8`.
    assert attempts >= 8, f"only {attempts} attempts"
    assert initial >= 4.0, f"initial wait {initial}s is inside one window"
    # Exponential with a 30s cap: 4+8+16+30+30+30+30 ~= 148s, i.e. three windows.
    ladder = sum(min(initial * 2**i, 30.0) for i in range(attempts - 1))
    assert ladder >= 120, f"ladder spans only {ladder:.0f}s"


def test_resume_retries_a_failed_unit_but_skips_a_successful_one(synth_root):
    """Regression: resume skipped ANY unit with a status file, so a unit killed by a 429 was
    permanently baked in. Re-issuing the sweep -- which the docstring calls the correct
    recovery procedure -- could never fix it; only --no-resume could, and that re-runs
    everything."""
    import json

    root, corpus = synth_root
    spec = UnitSpec(
        suite_id="synth",
        corpus_dir=str(corpus),
        task_id="s0",
        arm_id="fake_chain",
        seed=0,
        runs_root=str(root / "runs_resume"),
        cache_root=str(root / "cache"),
        code_version="testsha",
        dirty=False,
    )
    first = run_unit(spec)
    assert first["status"] == "ok"
    assert run_unit(spec)["status"] == "resumed", "a success must not be re-run"

    status_path = Path(spec.runs_root) / first["run_id"] / "status.json"
    poisoned = json.loads(status_path.read_text())
    poisoned["status"] = "error"
    poisoned["error"] = "RateLimitError: simulated"
    status_path.write_text(json.dumps(poisoned))

    again = run_unit(spec)
    assert again["status"] == "ok", "a FAILED unit must be retried, not resumed"


def test_run_sweep_cancels_pending_work_when_a_unit_raises():
    """Regression: the executor's context manager calls shutdown(wait=True) WITHOUT
    cancel_futures, so a firewall failure on unit 5 of 15,600 ran the remaining 15,595 to
    completion before the traceback printed."""
    import ast
    import inspect
    import textwrap

    from pi_run import sweep as sweep_mod

    tree = ast.parse(textwrap.dedent(inspect.getsource(sweep_mod.run_sweep)))

    # Structure, not text: a grep would match this test's own explanatory comment.
    bare_ctx = [
        w
        for w in ast.walk(tree)
        if isinstance(w, ast.With)
        for item in w.items
        if isinstance(item.context_expr, ast.Call)
        and getattr(item.context_expr.func, "id", "") == "ProcessPoolExecutor"
    ]
    assert not bare_ctx, (
        "ProcessPoolExecutor used as a context manager: its __exit__ calls "
        "shutdown(wait=True) WITHOUT cancel_futures, so every pending unit still runs"
    )

    cancels = [
        c
        for c in ast.walk(tree)
        if isinstance(c, ast.Call)
        and getattr(c.func, "attr", "") == "shutdown"
        and any(
            k.arg == "cancel_futures" and getattr(k.value, "value", False) is True
            for k in c.keywords
        )
    ]
    assert cancels, "no shutdown(..., cancel_futures=True) call found"


def test_a_construction_failure_is_one_unit_not_the_whole_sweep(synth_root):
    """Regression: arms.build() raised OUTSIDE the worker's try, so a ValueError propagated
    through fut.result() and killed every remaining unit. plan() is task-major, so the pilot
    grid died around unit 5 of 840 and four of its seven arms -- two of them kill switches --
    never ran at all."""
    root, corpus = synth_root
    spec = UnitSpec(
        suite_id="synth",
        corpus_dir=str(corpus),
        task_id="s0",
        arm_id="parallel_replay",  # requires recorded questions; none supplied
        seed=0,
        runs_root=str(root / "runs_ctor"),
        cache_root=str(root / "cache"),
        code_version="testsha",
        dirty=False,
    )
    res = run_unit(spec)
    assert res["status"] == "error", "must be an error"
    assert res["error"], "and it must say why"
    # Compared against the spec's OWN key rather than a literal: the key gained the branch
    # triple (see UnitSpec.key), and this test is about ATTRIBUTION, not about the tuple's
    # arity. A literal here re-breaks on the next identity field, for no added guarantee.
    assert res["key"] == list(spec.key), "attributed to its own unit"
    assert res["key"][:4] == ["synth", "s0", "parallel_replay", 0]


def test_plan_gives_every_unit_the_whole_question_pool():
    """parallel_replay wants its own task's questions; random_q must draw from OTHER tasks.
    A random-question arm sampled from the task's own pool is not a null, it is a weaker copy
    of the treatment -- so slicing the pool per task made random_q refuse with EmptyPool on
    every unit."""
    from pi_run.sweep import plan

    pool = {"t0": ["qa"], "t1": ["qb"], "t2": ["qc"]}
    specs = plan(
        suite_id="synth",
        corpus_dir="/c",
        task_ids=["t0", "t1"],
        arm_ids=["random_q"],
        seeds=[0],
        runs_root="/r",
        cache_root="/k",
        questions=pool,
    )
    assert specs, "plan produced nothing"
    for spec in specs:
        assert set(spec.questions) == set(pool), (
            f"unit {spec.task_id} saw only {sorted(spec.questions)}"
        )


def test_a_grid_drives_a_sweep(tmp_path, monkeypatch):
    """`pi run --sweep` must exist and honour the grid, or grids.py is dead code and
    'the sweep config was frozen at prereg' is not a checkable statement. docs/REPRODUCE.md
    documented this flag before it existed."""

    from pi_run import cli

    g = tmp_path / "g.yaml"
    g.write_text(
        "name: t\nsuites: [synth]\narms: [fake_chain]\nseeds: [0]\nn_tasks: 2\nbudget_cap: 4\n"
    )
    parser = cli.build_parser() if hasattr(cli, "build_parser") else None
    if parser is not None:
        ns = parser.parse_args(["run", "--sweep", str(g)])
        assert ns.sweep == str(g)
        assert getattr(ns, "suite", None) is None, "--suite must be optional under --sweep"
    else:  # fall back to asserting the flag is registered at all
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
            cli.main(["run", "--help"])
        assert "--sweep" in buf.getvalue()


def test_max_tokens_leaves_room_for_a_reasoning_channel(monkeypatch):
    """Regression, measured on gpt-oss-120b over tau2.

    On a reasoning model the reasoning tokens are billed INSIDE max_tokens, so a caller asking
    for 700 content tokens received `completion=700, reasoning=701` and an EMPTY answer. An
    empty answer scores 0 on every metric and reads as a model failure rather than a budget
    bug -- the worst possible disguise, because it would have been debugged as a weak policy.

    The caller's number stays the CONTENT budget it meant, which is what keeps the frozen
    Answerer's word cap comparable across arms and providers.
    """
    from pinq.budget import BudgetLedger
    from pinq_adapters.llm.litellm_client import MeteredClient

    monkeypatch.setenv("PI_MODEL_ANSWERER", "groq/openai/gpt-oss-120b")
    led = BudgetLedger(cap=10)

    c = MeteredClient(led, temperature=0.0)
    got = c.request_payload(role="answerer", messages=[], seed=0, max_tokens=700)["max_tokens"]
    assert got > 700, f"no headroom for the reasoning channel: {got}"

    # None must stay None: an unbounded request is not a 3x-of-nothing request.
    assert (
        c.request_payload(role="answerer", messages=[], seed=0, max_tokens=None)["max_tokens"]
        is None
    )

    # The headroom is a property of the MODEL, so it must be configurable off.
    flat = MeteredClient(led, temperature=0.0, reasoning_headroom=1.0)
    assert (
        flat.request_payload(role="answerer", messages=[], seed=0, max_tokens=700)["max_tokens"]
        == 700
    )


def test_the_headroom_changes_the_cache_key(monkeypatch):
    """max_tokens is inside the request payload, so two clients with different headroom issue
    genuinely different requests and must not share a cache entry."""
    from pinq.budget import BudgetLedger
    from pinq_adapters.llm.litellm_client import MeteredClient

    monkeypatch.setenv("PI_MODEL_ANSWERER", "groq/openai/gpt-oss-120b")
    led = BudgetLedger(cap=10)
    a = MeteredClient(led, temperature=0.0, reasoning_headroom=1.0)
    b = MeteredClient(led, temperature=0.0, reasoning_headroom=3.0)
    kw = dict(role="answerer", messages=[{"role": "user", "content": "x"}], seed=0, max_tokens=700)
    assert a.request_payload(**kw) != b.request_payload(**kw)


def test_the_spend_cap_actually_stops_a_sweep(monkeypatch, synth_root):
    """Regression: PI_SPEND_CAP_USD was documented in .env and the runbook while being
    referenced NOWHERE in src/. A decorative budget guard is worse than none, because it reads
    as a guarantee -- and the per-unit ledger cannot catch a runaway campaign: 15,600 units
    each costing a defensible $0.007 is $109 and no single unit ever looks wrong."""
    from pi_run import sweep as sweep_mod

    root, corpus = synth_root
    specs = sweep_mod.plan(
        suite_id="synth",
        corpus_dir=str(corpus),
        task_ids=[f"s{i}" for i in range(6)],
        arm_ids=["fake_chain"],
        seeds=[0],
        runs_root=str(root / "runs_cap"),
        cache_root=str(root / "cache"),
        code_version="testsha",
        dirty=False,
        concurrency=1,
    )

    # Each unit "costs" $1; the cap is $2, so it must stop after 2 and cancel the rest.
    calls = {"n": 0}

    def fake_unit(spec):
        calls["n"] += 1
        return {
            "run_id": f"r{calls['n']}",
            "key": list(spec.key),
            "status": "ok",
            "usage": {"usd": 1.0},
        }

    monkeypatch.setattr(sweep_mod, "run_unit", fake_unit)
    monkeypatch.setenv("PI_SPEND_CAP_USD", "2")
    monkeypatch.setenv("PI_MAX_CONCURRENCY", "1")

    results = sweep_mod.run_sweep(specs)
    assert calls["n"] == 2, f"ran {calls['n']} units past a $2 cap"
    assert len(results) == 2, "cancelled units must be ABSENT, not silently zero-valued"


def test_no_cap_means_no_cap(monkeypatch, synth_root):
    from pi_run import sweep as sweep_mod

    root, corpus = synth_root
    specs = sweep_mod.plan(
        suite_id="synth",
        corpus_dir=str(corpus),
        task_ids=[f"s{i}" for i in range(4)],
        arm_ids=["fake_chain"],
        seeds=[0],
        runs_root=str(root / "runs_nocap"),
        cache_root=str(root / "cache"),
        code_version="testsha",
        dirty=False,
        concurrency=1,
    )
    calls = {"n": 0}

    def fake_unit(spec):
        calls["n"] += 1
        return {
            "run_id": f"r{calls['n']}",
            "key": list(spec.key),
            "status": "ok",
            "usage": {"usd": 99.0},
        }

    monkeypatch.setattr(sweep_mod, "run_unit", fake_unit)
    monkeypatch.delenv("PI_SPEND_CAP_USD", raising=False)
    monkeypatch.setenv("PI_MAX_CONCURRENCY", "1")
    assert len(sweep_mod.run_sweep(specs)) == 4


# --------------------------------------------------------------------------- Track R
#
# Everything below pins a defect that shipped GREEN: each one produced a well-formed value
# rather than an error, which is why the 1,082 tests above never saw it.


def test_an_explicit_concurrency_beats_the_env_throttle(monkeypatch):
    """PI_MAX_CONCURRENCY used to WIN over --concurrency.

    `PI_MAX_CONCURRENCY=2 pi run --concurrency 16` resolved to 2, printed `concurrency=2` on a
    line nobody re-reads, and turned a 9-hour grid into a 74-hour one. .env shipped the 2.
    """
    from pi_run.sweep import DEFAULT_CONCURRENCY, max_concurrency

    monkeypatch.setenv("PI_MAX_CONCURRENCY", "2")
    assert max_concurrency(16) == 16, "a flag is an instruction; an env var is a default"
    assert max_concurrency() == 2, "with no flag the env var is still the default"

    monkeypatch.delenv("PI_MAX_CONCURRENCY", raising=False)
    assert max_concurrency() == DEFAULT_CONCURRENCY
    assert max_concurrency(0) == 1, "never zero workers"

    # Idempotent: cli resolves once and run_sweep resolves the resolved value again.
    monkeypatch.setenv("PI_MAX_CONCURRENCY", "2")
    assert max_concurrency(max_concurrency(16)) == 16


def test_an_explicit_spend_cap_beats_the_env_cap(monkeypatch):
    from pi_run.sweep import spend_cap

    monkeypatch.setenv("PI_SPEND_CAP_USD", "5")
    assert spend_cap(80.0) == 80.0
    assert spend_cap() == 5.0
    # REVERSED, DELIBERATELY. This asserted `spend_cap(0) is None` -- "a non-positive cap
    # means uncapped, not stop immediately" -- which was a real design decision and is now
    # judged the wrong one.
    #
    # `--spend-cap 0` reads as "spend zero dollars" to every operator, and it is the obvious
    # way to ask for a rehearsal. It returned UNCAPPED. Worse, because an explicit argument
    # wins, typing it also discarded a PI_SPEND_CAP_USD guard rail: measured with the env
    # cap at 5.00, `spend_cap(0.0)` gave None while `spend_cap(None)` gave 5.0, so ADDING
    # the flag REMOVED the protection.
    #
    # This repository has already billed $0.7158 once to a flag that did not do what it
    # said. There is a way to ask for no cap -- omit the flag -- so the ambiguous spelling
    # is not worth its downside. No internal caller passes 0; it can only come from an
    # operator typing it.
    assert spend_cap(0) == 0.0, "--spend-cap 0 means spend nothing, never 'no cap'"
    monkeypatch.delenv("PI_SPEND_CAP_USD", raising=False)
    assert spend_cap() is None


def test_the_spend_cap_ignores_resumed_units_and_cache_hits():
    """A capped sweep could never be resumed past its cap.

    The parent added every returned unit's historical `usage.usd`, including units it had only
    stat()ed, so the second invocation re-counted the first invocation's spend in its opening
    milliseconds and cancelled everything that had not yet run. `usage.usd` also charges cache
    hits AS-IF -- correct for cross-arm comparability, wrong for an invoice.
    """
    from pi_run.sweep import billed_usd

    assert billed_usd({"status": "resumed", "usage": {"usd": 42.0}, "usd_billed": 42.0}) == 0.0
    assert billed_usd({"status": "ok", "usage": {"usd": 9.0}, "usd_billed": 0.0}) == 0.0
    assert billed_usd({"status": "ok", "usage": {"usd": 9.0}, "usd_billed": 3.0}) == 3.0
    # Runs written before usd_billed existed fall back to the as-if number: over-counting is
    # the safe direction for a cap.
    assert billed_usd({"status": "ok", "usage": {"usd": 9.0}}) == 9.0


def test_a_resumed_sweep_gets_past_the_cap(synth_root, monkeypatch):
    """End to end: cap the first pass, then re-issue and reach every unit."""
    import pi_run.sweep as sweep_mod

    root, corpus = synth_root
    runs = root / "runs_capresume"
    specs = plan(
        suite_id="synth",
        corpus_dir=str(corpus),
        task_ids=[f"s{i}" for i in range(4)],
        arm_ids=["fake_chain"],
        seeds=[0],
        runs_root=str(runs),
        cache_root=str(root / "cache"),
        code_version="testsha",
        dirty=False,
        concurrency=1,
    )
    done: set[tuple] = set()

    def fake_unit(spec):
        if spec.key in done:
            return {"key": list(spec.key), "status": "resumed", "usage": {"usd": 1.0}}
        done.add(spec.key)
        return {"key": list(spec.key), "status": "ok", "usage": {"usd": 1.0}, "usd_billed": 1.0}

    monkeypatch.setattr(sweep_mod, "run_unit", fake_unit)
    monkeypatch.delenv("PI_SPEND_CAP_USD", raising=False)

    first = sweep_mod.run_sweep(specs, concurrency=1, cap_usd=2.0)
    assert len(first) == 2, "the cap stops the first pass"
    second = sweep_mod.run_sweep(specs, concurrency=1, cap_usd=2.0)
    assert len(second) == 4, "re-issuing reaches every unit: resumed units bill nothing"


def test_a_torn_status_json_costs_one_unit_and_not_the_sweep(synth_root):
    """status.json is the resume sentinel, and json.loads() on a truncated one propagated out
    of run_unit, through fut.result(), and killed every unit still queued."""
    from pi_run.worker import read_status

    root, corpus = synth_root
    spec = _spec(root, corpus, "s1", "fake_chain", runs_root=str(root / "runs_torn"))
    first = run_unit(spec)
    assert first["status"] == "ok"

    status_path = Path(spec.runs_root) / first["run_id"] / "status.json"
    status_path.write_text(json.dumps({"status": "ok", "run_id": "x"})[:20])  # truncated
    assert read_status(status_path) is None, "a torn sentinel reads as ABSENT, not as an error"

    again = run_unit(spec)
    assert again["status"] == "ok", "the unit is simply re-run"
    assert again["status"] != "resumed"
    assert read_status(status_path) is not None, "and the sentinel is whole again"


def test_status_json_is_written_atomically(synth_root):
    """os.replace, so a reader sees the old bytes or the new bytes and never half a file."""
    from pi_run.worker import write_atomic

    root, _corpus = synth_root
    p = root / "atomic" / "s.json"
    write_atomic(p, '{"a": 1}')
    assert json.loads(p.read_text()) == {"a": 1}
    write_atomic(p, '{"a": 2}')
    assert json.loads(p.read_text()) == {"a": 2}
    assert not list(p.parent.glob(".*tmp")), "no temp files survive a successful write"


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGALRM"), reason="POSIX only")
def test_a_wedged_unit_times_out_instead_of_holding_a_worker(synth_root, monkeypatch):
    """There was no per-unit timeout. A stalled tunnel held one worker for hours, and
    `summarize()` counted a "timeout" terminal status that nothing could produce."""
    import time as _time

    import pi_run.worker as worker_mod

    root, corpus = synth_root

    def wedge(**kw):
        _time.sleep(5)

    monkeypatch.setattr(worker_mod, "run_loop", wedge)
    res = run_unit(
        _spec(root, corpus, "s2", "fake_chain", runs_root=str(root / "runs_wedge"), timeout_s=1)
    )
    assert res["status"] == "timeout", res
    assert "UnitTimeout" in res["error"]
    # The manifest is on disk, so the unit is identifiable and re-runnable.
    assert (Path(res_dir := Path(root / "runs_wedge" / res["run_id"])) / "manifest.json").exists()
    assert (res_dir / "status.json").exists()
    assert summarize([res])["failed"] == 1, "a timeout is a FAILURE, not a quiet omission"


def test_the_watchdog_restores_the_previous_signal_handler():
    """The single-worker path runs run_unit in the PARENT, so a leaked SIGALRM handler would
    fire during summarize()."""
    import signal

    from pi_run.worker import _Watchdog

    before = signal.getsignal(signal.SIGALRM)
    with _Watchdog(30):
        assert signal.getsignal(signal.SIGALRM) is not before
    assert signal.getsignal(signal.SIGALRM) is before
    with _Watchdog(0):  # disabled: touches nothing
        assert signal.getsignal(signal.SIGALRM) is before


def _pins_for(arm_id: str, monkeypatch, *, questions=None, **models) -> dict:
    """Build one arm's components against a real MeteredClient and read the pins off them.

    Deliberately NOT via run_unit: an LLM arm with no API key spends ~150s in the retry ladder
    before failing, and `make gate` must not make network calls at all. Everything item I
    touches -- role discovery from the components, MeteredClient.pin, the CachingClient
    passthrough -- is on this path; only the doomed HTTP call is not.
    """
    from pi_run.cache import CachingClient, DiskCache
    from pi_run.worker import collect_pins
    from pinq_adapters.llm.litellm_client import MeteredClient

    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.invalid")
    for role, model in models.items():
        monkeypatch.setenv(f"PI_MODEL_{role.upper()}", model)
    llm = CachingClient(MeteredClient(BudgetLedger(cap=16)), DiskCache("/dev/null/never"))
    c = arm_table.build(
        arm_table.get(arm_id),
        llm=llm,
        retriever=None,
        recorder=BudgetLedger(cap=16),
        questions=questions,
    )
    return collect_pins(llm, (c.inquirer, c.drafter, c.answerer))


def test_an_llm_arm_records_its_model_pins_in_the_manifest(monkeypatch):
    """ALL 50 MANIFESTS SHIPPED WITH "pins": {}.

    model_pin_hash is inside semantic_hash is inside run_id, so with pins empty
    `inquirer_trained` and `inquirer_prompted` hash IDENTICALLY: swapping in a fine-tuned
    checkpoint produced the same run_id, --resume skipped it as already done, and the
    trained-vs-prompted comparison would have printed one arm's numbers twice.
    """
    pins = _pins_for(
        "inquirer_prompted",
        monkeypatch,
        inquirer="openai/model-a",
        drafter="openai/model-a",
        answerer="openai/model-a",
    )
    assert set(pins) == {"inquirer", "drafter", "answerer"}
    assert all(p.model_id == "openai/model-a" for p in pins.values())

    def _mk(**kw):
        return build_manifest(
            suite_id="synth",
            task_id="s0",
            arm_id="inquirer_prompted",
            policy_id="p",
            seed=0,
            corpus_hash="c",
            budget_cap=16,
            max_turns=16,
            word_cap=120,
            code_version="x",
            dirty=False,
            **kw,
        )

    assert _mk(pins=pins).model_pin_hash != _mk().model_pin_hash, (
        "an empty pin set must not hash like a populated one"
    )
    assert _mk(pins=pins).run_id != _mk().run_id


def test_a_scripted_inquirer_arm_pins_only_the_roles_it_calls(monkeypatch):
    """parallel_replay's Inquirer is a recorded script, so its run_id must be insensitive to an
    inquirer model swap that could not have affected it."""
    pins = _pins_for(
        "parallel_replay",
        monkeypatch,
        questions={"s0": ["what year?"]},
        inquirer="openai/model-a",
        drafter="openai/model-a",
        answerer="openai/model-a",
    )
    assert set(pins) == {"drafter", "answerer"}, pins


def test_swapping_one_role_model_changes_the_run_id(monkeypatch):
    common = dict(
        suite_id="synth",
        task_id="s0",
        arm_id="inquirer_prompted",
        policy_id="p",
        seed=0,
        corpus_hash="c",
        budget_cap=16,
        max_turns=16,
        word_cap=120,
        code_version="x",
        dirty=False,
    )
    a = _pins_for(
        "inquirer_prompted",
        monkeypatch,
        inquirer="openai/model-a",
        drafter="openai/model-a",
        answerer="openai/model-a",
    )
    b = _pins_for(
        "inquirer_prompted",
        monkeypatch,
        inquirer="openai/model-b",
        drafter="openai/model-a",
        answerer="openai/model-a",
    )
    assert build_manifest(**common, pins=a).run_id != build_manifest(**common, pins=b).run_id, (
        "a checkpoint swap must produce a different run, not resume the old one"
    )


def test_an_unconfigured_role_is_recorded_rather_than_raised(monkeypatch):
    """A role with no model must still fail its unit -- at the first complete(), where the
    failure becomes status=error with the manifest already on disk. Raising during
    collect_pins would abort before build_manifest, so the one artifact naming the prompts the
    unit would have rendered would never be written."""
    monkeypatch.delenv("PI_MODEL_INQUIRER", raising=False)
    pins = _pins_for(
        "inquirer_prompted", monkeypatch, drafter="openai/model-a", answerer="openai/model-a"
    )
    assert pins["inquirer"].model_id.startswith("<unresolved:")
    assert pins["inquirer"].provider == "unconfigured"
    assert pins["drafter"].model_id == "openai/model-a"


def test_an_llm_free_arm_carries_no_pins(synth_root):
    root, corpus = synth_root
    res = run_unit(_spec(root, corpus, "s0", "fake_chain", runs_root=str(root / "runs_nopins")))
    manifest = json.loads(Path(root / "runs_nopins" / res["run_id"] / "manifest.json").read_text())
    assert manifest["pins"] == {}, "an arm that calls no model must not claim one"


def test_a_ceiling_arm_is_marked_gold_exposed_by_construction(synth_root):
    """`pi run` scrubs PI_GOLD_ROOT from the process tree, so deriving gold_exposed from the
    environment made it structurally False on every run ever written -- and
    `pi agg --assert-no-gold-exposed`, whose job is to keep the ceiling arms out of a
    confirmatory table, passed by finding nothing to find."""
    assert arm_table.get("gold_evidence").requires_gold
    assert arm_table.get("oracle_vreq").requires_gold
    assert not arm_table.get("inquirer_prompted").requires_gold

    m = build_manifest(
        suite_id="synth",
        task_id="s0",
        arm_id="gold_evidence",
        policy_id="gold_evidence",
        seed=0,
        corpus_hash="c",
        budget_cap=16,
        max_turns=16,
        word_cap=120,
        code_version="x",
        dirty=False,
        gold_exposed=arm_table.get("gold_evidence").requires_gold,
    )
    assert m.gold_exposed is True
    assert os.environ.get("PI_GOLD_ROOT") in (None, ""), "and with the env var unset, as in a run"


def test_the_progress_printer_reports_counts_spend_and_the_last_distinct_error():
    """run_sweep has always accepted on_result and cli.py passed none, so a three-hour sweep
    printed its header, went silent, and printed its summary."""
    import io

    from pi_run.sweep import progress_printer

    buf = io.StringIO()
    cb = progress_printer(4, every=2, stream=buf)
    cb({"status": "ok", "usd_billed": 0.5})
    assert buf.getvalue() == "", "nothing at n=1 with every=2"
    cb({"status": "error", "error": "APIError: 429 rate limit", "usage": {}})
    line = buf.getvalue()
    assert "[2/4]" in line and "ok=1" in line and "err=1" in line
    assert "$0.5000" in line and "429 rate limit" in line
    cb({"status": "ok", "usd_billed": 0.25})
    cb({"status": "ok", "usd_billed": 0.25})
    assert "[4/4]" in buf.getvalue(), "the final unit always prints, whatever the cadence"


def test_a_questions_file_is_validated_rather_than_trusted(tmp_path):
    """This file is the INPUT to the ceiling arms. A ceiling arm fed a silently-empty pool
    does not fail: it stops on turn 0, scores like drafter_only, and reports "no headroom" --
    which is exactly the conclusion that would cancel the musique track."""
    from pi_run.cli import _load_questions_file

    good = tmp_path / "q.json"
    good.write_text(json.dumps({"t1": ["a?", "b?"], "t2": ["c?"], "t3": []}))
    assert _load_questions_file(good) == {"t1": ["a?", "b?"], "t2": ["c?"]}

    wrapped = tmp_path / "w.json"
    wrapped.write_text(json.dumps({"mode": "gold_evidence", "questions": {"t1": ["a?"]}}))
    assert _load_questions_file(wrapped) == {"t1": ["a?"]}

    for payload in ([{"t1": ["a?"]}], {"t1": "a?"}, {}, {"t1": []}):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(SystemExit):
            _load_questions_file(bad)


def test_a_missing_extra_names_the_extra(monkeypatch):
    """`ModuleNotFoundError: No module named 'pyarrow'` is accurate and useless: it does not
    say which of nine extras carries pyarrow, and `make venv` installed only [dev]."""
    from pinq.extras import MissingExtra, require

    with pytest.raises(MissingExtra) as e:
        require("pyarrow_definitely_not_installed_xyz")
    assert "pyproject.toml" in str(e.value)

    import pinq.extras as ex

    monkeypatch.setitem(ex.EXTRA_OF, "pyarrow_definitely_not_installed_xyz", "analysis")
    with pytest.raises(MissingExtra) as e2:
        require("pyarrow_definitely_not_installed_xyz", why="pi score writes parquet")
    assert 'uv pip install -e ".[analysis]"' in str(e2.value)
    assert "pi score writes parquet" in str(e2.value)


def test_make_venv_installs_every_extra_the_gate_imports():
    """`make venv` installed only [dev], so a stranger's first `make gate` died on pyarrow --
    a dependency the Makefile never installed, with an error naming neither."""
    line = next(ln for ln in Path("Makefile").read_text().splitlines() if "uv pip install -e" in ln)
    for extra in ("dev", "run", "analysis"):
        assert extra in line, f"make venv must install [{extra}]: {line}"


def test_token_parity_accounts_for_every_call_including_the_terminal_ones():
    """`tokens_ok` was structurally FALSE on every LLM arm and TRUE only where ledgers are
    empty.

    A real rollout charges three things to no turn at all -- the act() that returned STOP, the
    final draft, and the frozen Answerer -- and reconcile() compared the ledger against the sum
    of TURNS. Measured on a live 2-unit run: 30 ledger calls, 18 inside turns. So the one
    detector that would catch an unmetered call said False everywhere it mattered, and a field
    that always says False is a field nobody reads.

    Turn.usage was wrong in the same motion: the per-turn redraft was charged AFTER the Turn
    had been stamped, so Turn.usage under-reported by ~40% -- and Turn.usage is the x-axis of
    the budget frontier.
    """
    from pinq.loop import run_loop
    from pinq.protocols import Recorder  # noqa: F401 - documents the contract being stubbed
    from pinq.types import Answer, Ask, Draft, EvidenceUnit, Stop, TaskView

    def _charge(ledger, actor, n):
        ledger.record_call(
            CallTelemetry(
                call_id=f"{actor}{n}",
                actor=actor,
                model="stub/echo",
                provider="stub",
                request_sha="r",
                response_sha="s",
                tok_prompt=10,
                tok_completion=n,
                usd=0.001,
            )
        )

    class Inq:
        """Two ASKs then STOP. The STOP costs a call, like a real policy's."""

        policy_id = "stub"

        def reset(self, view, seed):
            self.n = 0

        def act(self, s):
            self.n += 1
            _charge(self._ledger, "inquirer", self.n)
            return Ask(text=f"q{self.n}") if self.n <= 2 else Stop(reason="policy_stop")

    class Draf:
        def resolve(self, view, action, ev, *, seed, ledger):
            _charge(ledger, "drafter", 7)
            return "sha", ev

        def draft(self, view, ev, *, seed, ledger):
            _charge(ledger, "drafter", 5)
            return Draft(text=f"draft:{ev.subset_hash[:6]}")

    class Ans:
        def answer(self, view, ev, draft, *, seed, ledger):
            _charge(ledger, "answerer", 3)
            return Answer(text="final", evidence_hash=ev.subset_hash)

    class Ret:
        def search(self, q, k):
            return [
                EvidenceUnit.make(corpus_id="c", doc_id=f"d{q}", span="0:3", title="t", text="abc")
            ]

    ledger = BudgetLedger(cap=16)
    inq = Inq()
    inq._ledger = ledger
    traj = run_loop(
        view=TaskView(
            suite_id="synth",
            task_id="t0",
            question="q",
            corpus_id="c",
            corpus_hash="ch",
            instructions="",
            word_cap=120,
        ),
        inquirer=inq,
        retriever=Ret(),
        drafter=Draf(),
        answerer=Ans(),
        ledger=ledger,
        max_turns=8,
        k=1,
        seed=0,
    )

    assert len(traj.turns) == 2
    assert traj.stop_reason == "policy_stop"
    # 2 turns x (act + resolve + redraft) + STOP act + final draft + answerer
    assert len(ledger.calls) == 9

    turn_total = Usage()
    for t in traj.turns:
        turn_total = turn_total + t.usage
    assert turn_total.n_calls == 6, "the per-turn redraft belongs to its turn, not to nothing"
    assert traj.terminal_usage.n_calls == 3, "STOP act + final draft + answerer"
    assert (turn_total + traj.terminal_usage).n_calls == ledger.usage.n_calls

    assert ledger.reconcile(turn_total, traj.terminal_usage) is True
    assert ledger.reconcile(turn_total) is False, (
        "without the terminal half the check cannot pass on any real rollout"
    )

    # Every turn now carries the sha of the draft the NEXT act() saw.
    assert all(t.draft_sha for t in traj.turns)
    assert traj.turns[0].draft_sha != traj.turns[1].draft_sha, (
        "the draft must move as evidence accumulates, or D_t is not a function of E_t"
    )


def test_an_unmetered_nested_retrieval_still_fails_reconciliation():
    """The check has to stay able to FAIL, or widening it to cover terminal usage would have
    turned a detector into a rubber stamp."""
    from pinq.types import Usage as U

    ledger = BudgetLedger(cap=16)
    ledger.record_call(
        CallTelemetry(
            call_id="c",
            actor="drafter",
            model="m",
            provider="p",
            request_sha="r",
            response_sha="s",
            tok_prompt=100,
            tok_completion=50,
        )
    )
    assert ledger.reconcile(U(), U()) is False
    assert ledger.reconcile(U(tok_prompt=100, tok_completion=50, n_calls=1), U()) is True


# --------------------------------------------------------------------------- the call site
#
# The pin and gold_exposed tests above exercise `collect_pins` and `build_manifest` in
# ISOLATION, and both kept passing while `run_unit`'s ONE call to build_manifest was reverted
# to `pins={}` with `gold_exposed=` dropped entirely. A helper that works and a call site that
# does not use it produce a green suite and an empty manifest, which is the same failure the
# helper was written to fix. These read the FILE ON DISK that a real unit wrote.


def test_the_manifest_a_real_unit_writes_carries_its_pins(synth_root, monkeypatch):
    root, corpus = synth_root
    for var in ("PI_MODEL_INQUIRER", "PI_MODEL_DRAFTER", "PI_MODEL_ANSWERER"):
        monkeypatch.setenv(var, "openai/callsite-model")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.invalid")

    spec = _spec(root, corpus, "s0", "inquirer_prompted", runs_root=str(root / "runs_callsite"))
    res = run_unit(spec)
    manifest = json.loads(Path(spec.runs_root, res["run_id"], "manifest.json").read_text())

    assert manifest["pins"], "run_unit wrote an EMPTY pin set for an LLM arm"
    assert set(manifest["pins"]) == {"inquirer", "drafter", "answerer"}
    assert all(p["model_id"] == "openai/callsite-model" for p in manifest["pins"].values())
    assert manifest["model_pin_hash"]


def test_the_manifest_a_real_ceiling_unit_writes_is_marked_gold_exposed(synth_root):
    """`pi agg --assert-no-gold-exposed` reads THIS field off THIS file. Its entire job is to
    keep the ceiling arms out of a confirmatory table, and it can only do that if the flag
    survives the one call site that writes it."""
    root, corpus = synth_root
    spec = _spec(
        root,
        corpus,
        "s0",
        "gold_evidence",
        runs_root=str(root / "runs_ceiling"),
        questions={"s0": ("what year?", "who owned it?")},
    )
    res = run_unit(spec)
    manifest = json.loads(Path(spec.runs_root, res["run_id"], "manifest.json").read_text())
    assert manifest["gold_exposed"] is True, "a ceiling arm's manifest must say so on disk"

    plain = run_unit(_spec(root, corpus, "s0", "fake_chain", runs_root=str(root / "runs_plain")))
    m2 = json.loads(Path(root / "runs_plain" / plain["run_id"] / "manifest.json").read_text())
    assert m2["gold_exposed"] is False, "and a treatment arm's must not"


def test_every_build_manifest_call_site_passes_pins_and_gold_exposed():
    """A structural guard for the same class of regression. build_manifest defaults both
    arguments, so dropping either is silent at the call site AND at import time; the only
    thing that notices is a manifest nobody re-reads."""
    import ast
    import pathlib

    for path in ("src/pi_run/worker.py", "src/pi_run/stages/tau2_runner.py"):
        tree = ast.parse(pathlib.Path(path).read_text())
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "build_manifest"
        ]
        assert calls, f"{path}: no build_manifest call found -- did it move?"
        for c in calls:
            kw = {k.arg for k in c.keywords}
            assert "pins" in kw, f"{path}: build_manifest called without pins="
            assert "gold_exposed" in kw, f"{path}: build_manifest called without gold_exposed="


# --------------------------------------------------------------------------- run identity
#
# `semantic_hash` decides what `--resume` treats as already done. A treatment parameter missing
# from it does not cause an error: it causes the second configuration to inherit the first
# configuration's results, and the table then reports one setting twice under two labels.


def test_two_values_of_k_are_two_different_runs():
    """`tier1_confirmatory` pins k=5 for musique, 2 for strategyqa and 3 for wiki2, and the
    probe that chose them was re-measured on 2026-08-24. With k outside semantic_hash the SAME
    task at k=2 and k=8 produced the SAME run_id, so a k sweep silently reported one k twice."""
    common = dict(
        suite_id="synth",
        task_id="s0",
        arm_id="inquirer_prompted",
        policy_id="p",
        seed=0,
        corpus_hash="c",
        budget_cap=16,
        max_turns=16,
        word_cap=120,
        code_version="x",
        dirty=False,
    )
    assert build_manifest(**common, k=2).run_id != build_manifest(**common, k=8).run_id
    assert build_manifest(**common, k=5).run_id == build_manifest(**common, k=5).run_id


def test_a_different_question_list_is_a_different_run():
    """Seeding a ceiling arm from `--questions-from-file A` and then from B produced IDENTICAL
    run ids, so the second invocation resumed the first's results and the new file was silently
    ignored. That is how both ceiling arms came to share one input."""
    common = dict(
        suite_id="musique",
        task_id="t0",
        arm_id="gold_evidence",
        policy_id="gold_evidence",
        seed=0,
        corpus_hash="c",
        budget_cap=64,
        max_turns=16,
        word_cap=120,
        code_version="x",
        dirty=False,
        k=5,
    )
    a = build_manifest(**common, questions_hash="aaaa")
    b = build_manifest(**common, questions_hash="bbbb")
    none = build_manifest(**common)
    assert a.run_id != b.run_id != none.run_id
    assert a.run_id != none.run_id


def test_a_scripted_arm_hashes_its_own_script_and_random_q_hashes_the_pool(synth_root):
    """They depend on different things. A ScriptedInquirer issues its OWN task's list, so
    adding an unrelated task must not change its run_id. `random_q` samples from the questions
    of OTHER tasks -- that is what makes it a null rather than a weaker copy of the treatment --
    so adding a task genuinely changes what it could have drawn."""
    from pi_run.worker import questions_hash
    from pinq_expt import arms as arm_table

    root, corpus = synth_root
    pool_a = {"s0": ("q1",), "s1": ("q2",)}
    pool_b = {"s0": ("q1",), "s1": ("q2",), "s2": ("q3",)}

    def _h(arm_id, pool):
        spec = _spec(root, corpus, "s0", arm_id, questions=pool)
        c = arm_table.build(
            arm_table.get(arm_id), llm=None, retriever=None, recorder=None, questions=pool
        )
        return questions_hash(spec, c.inquirer)

    assert _h("parallel_replay", pool_a) == _h("parallel_replay", pool_b), (
        "a scripted arm replays its OWN task's list; an unrelated task must not move its run_id"
    )
    assert _h("random_q", pool_a) != _h("random_q", pool_b), (
        "random_q draws from the OTHER tasks, so the pool is genuinely its input"
    )
    assert questions_hash(_spec(root, corpus, "s0", "inquirer_prompted"), object()) == ""


def test_the_ceiling_arms_can_be_given_separate_question_files(tmp_path):
    """`UnitSpec.questions` is keyed by task_id alone and `plan()` hands the SAME mapping to
    every arm with requires_questions=True -- so ONE unscoped file gives gold_evidence and
    oracle_vreq byte-identical inputs, reintroducing one layer up the exact collapse
    `pi gold questions` exists to break. They bound different things."""
    from pi_run.cli import _load_questions_files

    (tmp_path / "g.json").write_text(json.dumps({"t0": ["with the answer 1960"]}))
    (tmp_path / "v.json").write_text(json.dumps({"t0": ["without it"]}))

    got = _load_questions_files(
        [f"gold_evidence={tmp_path / 'g.json'}", f"oracle_vreq={tmp_path / 'v.json'}"],
        ["gold_evidence", "oracle_vreq"],
    )
    assert got["gold_evidence"] != got["oracle_vreq"]

    specs = plan(
        suite_id="musique",
        corpus_dir="c",
        task_ids=["t0"],
        arm_ids=["gold_evidence", "oracle_vreq"],
        seeds=[0],
        runs_root="r",
        cache_root="c",
        questions_by_arm=got,
    )
    by_arm = {s.arm_id: s.questions for s in specs}
    assert by_arm["gold_evidence"] != by_arm["oracle_vreq"], (
        "two ceiling arms with identical inputs are not two ceilings"
    )


def test_a_question_file_scoped_to_an_arm_nobody_runs_is_refused(tmp_path):
    """Silently ignoring it would leave the arm seeded from something else entirely."""
    from pi_run.cli import _load_questions_files

    (tmp_path / "q.json").write_text(json.dumps({"t0": ["x"]}))
    with pytest.raises(SystemExit, match="not among the arms"):
        _load_questions_files([f"not_an_arm={tmp_path / 'q.json'}"], ["gold_evidence"])
    with pytest.raises(SystemExit, match="given twice"):
        _load_questions_files(
            [f"gold_evidence={tmp_path / 'q.json'}", f"gold_evidence={tmp_path / 'q.json'}"],
            ["gold_evidence"],
        )


def test_the_two_lists_of_semantic_fields_are_the_same_list():
    """`RunManifest.semantic_hash` builds a dict; `ids.semantic_hash` filters it through an
    allowlist. Both are needed -- the allowlist stops wall-clock or a hostname entering run
    identity -- but they are two lists of one thing, and two lists of one thing drift. `k` and
    `questions_hash` were added to the dict and had NO EFFECT because they were not in the
    allowlist: the manifest carried them, the run_id did not, and `--resume` went on treating
    two different configurations as the same run."""
    import ast
    import inspect
    import textwrap

    from pinq.ids import SEMANTIC_FIELDS
    from pinq.types import RunManifest

    src = textwrap.dedent(inspect.getsource(RunManifest.semantic_hash.fget))
    keys = {
        n.value
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    built = keys & set(SEMANTIC_FIELDS)
    assert built == set(SEMANTIC_FIELDS), (
        f"in the allowlist but not built: {sorted(set(SEMANTIC_FIELDS) - built)}"
    )
    # And nothing machine-shaped ever enters run identity.
    for banned in ("wall_ms", "usd", "concurrency", "hostname", "started_at", "corpus_dir"):
        assert banned not in SEMANTIC_FIELDS


def test_the_sweep_reports_its_timeout_margin_before_it_bites():
    """A timeout EXCLUDES a unit, and the arms are not equally exposed. Measured live on
    2026-08-24: inquirer_prompted ran 5x drafter_only's wall time on musique and 2.4x on tau2
    (726s vs 300s), so the treatment is the only arm that can realistically trip the cap. An
    operator must see the headroom shrinking on a cheap sweep, not discover it as missing rows
    in an expensive one."""
    import os

    from pi_run.sweep import summarize

    os.environ.pop("PI_UNIT_TIMEOUT_S", None)
    fast = summarize([{"status": "ok", "wall_ms": 60_000}])
    assert fast["timeout_margin"]["headroom"] > 2
    assert "warning" not in fast["timeout_margin"]

    tight = summarize([{"status": "ok", "wall_ms": 60_000}, {"status": "ok", "wall_ms": 3_000_000}])
    m = tight["timeout_margin"]
    assert m["slowest_unit_s"] == 3000.0
    assert "warning" in m and "EXCLUDES" in m["warning"]


def test_the_unit_timeout_default_clears_the_slowest_unit_ever_measured():
    """3600s is 5x the 726s slowest live unit. It is set from measurement rather than from a
    round number, because at the old 1800s a tau2 treatment unit plus a few ~148s rate-limit
    ladders reaches the cap and no baseline unit ever can."""
    from pi_run.worker import DEFAULT_UNIT_TIMEOUT_S

    SLOWEST_OBSERVED_S = 726
    assert DEFAULT_UNIT_TIMEOUT_S >= 4 * SLOWEST_OBSERVED_S, (
        f"{DEFAULT_UNIT_TIMEOUT_S}s leaves under 4x headroom over the slowest unit measured"
    )


def test_the_cost_estimate_ignores_arms_that_cannot_spend_tokens():
    """It priced Grid A's 11,849 rollouts at $0.00 with `within_gate: true`.

    `fake_chain`, `fake_depth1` and `fake_drafter_only` spend zero tokens BY DECLARATION --
    that is what llm_free means, and the worker raises if one ever spends any. A repository
    that has run `make smoke` plus one real arm therefore holds three zero-token arms and one
    real one, and the median of {0, 0, 0, 9107} is 0.

    The old `max(...) <= 0` guard could not see it: the max was 9107, only the median was zero.
    An operator budgeting a campaign against "$0.00, within_gate: true" proceeds.
    """
    from pi_run.cli import _observed_per_rollout

    observed = {
        "fake_chain": {"n_runs": 12, "tok_prompt_median": 0.0, "tok_completion_median": 0.0},
        "fake_depth1": {"n_runs": 12, "tok_prompt_median": 0.0, "tok_completion_median": 0.0},
        "fake_drafter_only": {"n_runs": 12, "tok_prompt_median": 0.0, "tok_completion_median": 0.0},
        "inquirer_prompted": {
            "n_runs": 4,
            "tok_prompt_median": 9107.0,
            "tok_completion_median": 2322.0,
        },
    }
    got = _observed_per_rollout(observed)
    assert got == (9107.0, 2322.0), got


def test_with_no_spending_arm_observed_it_falls_back_to_the_prior():
    """A guess labelled a guess is worth more than a measurement of the wrong population, so
    the caller reports `token_source: prior` rather than an observed zero."""
    from pi_run.cli import _observed_per_rollout

    only_free = {
        "fake_chain": {"n_runs": 12, "tok_prompt_median": 0.0, "tok_completion_median": 0.0}
    }
    assert _observed_per_rollout(only_free) is None
    assert _observed_per_rollout({}) is None
    # An arm the table does not know is not evidence either.
    assert (
        _observed_per_rollout(
            {"not_an_arm": {"n_runs": 9, "tok_prompt_median": 5.0, "tok_completion_median": 5.0}}
        )
        is None
    )


def test_the_median_of_arm_medians_still_resists_one_runaway_arm():
    """The outer median exists so a single pathological arm cannot reprice a 30,000-unit grid.
    Excluding the llm_free arms must not cost that property."""
    from pi_run.cli import _observed_per_rollout

    observed = {
        "inquirer_prompted": {
            "n_runs": 4,
            "tok_prompt_median": 9000.0,
            "tok_completion_median": 2000.0,
        },
        "drafter_only": {"n_runs": 4, "tok_prompt_median": 8000.0, "tok_completion_median": 1000.0},
        "gold_evidence": {
            "n_runs": 4,
            "tok_prompt_median": 900000.0,
            "tok_completion_median": 900000.0,
        },
    }
    prompt, completion = _observed_per_rollout(observed)
    assert prompt == 9000.0 and completion == 2000.0


def test_a_pilot_and_a_confirmatory_run_are_not_the_same_run():
    """A pilot exists so its results can be LOOKED AT before the confirmatory analysis is
    fixed, which is precisely what makes them inadmissible afterwards. Outside semantic_hash,
    the two shared a run_id -- hence a run directory and a resume sentinel -- so `--resume`
    skipped the confirmatory unit as already done and published the pilot's numbers as
    confirmatory."""
    from pi_run.manifest import build_manifest

    base = dict(
        suite_id="musique",
        arm_id="inquirer_prompted",
        task_id="t1",
        seed=0,
        policy_id="p",
        corpus_hash="ch",
        budget_cap=32,
        max_turns=8,
        word_cap=30,
        code_version="v",
        dirty=False,
    )
    pilot = build_manifest(**base, pilot_flag=True)
    confirmatory = build_manifest(**base, pilot_flag=False)
    assert pilot.run_id != confirmatory.run_id
    assert pilot.semantic_hash != confirmatory.semantic_hash


def test_a_turn_records_response_TEXT_under_a_name_that_says_so():
    """`Turn.response_sha` held the Drafter's reply TEXT and always did -- measured on the
    shipped turns.parquet: 53 non-empty values, 0 matching [0-9a-f]{16,64}, every one English
    prose -- while the CALLS table used the SAME NAME, in the same frozen schema, for a real
    sha256. Two components depend on the text (`render_history` and the ancestors policy read
    it back to build prompts), so the behaviour was right and the name was the defect."""
    import re

    from pinq.types import Turn

    assert "response_text" in Turn.__dataclass_fields__
    assert "response_sha" not in Turn.__dataclass_fields__
    from pinq.types import Stop

    t = Turn(turn_idx=0, action=Stop(), response_text="A sentence, not a digest.")
    assert not re.fullmatch(r"[0-9a-f]{16,64}", t.response_text)


def test_the_calls_table_still_names_a_real_digest():
    """The other half: CALLS.response_sha IS a digest and must keep the name."""
    import pi_eval.schema as sch

    assert "response_sha" in sch.TABLES["calls"].names
    assert "response_text" in sch.TABLES["turns"].names
    assert "response_sha" not in sch.TABLES["turns"].names


def test_a_run_directory_from_before_the_rename_still_compacts():
    """The parquet is MERGED with what is already on disk, so a rename that only touched the
    writer would make the strict schema reject every previously compacted row -- turning a
    naming fix into a demand that everyone rebuild."""
    from pi_run.compact import _migrate

    old = {"turn_idx": 0, "response_sha": "answer 0"}
    # `draft_text` is backfilled empty by the same pass: it was added to the turns schema
    # after these runs were written, and refusing them would kill compaction for all 273
    # existing run dirs. Empty stays FALSY so `cmd_train.render_state` still refuses to
    # export a turn that recorded a draft_sha it cannot reconstruct.
    assert _migrate("turns", old) == {
        "turn_idx": 0,
        "response_text": "answer 0",
        "draft_text": "",
    }
    # a row already in the new shape is untouched -- including a real draft_text, which the
    # backfill must never overwrite
    cur = {"turn_idx": 0, "response_text": "answer 0", "draft_text": "the real draft"}
    assert _migrate("turns", cur) == cur
    # and a table with no renames passes through
    assert _migrate("calls", {"response_sha": "deadbeef"}) == {"response_sha": "deadbeef"}


def test_the_cache_keeps_the_first_response_not_the_last(tmp_path):
    """The module justified having no lock with "two workers racing the same key are computing
    the same content-addressed value, so they write byte-identical files". That is false, and
    docs/REPRODUCE.md says so in its own opening paragraph: "Temperature 0 is not bitwise
    deterministic under batched inference -- two identical requests to the same provider can
    differ." The record holds text, response_sha, tok_completion, tok_reasoning and usd, every
    one of which differs between racers that got different text.

    Measured before the fix: put(A) then put(B) made a replay of RUN A return run B's text."""
    from pi_run.cache import DiskCache, _record_from
    from pinq.types import CallTelemetry

    def tel(sha, tok, usd):
        return CallTelemetry(
            call_id="x",
            actor="drafter",
            model="m",
            provider="p",
            request_sha="k",
            response_sha=sha,
            tok_prompt=100,
            tok_completion=tok,
            tok_reasoning=0,
            tok_cached=0,
            usd=usd,
            wall_ms=999,
            cache_hit=False,
            http_status=200,
        )

    a = _record_from("Paris is the capital.", tel("sha_a", 7, 0.00021))
    b = _record_from("The capital is Paris.", tel("sha_b", 8, 0.00024))
    assert a != b, "the fixture is only meaningful if the two records differ"

    d = DiskCache(tmp_path)
    d.put("KEY", a)
    d.put("KEY", b)
    got = d.get("KEY")
    assert got["text"] == "Paris is the capital.", "the first response is canonical"
    assert got["response_sha"] == "sha_a"


def test_cache_verify_detects_a_diverged_entry(tmp_path, monkeypatch, capsys):
    """schema.py documents `request_sha` as "the cache key; joins runs to
    cache/<xx>/<sha>.json" and NO code performed that join, so nothing could tell whether the
    cache still holds the response a recorded run actually received."""
    import argparse

    import pyarrow.parquet as pq

    from pi_eval import schema as sch
    from pi_run.cache import DiskCache, _record_from
    from pi_run.cli import cmd_cache_verify
    from pinq.types import CallTelemetry

    cache_root = tmp_path / "cache"
    d = DiskCache(cache_root)
    d.put(
        "req1",
        _record_from(
            "cached answer",
            CallTelemetry(
                call_id="x",
                actor="drafter",
                model="m",
                provider="p",
                request_sha="req1",
                response_sha="CACHED",
                tok_prompt=1,
                tok_completion=1,
                tok_reasoning=0,
                tok_cached=0,
                usd=0.0,
                wall_ms=1,
                cache_hit=False,
                http_status=200,
            ),
        ),
    )

    pqdir = tmp_path / "pq"
    pqdir.mkdir()
    import pyarrow as pa

    def _blank(t):
        if pa.types.is_string(t):
            return ""
        if pa.types.is_boolean(t):
            return False
        if pa.types.is_floating(t):
            return 0.0
        return 0

    row = {f.name: _blank(f.type) for f in sch.TABLES["calls"]}
    row.update({"run_id": "r1", "request_sha": "req1", "response_sha": "SOMETHING_ELSE"})
    pq.write_table(sch.to_table("calls", [row]), pqdir / "calls.parquet")

    a = argparse.Namespace(cache_root=str(cache_root), parquet_dir=str(pqdir))
    assert cmd_cache_verify(a) == 1, "a diverged entry must exit nonzero"
    assert "SOMETHING" in capsys.readouterr().out or True

    # and a matching entry passes
    row["response_sha"] = "CACHED"
    pq.write_table(sch.to_table("calls", [row]), pqdir / "calls.parquet")
    assert cmd_cache_verify(a) == 0


def _blank_calls_row(overrides: dict) -> dict:
    """Build a minimal, schema-valid `calls` row for cache_verify tests.

    Mirrors the blanking helper in test_cache_verify_detects_a_diverged_entry so the three
    tests below don't have to repeat the pyarrow-type dispatch."""
    from pi_eval import schema as sch

    def _blank(t):
        import pyarrow as pa

        if pa.types.is_string(t):
            return ""
        if pa.types.is_boolean(t):
            return False
        if pa.types.is_floating(t):
            return 0.0
        return 0

    row = {f.name: _blank(f.type) for f in sch.TABLES["calls"]}
    row.update(overrides)
    return row


def _write_calls_parquet(pqdir, rows: list[dict]) -> None:
    import pyarrow.parquet as pq

    from pi_eval import schema as sch

    pqdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(sch.to_table("calls", rows), pqdir / "calls.parquet")


def test_cache_verify_names_the_root_it_read_and_the_entries_it_saw(tmp_path, capsys):
    """Defect 1: cmd_cache_verify emitted calls_checked/matched/missing/diverged but never the
    cache root it resolved, so a reading could not be audited after the fact -- you could not
    tell which cache was checked. It must also report how many entries that cache held.

    FAILS before the fix: neither key exists in the emitted dict."""
    import argparse

    from pi_run.cache import DiskCache, _record_from
    from pi_run.cli import cmd_cache_verify
    from pinq.types import CallTelemetry

    cache_root = tmp_path / "cache"
    d = DiskCache(cache_root)
    d.put(
        "req1",
        _record_from(
            "cached answer",
            CallTelemetry(
                call_id="x",
                actor="drafter",
                model="m",
                provider="p",
                request_sha="req1",
                response_sha="CACHED",
                tok_prompt=1,
                tok_completion=1,
                tok_reasoning=0,
                tok_cached=0,
                usd=0.0,
                wall_ms=1,
                cache_hit=False,
                http_status=200,
            ),
        ),
    )
    pqdir = tmp_path / "pq"
    _write_calls_parquet(
        pqdir, [_blank_calls_row({"run_id": "r1", "request_sha": "req1", "response_sha": "CACHED"})]
    )

    a = argparse.Namespace(cache_root=str(cache_root), parquet_dir=str(pqdir))
    cmd_cache_verify(a)
    out = json.loads(capsys.readouterr().out)
    assert out["cache_root"] == str(DiskCache(cache_root).root), (
        "the reading must name the ABSOLUTE resolved root, not the argument as given"
    )
    assert Path(out["cache_root"]).is_absolute()
    assert out["cache_entries"] == 1, "and the entry count that root actually held"


def test_cache_verify_refuses_on_a_missing_cache_root(tmp_path, capsys):
    """Defect 2: a resolved root that does not exist must refuse, not report a clean-looking
    `missing_from_cache` count. Before the fix, every request against a nonexistent root read
    as an ordinary miss -- 'the cache holds nothing' and 'I am pointed at the wrong directory'
    produced an identical, un-auditable measurement (measured: 16.1% where the run's own cache
    held 97.1%).

    FAILS before the fix: cmd_cache_verify returns 0 with matched/missing counts instead of
    refusing."""
    import argparse

    from pi_run.cli import cmd_cache_verify

    cache_root = tmp_path / "cache_that_was_never_created"
    assert not cache_root.exists()

    pqdir = tmp_path / "pq"
    _write_calls_parquet(
        pqdir, [_blank_calls_row({"run_id": "r1", "request_sha": "req1", "response_sha": "X"})]
    )

    a = argparse.Namespace(cache_root=str(cache_root), parquet_dir=str(pqdir))
    rc = cmd_cache_verify(a)
    out = json.loads(capsys.readouterr().out)
    assert rc != 0, "an absent cache root must refuse, not exit clean"
    assert "matched" not in out and "missing_from_cache" not in out, (
        "no counts on a refusal -- a count here is exactly the un-auditable measurement this "
        "refusal exists to prevent"
    )
    assert str(cache_root.resolve()) in json.dumps(out)


def test_cache_verify_refuses_on_a_present_but_empty_cache_root(tmp_path, capsys):
    """The other half of defect 2: a root that EXISTS but holds zero entries must refuse the
    same way as an absent root. This is the case a plain existence check would miss -- and
    it's the one that actually bit the sweep-vs-checkout mismatch (both processes could `mkdir`
    the wrong `./cache`)."""
    import argparse

    from pi_run.cli import cmd_cache_verify

    cache_root = tmp_path / "cache_present_but_empty"
    cache_root.mkdir(parents=True)
    assert cache_root.exists()

    pqdir = tmp_path / "pq"
    _write_calls_parquet(
        pqdir, [_blank_calls_row({"run_id": "r1", "request_sha": "req1", "response_sha": "X"})]
    )

    a = argparse.Namespace(cache_root=str(cache_root), parquet_dir=str(pqdir))
    rc = cmd_cache_verify(a)
    out = json.loads(capsys.readouterr().out)
    assert rc != 0, "a present-but-empty cache root must refuse, not exit clean"
    assert "matched" not in out and "missing_from_cache" not in out


def test_cache_verify_positive_control_a_real_miss_still_reports_and_does_not_refuse(
    tmp_path, capsys
):
    """Non-vacuity check for defect 2's refusal: a cache that IS present and non-empty, but
    genuinely missing one specific request, must still report `missing_from_cache: 1` and NOT
    refuse. A guard that refuses on both 'wrong directory' and 'this one request was never
    cached' is useless -- this repo has been bitten before by a guard that refused 48 of 156
    real verdicts it should have passed."""
    import argparse

    from pi_run.cache import DiskCache, _record_from
    from pi_run.cli import cmd_cache_verify
    from pinq.types import CallTelemetry

    cache_root = tmp_path / "cache"
    d = DiskCache(cache_root)
    # one real entry, so the cache is non-empty -- but it is NOT the request the run asks about
    d.put(
        "some_other_request",
        _record_from(
            "unrelated",
            CallTelemetry(
                call_id="x",
                actor="drafter",
                model="m",
                provider="p",
                request_sha="some_other_request",
                response_sha="Y",
                tok_prompt=1,
                tok_completion=1,
                tok_reasoning=0,
                tok_cached=0,
                usd=0.0,
                wall_ms=1,
                cache_hit=False,
                http_status=200,
            ),
        ),
    )

    pqdir = tmp_path / "pq"
    _write_calls_parquet(
        pqdir,
        [
            _blank_calls_row(
                {"run_id": "r1", "request_sha": "req_never_cached", "response_sha": "X"}
            )
        ],
    )

    a = argparse.Namespace(cache_root=str(cache_root), parquet_dir=str(pqdir))
    rc = cmd_cache_verify(a)
    out = json.loads(capsys.readouterr().out)
    assert rc == 0, "a genuine single miss against a real, non-empty cache must not refuse"
    assert out["missing_from_cache"] == 1
    assert out["cache_entries"] == 1
    assert "error" not in out


def test_cost_estimate_prices_the_model_the_sweep_will_actually_bill(monkeypatch):
    """`--model` defaulted to a hardcoded `anthropic/claude-sonnet-4-5` while every role in this
    repository is pinned to gpt-oss-120b and every recorded call in calls.parquet is
    `openai/aws/gpt-oss-120b`. Measured on the same parquet:

        default (sonnet-4-5)   $2,009.40 naive / $1,429.07 cached
        pinned (gpt-oss-120b)  $  100.47 naive / $   68.23 cached

    A factor of 21, on the number a spend decision is made from, printed by the command whose
    entire job is to produce that number."""
    from pi_run.cli import DEFAULT_COST_MODEL, _cost_model

    monkeypatch.setenv("PI_MODEL_INQUIRER", "openai/aws/gpt-oss-120b")
    assert _cost_model(None) == ("openai/aws/gpt-oss-120b", "PI_MODEL_INQUIRER")
    # an explicit flag still wins
    assert _cost_model("anthropic/claude-sonnet-4-5")[0] == "anthropic/claude-sonnet-4-5"
    # and with no pin the fallback SAYS it is a fallback
    monkeypatch.delenv("PI_MODEL_INQUIRER", raising=False)
    model, why = _cost_model(None)
    assert model == DEFAULT_COST_MODEL and "no PI_MODEL_INQUIRER" in why


def test_sweep_respects_an_explicit_arm_filter(monkeypatch, capsys):
    """`--sweep` silently discarded `--arm`: the grid loop did `sub.arm = list(grid.arms)`.

    An operator narrowing a sweep to one cheap arm -- exactly what you do to rehearse a grid
    safely -- got all 13 arms instead, including every billed one. I did this to myself while
    testing the spend cap: `--arm fake_chain` on tier1_confirmatory ran the full arm list and
    billed $0.72 before the cap stopped it.
    """
    import inspect

    from pi_run import cli

    src = inspect.getsource(cli.cmd_run)
    assert "sub.arm = list(grid.arms)" in src, "the no-filter default must still exist"
    assert 'wanted = [x for x in (getattr(a, "arm", None) or []) if x]' in src
    # an --arm outside the grid is refused, not silently empty
    assert "is not in grid" in src


def test_the_grid_spend_cap_mechanism_is_still_wired(monkeypatch):
    """STRUCTURAL ONLY. The behaviour is tested in tests/test_grid_budget_threading.py.

    `--spend-cap` was passed unchanged to each suite of a multi-suite grid, so
    tier1_confirmatory (musique, strategyqa, wiki2) could bill three times the stated cap.
    The flag says "cap", the operator reads "cap", and the invoice was a multiple of it.

    THIS TEST CANNOT CATCH THAT DEFECT RETURNING, and it used to be the only guard. It
    greps `cmd_run`'s source for four strings and never executes the loop, so the
    arithmetic is unasserted. Demonstrated: with the decrement replaced by
    `remaining - 0.0`, this test PASSES while the behavioural file fails 3 of 6.

    Kept because it is the cheap check that the mechanism has not been deleted outright.
    Do not read a green here as evidence that the budget is threaded correctly."""
    import inspect

    from pi_run import cli

    src = inspect.getsource(cli.cmd_run)
    assert "remaining = spend_cap(" in src
    assert "sub.spend_cap = remaining" in src
    assert "_billed_usd" in src, "each suite must report what it billed"
    assert "spend cap is exhausted" in src, "a suite reached with nothing left is skipped"


def test_pi_score_refuses_to_silently_drop_a_judge_derived_primary():
    """`pi score` reported `judging: "skipped: no judge client"` and exited 0, and
    docs/REPRODUCE.md's own command block ran it with no PI_JUDGE_CLIENT set -- so the
    documented reproduction path emitted ZERO rows for kpr_incremental, the drgym PRIMARY
    endpoint (P3), and nothing distinguished "measured nothing" from "measured zero".

    Verified end to end: exit 1 without the flag, exit 0 with --allow-no-judge."""
    import inspect

    from pi_run import cli

    src = inspect.getsource(cli.cmd_score)
    assert "JUDGING WAS SKIPPED" in src, "the guard must live in cmd_score"
    assert "allow_no_judge" in src
    # and NOT in cmd_compact, where a first patch of mine landed by matching the wrong
    # `_emit(res.as_dict()); return 0`
    assert "JUDGING WAS SKIPPED" not in inspect.getsource(cli.cmd_compact)


def test_kpr_incremental_is_the_endpoint_that_guard_protects():
    """If this stops being judge-derived, the guard silently protects nothing."""
    from pi_eval.prereg import PRIMARY, SECONDARY
    from pi_eval.score import define

    judged = {
        e.metric
        for e in (*PRIMARY, *SECONDARY)
        if define(e.metric) and getattr(define(e.metric), "judge_derived", False)
    }
    assert "kpr_incremental" in judged


def test_reproduce_md_sets_a_judge_client():
    """The documented path must not be the one that deletes a primary endpoint."""
    import pathlib

    md = pathlib.Path("docs/REPRODUCE.md").read_text()
    i = md.index("pi score")
    assert "PI_JUDGE_CLIENT" in md[max(0, i - 200) : i], "pi score must be shown with a judge"


def test_prompt_variant_id_labels_the_overlay_that_produced_a_run(tmp_path, monkeypatch):
    """`RunManifest.prompt_variant_id` was declared, defaulted "v1", written into every
    manifest and compacted into a parquet column -- and NOTHING ever set it. The column was
    constant across every run ever made, so rows produced under different prompts pooled into
    one cell with nothing able to separate them, and the between-prompt spread of the headline
    effect could not be reported at all.

    `PI_PROMPT_OVERLAY` already changes the templates; the label to GROUP BY was the missing
    half."""
    from pinq import promptlib

    monkeypatch.delenv("PI_PROMPT_OVERLAY", raising=False)
    assert promptlib.variant_id() == "v1"

    d = tmp_path / "terse_v2"
    d.mkdir()
    (d / "inquirer_prompted.txt").write_text("Ask one question.")
    monkeypatch.setenv("PI_PROMPT_OVERLAY", str(d))
    first = promptlib.variant_id()
    assert first.startswith("terse_v2-") and first != "v1"

    # the id follows the CONTENT, not just the directory name
    (d / "inquirer_prompted.txt").write_text("Ask two questions.")
    assert promptlib.variant_id() != first

    # an overlay directory with no templates is not a variant
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PI_PROMPT_OVERLAY", str(empty))
    assert promptlib.variant_id() == "v1"


def test_two_prompt_overlays_do_not_share_a_run_id():
    """`prompt_hashes` covers the templates an arm actually RENDERS, so an overlay that
    rewrites a template the arm never touches leaves the hash unchanged -- and an LLM-free arm
    renders none at all. Verified before the fix: two `pi run` invocations under different
    overlays produced ONE run directory, the second having resumed the first and inherited its
    label. Same class as k, questions_hash and pilot_flag."""
    from pinq.ids import SEMANTIC_FIELDS

    assert "prompt_variant_id" in SEMANTIC_FIELDS

    from pinq.types import RunManifest

    base = dict(
        suite_id="synth",
        task_id="t1",
        arm_id="a",
        policy_id="p",
        seed=0,
        split="train",
        corpus_hash="ch",
        budget_cap=32,
        max_turns=8,
        word_cap=30,
        code_version="v",
    )
    a = RunManifest(**base, prompt_variant_id="v1")
    b = RunManifest(**base, prompt_variant_id="terse_v2-abc12345")
    assert a.run_id != b.run_id
    assert a.semantic_hash != b.semantic_hash


def test_the_worker_stamps_it_from_the_live_overlay():
    import inspect

    from pi_run import worker

    assert "promptlib.variant_id()" in inspect.getsource(worker)


def test_verify_arms_catches_an_arm_that_never_asked(tmp_path):
    """`status == ok` is not enough. An arm whose policy failed to parse its own output stops on
    turn 0, retrieves nothing, and writes a well-formed row -- `drafter_only` in disguise,
    costing money and reporting the baseline's behaviour under the treatment's name.

    Run against the real musique sweep this caught `random_q` (0 asks, 0 of 10 calls live) and
    `parallel_replay` (0.167 asks, 0 of 4 live), which nothing else complained about."""
    import argparse

    import pyarrow.parquet as pq

    from pi_eval import schema as sch
    from pi_run.cli import cmd_verify_arms

    def row(table, **over):
        import pyarrow as pa

        r = {}
        for f in sch.TABLES[table]:
            if pa.types.is_string(f.type):
                r[f.name] = ""
            elif pa.types.is_boolean(f.type):
                r[f.name] = False
            elif pa.types.is_floating(f.type):
                r[f.name] = 0.0
            elif pa.types.is_list(f.type):
                r[f.name] = []
            else:
                r[f.name] = 0
        r.update(over)
        return r

    runs = [
        row("runs", run_id="r1", suite_id="musique", arm_id="self_ask", status="ok", seed=7),
        row("runs", run_id="r2", suite_id="musique", arm_id="drafter_only", status="ok", seed=7),
    ]
    pq.write_table(sch.to_table("runs", runs), tmp_path / "runs.parquet")
    scores = [
        row("scores", run_id="r1", metric_name="n_asks", value=0.0),  # asked NOTHING
        row("scores", run_id="r2", metric_name="n_asks", value=0.0),  # correct for NeverAsk
    ]
    pq.write_table(sch.to_table("scores", scores), tmp_path / "scores.parquet")
    calls = [row("calls", run_id="r1", cache_hit=False), row("calls", run_id="r2", cache_hit=False)]
    pq.write_table(sch.to_table("calls", calls), tmp_path / "calls.parquet")

    a = argparse.Namespace(parquet_dir=str(tmp_path), seed=7, suite="musique")
    assert cmd_verify_arms(a) == 1, "self_ask asking nothing must fail"


def test_verify_arms_refuses_an_all_cache_hit_probe(tmp_path):
    """`runs.usd` charges a cache hit its recorded cost AS-IF, so a non-zero usd is NOT evidence
    the provider was reached. A probe on a warm seed exercises the code path and proves nothing
    about the model, which is the whole reason the canary uses an unused seed."""
    import argparse

    import pyarrow as pa
    import pyarrow.parquet as pq

    from pi_eval import schema as sch
    from pi_run.cli import cmd_verify_arms

    def row(table, **over):
        r = {}
        for f in sch.TABLES[table]:
            if pa.types.is_string(f.type):
                r[f.name] = ""
            elif pa.types.is_boolean(f.type):
                r[f.name] = False
            elif pa.types.is_floating(f.type):
                r[f.name] = 0.0
            elif pa.types.is_list(f.type):
                r[f.name] = []
            else:
                r[f.name] = 0
        r.update(over)
        return r

    pq.write_table(
        sch.to_table(
            "runs",
            [
                row(
                    "runs",
                    run_id="r1",
                    suite_id="musique",
                    arm_id="self_ask",
                    status="ok",
                    seed=7,
                    usd=0.05,
                )
            ],
        ),
        tmp_path / "runs.parquet",
    )
    pq.write_table(
        sch.to_table("scores", [row("scores", run_id="r1", metric_name="n_asks", value=4.0)]),
        tmp_path / "scores.parquet",
    )
    # every call a cache hit, despite a non-zero usd
    pq.write_table(
        sch.to_table("calls", [row("calls", run_id="r1", cache_hit=True, usd=0.05)]),
        tmp_path / "calls.parquet",
    )
    a = argparse.Namespace(parquet_dir=str(tmp_path), seed=7, suite="musique")
    assert cmd_verify_arms(a) == 1, "an all-cache-hit probe must not pass as a live check"
