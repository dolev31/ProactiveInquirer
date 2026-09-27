"""Test-first for the graph_version recovery sidecar (CONTRIBUTING.md rule 2).

`pinq_train.gate.run_gate` now writes `selection.graph_version` on every verdict it produces
(commit 68f4f49, merged to main in 4dc17c3), and
`test_a_verdict_carries_the_graph_version_beside_the_scorer_hash` in `test_train_gate.py`
already proves that going forward. What that leaves open is every verdict written before the
fix landed: measured on the shared `artifacts/gate/` tree (see
`artifacts/graph_version_provenance_20260918/RESULT.md`), 0 of 176 verdict files on disk carry
`graph_version` anywhere, all 176 name a `parquet_dir` that still holds the `scores.parquet`
their numbers came from, and none of those directories carries more than one distinct
`graph_version` under the verdict's own `scorer_hash`. So the field is recoverable, not lost --
this is a promotion, not a re-score, and CONTRIBUTING.md's "recorded verdicts are immutable" means the
recovery must read a verdict file and never write to it.

Before `scripts/recover_graph_version.py` existed, `import recover_graph_version` raised
`ModuleNotFoundError` -- there was nothing to import and nothing that could recover the field.
That is the failing state this test-first rule (CONTRIBUTING.md rule 2) asks for; the module below
makes it pass.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_train_gate import GRAPH, SCORER, _build, _gate, _pairs  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "recover_graph_version",
    Path(__file__).resolve().parents[1] / "scripts" / "recover_graph_version.py",
)
recover_graph_version = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(recover_graph_version)


def _legacy_verdict(tmp_path: Path, parquet_dir: Path, name: str = "legacy_verdict.json") -> Path:
    """A verdict shaped exactly like the 176 on disk today: `scorer_hash` present,
    `graph_version` absent anywhere, because it was written before 68f4f49."""
    v = _gate(parquet_dir)
    assert v["selection"]["graph_version"] == GRAPH  # sanity: the fixture is post-fix
    del v["selection"]["graph_version"]
    assert "graph_version" not in v["selection"]
    p = tmp_path / name
    p.write_text(json.dumps(v))
    return p


def test_recovers_the_graph_version_a_legacy_verdict_is_missing(tmp_path):
    d = _build(tmp_path, _pairs())
    p = _legacy_verdict(tmp_path, d)
    before = p.read_text()

    out = recover_graph_version.recover_one(p)

    assert out["status"] == "recovered"
    assert out["graph_version"] == GRAPH
    assert out["scorer_hash"] == SCORER
    assert out["n_runs"] > 0
    # CONTRIBUTING.md: recorded verdicts are immutable. Recovery reads; it never writes the file.
    assert p.read_text() == before


def test_an_already_recorded_verdict_is_left_alone(tmp_path):
    d = _build(tmp_path, _pairs())
    v = _gate(d)
    p = tmp_path / "post_fix_verdict.json"
    p.write_text(json.dumps(v))

    out = recover_graph_version.recover_one(p)

    assert out["status"] == "already_recorded"
    assert out["graph_version"] == GRAPH


def test_two_graph_versions_in_the_parquet_are_refused_not_averaged(tmp_path):
    from pinq_train.gate import GraphVersionAmbiguous

    d = _build(tmp_path, _pairs())
    p = _legacy_verdict(tmp_path, d)

    t = pq.read_table(d / "scores.parquet").to_pylist()
    t[0]["graph_version"] = "v2"
    pq.write_table(pa.Table.from_pylist(t), d / "scores.parquet")

    with pytest.raises(GraphVersionAmbiguous, match="graph_version"):
        recover_graph_version.recover_one(p)


def test_a_missing_parquet_dir_is_a_named_gap_not_a_crash(tmp_path):
    d = _build(tmp_path, _pairs())
    p = _legacy_verdict(tmp_path, d)
    v = json.loads(p.read_text())
    v["selection"]["parquet_dir"] = str(tmp_path / "nowhere")
    p.write_text(json.dumps(v))

    out = recover_graph_version.recover_one(p)
    assert out["status"] == "gap"
    assert out["gap"] == "parquet_dir_missing"
