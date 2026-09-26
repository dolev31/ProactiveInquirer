"""A rung 0 search must survive being stopped, or a laptop lid ends a ten-hour run.

Rollouts already resume: `status.json` is the sentinel and is written LAST, so a unit killed
mid-write leaves no status file and is simply redone. The SEARCH state was different -- the
Pareto front, the per-candidate scores and the candidate pool lived only in memory.

That made a restart worse than it looks. The seed re-scores for free (its rollouts are
`resumed`), but the reflector proposes FRESH mutations on the way back up, and a different
mutation is a different prompt, a different `prompt_hashes`, a different run_id -- so every
generation after the first is paid for again.

The checkpoint is written after each generation and read at startup. It stores the candidate
TEXTS, not just their ids: a candidate whose text is lost cannot be re-evaluated or shipped as
the winner, and its cid alone would name something nothing can reconstruct.
"""

from __future__ import annotations

import json

from pinq_train.rung0_gepa.search import Candidate, load_checkpoint, save_checkpoint


def _pool():
    a = Candidate(text="seed text", generation=0, note="seed")
    b = Candidate(text="mutated text", generation=1, note="child", parent=a.cid)
    return {a.cid: a, b.cid: b}, a, b


def test_a_checkpoint_round_trips(tmp_path) -> None:
    pool, a, b = _pool()
    scores = {a.cid: {"musique/t1": 0.5}, b.cid: {"musique/t1": 0.9}}
    save_checkpoint(tmp_path, pool=pool, scores=scores, generation=2)
    got = load_checkpoint(tmp_path)
    assert got is not None
    assert got["generation"] == 2
    assert got["scores"] == scores
    assert {c.cid for c in got["pool"].values()} == set(pool)


def test_the_candidate_TEXT_survives_not_just_its_id(tmp_path) -> None:
    """A cid names a text nothing else stores; losing the text loses the candidate."""
    pool, a, b = _pool()
    save_checkpoint(tmp_path, pool=pool, scores={}, generation=1)
    got = load_checkpoint(tmp_path)
    assert got["pool"][b.cid].text == "mutated text"
    assert got["pool"][b.cid].parent == a.cid


def test_a_reloaded_candidate_keeps_its_identity(tmp_path) -> None:
    """`cid` is derived from the text. If it were recomputed differently on reload, resumed
    rollouts would no longer match and every one would be paid for again."""
    pool, a, _ = _pool()
    save_checkpoint(tmp_path, pool=pool, scores={}, generation=0)
    back = load_checkpoint(tmp_path)["pool"][a.cid]
    assert back.cid == a.cid


def test_no_checkpoint_returns_None_rather_than_raising(tmp_path) -> None:
    """A first run has none, and that is not an error."""
    assert load_checkpoint(tmp_path) is None


def test_a_corrupt_checkpoint_is_ignored_not_fatal(tmp_path) -> None:
    """A process killed mid-write must not make the search unstartable -- it should begin
    afresh, exactly as it would have without a checkpoint."""
    (tmp_path / "rung0.checkpoint.json").write_text("{not json")
    assert load_checkpoint(tmp_path) is None


def test_the_checkpoint_is_written_atomically(tmp_path) -> None:
    """Killed mid-write is the case this exists for, so the write must not be the thing that
    corrupts it."""
    pool, _, _ = _pool()
    save_checkpoint(tmp_path, pool=pool, scores={}, generation=1)
    p = tmp_path / "rung0.checkpoint.json"
    assert p.exists()
    json.loads(p.read_text())  # parses
    assert not list(tmp_path.glob("*.tmp")), "a temp file was left behind"
