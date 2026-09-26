"""Checkpoint after every CANDIDATE, not every generation.

Per-generation was the first design, and its stated reason was wrong: "a generation is the unit
the loop resumes at, so a mid-generation checkpoint would claim progress the restart discards."
The restart does NOT discard it. `pool` and `scores` are restored wholesale, and every scored
candidate goes on contributing to the Pareto front regardless of which generation produced it.

IT FAILED TWICE IN PRACTICE, both times to the same VPN drop. A generation is 6 candidates x 12
tasks and takes hours; the network dropped roughly hourly. The search died mid-generation-1 on
both attempts -- after 3 candidates, then after 5 -- and wrote no checkpoint either time, so
every rollout that was not already `resumed` was paid for again.

A checkpoint per candidate can at worst duplicate ONE candidate's work. A generation's worth
was being lost instead.
"""

from __future__ import annotations

from pinq_train.rung0_gepa.search import Candidate, load_checkpoint, save_checkpoint


def test_a_partial_generation_is_recoverable(tmp_path) -> None:
    """The exact case that lost work twice: died after 5 of 6 candidates."""
    seed = Candidate(text="seed", generation=0, note="seed")
    kids = [Candidate(text=f"c{i}", generation=1, parent=seed.cid, note="m") for i in range(4)]
    pool = {c.cid: c for c in (seed, *kids)}
    scores = {c.cid: {"musique/t1": 0.5 + i * 0.01} for i, c in enumerate(pool.values())}
    save_checkpoint(tmp_path, pool=pool, scores=scores, generation=1)

    got = load_checkpoint(tmp_path)
    assert len(got["pool"]) == 5, "a partially-completed generation was not preserved"
    assert len(got["scores"]) == 5
    assert all(c.cid in got["pool"] for c in kids)


def test_scored_candidates_survive_regardless_of_generation(tmp_path) -> None:
    """Every scored candidate feeds the Pareto front, whichever generation produced it -- which
    is why restoring a partial generation loses nothing."""
    a = Candidate(text="gen0", generation=0, note="seed")
    b = Candidate(text="gen1", generation=1, parent=a.cid, note="m")
    c = Candidate(text="gen2", generation=2, parent=b.cid, note="m")
    pool = {x.cid: x for x in (a, b, c)}
    save_checkpoint(
        tmp_path, pool=pool, scores={x.cid: {"t": 1.0} for x in pool.values()}, generation=2
    )
    got = load_checkpoint(tmp_path)
    assert {x.generation for x in got["pool"].values()} == {0, 1, 2}


def test_repeated_saves_overwrite_cleanly(tmp_path) -> None:
    """Called once per candidate now, so it runs many times per generation."""
    seed = Candidate(text="seed", generation=0, note="seed")
    for i in range(5):
        kid = Candidate(text=f"c{i}", generation=1, parent=seed.cid, note="m")
        pool = {seed.cid: seed, kid.cid: kid}
        save_checkpoint(tmp_path, pool=pool, scores={kid.cid: {"t": float(i)}}, generation=1)
    got = load_checkpoint(tmp_path)
    assert got["generation"] == 1
    assert not list(tmp_path.glob("*.tmp")), "temp files accumulated across repeated saves"
