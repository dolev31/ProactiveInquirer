"""No single state may dominate the preference dataset.

`export_pairs` emits all C(n,2) pairs per state, which its own docstring says is safe "ONLY
while every state is sampled with the SAME candidate count" -- and warns that under a varying n
"a state with 8 candidates contributes 28 pairs against another's 1 and dominates the loss in
proportion to how many candidates it happened to get", concluding "this needs a per-state cap,
not a bigger margin".

MEASURED on the 1,525-pair export, the condition it warns about holds: implied candidate counts
run from 2 to 12, pairs per state from 1 to 74, and the top TEN states contribute 19.8% of the
whole dataset. n varies because candidates FAIL -- a rate-limited or errored branch simply is
not there -- so the count a state gets is an accident of the proxy, not a property of the state.

Selection under the cap is deterministic and UNBIASED: sorted by `pair_id`, which is a hash of
the state and the two candidate run ids. Taking the top-k by margin instead would keep the
easiest contrasts and systematically drop the hard ones, which is the opposite of what a
preference model needs to learn.
"""

from __future__ import annotations

import collections

from pinq_train.export.dataset import MAX_PAIRS_PER_STATE, export_pairs


def _c(run_id, value, action=None):
    return {
        "suite_id": "musique",
        "task_id": "t1",
        "template_id": None,
        "run_id": run_id,
        "turn_idx": 1,
        "state_text": "S",
        "action_json": action or f'{{"action":"ASK","question":"q{run_id}"}}',
        "value": value,
        "phi_tilde": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "branch_of_run_id": "p",
        "branch_turn_idx": 1,
    }


def test_a_state_cannot_exceed_the_cap() -> None:
    rows = [_c(f"r{i}", 1.0 - i * 0.01) for i in range(15)]  # C(15,2) = 105 uncapped
    pairs, man = export_pairs(rows, margin_threshold=0.0, len_delta_max=1000)
    assert len(pairs) == MAX_PAIRS_PER_STATE
    assert man.n_over_cap_dropped == 105 - MAX_PAIRS_PER_STATE


def test_a_small_state_is_untouched() -> None:
    rows = [_c(f"r{i}", 1.0 - i * 0.1) for i in range(3)]  # C(3,2) = 3
    pairs, man = export_pairs(rows, margin_threshold=0.0, len_delta_max=1000)
    assert len(pairs) == 3
    assert man.n_over_cap_dropped == 0


def test_two_states_are_capped_independently() -> None:
    rows = [_c(f"a{i}", 1.0 - i * 0.01) for i in range(15)]
    rows += [dict(_c(f"b{i}", 1.0 - i * 0.01), task_id="t2") for i in range(3)]
    pairs, _ = export_pairs(rows, margin_threshold=0.0, len_delta_max=1000)
    per = collections.Counter((p.task_id, p.turn_idx) for p in pairs)
    assert per[("t1", 1)] == MAX_PAIRS_PER_STATE
    assert per[("t2", 1)] == 3


def test_selection_is_deterministic() -> None:
    """Two exports of one dataset must contain the same pairs, or a rerun is a different
    dataset with the same name."""
    rows = [_c(f"r{i}", 1.0 - i * 0.01) for i in range(15)]
    a, _ = export_pairs(rows, margin_threshold=0.0, len_delta_max=1000)
    b, _ = export_pairs(list(reversed(rows)), margin_threshold=0.0, len_delta_max=1000)
    assert {p.pair_id for p in a} == {p.pair_id for p in b}


def test_the_cap_does_not_keep_only_the_easiest_contrasts() -> None:
    """Top-k by margin would keep the widest gaps and drop every close call. The kept set must
    still span the margin range."""
    rows = [_c(f"r{i}", 1.0 - i * 0.05) for i in range(15)]
    pairs, _ = export_pairs(rows, margin_threshold=0.0, len_delta_max=1000)
    margins = sorted(p.margin for p in pairs)
    assert margins[0] < 0.15, f"smallest kept margin {margins[0]:.3f}: close calls were dropped"
