"""Lane L1.2: contributions A and B recomputed on the held-out test split.

Tests the two pieces of pure logic `scripts/contributions_on_test/lib.py` adds: the run-id
list parser (two incompatible formats live on disk in this repo, and the wrong choice silently
collects task_ids as run_ids with no error) and the population assertion the ANALYSIS RULES
require before any number is computed. Neither test touches gold, an LLM, or `scores/parquet/`.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts.contributions_on_test.lib import (
    PooledCodeVersions,
    PopulationIncomplete,
    TurnsDropped,
    assert_one_code_version_and_no_duplicate_keys,
    assert_population_scored,
    assert_turns_not_dropped,
    elects_to_stop,
    main_checkout,
    near_zero_bounds,
    read_run_ids,
    stability_verdict,
)
from scripts.contributions_on_test.teacher_token_charge import symmetric_contrast
from scripts.matched_cost import Ladder

# --------------------------------------------------------------------- main_checkout


def test_main_checkout_strips_the_claude_worktrees_segment(tmp_path):
    """The exact shape a worktree agent runs in. A hardcoded absolute path would fail
    scripts/check_no_home_paths.sh; this derives it from a fake `__file__` instead, so the
    check never sees a literal `/Users/...` (or similar) string in this file's source."""
    repo = tmp_path / "some_repo"
    fake = (
        repo
        / ".claude"
        / "worktrees"
        / "agent-abc123"
        / "scripts"
        / "contributions_on_test"
        / "lib.py"
    )
    assert main_checkout(fake) == repo.resolve()


def test_main_checkout_falls_back_to_the_repo_root_outside_a_worktree(tmp_path):
    repo = tmp_path / "some_repo"
    fake = repo / "scripts" / "contributions_on_test" / "lib.py"
    assert main_checkout(fake) == repo.resolve()


# --------------------------------------------------------------------- read_run_ids


def test_a_bare_run_id_list_with_a_comment_header_is_read_unchanged(tmp_path):
    p = tmp_path / "run_ids.trained.musique.txt"
    p.write_text(
        "# run_ids.trained.musique.txt -- some header\n"
        "# n=3\n"
        "\n"
        "00070432540e63945a696065fafcc6bf\n"
        "006f999e80a689b7df874db1e6b8f78d\n"
        "01b8ffaf6c490f2f1d97574964f3c673\n"
    )
    assert read_run_ids(p) == [
        "00070432540e63945a696065fafcc6bf",
        "006f999e80a689b7df874db1e6b8f78d",
        "01b8ffaf6c490f2f1d97574964f3c673",
    ]


def test_a_task_id_seed_run_id_tsv_yields_run_ids_not_task_ids(tmp_path):
    """The exact `artifacts/frontier_strategyqa/run_ids.*.cap*.tsv` shape (run_id LAST).
    Reading a fixed column index here would silently return task_ids -- same-shaped-enough
    strings, no parse error, a wrong population with no symptom. This is the regression the
    content-based reader exists to prevent."""
    p = tmp_path / "run_ids.teacher.cap4.tsv"
    p.write_text(
        "# task_id\tseed\trun_id\n"
        "00dc05718aedf2370213\t0\t575419739f057979895ef9607e2aa8d6\n"
        "00dc05718aedf2370213\t1\t239bf0f885a08bf885770e2a13221b56\n"
    )
    ids = read_run_ids(p)
    assert ids == [
        "575419739f057979895ef9607e2aa8d6",
        "239bf0f885a08bf885770e2a13221b56",
    ]
    # the wrong-column reading, named explicitly so a future edit cannot reintroduce it silently
    wrong = ["00dc05718aedf2370213", "00dc05718aedf2370213"]
    assert ids != wrong


def test_a_run_id_task_id_seed_txt_also_yields_run_ids_not_task_ids(tmp_path):
    """The exact `artifacts/frontier/run_ids.*.cap*.txt` shape (run_id FIRST, task_id has
    letters and underscores so it can never be mistaken for the 32-hex run_id). This is the
    OPPOSITE column order from the strategyqa `.tsv` above -- a fixed index (first OR last)
    gets one of the two formats wrong; content-based selection is what makes one reader right
    for both. Caught for real: an earlier last-field-only version of this reader misread this
    exact file's `seed` column as run_id."""
    p = tmp_path / "run_ids.teacher.cap4.txt"
    p.write_text(
        "# artifacts/frontier/run_ids.teacher.cap4.txt\n"
        "# grid=frontier_trained_musique_cap4\n"
        "1c5e67b05dd34a97d7e8d740bad5a4fc\t4hop3__838995_608613_54405_4107\t1\n"
        "2918480cddd9be5d36d89e76b858bae7\t4hop3__838995_608613_83398_4107\t0\n"
    )
    ids = read_run_ids(p)
    assert ids == [
        "1c5e67b05dd34a97d7e8d740bad5a4fc",
        "2918480cddd9be5d36d89e76b858bae7",
    ]


def test_an_ambiguous_or_run_id_free_line_is_refused_not_guessed(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text("00dc05718aedf2370213\t0\n")  # neither field is a 32-hex run_id
    with pytest.raises(ValueError):
        read_run_ids(p)


def test_the_n6_seven_column_format_is_resolved_by_its_header_not_content(tmp_path):
    """The exact `artifacts/n6/run_ids.inquirer_prompted.txt` shape: run_id FIRST, but a
    wiki2 task_id can ALSO be 32 lowercase hex characters (measured on the real file:
    '00c727580bde11eba7f7acde48001122'), so content alone is ambiguous -- two fields match
    the run_id shape. Caught for real: an earlier content-only version of this reader refused
    every row of this file outright. The header line naming the columns must be what
    disambiguates it."""
    p = tmp_path / "run_ids.inquirer_prompted.txt"
    p.write_text(
        "# grid=tier1_trained_qa_teacher arm=inquirer_prompted\n"
        "# run_id\tsuite\tseed\tsplit\tstatus\tusd\ttask_id\n"
        "a0f9a9fcebbe7b0edea2acca06861c8c\twiki2\t0\ttest\tok\t0.003908\t00c727580bde11eba7f7acde48001122\n"
        "cbf3db8ce8a121763b0fc6c0172dd280\tmusique\t0\ttest\tok\t0.002858\t2hop__11598_11596\n"
    )
    ids = read_run_ids(p)
    assert ids == [
        "a0f9a9fcebbe7b0edea2acca06861c8c",
        "cbf3db8ce8a121763b0fc6c0172dd280",
    ]


def test_a_header_naming_a_column_that_is_not_run_id_shaped_is_refused(tmp_path):
    """A header claiming index 0 is run_id, but the actual value there is not 32-hex, must be
    caught rather than trusted blindly -- the header is a strong signal, not a blank check."""
    p = tmp_path / "bad_header.tsv"
    p.write_text("# run_id\ttask_id\nnot-a-run-id\tsomething\n")
    with pytest.raises(ValueError):
        read_run_ids(p)


def test_blank_lines_are_skipped_not_collected_as_an_empty_run_id(tmp_path):
    p = tmp_path / "run_ids.txt"
    p.write_text("# header\n\nabc\n\ndef\n")
    assert read_run_ids(p) == ["abc", "def"]


# --------------------------------------------------------------------- elects_to_stop


def test_elects_to_stop_is_total_minus_ceiling_hit():
    # artifacts/frontier/FRONTIER.md base8b cap24: ceiling-hit 77/400 -> elects to stop 323/400,
    # the exact figure introduction.tex:80 quotes.
    assert elects_to_stop(400, 77) == 323


def test_elects_to_stop_refuses_a_ceiling_hit_count_above_the_total():
    with pytest.raises(ValueError):
        elects_to_stop(400, 401)


def test_elects_to_stop_refuses_negative_inputs():
    with pytest.raises(ValueError):
        elects_to_stop(400, -1)
    with pytest.raises(ValueError):
        elects_to_stop(-1, 0)


# --------------------------------------------------------------------- assert_population_scored


def _write_store(root, runs_rows, scores_rows):
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(runs_rows), root / "runs.parquet")
    pq.write_table(pa.table(scores_rows), root / "scores.parquet")


def test_population_assertion_passes_when_every_run_id_is_scored_at_the_named_hash(tmp_path):
    store = tmp_path / "store_ok"
    _write_store(
        store,
        {"run_id": ["a", "b", "c"]},
        {
            "run_id": ["a", "b", "c"],
            "metric_name": ["evidence_coverage"] * 3,
            "value": [1.0, 0.5, 0.0],
            "scorer_hash": ["HASH1"] * 3,
            "graph_version": ["v1"] * 3,
        },
    )
    report = assert_population_scored(
        store, ["a", "b", "c"], expected_scorer_hash="HASH1", expected_graph_version="v1"
    )
    assert report.n_checked == 3
    assert report.scorer_hash == "HASH1"


def test_population_assertion_names_a_run_id_missing_from_runs_parquet(tmp_path):
    store = tmp_path / "store_missing_run"
    _write_store(
        store,
        {"run_id": ["a", "b"]},
        {
            "run_id": ["a", "b"],
            "metric_name": ["evidence_coverage"] * 2,
            "value": [1.0, 0.5],
            "scorer_hash": ["HASH1"] * 2,
            "graph_version": ["v1"] * 2,
        },
    )
    with pytest.raises(PopulationIncomplete) as exc:
        assert_population_scored(
            store, ["a", "b", "c"], expected_scorer_hash="HASH1", expected_graph_version="v1"
        )
    assert exc.value.missing_runs == ["c"]
    assert exc.value.missing_scores == []
    assert exc.value.wrong_hash == []


def test_population_assertion_names_a_run_id_scored_under_the_wrong_hash(tmp_path):
    """Exactly the failure mode Lane L0.2's compaction hit today: the run exists and is
    scored, but under a scorer_hash not yet cleared for reporting. Must be distinguished
    from "missing" -- the fix is different (poll for a green light, not re-launch)."""
    store = tmp_path / "store_wrong_hash"
    _write_store(
        store,
        {"run_id": ["a", "b"]},
        {
            "run_id": ["a", "b"],
            "metric_name": ["evidence_coverage"] * 2,
            "value": [1.0, 0.5],
            "scorer_hash": ["HASH1", "PENDING_HASH"],
            "graph_version": ["v1", "v1"],
        },
    )
    with pytest.raises(PopulationIncomplete) as exc:
        assert_population_scored(
            store, ["a", "b"], expected_scorer_hash="HASH1", expected_graph_version="v1"
        )
    assert exc.value.missing_runs == []
    assert exc.value.wrong_hash == ["b"]


def test_population_assertion_names_a_run_id_with_no_score_row_at_all(tmp_path):
    store = tmp_path / "store_unscored"
    _write_store(
        store,
        {"run_id": ["a", "b"]},
        {
            "run_id": ["a"],
            "metric_name": ["evidence_coverage"],
            "value": [1.0],
            "scorer_hash": ["HASH1"],
            "graph_version": ["v1"],
        },
    )
    with pytest.raises(PopulationIncomplete) as exc:
        assert_population_scored(
            store, ["a", "b"], expected_scorer_hash="HASH1", expected_graph_version="v1"
        )
    assert exc.value.missing_scores == ["b"]


# --------------------------------------------------------------------- coordinator rule:
# near-zero bounds get a 50k/3-seed stability check, and an unstable one reads as undecided


def test_a_bound_of_zero_point_zero_zero_nine_is_flagged_near_zero():
    # strategyqa matched_question_tokens on test: [+0.00956, +0.09348] -- the actual cell this
    # rule exists for. The lower bound must be flagged; the upper must not.
    assert near_zero_bounds(0.00956, 0.09348) == ["lo"]


def test_a_bound_of_exactly_the_threshold_is_not_flagged():
    assert near_zero_bounds(0.01, 0.5) == []


def test_both_bounds_can_be_flagged_at_once():
    assert near_zero_bounds(-0.001, 0.004) == ["lo", "hi"]


def test_a_far_from_zero_interval_flags_nothing():
    assert near_zero_bounds(0.0210, 0.0912) == []


def test_stability_verdict_is_stable_when_the_flagged_bound_keeps_its_sign():
    bounds = {0: (0.0096, 0.0935), 1: (0.0021, 0.0890), 2: (0.0055, 0.0910)}
    assert stability_verdict(bounds, ["lo"]) == "stable"


def test_stability_verdict_is_undecided_when_the_flagged_bound_flips_sign():
    bounds = {0: (0.0096, 0.0935), 1: (-0.0012, 0.0890), 2: (0.0055, 0.0910)}
    assert stability_verdict(bounds, ["lo"]) == "undecided"


def test_stability_verdict_ignores_an_unflagged_bound_that_flips():
    # only 'lo' was near zero; 'hi' flipping sign across seeds (unrealistic here, but the
    # function must not go looking at bounds nothing flagged) must not affect the verdict.
    bounds = {0: (0.02, 0.05), 1: (0.02, -0.01), 2: (0.02, 0.04)}
    assert stability_verdict(bounds, ["lo"]) == "stable"


# --------------------------------------------------------------------- assert_turns_not_dropped
#
# Names the 2026-09-18 compaction defect directly: a run whose turns.jsonl is intact and
# non-empty on disk compacted to n_turns=0, calls rows missing.


def _write_turns_store(root, runs_rows, calls_rows):
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(runs_rows), root / "runs.parquet")
    pq.write_table(pa.table(calls_rows), root / "calls.parquet")


def _write_fake_run(runs_root, run_id, turns_jsonl_content):
    d = runs_root / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "turns.jsonl").write_text(turns_jsonl_content)


def test_a_run_whose_turns_jsonl_is_intact_and_n_turns_is_positive_passes(tmp_path):
    store = tmp_path / "store"
    _write_turns_store(
        store,
        {"run_id": ["a"], "n_turns": [3]},
        {"run_id": ["a"], "actor": ["inquirer"]},
    )
    runs_root = tmp_path / "runs"
    _write_fake_run(runs_root, "a", '{"turn_idx": 0}\n{"turn_idx": 1}\n{"turn_idx": 2}\n')
    report = assert_turns_not_dropped(store, ["a"], runs_root=runs_root)
    assert report.n_nonempty_on_disk == 1
    assert report.n_zeroed_n_turns == 0
    assert report.n_missing_inquirer_calls == 0


def test_a_run_with_no_turns_on_disk_is_skipped_not_flagged(tmp_path):
    """A genuinely 0-ask run (drafter_only, never-ask) has an empty or absent turns.jsonl --
    n_turns=0 is then correct, not the defect this check exists to catch."""
    store = tmp_path / "store"
    _write_turns_store(
        store,
        {"run_id": ["a"], "n_turns": [0]},
        {"run_id": [], "actor": []},
    )
    runs_root = tmp_path / "runs"
    (runs_root / "a").mkdir(parents=True)
    (runs_root / "a" / "turns.jsonl").write_text("")  # empty file: genuinely 0 turns
    report = assert_turns_not_dropped(store, ["a"], runs_root=runs_root)
    assert report.n_nonempty_on_disk == 0


def test_the_exact_defect_is_caught_turns_intact_on_disk_but_n_turns_zeroed(tmp_path):
    store = tmp_path / "store"
    _write_turns_store(
        store,
        {"run_id": ["a", "b"], "n_turns": [0, 2]},  # 'a' dropped, 'b' fine
        # both have an inquirer calls row, so 'a' fails ONLY the n_turns check -- isolates it
        # from the missing-calls check exercised separately below.
        {"run_id": ["a", "b"], "actor": ["inquirer", "inquirer"]},
    )
    runs_root = tmp_path / "runs"
    _write_fake_run(runs_root, "a", '{"turn_idx": 0}\n{"turn_idx": 1}\n')  # intact on disk
    _write_fake_run(runs_root, "b", '{"turn_idx": 0}\n{"turn_idx": 1}\n')
    with pytest.raises(TurnsDropped) as exc:
        assert_turns_not_dropped(store, ["a", "b"], runs_root=runs_root)
    assert exc.value.report.zeroed_run_ids == ("a",)
    assert exc.value.report.n_missing_inquirer_calls == 0


def test_a_run_missing_its_inquirer_calls_row_is_caught_even_with_n_turns_intact(tmp_path):
    store = tmp_path / "store"
    _write_turns_store(
        store,
        {"run_id": ["a"], "n_turns": [2]},
        {"run_id": [], "actor": []},  # no inquirer row for 'a' at all
    )
    runs_root = tmp_path / "runs"
    _write_fake_run(runs_root, "a", '{"turn_idx": 0}\n{"turn_idx": 1}\n')
    with pytest.raises(TurnsDropped) as exc:
        assert_turns_not_dropped(store, ["a"], runs_root=runs_root)
    assert exc.value.report.missing_call_run_ids == ("a",)


# --------------------------------------------------------- assert_one_code_version_and_no_duplicate_keys
#
# Names a hazard measured on the shared store: `_select_runs` has no code_version filter, so a
# store holding one arm's runs from more than one rollout code version pools them into one mean
# silently, and the same nominal (suite, task, seed) key can carry different evidence_coverage
# values across them.


def _write_runs_store(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), root / "runs.parquet")


def test_one_code_version_and_no_duplicate_keys_passes_clean(tmp_path):
    store = tmp_path / "store"
    _write_runs_store(
        store,
        {
            "run_id": ["a", "b"],
            "suite_id": ["musique", "strategyqa"],
            "task_id": ["t1", "t2"],
            "seed": [0, 0],
            "code_version": ["cv1", "cv1"],
        },
    )
    report = assert_one_code_version_and_no_duplicate_keys(store, ["a", "b"], arm="prompted")
    assert report["code_versions"] == {"cv1": 2}
    assert report["n_duplicate_keys"] == 0


def test_two_code_versions_in_one_arm_is_caught(tmp_path):
    store = tmp_path / "store"
    _write_runs_store(
        store,
        {
            "run_id": ["a", "b"],
            "suite_id": ["musique", "musique"],
            "task_id": ["t1", "t2"],
            "seed": [0, 0],
            "code_version": ["cv1", "cv2"],
        },
    )
    with pytest.raises(PooledCodeVersions) as exc:
        assert_one_code_version_and_no_duplicate_keys(store, ["a", "b"], arm="prompted")
    assert exc.value.code_versions == {"cv1": 1, "cv2": 1}


def test_a_duplicated_suite_task_seed_key_is_caught_even_at_one_code_version(tmp_path):
    """The exact silent-pooling shape: same key, two rows, one code_version -- a mean over
    them would average two populations without either a code_version difference or an error
    to notice by."""
    store = tmp_path / "store"
    _write_runs_store(
        store,
        {
            "run_id": ["a", "b"],
            "suite_id": ["musique", "musique"],
            "task_id": ["t1", "t1"],
            "seed": [0, 0],
            "code_version": ["cv1", "cv1"],
        },
    )
    with pytest.raises(PooledCodeVersions) as exc:
        assert_one_code_version_and_no_duplicate_keys(store, ["a", "b"], arm="prompted")
    assert exc.value.duplicate_keys == {("musique", "t1", 0): 2}


# --------------------------------------------------------------------- symmetric_contrast
#
# Names the matched-cost-helper defect a coordinator review found: matched_cost.py's
# contrast() reads the treatment (`va`) at its own n_asks UNCONDITIONALLY (matched_cost.py:626)
# and only clamps the comparator via Ladder.asks_at's min(k, self.n_asks) (matched_cost.py:248).
# On a pair where the treatment asks MORE than the comparator, the treatment is never brought
# down, so that pair silently reverts to an unmatched reading under a "matched" label.


def _ladder(run_id, task_id, n_asks, cov, arm_id="inquirer_trained", seed=0, q_tok=()):
    return Ladder(
        run_id=run_id,
        suite_id="musique",
        task_id=task_id,
        cluster_id=task_id,
        arm_id=arm_id,
        seed=seed,
        n_asks=n_asks,
        stop_reason="policy_stop",
        cov=cov,
        cad2_hit=(),
        cad2_n=0,
        cad_hit={},
        cad_n={},
        q_tok=q_tok,
    )


def test_symmetric_contrast_reads_both_arms_at_the_lower_question_count():
    """The exact unsafe pair, ask-priced (comparator='matched_k'): treatment (trained) asks 4,
    comparator (peer) asks 2. The asymmetric contrast() would read trained at cov[4]=0.8 and
    peer at cov[min(4,2)]=cov[2]=0.9, giving -0.1. The symmetric fix must read BOTH at
    k_sym=min(4,2)=2: trained cov[2]=0.4, peer cov[2]=0.9, giving -0.5. This is the regression
    test: an implementation that reverted to reading the treatment at its own n_asks would
    silently give -0.1 here instead."""
    trained = _ladder("t1", "task1", n_asks=4, cov=(0.0, 0.2, 0.4, 0.6, 0.8))
    peer = _ladder("p1", "task1", n_asks=2, cov=(0.0, 0.5, 0.9), arm_id="inquirer_prompted")
    peer_dict = {("musique", "task1", 0): peer}

    est, n_pairs, n_unsafe = symmetric_contrast(
        [trained], peer_dict, comparator="matched_k", n_boot=10, n_perm=10, seed=0
    )

    assert n_pairs == 1
    assert n_unsafe == 1  # peer.n_asks (2) < trained.n_asks (4): the unsafe condition
    assert est.point == pytest.approx(0.4 - 0.9)
    assert est.point != pytest.approx(0.8 - 0.9)  # the asymmetric (wrong) answer, named


def test_symmetric_contrast_is_a_noop_when_the_treatment_already_asks_fewer():
    """The SAFE direction, stated as a positive control: when treatment already asks fewer
    than the comparator, k_sym = treatment's own n_asks, so the symmetric reading coincides
    with what the (correct, in this direction) asymmetric contrast() would already give."""
    trained = _ladder("t1", "task1", n_asks=2, cov=(0.0, 0.3, 0.5))
    peer = _ladder(
        "p1", "task1", n_asks=4, cov=(0.0, 0.1, 0.2, 0.3, 0.9), arm_id="inquirer_prompted"
    )
    peer_dict = {("musique", "task1", 0): peer}

    est, n_pairs, n_unsafe = symmetric_contrast(
        [trained], peer_dict, comparator="matched_k", n_boot=10, n_perm=10, seed=0
    )

    assert n_pairs == 1
    assert n_unsafe == 0  # peer.n_asks (4) is NOT below trained.n_asks (2): safe
    assert est.point == pytest.approx(0.5 - 0.2)  # k_sym = min(2,4) = 2 for both


def test_symmetric_contrast_seed_averages_within_a_task():
    """Two seeds of the same task must average into one task-level pair, not enter the
    bootstrap as two independent units (the same convention matched_cost.py's own
    _seed_averaged uses, restated here since this function does not call it)."""
    trained0 = _ladder("t1", "task1", n_asks=2, cov=(0.0, 0.4, 0.6), seed=0)
    trained1 = _ladder("t2", "task1", n_asks=2, cov=(0.0, 0.2, 0.8), seed=1)
    peer0 = _ladder(
        "p1", "task1", n_asks=2, cov=(0.0, 0.5, 0.7), arm_id="inquirer_prompted", seed=0
    )
    peer1 = _ladder(
        "p2", "task1", n_asks=2, cov=(0.0, 0.1, 0.3), arm_id="inquirer_prompted", seed=1
    )
    peer_dict = {("musique", "task1", 0): peer0, ("musique", "task1", 1): peer1}

    est, n_pairs, n_unsafe = symmetric_contrast(
        [trained0, trained1], peer_dict, comparator="matched_k", n_boot=10, n_perm=10, seed=0
    )

    assert n_pairs == 2  # two (suite,task,seed) pairs seen
    # seed 0: va=0.6, vb=0.7 (k_sym=2 both); seed 1: va=0.8, vb=0.3 (k_sym=2 both)
    # averaged into one task: va=(0.6+0.8)/2=0.7, vb=(0.7+0.3)/2=0.5, delta=0.2
    assert est.point == pytest.approx(0.7 - 0.5)


def test_symmetric_contrast_charges_tokens_not_asks_for_a_budget_priced_comparator():
    """THE regression test for the real defect: an earlier version of `symmetric_contrast`
    ignored `comparator` entirely and always used `min(trained.n_asks, peer.n_asks)` -- ask
    counts -- even when called for `matched_question_tokens`, a TOKEN-priced comparator. That
    silently computed the ask-count (`matched_k`-style) answer under a token-charged label. It
    was caught because the mislabelled output agreed with an independently-computed symmetric
    calls-matched figure to four decimals on three suites, which is not a plausible coincidence
    between two different cost bases.

    This fixture is built so the two bases give MEASURABLY DIFFERENT answers, so a reversion to
    the ask-count fallback cannot pass by accident: trained asks 3 questions at 10 tokens each
    (cumulative 10/20/30); peer asks 5 short questions at 2 tokens each (cumulative
    2/4/6/8/10). By ASK COUNT, k_sym=min(3,5)=3: trained.cov[3]=0.9, peer.cov[3]=0.3, delta
    +0.6. By TOKEN COST (the correct basis here), trained's own total spend is 30 tokens,
    peer's is 10; budget_sym=min(30,10)=10; trained.k_within(10,...)=1 (10<=10, 20>10) and
    peer.k_within(10,...)=5 (peer's whole trajectory costs exactly 10): trained.cov[1]=0.3,
    peer.cov[5]=1.0, delta -0.7. The two are far enough apart that they cannot be confused.
    """
    trained = _ladder("t1", "task1", n_asks=3, cov=(0.0, 0.3, 0.6, 0.9), q_tok=(10, 10, 10))
    peer = _ladder(
        "p1",
        "task1",
        n_asks=5,
        cov=(0.0, 0.1, 0.2, 0.3, 0.4, 1.0),
        arm_id="inquirer_prompted",
        q_tok=(2, 2, 2, 2, 2),
    )
    peer_dict = {("musique", "task1", 0): peer}

    est, n_pairs, n_unsafe = symmetric_contrast(
        [trained],
        peer_dict,
        comparator="matched_question_tokens",
        n_boot=10,
        n_perm=10,
        seed=0,
    )

    assert n_pairs == 1
    assert n_unsafe == 1  # peer's own total token cost (10) < trained's own total cost (30)
    assert est.point == pytest.approx(0.3 - 1.0)  # the TOKEN-basis answer
    assert est.point != pytest.approx(0.9 - 0.3)  # the ASK-COUNT-fallback (wrong) answer, named
