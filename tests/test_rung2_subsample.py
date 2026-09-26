"""A MATCHED-n CONTROL NEEDS A SUBSAMPLE, AND A SUBSAMPLE IS NOT A FILTER.

A11 ("gain key") asks whether the pairs only the gain rule could order carry signal:
`dpo_outcome_only` (those pairs dropped) against `dpo_control` AT THE SAME n. Every selector
rung 2 had was a PREDICATE -- keep the rows that look like this -- and no predicate can
express "any k of them". Without the mechanism the control trains on the whole file, the two
arms differ in n as well as in labelling, and the ablation's own sentence ("if equal, the
gain-decided pairs are noise") is unavailable: equal at different n says nothing.

MEASURED on data/rl/pairs.jsonl (the headline export is not on this machine; same shape):
14,927 ask_ask pairs, of which 6,827 (45.7%) carry decided_by="gain" -- the 44% the ablation's
claim names. `--exclude-decided-by gain` is what leaves 8,100 of them, and this is what draws
8,100 of the 14,927 to put beside it.

WHAT IS PINNED HERE, and why each one is a way to publish a wrong number rather than a crash:

* EXACTLY n, or a loud refusal. Silently returning fewer is a matched-cost arm that is not
  matched, under a config sha that says it is.
* DETERMINISTIC in (seed, rows). A subset that moves between the preflight and the trainer,
  or between a resumed run and the run it continues, is two datasets under one identity.
* A RANDOM subset, not the first n. MEASURED on data/rl/pairs.jsonl: the file is grouped by
  suite (musique 23,039 rows, then strategyqa 22,003, then wiki2 2,947) and its first 5,000
  rows are 81 musique tasks of the 1,933 in the file. Truncation would make the control a
  different POPULATION, not a smaller sample of the same one, and the contrast would measure
  the suite mix.
* ORDER-INDEPENDENT. Two exports of one row set differ in line order (counters, dict order);
  the sample must not. The selection key is `pair_id`, which the exporter derives from
  (suite, task, run, turn, both candidate run ids, kind) -- MEASURED on data/rl/pairs.jsonl:
  47,989 rows, 47,989 distinct pair_ids, none missing.
* IN `cfg.sha`. `dpo_control` cut to the other arm's n at seed 0 and at seed 1 are two
  datasets and two checkpoints; sharing the full run's identity would let them collide in the
  registry. ABSENT at the default, so every rung-2 run already trained reproduces its sha.
"""

from __future__ import annotations

import json

import pytest

from pinq.actions import ask_action_json
from pinq_train.rung2_dpo import DPOConfig, TooFewPairsToSubsample, load_pairs

SHA = "0" * 64

# MEASURED before this commit (tests/test_rung2_objective.py pins the same number for the same
# reason): the sha of the default config. Two optional fields added at their absent default
# must render exactly these bytes, or this commit renames every finished rung-2 run.
PRE_CHANGE_DEFAULT_SHA = "63dcf89f421d0284772c20270d92103e2438a3dde6e6b25525dd9a7b814e1048"


def _p(i: int, **over):
    r = {
        "suite_id": "musique",
        "task_id": f"t{i}",
        "run_id": "p",
        "turn_idx": 1,
        "state_text": "S",
        "chosen_json": ask_action_json("who?"),
        "rejected_json": ask_action_json("when?"),
        "margin": 0.5,
        "pair_kind": "ask_ask",
        "pair_id": f"{i:016x}",
        "decided_by": "outcome",
    }
    r.update(over)
    return r


def _write(tmp_path, rows, name="pairs.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _ids(rows):
    return [r["pair_id"] for r in rows]


def _cfg(**over) -> DPOConfig:
    return DPOConfig(base_model="Qwen/Qwen3-8B", adapter="artifacts/rung1", **over)


# ---- exactly n, or nothing


def test_the_subsample_yields_exactly_the_target_n(tmp_path):
    p = _write(tmp_path, [_p(i) for i in range(40)])
    rows, drops = load_pairs(p, subsample_pairs=10, subsample_seed=0)
    assert len(rows) == 10
    assert len(set(_ids(rows))) == 10
    assert drops["subsample"] == 30, "the 30 rows it did not take are a drop like any other"


def test_a_target_above_the_available_rows_refuses_rather_than_returning_fewer(tmp_path):
    """The failure mode this exists for: an arm that reports the n it was asked for and
    trained on fewer, because a filter above it removed more than the operator expected."""
    p = _write(tmp_path, [_p(i) for i in range(5)])
    with pytest.raises(TooFewPairsToSubsample) as exc:
        load_pairs(p, subsample_pairs=9, subsample_seed=0)
    assert "9" in str(exc.value) and "5" in str(exc.value)


def test_the_target_counts_the_rows_that_survived_the_filters(tmp_path):
    """SUBSAMPLE AFTER FILTER. The matched control is matched to the n of the OTHER arm, which
    is itself a filtered count -- so the target is checked against the filtered population, and
    a file with enough rows overall does not make an impossible target possible."""
    rows_in = [_p(i, decided_by="gain") for i in range(30)] + [_p(100 + i) for i in range(6)]
    p = _write(tmp_path, rows_in)
    rows, drops = load_pairs(p, exclude_decided_by=("gain",), subsample_pairs=4, subsample_seed=0)
    assert len(rows) == 4 and drops["decided_by"] == 30 and drops["subsample"] == 2
    with pytest.raises(TooFewPairsToSubsample):
        load_pairs(p, exclude_decided_by=("gain",), subsample_pairs=8, subsample_seed=0)


# ---- deterministic, random, order-independent


def test_the_same_seed_over_the_same_rows_chooses_the_same_subset(tmp_path):
    p = _write(tmp_path, [_p(i) for i in range(40)])
    a, _ = load_pairs(p, subsample_pairs=10, subsample_seed=3)
    b, _ = load_pairs(p, subsample_pairs=10, subsample_seed=3)
    assert _ids(a) == _ids(b)


def test_two_seeds_choose_different_subsets(tmp_path):
    """The non-vacuity half of determinism: a function that ignored its seed would pass every
    assertion above."""
    p = _write(tmp_path, [_p(i) for i in range(40)])
    a, _ = load_pairs(p, subsample_pairs=10, subsample_seed=0)
    b, _ = load_pairs(p, subsample_pairs=10, subsample_seed=1)
    assert set(_ids(a)) != set(_ids(b))


def test_the_subset_is_not_the_first_n_rows(tmp_path):
    """Truncation is deterministic, exact and seed-stable -- it passes every test above and is
    a different population. Checked at two seeds so the assertion is not one lucky draw."""
    p = _write(tmp_path, [_p(i) for i in range(40)])
    head = [f"{i:016x}" for i in range(10)]
    for seed in (0, 1, 2):
        rows, _ = load_pairs(p, subsample_pairs=10, subsample_seed=seed)
        assert _ids(rows) != head
        assert sorted(_ids(rows)) != head


def test_the_subset_does_not_depend_on_the_order_of_the_file(tmp_path):
    """Two exports of one row set differ in line order. INVARIANCE, so it carries its own
    non-vacuity check: the same reordered file at a different seed MUST move the subset, or
    this test could not tell an order-independent sample from one that ignores its input."""
    rows_in = [_p(i) for i in range(40)]
    straight = _write(tmp_path, rows_in, "a.jsonl")
    shuffled = _write(tmp_path, list(reversed(rows_in)), "b.jsonl")
    a, _ = load_pairs(straight, subsample_pairs=10, subsample_seed=5)
    b, _ = load_pairs(shuffled, subsample_pairs=10, subsample_seed=5)
    assert set(_ids(a)) == set(_ids(b))
    moved, _ = load_pairs(shuffled, subsample_pairs=10, subsample_seed=6)
    assert set(_ids(moved)) != set(_ids(b)), (
        "forced change: the statistic the invariance test reads must be able to move"
    )


def test_the_rows_come_back_in_the_order_the_file_holds_them(tmp_path):
    """The sample is a SUBSET of the file, not a reordering of it: the trainer's own shuffle is
    seeded by `cfg.seed` and must stay the only thing that decides presentation order."""
    p = _write(tmp_path, [_p(i) for i in range(40)])
    rows, _ = load_pairs(p, subsample_pairs=10, subsample_seed=4)
    assert _ids(rows) == sorted(_ids(rows))


# ---- the selection key


def test_a_row_the_sample_cannot_name_refuses(tmp_path):
    """`pair_id` is the only stable identity a pair carries. With a row that has none the
    sample would have to fall back on position, which is exactly what the order-independence
    test forbids -- so it refuses instead, and says which row."""
    rows_in = [_p(i) for i in range(5)] + [_p(9, pair_id="")]
    p = _write(tmp_path, rows_in)
    with pytest.raises(ValueError, match="pair_id"):
        load_pairs(p, subsample_pairs=3, subsample_seed=0)


def test_two_rows_sharing_one_pair_id_refuse(tmp_path):
    """A duplicated key breaks the exact-n property from underneath: selecting k keys would
    return more than k rows, and `n_pairs` would disagree with the target under a sha that
    names it."""
    p = _write(tmp_path, [_p(1), _p(2), _p(1)])
    with pytest.raises(ValueError, match="pair_id"):
        load_pairs(p, subsample_pairs=2, subsample_seed=0)


# ---- identity


def test_the_absent_default_hashes_what_the_older_code_hashed():
    cfg = _cfg()
    assert cfg.subsample_pairs is None and cfg.subsample_seed is None
    assert cfg.sha == PRE_CHANGE_DEFAULT_SHA


def test_a_subsampled_arm_does_not_share_the_full_run_s_identity():
    """The non-vacuity half of the test above. `dpo_control` and `dpo_control` cut to an n are
    two checkpoints; one `config_sha` over both is a registry collision."""
    assert _cfg(subsample_pairs=100, subsample_seed=0).sha != PRE_CHANGE_DEFAULT_SHA


def test_the_subsample_seed_is_part_of_the_identity():
    a = _cfg(subsample_pairs=100, subsample_seed=0).sha
    b = _cfg(subsample_pairs=100, subsample_seed=1).sha
    assert a != b, "two draws of 100 pairs are two datasets"


def test_a_target_with_no_seed_refuses():
    """NO DEFAULT SEED. Borrowing `cfg.seed` would make the three seed replicates of the
    matched arm train on three different subsets, so the arm's spread would mix training-seed
    variance with subset choice while the table calls it seed variance; defaulting to 0 would
    hide the choice in a field nobody wrote."""
    with pytest.raises(ValueError, match="subsample_seed"):
        _cfg(subsample_pairs=100).validate()


def test_a_seed_with_no_target_refuses():
    """A seed that draws nothing still enters `cfg.sha`: it would rename a run without changing
    a row of its data."""
    with pytest.raises(ValueError, match="subsample_pairs"):
        _cfg(subsample_seed=0).validate()


@pytest.mark.parametrize("n", [0, -1])
def test_a_non_positive_target_refuses(n):
    with pytest.raises(ValueError, match="subsample_pairs"):
        _cfg(subsample_pairs=n, subsample_seed=0).validate()


# ---- the two ways an operator can set it


def _prep(tmp_path, n_rows=20):
    pairs = _write(tmp_path, [_p(i) for i in range(n_rows)])
    return pairs, tmp_path / "rung2"


def _run_cli(argv, monkeypatch):
    import pinq_train.rung2_dpo as rung2
    from pi_run.cli import build_parser

    seen: dict = {}

    def fake_train(cfg, **kw):
        seen["cfg"] = cfg
        seen["n"] = len(kw["rows"])
        return {"out_dir": str(cfg.out_dir)}

    monkeypatch.setattr(rung2, "train", fake_train)
    args = build_parser().parse_args(argv)
    assert args.fn(args) == 0
    return seen


def test_the_flag_reaches_the_trainer_and_the_config(tmp_path, monkeypatch, capsys):
    pairs, out = _prep(tmp_path)
    seen = _run_cli(
        [
            "train",
            "rung2",
            "--base-model",
            "Qwen/Qwen3-4B",
            "--adapter",
            "artifacts/rung1",
            "--pairs",
            str(pairs),
            "--out",
            str(out),
            "--subsample-pairs",
            "6",
            "--subsample-seed",
            "2",
            "--train",
            "--acknowledge-untested",
        ],
        monkeypatch,
    )
    report = json.loads(_first_json(capsys.readouterr().out))
    assert report["n_pairs"] == 6 and seen["n"] == 6
    assert seen["cfg"].subsample_pairs == 6 and seen["cfg"].subsample_seed == 2


def test_a_config_file_can_set_it(tmp_path, monkeypatch, capsys):
    """THE ROUTE THAT MATTERS ON THE CLUSTER. `scripts/hpc/run_experiment.sh` invokes
    `pi train rung2` with `--config`, `--base-model`, the reference, `--pairs`, `--out` and
    `--seed` and nothing else, so a knob reachable only by flag is a knob no queued experiment
    can use."""
    pairs, out = _prep(tmp_path)
    conf = tmp_path / "rung2.json"
    conf.write_text(json.dumps({"subsample_pairs": 6, "subsample_seed": 2}) + "\n")
    seen = _run_cli(
        [
            "train",
            "rung2",
            "--config",
            str(conf),
            "--base-model",
            "Qwen/Qwen3-4B",
            "--adapter",
            "artifacts/rung1",
            "--pairs",
            str(pairs),
            "--out",
            str(out),
            "--train",
            "--acknowledge-untested",
        ],
        monkeypatch,
    )
    report = json.loads(_first_json(capsys.readouterr().out))
    assert report["n_pairs"] == 6 and seen["n"] == 6
    assert seen["cfg"].subsample_pairs == 6 and seen["cfg"].subsample_seed == 2


def test_an_unset_flag_is_absent_and_not_an_empty_string(tmp_path, monkeypatch, capsys):
    """The default must be ABSENT: a flag whose unset value is "" is a value argparse hands on
    to a config field that then enters -- or corrupts -- the sha."""
    pairs, out = _prep(tmp_path)
    seen = _run_cli(
        [
            "train",
            "rung2",
            "--base-model",
            "Qwen/Qwen3-4B",
            "--adapter",
            "artifacts/rung1",
            "--pairs",
            str(pairs),
            "--out",
            str(out),
            "--train",
            "--acknowledge-untested",
        ],
        monkeypatch,
    )
    capsys.readouterr()
    assert seen["cfg"].subsample_pairs is None and seen["cfg"].subsample_seed is None
    assert seen["n"] == 20, "no target set: every row the filters kept"


def _first_json(out: str) -> str:
    start = out.index("{")
    depth, end = 0, start
    for i, ch in enumerate(out[start:], start):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            end = i + 1
            break
    return out[start:end]


def test_an_impossible_target_refuses_on_the_laptop_that_wrote_it(tmp_path, monkeypatch, capsys):
    """REFUSED THROUGH THE COMMAND'S OWN PREFLIGHT PATH, with a non-zero exit -- not as a
    traceback. `scripts/hpc/run_experiment.sh` runs under `set -e` on a queued node: either way
    the job stops, but only one of the two prints the sentence that says which filter took the
    rows, and a traceback out of a queue log reads as a crash rather than a refusal."""
    import pinq_train.rung2_dpo as rung2
    from pi_run.cli import build_parser

    pairs, out = _prep(tmp_path, n_rows=5)

    def never(*a, **k):  # pragma: no cover - the point is that it is not reached
        raise AssertionError("the trainer must not be reached")

    monkeypatch.setattr(rung2, "train", never)
    args = build_parser().parse_args(
        [
            "train",
            "rung2",
            "--base-model",
            "Qwen/Qwen3-4B",
            "--adapter",
            "artifacts/rung1",
            "--pairs",
            str(pairs),
            "--out",
            str(out),
            "--subsample-pairs",
            "9",
            "--subsample-seed",
            "0",
            "--train",
            "--acknowledge-untested",
        ]
    )
    assert args.fn(args) == 1
    err = capsys.readouterr().err
    assert "PREFLIGHT FAILED" in err and "TooFewPairsToSubsample" in err


def test_the_preflight_report_names_the_subsampled_arm(tmp_path, monkeypatch, capsys):
    """`config_sha` is in the report the operator reads before launching, so the two arms of
    A11 can be told apart from their preflights alone."""
    pairs, out = _prep(tmp_path)
    argv = [
        "train",
        "rung2",
        "--base-model",
        "Qwen/Qwen3-4B",
        "--adapter",
        "artifacts/rung1",
        "--pairs",
        str(pairs),
        "--out",
        str(out),
        "--train",
        "--acknowledge-untested",
    ]
    _run_cli(argv, monkeypatch)
    full = json.loads(_first_json(capsys.readouterr().out))
    _run_cli(argv + ["--subsample-pairs", "6", "--subsample-seed", "2"], monkeypatch)
    cut = json.loads(_first_json(capsys.readouterr().out))
    assert full["config_sha"] != cut["config_sha"]
    assert cut["n_dropped_by_filter"]["subsample"] == 14
