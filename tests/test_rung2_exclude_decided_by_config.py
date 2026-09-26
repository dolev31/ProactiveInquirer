"""`exclude_decided_by` MUST BE REACHABLE FROM A `--config` FILE, or A11 cannot be launched.

A11 ("gain key") is `dpo_outcome_only` -- the pairs only the gain rule could order dropped --
against `dpo_control` at the same n. The subsample half of that pair landed in d4589a5 and is
settable from a config file. This half was not: `exclude_decided_by` was read straight off
`a.exclude_decided_by`, and the key was absent from `_rung2_defaults`, so

  * `pi train rung2 --config <grid>` REFUSED a file naming it ("which this command has no flag
    for") -- a refusal, not a silent miss, but a refusal is still an arm that cannot run; and
  * `scripts/hpc/run_experiment.sh` passes `--config`, `--base-model`, the reference,
    `--pairs`, `--out` and `--seed` and has no flag channel at all, so the outcome-only arm was
    unlaunchable on the cluster whatever the row said.

Both halves of the ablation therefore have to travel the same route, and this file pins that
route. `subsample_pairs` is the worked example next door (tests/test_rung2_subsample.py); the
difference is `exclude_decided_by`'s default, which is `()` and NOT absent: it has been inside
`cfg.sha` since the field was written, so wiring it must leave every finished rung-2 run
hashing exactly what it hashed. That is what `PRE_CHANGE_DEFAULT_SHA` below is for.
"""

from __future__ import annotations

import json

import pytest

from pinq.actions import ask_action_json
from pinq_train.rung2_dpo import DPOConfig

# MEASURED before this commit, and the SAME constant tests/test_rung2_subsample.py and
# tests/test_rung2_objective.py pin for the same reason: the sha of the default config. Wiring
# a field that already had a default of `()` must render exactly these bytes.
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


def _prep(tmp_path, n_gain=8, n_outcome=12):
    """A file whose two deciders are distinguishable by count alone, so `n_pairs` says which
    filter ran without having to look at a row."""
    rows = [_p(i, decided_by="gain") for i in range(n_gain)]
    rows += [_p(100 + i) for i in range(n_outcome)]
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p, tmp_path / "rung2"


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


def _argv(pairs, out, *extra):
    return [
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
        *extra,
        "--train",
        "--acknowledge-untested",
    ]


def _first_json(out: str) -> str:
    start = out.index("{")
    depth, end = 0, start
    for i, ch in enumerate(out[start:], start):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            end = i + 1
            break
    return out[start:end]


# ---- the route that matters on the cluster


def test_a_config_file_can_set_exclude_decided_by(tmp_path, monkeypatch, capsys):
    """THE GAP. `run_experiment.sh` builds a fixed argument list from the experiment row and
    passes the grid as `--config`; a knob reachable only by flag is a knob no queued experiment
    can use, and E48 (`dpo_outcome_only`) is exactly that experiment."""
    pairs, out = _prep(tmp_path)
    conf = tmp_path / "rung2_outcome_only.json"
    conf.write_text(json.dumps({"exclude_decided_by": ["gain"]}) + "\n")
    seen = _run_cli(_argv(pairs, out, "--config", str(conf)), monkeypatch)
    report = json.loads(_first_json(capsys.readouterr().out))
    assert seen["cfg"].exclude_decided_by == ("gain",)
    assert seen["n"] == 12, "the 8 gain-decided rows are dropped before the trainer sees them"
    assert report["n_pairs"] == 12
    assert report["n_dropped_by_filter"]["decided_by"] == 8


def test_the_config_file_value_is_in_the_run_s_identity(tmp_path, monkeypatch, capsys):
    """The non-vacuity half: a config key that resolved to the default would still produce a
    12-row run if the file happened to hold 12 -- so pin that the ARM's sha differs from the
    unfiltered one. Two datasets, two checkpoints, two identities."""
    pairs, out = _prep(tmp_path)
    conf = tmp_path / "rung2_outcome_only.json"
    conf.write_text(json.dumps({"exclude_decided_by": ["gain"]}) + "\n")
    cut = _run_cli(_argv(pairs, out, "--config", str(conf)), monkeypatch)["cfg"]
    full = _run_cli(_argv(pairs, tmp_path / "rung2b"), monkeypatch)["cfg"]
    capsys.readouterr()
    assert cut.sha != full.sha


# ---- the flag must keep working, and must still win


def test_the_flag_still_reaches_the_trainer(tmp_path, monkeypatch, capsys):
    """The route that already worked. Wiring the key through `_resolve_options` must not take
    it away: `--exclude-decided-by` is how this ablation was run by hand before it had a row."""
    pairs, out = _prep(tmp_path)
    seen = _run_cli(_argv(pairs, out, "--exclude-decided-by", "gain"), monkeypatch)
    capsys.readouterr()
    assert seen["cfg"].exclude_decided_by == ("gain",) and seen["n"] == 12


def test_the_flag_overrides_the_config_file(tmp_path, monkeypatch, capsys):
    """FLAG > `--config` > default, the rule `_resolve_options` documents, applied to a list
    field: an operator overriding a grid from the command line must not get the union."""
    pairs, out = _prep(tmp_path)
    conf = tmp_path / "rung2_outcome_only.json"
    conf.write_text(json.dumps({"exclude_decided_by": ["gain"]}) + "\n")
    seen = _run_cli(
        _argv(pairs, out, "--config", str(conf), "--exclude-decided-by", "outcome"), monkeypatch
    )
    capsys.readouterr()
    assert seen["cfg"].exclude_decided_by == ("outcome",)
    assert seen["n"] == 8, "the outcome-decided rows dropped, the gain-decided ones kept"


# ---- identity: an unset value must not move a single finished run


def test_an_unset_value_is_the_empty_tuple_and_not_an_empty_string(tmp_path, monkeypatch, capsys):
    """`exclude_decided_by` is NOT in `SHA_OMIT_WHEN_NONE`: `()` is hashed, and it is what
    every rung-2 run of record hashed. An unset knob that arrives as `""` or `(\"\",)` is a
    value that enters -- or corrupts -- the sha of runs that predate this commit."""
    pairs, out = _prep(tmp_path)
    seen = _run_cli(_argv(pairs, out), monkeypatch)
    capsys.readouterr()
    assert seen["cfg"].exclude_decided_by == ()
    assert seen["n"] == 20, "no exclusion set: every row the other filters kept"


def test_the_unset_default_hashes_what_the_older_code_hashed():
    """The identity pin. `_rung2_defaults` must return `DPOConfig`'s own `()` -- not `None`,
    not `[]`, not `""` -- so a config built without the key is byte-identical to one built
    before the key existed."""
    import pinq_train.rung2_dpo as rung2
    from pi_run import cmd_train as ct

    defaults = ct._rung2_defaults(rung2)
    assert defaults["exclude_decided_by"] == ()
    cfg = DPOConfig(
        base_model="Qwen/Qwen3-8B",
        adapter="artifacts/rung1",
        exclude_decided_by=tuple(defaults["exclude_decided_by"]),
    )
    assert cfg.sha == PRE_CHANGE_DEFAULT_SHA


def test_the_default_follows_a_changed_dpoconfig_field(monkeypatch):
    """AUTHORITY-FOLLOWING, as tests/test_train_cli.py requires of every other key in the map:
    the default is READ from the dataclass, not retyped here as a second literal that could
    drift away from it under a `cfg.sha` that says otherwise."""
    import pinq_train.rung2_dpo as rung2
    from pi_run import cmd_train as ct

    monkeypatch.setattr(
        rung2.DPOConfig.__dataclass_fields__["exclude_decided_by"], "default", ("rater",)
    )
    assert ct._rung2_defaults(rung2)["exclude_decided_by"] == ("rater",)


# ---- the grid files the launch actually names


@pytest.mark.parametrize(
    "path,key,value",
    [
        ("conf/train/rung2_4b_outcome_only.json", "exclude_decided_by", ["gain"]),
        ("conf/train/rung2_4b_outcome_matched.json", "subsample_pairs", None),
    ],
)
def test_the_a11_grids_are_accepted_by_this_command(path, key, value):
    """A grid file is refused wholesale if it names one key this command cannot consume, so the
    two A11 files are checked against the real defaults map rather than eyeballed. Also pins
    that each arm's own knob is actually IN its file: a matched control whose grid lost its
    `subsample_pairs` trains the full arm under the matched arm's name."""
    import argparse
    from pathlib import Path

    import pinq_train.rung2_dpo as rung2
    from pi_run import cmd_train as ct

    raw = json.loads(Path(path).read_text())
    defaults = ct._rung2_defaults(rung2)
    unknown = sorted(k for k in raw if not k.startswith("_") and k not in defaults)
    assert unknown == [], f"{path} names keys `pi train rung2` cannot consume: {unknown}"
    assert key in raw
    if value is not None:
        assert raw[key] == value
    ns = argparse.Namespace(**{k: None for k in defaults}, config=path)
    opt = ct._resolve_options(ns, defaults)
    assert opt[key] == raw[key]


def test_the_matched_control_does_not_also_exclude_the_decider():
    """WRITTEN AFTER THE DEFECT IT CATCHES, which the preflight could not: the first draft of
    `rung2_4b_outcome_matched.json` was generated from the outcome-only file and carried its
    `exclude_decided_by` along. The subsample then drew 6,341 of the 6,341 the exclusion had
    already left -- `drops['subsample']` is recorded only when non-zero, so the preflight
    printed `n_pairs: 6341`, EXACTLY the target, and the control's dataset was the other arm's.
    A matched control that excludes the same rows is not a control; it is E48 twice.

    The recipe test below cannot see this: it compares the two files with the selector keys
    removed, which is precisely the field that was wrong. Measured on login-node 2026-09-18."""
    from pathlib import Path

    raw = json.loads(Path("conf/train/rung2_4b_outcome_matched.json").read_text())
    assert "exclude_decided_by" not in raw
    assert raw["subsample_pairs"] == 6341 and raw["subsample_seed"] is not None
    only = json.loads(Path("conf/train/rung2_4b_outcome_only.json").read_text())
    assert "subsample_pairs" not in only and "subsample_seed" not in only


def test_the_two_a11_grids_share_one_recipe():
    """THE WHOLE OF THE ABLATION IS THE ONE KEY. If the two grids differ in a hyperparameter as
    well as in their selector, the contrast measures the pair of them and the ablation's own
    sentence ("if equal, the gain-decided pairs are noise") is unavailable."""
    from pathlib import Path

    selectors = {"exclude_decided_by", "subsample_pairs", "subsample_seed"}

    def recipe(p):
        raw = json.loads(Path(p).read_text())
        return {k: v for k, v in raw.items() if not k.startswith("_") and k not in selectors}

    only = recipe("conf/train/rung2_4b_outcome_only.json")
    matched = recipe("conf/train/rung2_4b_outcome_matched.json")
    assert only == matched
    assert only == recipe("conf/train/rung2_8b.json"), "both are the grid of record's recipe"
