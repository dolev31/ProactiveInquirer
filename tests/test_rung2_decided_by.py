"""WHO ORDERED A PAIR IS `decided_by`. WHAT FILE IT CAME FROM IS `label_source`.

MEASURED on `data/rl/pairs.rater.jsonl`: 532 pairs carry `label_source="rule"` and
`decided_by="rater"`. They are rows the mechanical rule could have ordered and a rater
majority actually did -- so a run that excludes rater judgement by filtering `label_source`
excludes none of them and reports, under a config sha that says otherwise, that it trained on
consequence labels alone. The two fields answer different questions and the loader has to be
able to filter on the one that names the DECIDER.

The second thing this file pins is not a filter but an identity: the row set `preflight`
reported and the row set the trainer optimised must be the same rows. `cmd_train_rung2` called
`load_pairs` WITHOUT `include_label_sources` while `train()` called it WITH -- one `cfg.sha`,
two datasets, and the printed report describing neither reliably. The fix is structural: one
mapping from config to rows (`rows_for`), and the CLI hands the trainer the list it already
loaded rather than asking it to load again.
"""

from __future__ import annotations

import importlib
import inspect
import json

from pinq.actions import ask_action_json
from pinq_train.rung2_dpo import DPOConfig, load_pairs

SHA = "0" * 64


def _p(**over):
    r = {
        "suite_id": "musique",
        "task_id": "t1",
        "run_id": "p",
        "turn_idx": 1,
        "state_text": "S",
        "chosen_json": ask_action_json("who?"),
        "rejected_json": ask_action_json("when?"),
        "margin": 0.5,
        "pair_kind": "ask_ask",
    }
    r.update(over)
    return r


def _write(tmp_path, rows):
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_a_rater_decided_pair_is_excluded_by_its_decider_not_by_its_file(tmp_path):
    """THE 532 ROWS. `label_source="rule"` and `decided_by="rater"` at once is not a
    contradiction -- it is a pair the rule could order and a rater actually ordered."""
    p = _write(
        tmp_path,
        [
            _p(label_source="rule", decided_by="rule"),
            _p(label_source="rule", decided_by="rater"),
            _p(label_source="rater", decided_by="rater"),
        ],
    )
    rows, drops = load_pairs(p, exclude_decided_by=("rater",))
    assert [r["decided_by"] for r in rows] == ["rule"]
    assert drops["decided_by"] == 2


def test_filtering_the_label_source_does_not_exclude_the_rater_who_decided(tmp_path):
    """Why the new filter had to exist: the old one keeps every one of those 532 rows."""
    p = _write(tmp_path, [_p(label_source="rule", decided_by="rater")])
    rows, drops = load_pairs(p, include_label_sources=("rule",))
    assert len(rows) == 1 and drops == {}


def test_a_row_that_claims_no_decider_is_not_excluded_by_one(tmp_path):
    """A row cannot be excluded by a value it does not claim; the corpus predates the field."""
    p = _write(tmp_path, [_p(), _p(decided_by="rater")])
    rows, drops = load_pairs(p, exclude_decided_by=("rater",))
    assert len(rows) == 1 and drops["decided_by"] == 1


def test_the_decider_filter_is_in_the_config_sha():
    base = DPOConfig(base_model="m", adapter="a", merged_base_sha=SHA)
    other = DPOConfig(
        base_model="m", adapter="a", merged_base_sha=SHA, exclude_decided_by=("rater",)
    )
    assert base.sha != other.sha


def test_every_filter_the_loader_accepts_is_forwarded_from_the_config(monkeypatch):
    """The drift this catches is the one that already happened, generalised: a filter that is on
    `DPOConfig` (and therefore in `cfg.sha`) but never reaches `load_pairs` is a run whose
    identity claims a subset it trained without."""
    # NOT `import pinq_train.rung2_dpo.train as t`: the package rebinds that name to the
    # FUNCTION, so the attribute path hands back `train()` rather than the module the loader
    # global lives in.
    t = importlib.import_module("pinq_train.rung2_dpo.train")

    seen: dict = {}

    def recorder(path, **kw):
        seen.update(kw)
        return [], {}

    monkeypatch.setattr(t, "load_pairs", recorder)
    t.rows_for(DPOConfig(base_model="m", adapter="a", merged_base_sha=SHA))

    shared = set(inspect.signature(load_pairs).parameters) & set(DPOConfig.__dataclass_fields__)
    assert shared, "expected the config and the loader to share filter names"
    assert not (shared - set(seen)), f"rows_for does not forward: {sorted(shared - set(seen))}"


def test_the_cli_hands_the_trainer_the_rows_it_reported(tmp_path, monkeypatch, capsys):
    """ONE ROW SET. Not "the same filters applied twice" -- the same list."""
    import pinq_train.rung2_dpo as rung2
    from pi_run.cli import build_parser

    pairs = _write(
        tmp_path,
        [
            _p(label_source="rule"),
            _p(label_source="rater"),
            _p(label_source="rater"),
        ],
    )
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "merge.manifest.json").write_text(
        json.dumps({"base_model": "b", "adapter": "a", "adapter_sha": SHA, "output_sha": SHA})
        + "\n"
    )

    seen: dict = {}

    def fake_train(cfg, **kw):
        # The real trainer, faithfully: it uses the rows it was handed, and only loads them
        # itself when it was handed none.
        rows = kw.get("rows")
        if rows is None:
            rows, _ = rung2.load_pairs(
                cfg.pairs,
                len_delta_max=cfg.len_delta_max,
                include_pair_kinds=cfg.include_pair_kinds,
                exclude_suites=cfg.exclude_suites,
                min_q_distinctness=cfg.min_q_distinctness,
                include_label_sources=cfg.include_label_sources,
            )
        seen["n"] = len(rows)
        return {"out_dir": str(cfg.out_dir)}

    monkeypatch.setattr(rung2, "train", fake_train)
    args = build_parser().parse_args(
        [
            "train",
            "rung2",
            "--reference",
            "merged",
            "--merged-base",
            str(merged),
            "--adapter",
            "artifacts/rung1",
            "--pairs",
            str(pairs),
            "--out",
            str(tmp_path / "rung2"),
            "--include-label-source",
            "rule",
            "--train",
            "--acknowledge-untested",
        ]
    )
    assert args.fn(args) == 0
    report = json.loads(_first_json(capsys.readouterr().out))
    assert report["n_pairs"] == 1, "preflight must apply the filter the config carries"
    assert seen["n"] == report["n_pairs"], (
        f"preflight reported {report['n_pairs']} pairs and the trainer trained on {seen['n']}: "
        "one cfg.sha, two datasets."
    )


def _first_json(out: str) -> str:
    """The preflight report is the first JSON object the command prints."""
    start = out.index("{")
    depth, end = 0, start
    for i, ch in enumerate(out[start:], start):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            end = i + 1
            break
    return out[start:end]
