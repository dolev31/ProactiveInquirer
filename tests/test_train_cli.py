"""`pi train` -- the row builder, the split refusal, and the import contract it must not break.

The most important test here is the boring one: `pi_run.cmd_train` must contain no static
import of `pinq_train`. Contract 4 ("Nothing may import pinq_train") is checked by
import-linter over the module graph, and grimp sees imports inside functions too, so a
"lazy" `import pinq_train` would break the build. It is asserted HERE as well, in the file's
own text, because the person who adds that import will be reading this module and not the
linter's configuration.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pytest

from pi_run import cmd_train
from pi_run.cmd_train import (
    StateMismatch,
    SuiteCache,
    action_json,
    collect_rows,
    render_state,
    split_disagreement,
)

SOURCE = Path(cmd_train.__file__)


# --------------------------------------------------------------------------- the contract


def test_cmd_train_never_statically_imports_the_trainer():
    """`pi_run` reaches `pi_eval`; a static edge to `pinq_train` would put
    `pinq_train -> pi_run -> pi_eval` in the graph and the gold firewall would be one hop from
    broken. The trainer is loaded by NAME, the way `pi run` launches a worker."""
    tree = ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith("pinq_train"), (
                f"{SOURCE.name} imports {name!r}. Import-linter contract 4 forbids it, and "
                "grimp sees function-level imports too. Use _train('<module>') instead."
            )


def test_the_trainer_is_still_reachable_by_name():
    """The dynamic path has to actually work, or the contract is kept by a broken command."""
    # Literal updated when `userbench` became the third eval-only suite. The belief that was
    # wrong is "there are two"; what this test is actually for -- that the dynamically
    # imported module is the real one and not a stub -- is unchanged, which is why it still
    # compares against a written-out set rather than against `pinq.splitting`'s own object.
    # Updated again when `frames` became the fourth; the belief that was wrong is "there are
    # three", and what this test is for is unchanged. Updated again on 2026-09-15 when the
    # user made `drgym` the fifth; same story, same unchanged purpose. And on 2026-09-23 when
    # `musique_x2` (composed pairs of held-out MuSiQue TEST tasks) became the sixth.
    assert cmd_train._train("split").EVAL_ONLY_SUITES == frozenset(
        {"tau2", "pare", "userbench", "frames", "drgym", "musique_x2"}
    )
    assert cmd_train._train("rung1_sft").DISTINCT3_FLOOR == 0.55


# --------------------------------------------------------------------------- the row builder


def _synth(tmp_path: Path):
    from pi_eval.build.synth_build import build
    from pinq_adapters.synth.suite import SynthSuite

    corpus, *_ = build(n_tasks=4, n_facets=2, depth=2, root=tmp_path)
    suite = SynthSuite(Path(corpus).parent)
    tid = str(suite.task_ids()[0])
    return suite, tid, list(suite.retriever(tid)._units)


def _turns(units, *, subset_hashes=True):
    from pinq.types import Evidence

    rows = []
    held: list = []
    for i, u in enumerate(units):
        before = Evidence.of(tuple(held)).subset_hash
        held.append(u)
        rows.append(
            {
                "turn_idx": i,
                "action_kind": "ask",
                "question": f"Retrieve record {i}.",
                "rationale": f"because {i}",
                "response_text": f"answer {i}",
                "retrieved_uids": [u.uid],
                "new_uids": [u.uid],
                "subset_hash_before": before if subset_hashes else "",
            }
        )
    return rows


def test_the_rendered_state_is_the_prompt_the_policy_saw(tmp_path):
    """Reconstruction, not approximation: the evidence block is a function of the retrieved uid
    SET and the history is the recorded Q/A text.

    The draft slot renders "(no draft yet)" HERE because this fixture's turns carry no
    draft_sha. That is no longer true of a real run: `pinq.loop` drafts INSIDE the loop so the
    next act() can see D_t, which is the two-agent mechanism, and 149 of 335 recorded turns
    carry a draft_sha. This docstring used to assert "`pinq.loop` drafts only after it exits",
    which described the pre-fix loop. A turn that DID carry a draft is now refused rather than
    misrendered -- see the test below."""
    suite, tid, units = _synth(tmp_path)
    cache = SuiteCache(tmp_path)
    turns = _turns(units[:3])

    first = render_state(cache, "synth", tid, turns, 0)
    assert "(nothing retrieved yet)" in first
    assert "(nothing asked yet)" in first
    assert "(no draft yet)" in first

    third = render_state(cache, "synth", tid, turns, 2)
    assert "Q1: Retrieve record 0." in third and "A1: answer 0" in third
    assert units[0].text[:40] in third  # the evidence block really carries the document
    assert "{{" not in third  # every placeholder resolved


def test_a_reconstruction_that_drifted_is_refused_not_exported(tmp_path):
    """The subset hash the run recorded is the check. Without it, a silently wrong state_text
    trains the policy on inputs it will never receive and nothing downstream can tell."""
    suite, tid, units = _synth(tmp_path)
    cache = SuiteCache(tmp_path)
    turns = _turns(units[:2])
    turns[1]["subset_hash_before"] = "deadbeef" * 8
    with pytest.raises(StateMismatch, match="would not be the state"):
        render_state(cache, "synth", tid, turns, 1)
    # ...and the check can be turned off explicitly, which is the only way it ever is.
    assert render_state(cache, "synth", tid, turns, 1, verify=False)


def test_the_action_is_serialised_in_one_canonical_shape():
    ask = action_json({"action_kind": "ask", "question": "q?", "rationale": "r"})
    assert json.loads(ask) == {"action": "ASK", "question": "q?", "rationale": "r"}
    stop = action_json({"action_kind": "stop", "rationale": "done"})
    assert json.loads(stop)["action"] == "STOP"


# --------------------------------------------------------------------------- end to end


def _write_run(
    runs: Path, run_id: str, *, task: str, split: str, turns: list, arm="inquirer_prompted"
):
    d = runs / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "suite_id": "synth",
                "task_id": task,
                "arm_id": arm,
                "split": split,
                "template_id": None,
            }
        )
    )
    (d / "turns.jsonl").write_text("\n".join(json.dumps(t) for t in turns))
    (d / "ledger.jsonl").write_text(
        json.dumps({"currency": "retrieval_calls", "cumulative": float(len(turns))})
    )
    (d / "status.json").write_text(
        json.dumps({"stop_reason": "policy_stop", "wall_ms": 5, "usage": {"tok_total": 50}})
    )


def test_collect_rows_exports_train_and_refuses_everything_else(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
    suite, tid, units = _synth(tmp_path)
    turns = _turns(units[:3])
    runs = tmp_path / "runs"
    _write_run(runs, "keep", task=tid, split="train", turns=turns)
    _write_run(runs, "drop_test", task=tid, split="test", turns=turns)
    _write_run(runs, "drop_dev", task=tid, split="dev", turns=turns)

    weights = cmd_train._train("reward").RewardWeights()
    rows, skipped = collect_rows(runs, tmp_path, graph_version="v1", weights=weights)

    assert {r["run_id"] for r in rows} == {"keep"}
    # WAS `== 3`: one row per recorded turn. Belief corrected: the fixture's status.json says
    # `policy_stop`, and a STOP the policy chose is a decision point too -- it is now the
    # fourth row, rendered on the evidence after turn 2 (see rows_from_run). A budget stop
    # would still give three.
    assert len(rows) == 4
    assert [r["is_stop"] for r in rows] == [False, False, False, True]
    assert skipped["split_refused"] == 2
    assert all(r["scorer_hash"] and r["graph_version"] == "v1" for r in rows)
    assert all("EVIDENCE RETRIEVED SO FAR" in r["state_text"] for r in rows)


def test_a_gold_exposed_arm_is_refused_by_name(tmp_path, monkeypatch):
    """`gold_evidence` chose its questions where gold is readable. Exporting it is oracle
    distillation, so the arm name is refused in addition to the manifest's flag."""
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
    suite, tid, units = _synth(tmp_path)
    runs = tmp_path / "runs"
    _write_run(
        runs, "ceiling", task=tid, split="train", turns=_turns(units[:2]), arm="gold_evidence"
    )
    weights = cmd_train._train("reward").RewardWeights()
    rows, skipped = collect_rows(runs, tmp_path, graph_version="v1", weights=weights)
    assert rows == [] and skipped["state_mismatch"] == 1


def test_the_exporter_writes_an_sft_file_and_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
    suite, tid, units = _synth(tmp_path)
    runs = tmp_path / "runs"
    _write_run(runs, "keep", task=tid, split="train", turns=_turns(units[:3]))

    dataset = cmd_train._train("export.dataset")
    weights = cmd_train._train("reward").RewardWeights()
    rows, _ = collect_rows(runs, tmp_path, graph_version="v1", weights=weights)
    # Only rows the EXPORTER's own bucket function also calls `train` survive; see
    # split_disagreement's docstring for why the two functions can differ.
    keep = [r for r in rows if not split_disagreement([r])]
    items, man = dataset.export_sft(keep, margin_threshold=-1.0)
    out = dataset.write_jsonl(tmp_path / "rl" / "sft.jsonl", items, man)

    assert out.exists()
    written = [json.loads(x) for x in out.read_text().splitlines()]
    # WAS `== len(keep)`: one example per row. Belief corrected -- the exporter is no longer a
    # 1:1 row->example map. This run stopped with required coverage at 0.5, so its recorded
    # STOP is a decision on an UNFINISHED task: refused as a target and counted, rather than
    # taught (see export.dataset's stop rule). Three ASK rows clear the floor and are exported.
    assert len(written) == man.n_examples == 3
    assert man.n_no_target_dropped == 1 and man.n_stop_done_before_dedupe == 0
    assert [w["label_rule"] for w in written] == ["ask_clears_floor"] * 3
    assert all(w["state_text"] and w["action_json"] for w in written)
    sidecar = json.loads((out.parent / "sft.manifest.json").read_text())
    assert sidecar["train_id_set_hash"] and sidecar["n_examples"] == man.n_examples


def test_split_disagreement_is_reported_rather_than_hidden():
    """The two bucket functions in this repo do not agree. The intersection is exported, which
    is safe; leaving the count invisible would make a lossy filter look like a blocked attack."""
    from pinq_train.split import split_of

    rows = [{"suite_id": "musique", "task_id": f"t{i}", "template_id": None} for i in range(200)]
    counts = split_disagreement(rows)
    expected = sum(1 for r in rows if split_of("musique", r["task_id"]) != "train")
    assert counts.get("musique", 0) == expected > 0


# --------------------------------------------------------------------------- the other commands


def test_pi_train_status_runs_and_names_the_eval_only_suites(tmp_path, capsys):
    from pi_run.cli import build_parser

    args = build_parser().parse_args(["train", "status", "--root", str(tmp_path)])
    assert args.fn(args) == 0
    out = capsys.readouterr().out
    assert "rung0" in out and "rung3" in out
    assert "tau2" in out and "pare" in out
    assert "2026-09-12" in out  # rung 3's hard kill date, printed where it is read


def test_pi_train_rung0_prints_its_cost_before_it_can_spend_anything(tmp_path, capsys):
    from pi_run.cli import build_parser

    p = build_parser()
    dry = p.parse_args(["train", "rung0", "--dry-run", "--root", str(Path.cwd())])
    assert dry.fn(dry) == 0
    out = capsys.readouterr().out
    assert "PROJECTED COST" in out and "rollouts" in out
    assert "nothing was rolled out" in out

    live = p.parse_args(["train", "rung0", "--root", str(Path.cwd())])
    assert live.fn(live) == 2, "must refuse to spend without --yes"
    assert "PROJECTED COST" in capsys.readouterr().out


def test_pi_train_export_refuses_without_a_gold_root(tmp_path, capsys):
    from pi_run.cli import build_parser

    args = build_parser().parse_args(["train", "export", "--root", str(tmp_path)])
    assert args.fn(args) == 2
    assert "PI_GOLD_ROOT is unset" in capsys.readouterr().err


def test_pi_train_rung1_preflight_fails_loudly_on_a_missing_dataset(tmp_path, capsys):
    from pi_run.cli import build_parser

    args = build_parser().parse_args(
        ["train", "rung1", "--base-model", "m", "--dataset", str(tmp_path / "nope.jsonl")]
    )
    assert args.fn(args) == 1
    assert "PREFLIGHT FAILED" in capsys.readouterr().err


def test_a_state_whose_turn_had_a_draft_is_refused_not_misrendered(tmp_path):
    """`render_state` hardcoded draft="(no draft yet)" for every turn. Since the loop now
    drafts inside the loop, that renders a `state_text` which is NOT the prompt the policy saw
    on 44% of recorded turns (149 of 335 carry a draft_sha, 26 of them at turn 0).

    The evidence guard cannot catch it: it re-derives `subset_hash_before` and says nothing
    about the draft. Silently training on a prompt the policy never saw is worse than not
    training, so this refuses -- the draft TEXT is not persisted, only its sha."""
    suite, tid, units = _synth(tmp_path)
    cache = SuiteCache(tmp_path)
    turns = _turns(units[:2])
    turns[0]["draft_sha"] = "abc123def456"

    # RENDERED AT 1, NOT 0. `turns[i]` carries D_{i+1} -- the draft is computed after turn
    # i's retrieval so the NEXT act() sees it -- so the draft turn 1 needs is the one
    # recorded on turn 0. This test asserted `render_state(..., 0)`, which is what the
    # off-by-one made it look like; turn 0 has no predecessor and correctly needs no draft.
    with pytest.raises(StateMismatch, match="carries no draft_text"):
        render_state(cache, "synth", tid, turns, 1)


def test_a_turn_that_recorded_its_draft_text_is_reconstructed(tmp_path):
    """`Turn.draft_text` now persists D_t, so a turn from a current sweep renders exactly the
    prompt `act()` was shown. The refusal above is reserved for turns recorded BEFORE the field
    existed, which carry a sha and no text -- 655 of 841 turns on disk at the time."""
    suite, tid, units = _synth(tmp_path)
    cache = SuiteCache(tmp_path)
    turns = _turns(units[:2])
    turns[0]["draft_sha"] = "abc123def456"
    turns[0]["draft_text"] = "The capital is Paris, pending confirmation."

    # At 1: D_1 is stamped on turns[0] and is what act() sees when deciding turn 1.
    out = render_state(cache, "synth", tid, turns, 1)
    assert "The capital is Paris, pending confirmation." in out
    assert "(no draft yet)" not in out


def test_a_draft_free_turn_still_renders(tmp_path):
    """The refusal is about the draft, not about every turn."""
    suite, tid, units = _synth(tmp_path)
    cache = SuiteCache(tmp_path)
    turns = _turns(units[:2])
    assert not turns[0].get("draft_sha")
    assert "(no draft yet)" in render_state(cache, "synth", tid, turns, 0)


def test_turn_zero_renders_no_draft_because_the_policy_saw_none(tmp_path):
    """OFF BY ONE, and it leaks the future into the training prompt.

    `pinq.loop` computes the draft AFTER turn i's retrieval and stamps it on Turn i:
    "the draft is refreshed INSIDE the loop so the NEXT act() sees it". So `turns[i]`
    carries D_{i+1} -- the belief the policy holds when deciding turn i+1, not turn i.

    `render_state(..., upto)` promises "the prompt as it stood BEFORE turns[upto] was
    decided" and read `turns[upto].draft_text`, which is the draft produced by the very
    retrieval the predicted action caused. At turn 0 this is provable rather than arguable:
    `run_loop` calls `act()` with `State(view=view)` and no draft at all, yet 138 of 174
    recorded turn-0 rows carry a draft_sha.

    The evidence guard cannot catch it -- it verifies `subset_hash_before` while the draft
    it renders was computed from `subset_hash_after`, so the rendered state is internally
    contradictory: two halves of one prompt taken from opposite sides of the decision.
    """
    suite, tid, units = _synth(tmp_path)
    cache = SuiteCache(tmp_path)
    turns = _turns(units[:2])
    turns[0]["draft_sha"] = "abc123def456"
    turns[0]["draft_text"] = "D_1: the draft that only exists AFTER turn 0 retrieved"

    text = render_state(cache, "synth", tid, turns, 0)
    assert "only exists AFTER" not in text, "turn 0's prompt must not contain D_1"
    assert "(no draft yet)" in text


def test_turn_one_renders_the_draft_stamped_on_turn_zero(tmp_path):
    """D_1 is what act() sees at turn 1, and it is recorded on turns[0]."""
    suite, tid, units = _synth(tmp_path)
    cache = SuiteCache(tmp_path)
    turns = _turns(units[:2])
    turns[0]["draft_sha"] = "abc123def456"
    turns[0]["draft_text"] = "D_1: the belief the policy holds at turn 1"
    turns[1]["draft_sha"] = "def456abc123"
    turns[1]["draft_text"] = "D_2: not visible at turn 1"

    text = render_state(cache, "synth", tid, turns, 1)
    assert "D_1: the belief" in text
    assert "D_2" not in text, "turn 1 must not see the draft its own action produced"


# --------------------------------------------------------------------------- track D status


def test_pi_train_status_reports_the_dev_artifacts_and_the_registry(tmp_path, capsys):
    """`pi train status` is where someone starting a session asks "what is on disk". Adding a
    dev gate and a checkpoint registry without adding them here leaves two artifacts whose
    absence is invisible until a command fails on them."""
    from pi_run.cli import build_parser

    args = build_parser().parse_args(["train", "status", "--root", str(tmp_path)])
    assert args.fn(args) == 0
    out = capsys.readouterr().out

    # the two lines the older tests pin are still here
    assert "2026-09-12" in out and "rung0" in out and "rung3" in out
    assert "tau2" in out and "pare" in out

    assert "sft.dev.jsonl" in out and "pairs.dev.jsonl" in out
    assert "checkpoint registry" in out
    assert "0.55" in out, "the distinct-3 floor must be printed where it is read"
    assert "0.4685" in out, (
        "the floor is UNREACHABLE on musique and the status line must say so with the measured "
        "number, or a reader takes 0.55 for an attainable target"
    )
    assert "0.65" in out, "the gate's within-task floor, which is the reachable one"


def test_pi_train_status_names_the_registered_checkpoints(tmp_path, monkeypatch, capsys):
    """An empty registry and a registry nobody loaded must not look alike."""
    import json

    from pi_run.cli import build_parser
    from pinq_adapters.llm import checkpoints

    reg = tmp_path / "ckpt.json"
    reg.write_text(
        json.dumps(
            {
                "qwen3-8b-sft": {
                    "adapter_sha": "a" * 64,
                    "training_manifest_sha": "b" * 64,
                    "base_model": "Qwen/Qwen3-8B",
                    "rung": 1,
                    "dataset_sha": "c" * 64,
                    # The same value under the name the stamp and the contamination check use.
                    # Required since 2026-09-15; `registry()` refuses a row without it, and
                    # refuses one where the two names disagree.
                    "train_id_set_hash": "c" * 64,
                }
            }
        )
    )
    monkeypatch.setenv("PI_CHECKPOINTS", str(reg))
    checkpoints.reset_cache()
    try:
        args = build_parser().parse_args(["train", "status", "--root", str(tmp_path)])
        assert args.fn(args) == 0
        out = capsys.readouterr().out
        assert "qwen3-8b-sft" in out and "Qwen/Qwen3-8B" in out and "rung 1" in out
    finally:
        checkpoints.reset_cache()


def test_the_graph_cache_is_keyed_on_the_gold_root(tmp_path, monkeypatch):
    """A MEMO THAT IGNORES WHERE IT READ FROM SERVES ANOTHER ROOT'S GRAPHS.

    `_GRAPHS` memoises `load_graphs` on (suite, graph_version) because an uncached read is one
    whole-suite file parse per RUN -- 720,000 graph constructions for one musique export. The key
    left out the only other input: `PI_GOLD_ROOT`. Inside one export that is invisible, because a
    process has one gold root; across two it serves the first root's graphs for the second's runs.

    AND THE FAILURE IS A SILENT ZERO, which is why this is worth a test rather than a comment.
    `rows_from_run` turns a missing graph into `NoGoldForTask`, and `collect_rows` counts that as
    a skip and carries on -- so the export finishes, exits 0, writes a manifest, and holds no
    rows. Caught here because a dev export built its corpus after another test had already warmed
    the cache: every dev run was refused `no_gold` for a task that exists.
    """
    from pi_eval.build.synth_build import build
    from pi_run.cmd_train import _graphs

    first = tmp_path / "a"
    monkeypatch.setenv("PI_GOLD_ROOT", str(first / "data" / "gold"))
    build(n_tasks=4, n_facets=2, depth=2, root=first)
    assert set(_graphs("synth", "v1")) == {"s0", "s1", "s2", "s3"}

    second = tmp_path / "b"
    monkeypatch.setenv("PI_GOLD_ROOT", str(second / "data" / "gold"))
    build(n_tasks=6, n_facets=2, depth=2, root=second)
    assert "s5" in _graphs("synth", "v1"), (
        "the cache served the previous gold root's graphs; every run on a task the old root "
        "never had would be skipped `no_gold` and the export would report a clean, empty success"
    )


# --- track F: cli ---
# EVERY HYPERPARAMETER THE GRID NAMES HAS TO REACH `cfg.sha`.
#
# A flag that argparse accepts and nothing forwards does not raise: the run trains at the
# dataclass default under a config sha that names it, which is the same failure as a filter that
# never reaches the loader. `pi train rung1 --max-seq-len 5120` was the only shape knob either
# rung exposed, so the 8B grid could not be expressed on the command line at all -- it had to be
# retyped into a python REPL, where it is provenance nothing records.
#
# `--config` exists for the same reason `--merged-base` reads the sha from the manifest instead
# of taking it on the command line: numbers an operator retypes are numbers that drift.


def _rung1_cfg(monkeypatch, tmp_path, *flags):
    """`pi train rung1` up to preflight, with the config it built captured.

    Preflight is the double rather than `train`: it is handed the same `cfg` object, and it is
    what stands between an invalid config and the GPU. The dataset only has to be readable.
    """
    import pinq_train.rung1_sft as rung1
    from pi_run.cli import build_parser

    seen: dict = {}
    ds = tmp_path / "sft.jsonl"
    ds.write_text("")
    monkeypatch.setattr(rung1, "preflight", lambda cfg, rows: seen.setdefault("cfg", cfg) and {})
    args = build_parser().parse_args(
        ["train", "rung1", "--base-model", "Qwen/Qwen3-8B", "--dataset", str(ds), *flags]
    )
    assert args.fn(args) == 0
    return seen["cfg"]


def _rung2_pair(**over):
    """One same-state ask_ask pair. `over` is for the flags whose effect is a FILTER: a floor
    the fixture cannot meet refuses the run, so the row has to carry what the flag selects on."""
    from pinq.actions import ask_action_json

    return {
        "suite_id": "musique",
        "task_id": "t1",
        "state_text": "S",
        "chosen_json": ask_action_json("who?"),
        "rejected_json": ask_action_json("when?"),
        "margin": 0.5,
        "pair_kind": "ask_ask",
    } | over


def _rung2_run(monkeypatch, tmp_path, *flags, rows=None):
    """`pi train rung2` up to the trainer, returning `(exit_code, cfg_or_None)`.

    Preflight runs for real -- it is what calls `cfg.validate()` -- so a configuration the
    dataclass refuses never reaches the double, and the exit code is the whole result.
    """
    import pinq_train.rung2_dpo as rung2
    from pi_run.cli import build_parser

    seen: dict = {}
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(r) + "\n" for r in (rows or [_rung2_pair()])))
    monkeypatch.setattr(rung2, "train", lambda cfg, **kw: seen.setdefault("cfg", cfg) and {})
    args = build_parser().parse_args(
        [
            "train",
            "rung2",
            "--base-model",
            "Qwen/Qwen3-8B",
            "--adapter",
            "artifacts/rung1",
            "--pairs",
            str(pairs),
            "--train",
            "--acknowledge-untested",
            *flags,
        ]
    )
    return args.fn(args), seen.get("cfg")


def _rung2_cfg(monkeypatch, tmp_path, *flags, rows=None):
    rc, cfg = _rung2_run(monkeypatch, tmp_path, *flags, rows=rows)
    assert rc == 0
    return cfg


def test_every_rung1_flag_reaches_the_config(monkeypatch, tmp_path, capsys):
    cfg = _rung1_cfg(
        monkeypatch,
        tmp_path,
        "--learning-rate",
        "2e-4",
        "--per-device-batch",
        "2",
        "--grad-accum",
        "8",
        "--lora-r",
        "16",
        "--max-seq-len",
        "5120",
        "--epochs",
        "3",
        "--no-bf16",
        "--no-chat-template",
    )
    capsys.readouterr()
    assert cfg.learning_rate == 2e-4
    assert cfg.per_device_batch == 2 and cfg.grad_accum == 8
    assert cfg.max_seq_len == 5120 and cfg.epochs == 3
    assert cfg.bf16 is False and cfg.chat_template is False
    assert (cfg.lora.r, cfg.lora.alpha) == (16, 32), "alpha is 2r, never a second free knob"


def test_the_stop_weight_reaches_the_rung1_config(monkeypatch, tmp_path, capsys):
    """A flag argparse accepts and nothing forwards trains at the dataclass default under a
    config sha that names the value nobody applied. `--stop-weight` is the whole ablation, so
    the failure would be two arms that are one run reported twice."""
    cfg = _rung1_cfg(monkeypatch, tmp_path, "--stop-weight", "0.5")
    capsys.readouterr()
    assert cfg.stop_weight == 0.5
    quarter = _rung1_cfg(monkeypatch, tmp_path, "--stop-weight", "0.25")
    capsys.readouterr()
    assert quarter.stop_weight == 0.25
    assert quarter.sha != cfg.sha


def test_the_stop_weight_is_absent_by_default_and_a_grid_file_may_set_it(
    monkeypatch, tmp_path, capsys
):
    """ABSENT, not 1.0: an optional knob at its absent default is not part of the identity, so
    every rung-1 run already trained keeps its `config_sha`."""
    base = _rung1_cfg(monkeypatch, tmp_path)
    capsys.readouterr()
    assert base.stop_weight is None

    conf = tmp_path / "sw.json"
    conf.write_text(json.dumps({"stop_weight": 0.25}) + "\n")
    cfg = _rung1_cfg(monkeypatch, tmp_path, "--config", str(conf))
    capsys.readouterr()
    assert cfg.stop_weight == 0.25
    # FLAG > --config file, like every other knob in the track-F block.
    flagged = _rung1_cfg(monkeypatch, tmp_path, "--config", str(conf), "--stop-weight", "0.5")
    capsys.readouterr()
    assert flagged.stop_weight == 0.5


# --- curriculum ---
def test_the_curriculum_reaches_the_rung1_config_and_names_its_own_arm(
    monkeypatch, tmp_path, capsys
):
    """The ordering IS the intervention, so a flag that argparse accepts and nothing forwards
    would train the null arm under a `config_sha` that names an ablation -- two arms that are
    one run reported twice, which is the failure this block of tests exists for."""
    easy = _rung1_cfg(monkeypatch, tmp_path, "--curriculum", "depth_easy_first")
    capsys.readouterr()
    assert easy.curriculum == "depth_easy_first"
    hard = _rung1_cfg(monkeypatch, tmp_path, "--curriculum", "depth_hard_first")
    capsys.readouterr()
    assert hard.curriculum == "depth_hard_first"

    base = _rung1_cfg(monkeypatch, tmp_path)
    capsys.readouterr()
    assert base.curriculum == "shuffled", "the default is the null arm, not a curriculum"
    assert easy.sha != base.sha and hard.sha != base.sha and easy.sha != hard.sha


def test_a_grid_file_may_set_the_curriculum_and_the_flag_still_wins(monkeypatch, tmp_path, capsys):
    """`--config` is how a sweep is written down; a key it has no flag for is REFUSED, so the
    knob has to be in `_rung1_defaults()` or the grid file for this arm cannot be written at all."""
    conf = tmp_path / "curriculum.json"
    conf.write_text(json.dumps({"curriculum": "depth_hard_first"}) + "\n")
    cfg = _rung1_cfg(monkeypatch, tmp_path, "--config", str(conf))
    capsys.readouterr()
    assert cfg.curriculum == "depth_hard_first"
    # FLAG > --config file, like every other knob in the track-F block.
    flagged = _rung1_cfg(
        monkeypatch, tmp_path, "--config", str(conf), "--curriculum", "depth_easy_first"
    )
    capsys.readouterr()
    assert flagged.curriculum == "depth_easy_first"


def test_the_curriculum_choices_are_the_ones_the_trainer_knows():
    """The parser cannot import `pinq_train` (contract 4), so the three names are spelled out
    in `cmd_train` a second time. This is the join: a name added to one side and not the other
    is either a flag that refuses a real arm or a flag that accepts one `validate()` will
    reject after the operator has waited for the preflight."""
    from pi_run.cli import build_parser

    action = next(
        a
        for a in _rung1_parser(build_parser())._actions
        if "--curriculum" in getattr(a, "option_strings", [])
    )
    assert tuple(action.choices) == tuple(cmd_train._train("rung1_sft").CURRICULA)
    assert action.default is None, "a default here would shadow --config, like every other knob"
    assert action.help and "cfg.sha" in action.help


def _subparser(parser, name: str):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices[name]
    raise AssertionError(f"{parser.prog} has no subcommands, so no {name!r}")


def _rung1_parser(parser):
    """The `pi train rung1` subparser, dug out of the top-level one."""
    return _subparser(_subparser(parser, "train"), "rung1")


def test_the_rung1_help_renders_with_the_curriculum_flag(capsys):
    """A help string that raises is a command nobody can discover; argparse formats lazily, so
    nothing else here would catch a bad `%` or an empty choices tuple."""
    from pi_run.cli import build_parser

    text = _rung1_parser(build_parser()).format_help()
    assert "--curriculum" in text
    for name in cmd_train._train("rung1_sft").CURRICULA:
        assert name in text


# --- end curriculum ---


def test_the_rung1_defaults_are_unchanged_when_no_flag_is_given(monkeypatch, tmp_path, capsys):
    """The flags are new; the run they produce without them must not be."""
    cfg = _rung1_cfg(monkeypatch, tmp_path)
    capsys.readouterr()
    assert (cfg.learning_rate, cfg.per_device_batch, cfg.grad_accum) == (1e-4, 1, 16)
    assert (cfg.lora.r, cfg.lora.alpha) == (32, 64)
    assert cfg.bf16 is True and cfg.chat_template is True
    assert cfg.max_seq_len == 12_288 and cfg.epochs == 2


def test_every_rung2_flag_reaches_the_config(monkeypatch, tmp_path, capsys):
    cfg = _rung2_cfg(
        monkeypatch,
        tmp_path,
        "--max-seq-len",
        "5120",
        "--max-prompt-len",
        "4608",
        "--include-pair-kind",
        "ask_ask",
        "--learning-rate",
        "5e-6",
        "--per-device-batch",
        "2",
        "--grad-accum",
        "8",
        "--beta",
        "0.2",
        "--no-bf16",
    )
    capsys.readouterr()
    assert cfg.max_seq_len == 5120 and cfg.max_prompt_len == 4608
    assert cfg.include_pair_kinds == ("ask_ask",)
    assert cfg.learning_rate == 5e-6
    assert cfg.per_device_batch == 2 and cfg.grad_accum == 8
    assert cfg.beta == 0.2 and cfg.bf16 is False


def test_the_rung2_defaults_are_unchanged_when_no_flag_is_given(monkeypatch, tmp_path, capsys):
    cfg = _rung2_cfg(monkeypatch, tmp_path)
    capsys.readouterr()
    assert cfg.max_seq_len == 12_288 and cfg.max_prompt_len == 11_264
    assert cfg.include_pair_kinds == ("ask_ask", "ask_stop")
    assert (cfg.learning_rate, cfg.per_device_batch, cfg.grad_accum) == (5e-6, 1, 16)
    assert cfg.beta == 0.1 and cfg.bf16 is True
    # The objective knobs are new; the run built without them must not be.
    assert cfg.loss_type == ("sigmoid",)
    assert cfg.loss_weights is None and cfg.ld_alpha is None
    assert cfg.min_q_distinctness == "" and cfg.min_stop_chosen_share == 0.0


# ---- authority-following: `_rung1_defaults`/`_rung2_defaults` must READ `SFTConfig`,
# `LoraSpec` and `DPOConfig`'s own field defaults, not retype them as a second, hand-maintained
# literal that a change to the dataclass could silently drift away from.


def test_the_rung1_defaults_follow_a_changed_sftconfig_or_loraspec_field(
    monkeypatch, tmp_path, capsys
):
    """`_rung1_defaults` used to retype `SFTConfig`'s and `LoraSpec`'s own field defaults as a
    literal dict; a change to either dataclass's default could silently diverge from the copy
    here and nothing would say so. Move `SFTConfig.learning_rate`, `SFTConfig.save_steps` and
    `LoraSpec.r` (whose `alpha` is always `2*r`, never a second free knob -- see
    `test_the_rung1_defaults_are_unchanged_when_no_flag_is_given`) and confirm the no-flag
    config follows all three: this is a live read of `dataclasses.fields(...)`, not a frozen
    copy taken once and forgotten."""
    import pinq_train.rung1_sft as rung1

    monkeypatch.setattr(rung1.SFTConfig.__dataclass_fields__["learning_rate"], "default", 4.5e-4)
    monkeypatch.setattr(rung1.SFTConfig.__dataclass_fields__["save_steps"], "default", 77)
    monkeypatch.setattr(rung1.LoraSpec.__dataclass_fields__["r"], "default", 17)

    cfg = _rung1_cfg(monkeypatch, tmp_path)
    capsys.readouterr()
    assert cfg.learning_rate == 4.5e-4
    assert cfg.save_steps == 77
    assert (cfg.lora.r, cfg.lora.alpha) == (17, 34)


def test_the_rung2_defaults_follow_a_changed_dpoconfig_field(monkeypatch, tmp_path, capsys):
    """Same proof for `_rung2_defaults` and `DPOConfig`: move two of its field defaults --
    `learning_rate` (a plain hyperparameter) and `save_steps` (validated only as `>= 1`, so an
    arbitrary sentinel cannot trip `cfg.validate()` the way an enum-constrained field like
    `min_q_distinctness` would) -- and confirm the no-flag config follows both."""
    import pinq_train.rung2_dpo as rung2

    monkeypatch.setattr(rung2.DPOConfig.__dataclass_fields__["learning_rate"], "default", 7.5e-6)
    monkeypatch.setattr(rung2.DPOConfig.__dataclass_fields__["save_steps"], "default", 55)

    cfg = _rung2_cfg(monkeypatch, tmp_path)
    capsys.readouterr()
    assert cfg.learning_rate == 7.5e-6
    assert cfg.save_steps == 55


# ---- config-hash invariance: moving `_rung1_defaults`/`_rung2_defaults` from literal dict
# defaults to `dataclasses.fields(...)` reads must not move `cfg.sha` for any run of record.
#
# THE STAKES. `cfg.sha` is the run's identity: a moved hash is not a moved number, it is every
# past AND future run trained under one of these five real grids silently renamed. These seven
# values were measured directly against the CURRENT (migrated) code -- through the exact
# `_resolve_options` + `SFTConfig`/`DPOConfig` construction path production training uses -- for
# every real 8B grid file this repository ships plus the no-`--config` case, at the time the
# migration from literal dict defaults to dataclass-field reads landed. A future, DELIBERATE
# change to one of these configs' own hyperparameter defaults is expected to move its hash --
# that is what `cfg.sha` is for -- and the fix then is to re-measure and update the pin with a
# comment saying why, not to weaken this test. An UNEXPLAINED failure here means the
# options-resolution or config-construction pipeline itself changed silently, which is exactly
# the defect class this migration was hardening against.
def _real_grid_sha(rung: str, config_path: str | None) -> str:
    import pinq_train.rung1_sft as rung1
    import pinq_train.rung2_dpo as rung2
    from pi_run import cmd_train as ct

    def replay_opt(defaults: dict, path: str | None) -> dict:
        ns = argparse.Namespace(**{k: None for k in defaults}, config=path)
        return ct._resolve_options(ns, defaults)

    if rung == "rung1":
        opt = replay_opt(ct._rung1_defaults(rung1), config_path)
        return rung1.SFTConfig(
            base_model="Qwen/Qwen3-8B",
            dataset="data/rl/headline/sft.jsonl",
            out_dir="artifacts/rung1",
            tau=0.05,
            sigma_j=0.0,
            epochs=opt["epochs"],
            max_seq_len=opt["max_seq_len"],
            learning_rate=opt["learning_rate"],
            per_device_batch=opt["per_device_batch"],
            grad_accum=opt["grad_accum"],
            seed=opt["seed"],
            lora=rung1.LoraSpec(r=opt["lora_r"], alpha=2 * opt["lora_r"]),
            bf16=opt["bf16"],
            chat_template=opt["chat_template"],
            reasoning_effort=opt["reasoning_effort"],
            stop_weight=opt["stop_weight"],
            curriculum=opt["curriculum"],
            resume=opt["resume"],
            save_steps=opt["save_steps"],
        ).sha

    opt = replay_opt(ct._rung2_defaults(rung2), config_path)

    def _loss_weights(value):
        if value is None or value == "":
            return None
        parts = [p.strip() for p in value.split(",")] if isinstance(value, str) else list(value)
        return tuple(float(p) for p in parts if p != "")

    return rung2.DPOConfig(
        base_model="Qwen/Qwen3-8B",
        adapter="",
        pairs="data/rl/headline/pairs.jsonl",
        out_dir="artifacts/rung2",
        beta=opt["beta"],
        epochs=opt["epochs"],
        max_seq_len=opt["max_seq_len"],
        max_prompt_len=opt["max_prompt_len"],
        learning_rate=opt["learning_rate"],
        per_device_batch=opt["per_device_batch"],
        grad_accum=opt["grad_accum"],
        seed=opt["seed"],
        bf16=opt["bf16"],
        include_pair_kinds=(
            tuple(opt["include_pair_kinds"])
            if opt["include_pair_kinds"]
            else rung2.SAMPLED_PAIR_KINDS
        ),
        reference=opt["reference"],
        merge_adapter=False,
        merged_base_sha="",
        label_smoothing=0.0,
        include_label_sources=rung2.LABEL_SOURCES,
        exclude_decided_by=(),
        include_code_versions=(),
        loss_type=tuple(opt["loss_type"]) if opt["loss_type"] else rung2.DEFAULT_LOSS_TYPE,
        loss_weights=_loss_weights(opt["loss_weights"]),
        ld_alpha=opt["ld_alpha"],
        min_q_distinctness=opt["min_q_distinctness"],
        min_stop_chosen_share=opt["min_stop_chosen_share"],
        resume=opt["resume"],
        save_steps=opt["save_steps"],
    ).sha


@pytest.mark.parametrize(
    "rung,config_path,pinned_sha",
    [
        ("rung1", None, "e347fbc24df04dd0d0f66562303d919745997a7e2989438c2ce9cd226e3ce3bd"),
        (
            "rung1",
            "conf/train/rung1_8b.json",
            "df9aa1462d1ca392d24f4fba42a31e0bc2a76846b246d2f65efe3c48aaabd8e8",
        ),
        ("rung2", None, "579371d6ecefb10be0ab89236123b7856623bfb43b56166dcd9be455aaa0f78d"),
        (
            "rung2",
            "conf/train/rung2_8b.json",
            "375e53ae2cf89cd611a9b0c2ca0237dc1c121609cd08f1d401e028c45d2e04b0",
        ),
        (
            "rung2",
            "conf/train/rung2_8b_from_base.json",
            "989d312c281a4c61adaff133f551c8c341b4b945b726a46dab12a1b0ed7bc829",
        ),
        (
            "rung2",
            "conf/train/rung2_8b_notdone_both.json",
            "4e3a8de9dc4964271839ab02be7844e194ae570dccf279ab15f3896eb809ed55",
        ),
        (
            "rung2",
            "conf/train/rung2_8b_notdone_only.json",
            "dac8944ed380c5fbf950b189f0e98875a17084d9dbc42b4053a3450ad084df39",
        ),
    ],
    ids=[
        "rung1-none",
        "rung1-8b",
        "rung2-none",
        "rung2-8b",
        "rung2-8b-from-base",
        "rung2-8b-notdone-both",
        "rung2-8b-notdone-only",
    ],
)
def test_the_real_grids_config_sha_is_pinned_to_what_the_migration_measured(
    rung, config_path, pinned_sha
):
    assert _real_grid_sha(rung, config_path) == pinned_sha


# ---- the objective, and the two filters that had a config field and no flag
#
# `min_q_distinctness` and `min_stop_chosen_share` have been `DPOConfig` fields -- and therefore
# in `cfg.sha` -- since they were written, and NOTHING could set them from a command line. A
# field that only a python REPL can reach is provenance nobody records, which is the failure the
# track-F block exists to close; the objective fields are new and arrive with flags for the same
# reason.


def test_every_rung2_objective_flag_reaches_the_config(monkeypatch, tmp_path, capsys):
    cfg = _rung2_cfg(
        monkeypatch,
        tmp_path,
        "--loss-type",
        "sigmoid",
        "--loss-type",
        "sft",
        "--loss-weights",
        "1.0,0.5",
        "--ld-alpha",
        "0.9",
    )
    capsys.readouterr()
    assert cfg.loss_type == ("sigmoid", "sft")
    assert cfg.loss_weights == (1.0, 0.5)
    assert cfg.ld_alpha == 0.9
    assert cfg.sha != _rung2_cfg(monkeypatch, tmp_path).sha


def test_the_two_pair_filters_reach_the_config(monkeypatch, tmp_path, capsys):
    """Both are floors, so the fixture has to satisfy them: a row with no `q_distinctness` at
    all is refused by `load_pairs` before the filter runs, and a floor on the STOP share over a
    file of ASK-wins is `TooFewStopPairs`. Either refusal would be an exit code, not a config."""
    from pinq.actions import STOP_ACTION_JSON

    cfg = _rung2_cfg(
        monkeypatch,
        tmp_path,
        "--min-q-distinctness",
        "related",
        rows=[_rung2_pair(q_distinctness="different")],
    )
    capsys.readouterr()
    assert cfg.min_q_distinctness == "related"

    stop = _rung2_pair(chosen_json=STOP_ACTION_JSON, pair_kind="ask_stop")
    floored = _rung2_cfg(monkeypatch, tmp_path, "--min-stop-chosen-share", "0.5", rows=[stop])
    capsys.readouterr()
    assert floored.min_stop_chosen_share == 0.5


@pytest.mark.parametrize(
    "flags",
    [
        ("--loss-type", "ipo"),
        ("--loss-type", "sigmoid", "--loss-type", "sft"),
        ("--ld-alpha", "0.5"),
        ("--min-q-distinctness", "different"),
    ],
    ids=["ipo", "mixture", "ld_alpha", "q_distinctness"],
)
def test_each_new_rung2_flag_moves_the_config_sha(monkeypatch, tmp_path, capsys, flags):
    """A flag argparse accepts and nothing forwards trains at the dataclass default under a
    config sha that names the value nobody applied -- two arms that are one run reported twice."""
    rows = [_rung2_pair(q_distinctness="different")]
    base = _rung2_cfg(monkeypatch, tmp_path, rows=rows)
    capsys.readouterr()
    cfg = _rung2_cfg(monkeypatch, tmp_path, *flags, rows=rows)
    capsys.readouterr()
    assert cfg.sha != base.sha


def test_a_weight_per_loss_is_required_and_the_mismatch_is_refused(monkeypatch, tmp_path, capsys):
    """Two names and one weight is a mixture whose second term is unweighted. TRL raises the
    same refusal in its own `__post_init__`; catching it here means the operator sees it before
    the trainer imports torch, and `pi train rung2` exits non-zero rather than 0."""
    rc, cfg = _rung2_run(
        monkeypatch,
        tmp_path,
        "--loss-type",
        "sigmoid",
        "--loss-type",
        "sft",
        "--loss-weights",
        "1.0",
    )
    err = capsys.readouterr().err
    assert rc == 1 and cfg is None
    assert "loss_weights" in err


def test_a_loss_name_the_installed_trl_cannot_run_is_refused(monkeypatch, tmp_path, capsys):
    """`kto_pair` was the OTHER accepted value of the old string field, and trl 1.13.0 does not
    dispatch it. The refusal has to be here, not in the loss: by then an 8B base and two
    adapters are resident on a rented card."""
    rc, cfg = _rung2_run(monkeypatch, tmp_path, "--loss-type", "kto_pair")
    err = capsys.readouterr().err
    assert rc == 1 and cfg is None
    assert "kto_pair" in err


def test_the_objective_can_be_set_from_a_grid_file_and_a_flag_still_wins(
    monkeypatch, tmp_path, capsys
):
    """`--config` has to accept every key a flag accepts, or a sweep is a thing you retype."""
    conf = tmp_path / "obj.json"
    conf.write_text(
        json.dumps({"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, 0.5], "ld_alpha": 0.8})
        + "\n"
    )
    cfg = _rung2_cfg(monkeypatch, tmp_path, "--config", str(conf))
    capsys.readouterr()
    assert cfg.loss_type == ("sigmoid", "sft")
    assert cfg.loss_weights == (1.0, 0.5) and cfg.ld_alpha == 0.8

    flagged = _rung2_cfg(monkeypatch, tmp_path, "--config", str(conf), "--ld-alpha", "0.2")
    capsys.readouterr()
    assert flagged.ld_alpha == 0.2


# ---- the grids of record


def test_the_rung1_grid_of_record_is_what_the_cli_builds(monkeypatch, tmp_path, capsys):
    cfg = _rung1_cfg(monkeypatch, tmp_path, "--config", "conf/train/rung1_8b.json")
    capsys.readouterr()
    assert cfg.max_seq_len == 5120
    assert cfg.bf16 is True
    assert (cfg.per_device_batch, cfg.grad_accum) == (1, 16)
    assert cfg.learning_rate == 1e-4 and cfg.epochs == 2


def test_the_rung2_grid_of_record_is_what_the_cli_builds(monkeypatch, tmp_path, capsys):
    cfg = _rung2_cfg(monkeypatch, tmp_path, "--config", "conf/train/rung2_8b.json")
    capsys.readouterr()
    assert (cfg.max_seq_len, cfg.max_prompt_len) == (5120, 4608)
    assert cfg.bf16 is True
    assert (cfg.per_device_batch, cfg.grad_accum) == (1, 16)
    assert cfg.learning_rate == 5e-6 and cfg.epochs == 1
    assert cfg.beta == 0.1
    assert cfg.include_pair_kinds == ("ask_ask",)
    assert cfg.reference == "adapter"
    # The grid names the objective explicitly. It names the DEFAULT, so the identity of the
    # 8B run of record is unchanged -- see `test_the_default_objective_hashes_exactly_what_the
    # _old_string_hashed` in tests/test_rung2_objective.py, which pins that against a sha
    # measured before this field became a tuple.
    assert cfg.loss_type == ("sigmoid",)
    assert cfg.loss_weights is None and cfg.ld_alpha is None


def test_a_flag_beats_the_config_file(monkeypatch, tmp_path, capsys):
    """Otherwise a grid file is a thing you have to edit to run a sweep, and the sweep is then
    not reproducible from the arguments anyone wrote down."""
    cfg = _rung1_cfg(
        monkeypatch, tmp_path, "--config", "conf/train/rung1_8b.json", "--max-seq-len", "4096"
    )
    capsys.readouterr()
    assert cfg.max_seq_len == 4096


# ---- resume after preemption
#
# The behaviour is tested in `tests/test_resume_from_checkpoint.py`; what is tested here is the
# only part of it an operator on the cluster touches. A flag that does not reach the config is a
# `--no-resume` that silently resumes, which is worse than no flag: the operator believes the
# run started clean and the manifest says so too.


def test_the_rungs_resume_by_default_and_checkpoint_every_two_hundred_steps(
    monkeypatch, tmp_path, capsys
):
    """The default is what a preempted job gets without anyone remembering a flag."""
    r1 = _rung1_cfg(monkeypatch, tmp_path)
    r2 = _rung2_cfg(monkeypatch, tmp_path)
    capsys.readouterr()
    assert r1.resume is True and r1.save_steps == 200
    assert r2.resume is True and r2.save_steps == 200


def test_no_resume_reaches_both_configs(monkeypatch, tmp_path, capsys):
    r1 = _rung1_cfg(monkeypatch, tmp_path, "--no-resume")
    r2 = _rung2_cfg(monkeypatch, tmp_path, "--no-resume")
    capsys.readouterr()
    assert r1.resume is False and r2.resume is False


def test_the_checkpoint_cadence_reaches_both_configs(monkeypatch, tmp_path, capsys):
    """In `cfg.sha`, so a flag that did not arrive would be a run whose identity names a
    cadence nobody applied."""
    r1 = _rung1_cfg(monkeypatch, tmp_path, "--save-steps", "50")
    r2 = _rung2_cfg(monkeypatch, tmp_path, "--save-steps", "50")
    capsys.readouterr()
    assert r1.save_steps == 50 and r2.save_steps == 50


# ---- the training seed
#
# `SFTConfig.seed` and `DPOConfig.seed` have existed since the rungs were written, both reach
# `seed` in the HF/TRL arguments, and both are in `cfg.sha` -- and NEITHER was settable. A seed
# that only a python REPL can change is a seed no run records: the three-seed arms in the
# programme would all have been seed 0 under three identical `config_sha`s, which is one run
# reported three times.


def test_the_seed_reaches_both_rung_configs(monkeypatch, tmp_path, capsys):
    r1 = _rung1_cfg(monkeypatch, tmp_path, "--seed", "1")
    r2 = _rung2_cfg(monkeypatch, tmp_path, "--seed", "1")
    capsys.readouterr()
    assert r1.seed == 1 and r2.seed == 1


def test_the_seed_changes_the_config_sha_on_both_rungs(monkeypatch, tmp_path, capsys):
    """The point of a seed sweep is three DISTINGUISHABLE runs. A seed that reaches the trainer
    but not the identity gives three checkpoints one `config_sha` can no longer tell apart."""
    a1 = _rung1_cfg(monkeypatch, tmp_path, "--seed", "0")
    b1 = _rung1_cfg(monkeypatch, tmp_path, "--seed", "1")
    a2 = _rung2_cfg(monkeypatch, tmp_path, "--seed", "0")
    b2 = _rung2_cfg(monkeypatch, tmp_path, "--seed", "1")
    capsys.readouterr()
    assert a1.sha != b1.sha
    assert a2.sha != b2.sha


def test_the_seed_defaults_to_zero_on_both_rungs(monkeypatch, tmp_path, capsys):
    """The flag is new; the run produced without it must not be. Every sha-pinning test in this
    repository was written against seed 0."""
    r1 = _rung1_cfg(monkeypatch, tmp_path)
    r2 = _rung2_cfg(monkeypatch, tmp_path)
    capsys.readouterr()
    assert r1.seed == 0 and r2.seed == 0
    assert r1.sha == _rung1_cfg(monkeypatch, tmp_path, "--seed", "0").sha
    assert r2.sha == _rung2_cfg(monkeypatch, tmp_path, "--seed", "0").sha
    capsys.readouterr()


def test_a_grid_file_may_set_the_seed_on_both_rungs(monkeypatch, tmp_path, capsys):
    """A sweep is written as a grid file, not as a command line, so `seed` has to be a key
    `--config` accepts -- today it is refused as "a key this command has no flag for"."""
    conf = tmp_path / "seed.json"
    conf.write_text(json.dumps({"seed": 1}) + "\n")
    r1 = _rung1_cfg(monkeypatch, tmp_path, "--config", str(conf))
    r2 = _rung2_cfg(monkeypatch, tmp_path, "--config", str(conf))
    capsys.readouterr()
    assert r1.seed == 1 and r2.seed == 1


def test_the_seed_flag_beats_the_grid_file(monkeypatch, tmp_path, capsys):
    conf = tmp_path / "seed.json"
    conf.write_text(json.dumps({"seed": 1}) + "\n")
    r1 = _rung1_cfg(monkeypatch, tmp_path, "--config", str(conf), "--seed", "2")
    capsys.readouterr()
    assert r1.seed == 2


def test_a_config_key_nothing_consumes_is_refused_not_ignored(monkeypatch, tmp_path, capsys):
    """A typo in a grid file is otherwise a run at the default under a config sha that names
    the value nobody applied -- the same failure as an unforwarded flag, one layer up."""
    from pi_run.cli import build_parser

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"max_seq_length": 5120}) + "\n")
    ds = tmp_path / "sft.jsonl"
    ds.write_text("")
    args = build_parser().parse_args(
        ["train", "rung1", "--base-model", "m", "--dataset", str(ds), "--config", str(bad)]
    )
    assert args.fn(args) == 1
    assert "max_seq_length" in capsys.readouterr().err


def test_a_why_field_is_documentation_and_not_a_key(monkeypatch, tmp_path, capsys):
    """The grids carry `_why_*` fields saying where each number came from. They are the reason
    the file is worth reading, so they must not have to be stripped before it can be used."""
    conf = tmp_path / "c.json"
    conf.write_text(json.dumps({"_why_max_seq_len": "measured", "max_seq_len": 5120}) + "\n")
    cfg = _rung1_cfg(monkeypatch, tmp_path, "--config", str(conf))
    capsys.readouterr()
    assert cfg.max_seq_len == 5120


# --- end track F: cli ---


def test_every_train_subcommand_renders_its_help():
    """`pi train export --help` crashed with `TypeError: %o format` because a help string said
    "19% of": argparse interpolates `%` in help text, so a bare percent sign is a format spec.
    The gate never renders help, so this is the test that does, for every subcommand."""
    from pi_run.cli import build_parser

    parser = build_parser()
    train = None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            train = action.choices.get("train")
    assert train is not None, "the top-level parser has no `train` subcommand"
    subs = [a for a in train._actions if isinstance(a, argparse._SubParsersAction)]
    assert subs, "`pi train` has no subcommands"
    rendered = {}
    for name, sp in subs[0].choices.items():
        rendered[name] = sp.format_help()  # raises on an unescaped % in any help string
    assert "export" in rendered and "--include-arm" in rendered["export"]
    for flag in (
        "--loss-type",
        "--loss-weights",
        "--ld-alpha",
        "--min-q-distinctness",
        "--min-stop-chosen-share",
    ):
        assert flag in rendered["rung2"], f"{flag} is not on `pi train rung2 --help`"
