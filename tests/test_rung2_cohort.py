"""Every row names the code that produced it, and no pair straddles two cohorts.

WHY A ROW NEEDS A `code_version`. The corpus was rolled under two loops. In the OLD one a
question was resolved against evidence gathered before its own retrieval had merged, so ~63% of
evidence-bearing prompts carry a stale per-turn answer in their history block; the FIXED loop is
commit 3ffa972 and its descendants. That is a covariate shift in the PROMPT, which behaviour
cloning survives -- so the primary trains on both and the FIXED-only slice is an ablation. An
ablation you cannot select is not an ablation, and until now nothing on a row said which loop
rendered it: `code_version` rode in `semantic_hash` and reached no artifact.

WHY A PAIR MAY NOT STRADDLE THEM. A preference pair asserts "at this state, A beats B", which is
a claim about the QUESTIONS only if everything else about the two rollouts matched. `pins_sha`
already refuses a pair whose sides ran different prompts or model pins, for a measured reason:
266 pairs (18.9%) once straddled two prompt eras and the newer side was chosen just 35.3% of the
time. The loop change is the same kind of difference and `pins_sha` does not carry it -- the
prompt TEMPLATE is identical across the fix; what changed is what got rendered into it.

AND THE LOADER MUST BE ABLE TO SELECT ONE. `include_code_versions` is the ALLOWLIST a run
declares, and it is in `cfg.sha` because two runs over different cohorts of one file would
otherwise claim one identity. A ROW WITHOUT A `code_version` IS NOT IN ANY COHORT: it cannot be
shown to belong to the allowlist, so under an active filter it is dropped -- and counted
SEPARATELY from the rows excluded by value, because "outside the cohort" and "predates the
field" are different facts about the corpus and summing them hides which one the operator is
looking at.

Both halves live here: the EXPORT half stamps the rows, the LOADER half selects on them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from pi_run import cmd_train
from pi_run.cmd_train import collect_rows
from pinq.actions import ask_action_json
from pinq_train.export.dataset import export_pairs, export_sft, write_jsonl
from pinq_train.rung2_dpo import DPOConfig, load_pairs

GOLDEN = Path(__file__).parent / "fixtures" / "pairs_control_golden.jsonl"
SHA = "0" * 64


# --------------------------------------------------------------------------- export half


def _synth(tmp_path: Path):
    from pi_eval.build.synth_build import build
    from pinq_adapters.synth.suite import SynthSuite

    corpus, *_ = build(n_tasks=4, n_facets=2, depth=2, root=tmp_path)
    suite = SynthSuite(Path(corpus).parent)
    tid = str(suite.task_ids()[0])
    return suite, tid, list(suite.retriever(tid)._units)


def _turns(units):
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
                "subset_hash_before": before,
            }
        )
    return rows


def _write_run(runs: Path, run_id: str, *, task: str, turns: list, code: str) -> Path:
    d = runs / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "suite_id": "synth",
                "task_id": task,
                "arm_id": "inquirer_prompted",
                "split": "train",
                "template_id": None,
                "code_version": code,
            }
        )
    )
    (d / "turns.jsonl").write_text("\n".join(json.dumps(t) for t in turns))
    (d / "ledger.jsonl").write_text(
        json.dumps({"currency": "retrieval_calls", "cumulative": float(len(turns))})
    )
    (d / "status.json").write_text(
        json.dumps({"stop_reason": "budget", "wall_ms": 5, "usage": {"tok_total": 50}})
    )
    return d


def _cand(run_id: str, value: float, question: str, *, code: str | None = "fixed", **over) -> dict:
    r = {
        "suite_id": "musique",
        "task_id": "t1",
        "template_id": None,
        "run_id": run_id,
        "turn_idx": 1,
        "state_text": "S",
        "action_json": ask_action_json(question),
        "value": value,
        "phi_tilde": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "arm_id": "inquirer_prompted",
        "branch_of_run_id": "p",
        "branch_turn_idx": 1,
        "pins_sha": "pins-A",
        "budget_cap": 24,
        "max_turns": 24,
        "is_stop": False,
        "done_before": False,
        "coverage_before": 0.4,
    }
    if code is not None:
        r["code_version"] = code
    r.update(over)
    return r


# ------------------------------------------------------------------ the row carries it


def test_export_rows_name_the_code_and_the_arm_that_produced_them(tmp_path, monkeypatch):
    """From the MANIFEST, not from the environment: a re-export of an old run must report what
    that run actually used, which is the rule `pins_sha` already follows."""
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
    _suite, tid, units = _synth(tmp_path)
    runs = tmp_path / "runs"
    _write_run(runs, "r1", task=tid, turns=_turns(units[:2]), code="3ffa972")
    weights = cmd_train._train("reward").RewardWeights()

    rows, _ = collect_rows(runs, tmp_path, graph_version="v1", weights=weights)

    assert rows, "the fixture must produce rows or this proves nothing"
    assert all(r["code_version"] == "3ffa972" for r in rows)
    assert all(r["arm_id"] == "inquirer_prompted" for r in rows)


def test_export_carries_the_cohort_onto_every_example_and_pair():
    """DATACLASS FIELDS, not row keys. `write_jsonl` serialises `asdict(it)`, so a key that is
    only on the row dict is computed, carried through the exporter and dropped on the floor at
    write time -- which is exactly how the shipped export once lost its provenance block."""
    rows = [_cand("a", 0.9, "who?"), _cand("b", 0.3, "when?")]

    items, _ = export_sft(rows, margin_threshold=0.05)
    pairs, _ = export_pairs(rows, margin_threshold=0.05)

    assert items and pairs
    assert all(e.code_version == "fixed" and e.arm_id == "inquirer_prompted" for e in items)
    assert all(p.code_version == "fixed" and p.arm_id == "inquirer_prompted" for p in pairs)
    # and they SURVIVE serialisation, which is the half that was lost before
    assert "code_version" in dataclasses.asdict(pairs[0])


# ------------------------------------------------------------------ the guard


def test_export_refuses_a_pair_whose_candidates_ran_different_code():
    """Counted, never silent: a guard that drops rows without saying how many is a guard nobody
    can audit. Placed with the cross-instrument guard because it is the same kind of fact."""
    rows = [_cand("a", 0.9, "who?", code="old"), _cand("b", 0.3, "when?", code="fixed")]

    pairs, man = export_pairs(rows, margin_threshold=0.05)

    assert pairs == []
    assert man.n_cross_code_version_dropped == 1


def test_export_pairs_within_one_cohort_are_untouched():
    """The other half: the guard must not eat the population it exists to keep clean."""
    rows = [_cand("a", 0.9, "who?", code="fixed"), _cand("b", 0.3, "when?", code="fixed")]

    pairs, man = export_pairs(rows, margin_threshold=0.05)

    assert len(pairs) == 1 and man.n_cross_code_version_dropped == 0


def test_export_two_unstamped_rows_still_pair():
    """The asymmetry `pins_sha` already applies: an unstamped row is refused AGAINST a stamped
    one, but two unstamped rows pair. Legacy exports are internally consistent, and refusing them
    would delete data over a difference that is not there."""
    rows = [_cand("a", 0.9, "who?", code=None), _cand("b", 0.3, "when?", code=None)]

    pairs, man = export_pairs(rows, margin_threshold=0.05)

    assert len(pairs) == 1 and man.n_cross_code_version_dropped == 0

    mixed = [_cand("a", 0.9, "who?", code=None), _cand("b", 0.3, "when?", code="fixed")]
    pairs2, man2 = export_pairs(mixed, margin_threshold=0.05)
    assert pairs2 == [] and man2.n_cross_code_version_dropped == 1


def test_export_refuses_a_cross_cohort_ask_stop_pair_too():
    """The ask_stop branch shares the instrument, cohort and leak guards by its own comment. A
    guard present on one branch and absent on the other is the shape of every bug this file
    exists to prevent."""
    from pinq.actions import STOP_ACTION_JSON

    rows = [
        _cand("a", 0.9, "who?", code="old", done_before=True, coverage_before=1.0),
        _cand(
            "b",
            0.0,
            "",
            code="fixed",
            action_json=STOP_ACTION_JSON,
            is_stop=True,
            done_before=True,
            coverage_before=1.0,
        ),
    ]

    pairs, man = export_pairs(rows, margin_threshold=0.05)

    assert man.n_cross_code_version_dropped == 1
    assert all(p.pair_kind != "ask_stop" for p in pairs), "the recorded ask_stop pair must not form"


# ----------------------------------------------- the guard is on the COHORT, not the commit

COHORT_FILE = Path(__file__).parent.parent / "conf" / "cohorts" / "fixed_loop.json"
FIXED = tuple(json.loads(COHORT_FILE.read_text())["members"])
COHORT = frozenset(FIXED)
# TWO COMMITS OF THE OLD LOOP, taken from the population the guard was refusing: one musique
# turn-1 state carried four candidates at cdbf5f9 and six at 430c7a0. Neither descends from
# 3ffa972 -- `git merge-base --is-ancestor 3ffa972 <sha>` exits 1 for both -- so both are
# OUTSIDE the fixed cohort, and they belong to each other's population.
OLD_A = "cdbf5f97d2bc9b1e2369f20cc47c8b833eb40c83"
OLD_B = "430c7a089d97835173315decd084bf0e5dabfb3a"


def test_export_two_commits_of_the_fixed_loop_are_one_population():
    """WHAT THE GUARD IS FOR is keeping a PRE-fix candidate (a stale per-turn answer in its
    history block) from being ranked against a POST-fix one. Two commits that both descend from
    3ffa972 rendered the same prompt, so the contrast between them is the question and nothing
    else -- and the whole point of committing `conf/cohorts/fixed_loop.json` was to be able to
    say so without a subprocess.

    Comparing the COMMIT instead refused this: measured on the 2026-09-12 prompted-only control
    export, 448 pairs (1.1% of 38,858), every one of them a state whose candidates were forked
    under two commits of ONE cohort -- 22 round-2 states that finished later, and so on."""
    assert len(FIXED) >= 2, "the committed cohort must name two shas or this proves nothing"
    rows = [_cand("a", 0.9, "who?", code=FIXED[0]), _cand("b", 0.3, "when?", code=FIXED[1])]

    pairs, man = export_pairs(rows, margin_threshold=0.05, cohort=COHORT)

    assert len(pairs) == 1 and man.n_cross_code_version_dropped == 0


def test_export_two_commits_of_the_old_loop_are_one_population_too():
    """THE OTHER SIDE, and the bigger half of the refused population: 188 of the corpus's 191
    distinct `code_version` values predate the fix. Two OLD commits share the same defect, so a
    pair across them is no more cross-cohort than a pair inside one commit is. A rule that only
    forgave the fixed side would still be comparing commits, just on a shorter list."""
    rows = [_cand("a", 0.9, "who?", code=OLD_A), _cand("b", 0.3, "when?", code=OLD_B)]

    pairs, man = export_pairs(rows, margin_threshold=0.05, cohort=COHORT)

    assert len(pairs) == 1 and man.n_cross_code_version_dropped == 0


def test_export_the_ask_stop_branch_reads_the_same_cohort():
    """The ask_stop branch carries its own copy of the instrument, cohort and leak guards. A
    guard fixed on one branch and left comparing commits on the other is the shape of every bug
    this file exists to prevent -- and it would be invisible, because the counter is shared."""
    from pinq.actions import STOP_ACTION_JSON

    rows = [
        _cand("a", 0.9, "who?", code=OLD_A, done_before=True, coverage_before=1.0),
        _cand(
            "b",
            0.0,
            "",
            code=OLD_B,
            action_json=STOP_ACTION_JSON,
            is_stop=True,
            done_before=True,
            coverage_before=1.0,
        ),
    ]

    pairs, man = export_pairs(rows, margin_threshold=0.05, cohort=COHORT)

    assert man.n_cross_code_version_dropped == 0
    assert [p.pair_kind for p in pairs] == ["ask_stop"]


def test_export_refuses_a_pair_that_straddles_the_cohort():
    """The refusal that must survive the change, in its new terms: one candidate rendered by the
    fixed loop, one by the old one. This is the 266-pair prompt-era straddle one level down --
    a systematic cohort difference ranked as if it were a difference in the question."""
    rows = [_cand("a", 0.9, "who?", code=FIXED[0]), _cand("b", 0.3, "when?", code=OLD_A)]

    pairs, man = export_pairs(rows, margin_threshold=0.05, cohort=COHORT)

    assert pairs == [] and man.n_cross_code_version_dropped == 1


@pytest.mark.parametrize("unstamped", [None, ""])
@pytest.mark.parametrize("stamped", ["FIXED", "OLD"])
def test_export_an_unstamped_candidate_is_unknown_rather_than_old(unstamped, stamped):
    """THE RULE: an empty or missing `code_version` is UNKNOWN, not "outside the cohort".

    Unknown is refused against ANY stamped candidate -- including one that is itself outside the
    cohort -- and counted under `n_cross_code_version_dropped`, the same counter. This is the
    asymmetry `pins_sha` and the cap guard already apply, and it is the case a membership test
    written as `(a in cohort) != (b in cohort)` gets WRONG: `"" not in cohort` and
    `OLD_A not in cohort` are both true, so the naive rule would pair an unstamped row with an
    old-loop one and call them one population. "We could not check" is not "we checked".

    Two unstamped rows still pair: a legacy export is internally consistent, and refusing it
    would delete data over a difference that is not shown to be there."""
    other = FIXED[0] if stamped == "FIXED" else OLD_A
    rows = [_cand("a", 0.9, "who?", code=unstamped), _cand("b", 0.3, "when?", code=other)]

    pairs, man = export_pairs(rows, margin_threshold=0.05, cohort=COHORT)

    assert pairs == [] and man.n_cross_code_version_dropped == 1

    both = [_cand("a", 0.9, "who?", code=unstamped), _cand("b", 0.3, "when?", code=unstamped)]
    pairs2, man2 = export_pairs(both, margin_threshold=0.05, cohort=COHORT)
    assert len(pairs2) == 1 and man2.n_cross_code_version_dropped == 0


def test_export_the_manifest_names_the_cohort_file_that_decided(tmp_path):
    """A GUARD WHOSE RULE LIVES IN A FILE MUST NAME THE FILE. `n_cross_code_version_dropped`
    alone cannot be read: the same export, run against a cohort file with one more sha in it,
    reports a different count for the same rows and nothing on the artifact says which
    definition produced which number. `cohort_file_sha` is that provenance, and `""` is what an
    export decided by NO cohort file records -- a fact, not a default.

    It has to survive `asdict`: `write_jsonl` serialises the manifest that way, which is exactly
    how the shipped export once dropped its provenance block on the floor at write time."""
    rows = [_cand("a", 0.9, "who?", code=FIXED[0]), _cand("b", 0.3, "when?", code=OLD_A)]
    cohort, sha = cmd_train._cohort(None)

    assert cohort == COHORT, "the CLI must load the committed cohort by default"
    assert sha == hashlib.sha256(COHORT_FILE.read_bytes()).hexdigest()

    pairs, man = export_pairs(rows, margin_threshold=0.05, cohort=cohort, cohort_file_sha=sha)

    assert pairs == [] and man.n_cross_code_version_dropped == 1
    assert man.cohort_file_sha == sha
    out = write_jsonl(tmp_path / "pairs.jsonl", pairs, man)
    on_disk = json.loads((out.parent / "pairs.manifest.json").read_text())
    assert on_disk["cohort_file_sha"] == sha

    _unnamed, man2 = export_pairs(rows, margin_threshold=0.05)
    assert man2.cohort_file_sha == "", "no cohort file decided this one, and it must say so"

    # and a file naming NO members is refused rather than loaded: an empty set is how the
    # exporter spells "no cohort definition", so accepting one would restore the commit
    # comparison under the flag that says it was replaced.
    bad = tmp_path / "empty.json"
    bad.write_text(json.dumps({"root": FIXED[0], "members": []}))
    with pytest.raises(ValueError):
        cmd_train._cohort(str(bad))


# ------------------------------------------------------------------ pair_id must not move


def test_export_pair_id_does_not_move_when_provenance_grows():
    """THE REGRESSION ADDING FIELDS COULD CAUSE. `pair_id` is the join key onto every annotation
    campaign already collected -- the A5/A6/A7 verdicts on disk are keyed by it -- so a pair_id
    that moved would orphan 2,429 rater preferences silently, and nothing downstream would say so.

    `_pair` hashes `suite|task|run|turn|chosen_run|rejected_run|pair_kind` and NOTHING else, so
    appending provenance fields cannot move it. That is an argument; this is the measurement,
    against the byte-level golden generated at commit af3bc48, long before these fields existed.
    """
    gen = GOLDEN.parent / "make_pairs_control_golden.py"
    ns: dict = {"__file__": str(gen), "__name__": "make_golden"}
    exec(compile(gen.read_text(), "make_golden", "exec"), ns)

    pairs, _ = export_pairs(ns["ROWS"], margin_threshold=0.05, len_delta_max=40)
    want = [json.loads(x) for x in GOLDEN.read_text().splitlines() if x.strip()]

    assert len(pairs) == len(want) == 6
    assert [p.pair_id for p in pairs] == [w["pair_id"] for w in want]
    # and the id is still a function of the seven fields it claims, not of the new ones
    p0 = pairs[0]
    assert (
        p0.pair_id
        == hashlib.sha256(
            f"{p0.suite_id}|{p0.task_id}|{p0.run_id}|{p0.turn_idx}"
            f"|{p0.chosen_run_id or 'synth-stop'}|{p0.rejected_run_id}|{p0.pair_kind}".encode()
        ).hexdigest()[:16]
    )


def test_export_the_golden_rows_carry_empty_provenance_rather_than_a_guess():
    """The golden's rows name no code. `""` is what an unstamped row must serialise as -- not a
    default sha, and not a dropped key: an audit has to be able to tell "ran under no recorded
    code" from "ran under the fixed loop"."""
    gen = GOLDEN.parent / "make_pairs_control_golden.py"
    ns: dict = {"__file__": str(gen), "__name__": "make_golden"}
    exec(compile(gen.read_text(), "make_golden", "exec"), ns)

    pairs, man = export_pairs(ns["ROWS"], margin_threshold=0.05, len_delta_max=40)

    assert all(p.code_version == "" for p in pairs)
    assert man.n_cross_code_version_dropped == 0, "unstamped rows must pair with each other"


@pytest.mark.parametrize("field", ["code_version", "arm_id"])
def test_export_the_new_fields_are_last_so_positional_construction_is_unmoved(field):
    """`_pair` builds a PreferencePair with its first TWELVE arguments POSITIONAL. A field
    inserted anywhere but the end would silently rebind `margin`, `len_delta` or the latent
    labels -- a corruption with no exception and no failing type check."""
    from pinq_train.export.dataset import Example, PreferencePair

    for cls in (Example, PreferencePair):
        names = [f.name for f in dataclasses.fields(cls)]
        assert names[-2:] == ["code_version", "arm_id"], f"{cls.__name__}: {names[-4:]}"
        assert field in names


# ------------------------------------------------------------------ the committed cohort file


def test_export_the_fixed_loop_cohort_file_is_committed_and_self_consistent():
    """The cohort is a GIT-ANCESTRY fact over 192 distinct `code_version` values, and re-deriving
    it at read time would put a subprocess and a populated object store in the path of every
    consumer. So it is computed once by `scripts/cohort_ids.py` and committed.

    A committed derived artifact needs a check that it is the artifact it claims to be (rule 1):
    the root must be named, must be a member of its own cohort (`git merge-base --is-ancestor`
    holds reflexively), every member must be a full sha, and the file must say how many values
    were examined -- otherwise a cohort of 3 and a cohort of 3-out-of-192 read identically.
    """
    cohort = json.loads(
        (Path(__file__).parent.parent / "conf" / "cohorts" / "fixed_loop.json").read_text()
    )

    assert cohort["root"] == "3ffa972472cdbc8510c431214beb9ca03a5208af"
    assert cohort["generated_by"] == "scripts/cohort_ids.py"
    assert cohort["root"] in cohort["members"], "a commit is an ancestor of itself"
    assert all(len(m) == 40 and set(m) <= set("0123456789abcdef") for m in cohort["members"])
    assert cohort["members"] == sorted(set(cohort["members"])), "sorted and deduplicated"
    assert cohort["n_checked"] >= len(cohort["members"]) > 0


# --------------------------------------------------------------------------- loader half


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


def test_the_allowlist_keeps_its_cohort_and_counts_what_it_dropped(tmp_path):
    p = _write(
        tmp_path,
        [_p(code_version="aaaa111"), _p(code_version="aaaa111"), _p(code_version="bbbb222")],
    )
    rows, drops = load_pairs(p, include_code_versions=("aaaa111",))
    assert [r["code_version"] for r in rows] == ["aaaa111", "aaaa111"]
    assert drops["code_version"] == 1


def test_a_row_with_no_code_version_is_dropped_under_an_active_filter(tmp_path):
    """It cannot be shown to be in the cohort. Counted under its own key, so the report says
    'predates the field' rather than folding it into 'outside the allowlist'."""
    p = _write(tmp_path, [_p(code_version="aaaa111"), _p(), _p(code_version="bbbb222")])
    rows, drops = load_pairs(p, include_code_versions=("aaaa111",))
    assert len(rows) == 1
    assert drops["code_version_missing"] == 1
    assert drops["code_version"] == 1


def test_with_no_filter_a_row_without_a_code_version_is_kept(tmp_path):
    """The whole shipped corpus predates the field; an empty allowlist is no filter at all."""
    p = _write(tmp_path, [_p(), _p(code_version="aaaa111")])
    rows, drops = load_pairs(p)
    assert len(rows) == 2 and drops == {}


def test_an_allowlist_no_row_can_satisfy_is_fatal_rather_than_empty(tmp_path):
    """Otherwise the run drops every row and the operator reads the resulting refusal as "no
    pairs were exported", which is a different problem with a different fix."""
    p = _write(tmp_path, [_p(), _p()])
    with pytest.raises(ValueError, match="code_version"):
        load_pairs(p, include_code_versions=("aaaa111",))


def test_the_cohort_is_in_the_config_sha():
    base = DPOConfig(base_model="m", adapter="a", merged_base_sha=SHA)
    other = DPOConfig(
        base_model="m", adapter="a", merged_base_sha=SHA, include_code_versions=("aaaa111",)
    )
    assert base.sha != other.sha
