"""Every gate in the structured-baselines reader must be able to FAIL.

WHY THESE TESTS EXIST. A gate that cannot fire is not a gate, and this repository has shipped two:
a slot-bias check that returned {} for every arm but one, and a stability check that compared reruns
to each other rather than to the thing they certify. Both passed forever. So each test below
mutates ONE field of an otherwise-clean population and asserts the corresponding gate fires, and
one asserts the clean population passes -- because a gate that fires on everything is equally
useless.

What these tests do NOT establish, stated so nobody reads them as more than they are: a fixture
proves a guard CAN fire. Only a replay against real history proves it does not refuse real work.
The clean fixture here is synthetic, so the non-refusal claim rests on running the script against
the live grid, which is recorded in the commit message rather than here.
"""

from __future__ import annotations

import copy

from scripts.structured_baselines.gate_and_read import gates


def _pop(n_tasks: int = 40) -> list[dict]:
    """A clean two-arm population: one code_version, one base_url_sha, only the inquirer differs."""
    out = []
    for arm, inq_key, inq_hash in (
        ("inquirer_prompted", "inquirer_prompted", "a" * 64),
        ("par2_rag", "par2_rag_plan", "b" * 64),
    ):
        for t in range(n_tasks):
            for seed in (0, 1):
                out.append(
                    {
                        "arm_id": arm,
                        "suite_id": "musique",
                        "task_id": f"t{t}",
                        "seed": seed,
                        "run_id": f"{arm}-{t}-{seed}",
                        "code_version": "a61c4f4be3e9" + "0" * 28,
                        "pins": {
                            r: {"base_url_sha": "c" * 64, "model_id": "m"}
                            for r in ("inquirer", "drafter", "answerer")
                        },
                        "prompt_hashes": {
                            "answerer": "d" * 64,
                            "answerer_frozen": "d" * 64,
                            "drafter_draft": "e" * 64,
                            "drafter_resolve": "f" * 64,
                            inq_key: inq_hash,
                        },
                        "_status": "ok",
                    }
                )
    return out


def test_the_clean_population_passes() -> None:
    """A gate that fires on everything is as useless as one that never fires."""
    assert gates(_pop()) == []


def test_two_code_versions_is_a_refusal() -> None:
    ms = copy.deepcopy(_pop())
    ms[0]["code_version"] = "deadbeefcafe" + "0" * 28
    fail = gates(ms)
    assert any("code_version" in f for f in fail), fail


def test_two_base_urls_is_a_refusal() -> None:
    """base_url_sha sits inside run identity, so two endpoints are two experiments."""
    ms = copy.deepcopy(_pop())
    ms[0]["pins"]["inquirer"]["base_url_sha"] = "9" * 64
    fail = gates(ms)
    assert any("base_url_sha" in f for f in fail), fail


def test_a_dev_prefix_is_a_refusal() -> None:
    ms = copy.deepcopy(_pop())
    ms[5]["run_id"] = "dev-" + ms[5]["run_id"]
    fail = gates(ms)
    assert any("dev-" in f for f in fail), fail


def test_identical_inquirer_prompts_is_a_refusal() -> None:
    """The arm whose prompt slot is never read: passes everything else, means nothing."""
    ms = copy.deepcopy(_pop())
    for m in ms:
        if m["arm_id"] == "par2_rag":
            del m["prompt_hashes"]["par2_rag_plan"]
            m["prompt_hashes"]["inquirer_prompted"] = "a" * 64
    fail = gates(ms)
    assert any("IDENTICAL inquirer prompts" in f for f in fail), fail


def test_differing_frozen_prompts_is_a_refusal() -> None:
    """If the drafter differs too, more than the inquirer differs."""
    ms = copy.deepcopy(_pop())
    for m in ms:
        if m["arm_id"] == "par2_rag":
            m["prompt_hashes"]["drafter_draft"] = "0" * 64
    fail = gates(ms)
    assert any("FROZEN prompts" in f for f in fail), fail


def test_a_selected_intersection_is_a_refusal_and_one_stray_error_is_not() -> None:
    """The distinction the first attempt at this grid got wrong.

    Losing 171 of 400 units on ONE arm selects the survivors and is fatal. Losing one unit costs
    one task and is not. Both are "unbalanced", which is why balance was the wrong test.
    """
    one_error = copy.deepcopy(_pop())
    one_error[0]["_status"] = "error"
    assert gates(one_error) == [], "a single stray error must not block a paired contrast"

    selected = copy.deepcopy(_pop())
    dropped = 0
    for m in selected:
        if m["arm_id"] == "par2_rag" and dropped < 40:
            m["_status"] = "error"
            dropped += 1
    fail = gates(selected)
    assert any("selected" in f for f in fail), fail


def test_an_empty_population_is_a_refusal_not_a_pass() -> None:
    """An empty store reads like a clean one to any check that only looks for violations."""
    assert gates([]) == ["no units with status ok"]


def test_re_runs_counted_as_progress_is_a_refusal() -> None:
    """The failure a peer hit: a sweep spanning a commit re-runs finished units.

    code_version sits inside run identity, so a commit moves every run_id and resume stops
    recognising completed units. Their ok counter counted ~79 such re-runs as progress. At one
    code_version resume overwrites in place, so ok and distinct cells are equal by construction --
    which is exactly why a divergence is worth refusing rather than tolerating.
    """
    ms = copy.deepcopy(_pop())
    dup = copy.deepcopy(ms[0])
    dup["run_id"] = ms[0]["run_id"] + "-rerun"
    ms.append(dup)
    fail = gates(ms)
    assert any("re-runs being counted as progress" in f for f in fail), fail


def test_a_single_arm_population_is_a_refusal_not_a_pass() -> None:
    """The hole a peer found in their equivalent, present in mine too.

    Every arm-comparing gate is guarded by `len(present) >= 2`, so on a one-arm store they skip and
    the function reports no failures. That is worse than the empty case: 80 ok runs at one
    code_version and one base_url_sha look healthy, and nothing in the output says no contrast is
    possible. Their version printed a 2x2 of zeros whose `prompted_only=0` was indistinguishable
    from a real result where the comparator never won.
    """
    one_arm = [m for m in _pop() if m["arm_id"] == "inquirer_prompted"]
    assert len(one_arm) == 80, "fixture changed; this test needs a substantial single-arm store"
    fail = gates(one_arm)
    assert any("no contrast is possible" in f for f in fail), fail


def test_a_missing_runs_root_refuses_distinctly_from_a_failed_gate() -> None:
    """A bad path and bad data must not look the same to a caller.

    A peer's find-based check was defeated by its tool: a rejected argument, an unreadable root and
    a wrong dialect all produced an empty string indistinguishable from an empty root, because the
    exit status went to a pipe. The Python analogue is letting FileNotFoundError out of os.listdir:
    it exits 1 with a traceback, which a caller testing only for non-zero cannot tell from the
    exit 2 that means the data failed the gates.
    """
    import pathlib

    import pytest
    from scripts.structured_baselines.gate_and_read import MissingRoot, load

    with pytest.raises(MissingRoot):
        load(pathlib.Path("/tmp/definitely-not-a-runs-root-xyzzy"))


def test_identical_prompts_on_different_weights_is_not_a_refusal() -> None:
    """The trained questioner runs the prompted template on its own weights.

    Two arms must differ on the inquirer side in SOMETHING, the prompt or the weights. The first
    version of this gate required the prompt to differ and so refused the trained-vs-prompted pair,
    which differs exactly and only in weights (found on the live grid, 2026-09-23). Identical prompts
    on IDENTICAL weights must still refuse; that is test_identical_inquirer_prompts_is_a_refusal.
    """
    ms = copy.deepcopy(_pop())
    for m in ms:
        if m["arm_id"] == "par2_rag":
            m["arm_id"] = "inquirer_trained"
            del m["prompt_hashes"]["par2_rag_plan"]
            m["prompt_hashes"]["inquirer_prompted"] = "a" * 64
            m["pins"]["inquirer"]["model_id"] = "trained-weights"
    assert gates(ms) == [], gates(ms)
