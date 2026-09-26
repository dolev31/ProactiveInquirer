"""The four rungs. Every property that can be checked without a GPU, and nothing else.

There is no CUDA device on this machine, so `train()` is never called here. What IS called is
everything the checkpoint's correctness actually rests on: the loss mask, the acceptance rule,
the mode-collapse gate, the same-state invariant, the Pareto search and the cost projection.
A rung whose only untested line is `trainer.train()` is a rung whose failure modes are visible;
a rung whose logic lives inside a Trainer subclass is a rung nobody can check at all.
"""

from __future__ import annotations

import dataclasses
import json
import random
import zlib
from collections import Counter
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from pinq.actions import STOP_ACTION_JSON
from pinq.wire import Health, ScoreResponse, TurnScore
from pinq_train.rung0_gepa import (
    Candidate,
    Rung0Config,
    Rung0ConfigError,
    TaskKey,
    TaskResult,
    accept_mutation,
    ownership,
    pareto_front,
    run_search,
    sample_parent,
)
from pinq_train.rung1_sft import (
    DISTINCT3_FLOOR,
    IGNORE_INDEX,
    ModeCollapse,
    SFTConfig,
    assert_diverse,
    build_masked_example,
    distinct_n,
    estimate_gpu_hours,
    margin_threshold,
    pack,
    preflight,
    truncate_left,
)
from pinq_train.rung2_dpo import (
    DPOConfig,
    NotSameState,
    assert_length_guard,
    assert_same_state,
    encode_pairs,
    load_pairs,
)
from pinq_train.rung3_grpo import (
    KILL_DATE,
    ROLLOUT_SERVER_CONTRACT,
    Rung3Config,
    Rung3NotStarted,
    Rung3NotValidated,
    assert_go,
    group_rewards,
    is_monotone,
)
from pinq_train.rung3_grpo import train as grpo_train

GOLDEN = Path(__file__).parent / "fixtures" / "train" / "golden_episode.json"


class FakeTokenizer:
    """A whitespace tokenizer with a stable vocab. The two-method surface `mask.Tokenizer`
    declares, and nothing more -- which is the point: a test that needed `transformers` would
    be a test that never runs, because the real one lives behind the `[train]` extra."""

    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # crc32, not hash(): str.__hash__ is salted per process, and a fixture-backed test
        # whose token ids change between runs is a test that cannot pin anything.
        ids = [(zlib.crc32(t.encode()) % 30_000) + 10 for t in text.split()]
        return ([1] + ids) if add_special_tokens else ids


# --------------------------------------------------------------------- rung 1: the loss mask


def test_loss_mask_sums_to_the_inquirer_json_length():
    """THE assertion this rung exists for, on a REAL recorded decision point.

    The example is a long state (a task, an evidence block, a Q/A history) followed by a short
    JSON action. Only the action was written by the Inquirer. If the supervised count exceeds
    the action's token length by even one, the loss is reaching into the evidence block, and
    the checkpoint drifts toward a document language model while every loss curve looks fine.
    """
    ep = json.loads(GOLDEN.read_text())
    tok = FakeTokenizer()
    ex = build_masked_example(tok, ep["state_text"], ep["action_json"], append_eos=False)

    n_action = len(tok.encode(ep["action_json"], add_special_tokens=False))
    assert ex.n_supervised == n_action
    assert sum(1 for x in ex.labels if x != IGNORE_INDEX) == n_action
    # And the mask is genuinely load-bearing: the state dwarfs the action.
    assert ex.n_prompt > 10 * n_action
    assert len(ex.input_ids) == len(ex.labels) == ex.n_prompt + n_action
    # The supervised positions are exactly the trailing ones, in order.
    assert list(ex.labels[ex.n_prompt :]) == list(ex.input_ids[ex.n_prompt :])
    assert set(ex.labels[: ex.n_prompt]) == {IGNORE_INDEX}


def test_the_eos_is_supervised_and_is_the_only_extra_token():
    ep = json.loads(GOLDEN.read_text())
    tok = FakeTokenizer()
    n_action = len(tok.encode(ep["action_json"], add_special_tokens=False))
    with_eos = build_masked_example(tok, ep["state_text"], ep["action_json"], append_eos=True)
    assert with_eos.n_supervised == n_action + 1
    assert with_eos.labels[-1] == tok.eos_token_id


def test_the_golden_state_is_the_prompt_the_policy_actually_saw():
    """The fixture is provenance, not decoration: it carries the run id and the subset hash of
    the evidence the policy held, so a reconstruction that drifted would be visible here."""
    ep = json.loads(GOLDEN.read_text())
    assert ep["run_id"] and ep["subset_hash_before"]
    assert "EVIDENCE RETRIEVED SO FAR" in ep["state_text"]
    assert "QUESTIONS ALREADY ASKED" in ep["state_text"]
    assert json.loads(ep["action_json"])["action"] == "ASK"


def test_truncation_takes_from_the_front_never_from_the_action():
    tok = FakeTokenizer()
    ex = build_masked_example(tok, " ".join(f"w{i}" for i in range(400)), STOP_ACTION_JSON)
    small = truncate_left(ex, 40)
    assert len(small.input_ids) == 40
    assert small.n_supervised == ex.n_supervised  # the action survived intact
    assert list(small.input_ids[-small.n_action :]) == list(ex.input_ids[-ex.n_action :])


def test_truncation_refuses_to_cut_into_the_action():
    tok = FakeTokenizer()
    ex = build_masked_example(tok, "short state", " ".join(f"a{i}" for i in range(50)))
    with pytest.raises(ValueError, match="action alone"):
        truncate_left(ex, 10)


def test_packing_preserves_every_supervised_token():
    tok = FakeTokenizer()
    exs = [build_masked_example(tok, f"state {i} " * 5, f'{{"i":{i}}}') for i in range(9)]
    windows = pack(exs, 64)
    supervised = sum(1 for _, labels in windows for x in labels if x != IGNORE_INDEX)
    assert supervised == sum(e.n_supervised for e in exs)
    assert all(len(i) == len(x) <= 64 for i, x in windows)


# --------------------------------------------------------------- rung 1: acceptance + collapse


def test_the_acceptance_margin_carries_the_sqrt_two():
    """The thresholded quantity is a DIFFERENCE of two judged values, so two independent noise
    draws enter it. Dropping the sqrt(2) admits ties as wins."""
    assert margin_threshold(0.05, 0.0) == pytest.approx(0.05)
    assert margin_threshold(0.0, 1.0) == pytest.approx(1.5 * 2**0.5)
    assert margin_threshold(0.05, 0.02) > margin_threshold(0.05, 0.01)
    with pytest.raises(ValueError):
        margin_threshold(0.05, -0.1)


def test_distinct3_is_pooled_across_questions_not_averaged_within_one():
    """Total collapse must read as collapse. A per-text mean would read ~1.0 here."""
    collapsed = ["what is the founding date of the institute"] * 40
    assert distinct_n(collapsed) < 0.1
    varied = [f"what did organisation {i} publish in {1900 + i} about topic {i}" for i in range(40)]
    assert distinct_n(varied) > DISTINCT3_FLOOR


def test_a_near_duplicate_policy_still_fails_the_gate():
    """Varying one noun defeats an exact-duplicate rate. It must not defeat trigram diversity."""
    texts = [f"what is the founding date of the institute that owned X{i}" for i in range(60)]
    assert len(set(texts)) == 60  # every string is unique
    with pytest.raises(ModeCollapse, match="distinct-3"):
        assert_diverse(texts)


def test_an_empty_sample_is_refused_rather_than_passed():
    with pytest.raises(ModeCollapse, match="empty sample|no questions"):
        assert_diverse([])


def test_sft_config_refuses_an_unmeasured_threshold():
    with pytest.raises(ValueError, match="tau and sigma_j"):
        SFTConfig(base_model="m").validate()
    with pytest.raises(ValueError, match="base_model"):
        SFTConfig(tau=0.05, sigma_j=0.01).validate()


# --- curriculum ---
def test_sft_config_refuses_a_curriculum_it_cannot_run():
    """Two refusals, both at `pi train rung1` time, before a GPU-hour is spent.

    A TYPO IS THE DANGEROUS ONE. `curriculum="easy_first"` would otherwise be carried into
    `cfg.sha`, named in the manifest, printed in the preflight -- and the run would train the
    NULL arm, because anything that is not one of the three names falls through to a full
    shuffle. The ablation would then report a difference of zero and be believed.
    """
    base = SFTConfig(base_model="m", tau=0.05, sigma_j=0.01)
    for good in ("shuffled", "depth_easy_first", "depth_hard_first"):
        dataclasses.replace(base, curriculum=good).validate()
    for bad in ("easy_first", "Depth_Easy_First", "", "none"):
        with pytest.raises(ValueError, match="curriculum"):
            dataclasses.replace(base, curriculum=bad).validate()
    # A packed window is a CONCATENATION of rows: the row boundary the ordering is defined
    # over does not survive it, so the sequence would be shuffled into windows anyway.
    with pytest.raises(ValueError, match="curriculum"):
        dataclasses.replace(
            base, curriculum="depth_easy_first", pack_sequences=True, chat_template=False
        ).validate()


def test_the_curriculums_absent_default_does_not_move_the_config_sha():
    """MEASURED at the pre-change commit f3f3f0f, before `curriculum` existed:

        SFTConfig(base_model="Qwen/Qwen3-8B", tau=0.05, sigma_j=0.01).sha
        = 44eee2b7d306bdb27ae7bc3b9626c847dfb523f1f0d8b5c21bae4f83b1fdcc08

    `"shuffled"` is both the absent default and the null arm's own name, so there is no third
    state for the sha to distinguish: at the default the config must render the bytes it
    rendered before the field existed, or every rung-1 run already trained is unjoined from
    its own manifest. `tests/test_harmony_template.py` and `tests/test_sft_weighting.py` pin
    the same constant for the same reason.
    """
    cfg = SFTConfig(base_model="Qwen/Qwen3-8B", tau=0.05, sigma_j=0.01)
    assert cfg.curriculum == "shuffled"
    assert cfg.sha == "44eee2b7d306bdb27ae7bc3b9626c847dfb523f1f0d8b5c21bae4f83b1fdcc08"
    # And a set value is a different checkpoint: it changes what the trainer saw and when.
    assert dataclasses.replace(cfg, curriculum="depth_easy_first").sha != cfg.sha


# --- end curriculum ---


def test_lora_alpha_is_coupled_to_r():
    from pinq_train.rung1_sft import LoraSpec

    LoraSpec(r=16, alpha=32).validate()
    with pytest.raises(ValueError, match="2\\*r"):
        LoraSpec(r=16, alpha=16).validate()


def test_preflight_reports_the_stop_share_and_the_diversity():
    rows = [
        {
            "suite_id": "musique",
            "action_json": json.dumps(
                {"action": "ASK", "question": f"who founded body {i} in {i}"}
            ),
            "is_stop": False,
        }
        for i in range(40)
    ] + [{"suite_id": "musique", "action_json": STOP_ACTION_JSON, "is_stop": True}]
    rep = preflight(SFTConfig(base_model="m", tau=0.05, sigma_j=0.01), rows)
    assert rep["n_examples"] == 41 and rep["n_stop"] == 1
    assert rep["stop_share"] == pytest.approx(1 / 41)
    assert rep["distinct_3"] > DISTINCT3_FLOOR
    assert rep["margin_threshold"] == pytest.approx(margin_threshold(0.05, 0.01))


def test_gpu_hours_are_arithmetic_not_a_guess():
    cfg = SFTConfig(base_model="m", tau=0.0, sigma_j=0.0, epochs=2, tokens_per_second=1000.0)
    assert estimate_gpu_hours(cfg, 1_800_000) == pytest.approx(1.0)


# --------------------------------------------------------------------- rung 2: the same state


def _pair(state="S", chosen='{"q":"a"}', rejected='{"q":"b"}', margin=0.5):
    return {
        "suite_id": "musique",
        "task_id": "t",
        "run_id": "r",
        "turn_idx": 0,
        "state_text": state,
        "chosen_json": chosen,
        "rejected_json": rejected,
        "margin": margin,
        "len_delta": abs(len(chosen) - len(rejected)),
    }


def test_a_cross_state_pair_is_fatal_not_filtered():
    """If the states differ, the within-state contrast does not cancel V(s_t) and the loss is
    measuring task difficulty while the training curve looks perfectly healthy."""
    bad = {"chosen_state_text": "S1", "rejected_state_text": "S2"}
    with pytest.raises(NotSameState, match="DIFFERENT states"):
        assert_same_state(bad)
    assert_same_state(_pair())  # one shared state_text: nothing can disagree


def test_a_pair_with_no_state_at_all_is_refused():
    with pytest.raises(NotSameState, match="state_text"):
        assert_same_state({"chosen_json": "a", "rejected_json": "b"})


def test_the_length_guard_survives_into_training():
    """Without it the preference model learns 'longer question wins' -- the judge confound,
    imported straight into the policy's weights."""
    assert_length_guard(_pair(), len_delta_max=40)
    long_pair = _pair(chosen='{"q":"' + "x" * 200 + '"}')
    with pytest.raises(ValueError, match="longer question wins"):
        assert_length_guard(long_pair, len_delta_max=40)


def test_load_pairs_re_checks_both_invariants(tmp_path):
    p = tmp_path / "pairs.jsonl"
    p.write_text(json.dumps(_pair()) + "\n")
    # `load_pairs` returns (rows, drops): the config's filters run here, and a filter that
    # removes rows without saying how many is a filter nobody can audit. The counts reach
    # `rung2.manifest.json` through `pairs_report`.
    rows, drops = load_pairs(p)
    assert len(rows) == 1 and drops == {}
    p.write_text(json.dumps(_pair(chosen='{"q":"' + "x" * 200 + '"}')) + "\n")
    with pytest.raises(ValueError):
        load_pairs(p)


def test_encoded_pairs_share_the_prompt_token_for_token():
    """The cancellation argument is about token sequences, not about intent."""
    enc = encode_pairs(FakeTokenizer(), [_pair(state="a long shared state here")], max_seq_len=512)
    assert len(enc) == 1
    assert enc[0].n_shared_prompt > 0
    assert enc[0].chosen_ids != enc[0].rejected_ids


def test_dpo_refuses_to_start_from_the_bare_base():
    with pytest.raises(ValueError, match="adapter"):
        DPOConfig(base_model="m").validate()
    with pytest.raises(ValueError, match="beta"):
        DPOConfig(base_model="m", adapter="a", beta=0.0).validate()
    # A COMPLETE CONFIG NAMES ITS REFERENCE POLICY. This line used to encode the belief that a
    # rung-2 config is complete without saying what `pi_ref` is; it is not -- see
    # tests/test_rung2_reference_policy.py. Under the default the reference is rung 1's adapter
    # loaded frozen, so the adapter alone completes it; under a merge the weights have to be
    # named too. What this test is for (rung 2 refuses the bare base, and refuses beta=0) is
    # unchanged, and both readings are exercised so neither can quietly stop validating.
    DPOConfig(base_model="m", adapter="a").validate()
    DPOConfig(
        base_model="m",
        adapter="a",
        reference="merged",
        merge_adapter=True,
        merged_base_sha="0" * 64,
    ).validate()


# --------------------------------------------------------------------------- rung 0: the search


def test_rung0_refuses_to_optimise_against_an_eval_only_suite():
    """tau2 and pare are the zero-shot transfer targets. Tuning a prompt on them turns the
    transfer claim into a within-suite result reported under a transfer claim's name."""
    with pytest.raises(Rung0ConfigError, match="zero-shot"):
        Rung0Config(suites=("musique", "tau2"))
    with pytest.raises(Rung0ConfigError, match="zero-shot"):
        Rung0Config(suites=("pare",))


def test_the_projected_cost_is_computed_before_anything_runs():
    cfg = Rung0Config(
        suites=("musique",), n_tasks=10, n_val_tasks=5, n_candidates=2, n_generations=3
    )
    # 10 * (1 + 3*2)  train sweeps           = 70
    #  + 5 * 2         held-out tie-break     = 10
    #  + 5             the SEED's held-out pass = 5   <- added with the baseline fix
    #                                           ---
    #                                           85
    # That last term is not bookkeeping: without it `baseline_mean` is the seed's TRAIN mean
    # against the winner's VAL mean, and the reported gain is a task-set contrast. The pass is
    # real money and a quoted cost that omits it is a number the operator budgets against and
    # then overruns.
    assert cfg.n_rollouts == 85
    proj = cfg.projected_cost(".")
    assert proj.n_rollouts == 85
    assert proj.tok_in == 85 * cfg.tok_in_per_rollout
    assert proj.usd_uncached > proj.usd_cached > 0
    assert "PROJECTED COST" in proj.render()


def test_rung0_default_rates_match_the_pinned_table():
    """Two sources of truth for a token price disagree the first day a provider moves."""
    from pinq_train.rung0_gepa.config import (
        FALLBACK_USD_PER_MTOK_IN,
        FALLBACK_USD_PER_MTOK_OUT,
    )

    table = json.loads(Path("scripts/price_tables/2026-08.json").read_text())
    row = table["models"]["openai/aws/gpt-oss-120b"]
    assert row["input"] == FALLBACK_USD_PER_MTOK_IN
    assert row["output"] == FALLBACK_USD_PER_MTOK_OUT


def test_the_pareto_front_keeps_a_candidate_that_wins_one_task():
    """The whole reason not to argmax the mean: a mutation that fixes the hard tasks and costs
    a hair on the easy ones loses on the mean and is the mutation worth keeping."""
    scores = {
        "generalist": {"t1": 1.0, "t2": 1.0, "t3": 1.0, "hard": 0.0},
        "specialist": {"t1": 0.0, "t2": 0.0, "t3": 0.0, "hard": 9.0},
    }
    assert set(pareto_front(scores)) == {"generalist", "specialist"}
    assert ownership(scores) == {"generalist": 3, "specialist": 1}


def test_ties_keep_every_tied_candidate_so_the_front_is_order_independent():
    scores = {"a": {"t": 1.0}, "b": {"t": 1.0}, "c": {"t": 0.5}}
    assert pareto_front(scores) == ("a", "b")
    assert pareto_front({k: scores[k] for k in reversed(list(scores))}) == ("a", "b")


def test_parent_sampling_is_seeded_and_weighted_by_ownership():
    scores = {"a": {"t1": 1.0, "t2": 1.0, "t3": 1.0}, "b": {"t1": 0.0, "t2": 0.0, "t3": 0.0}}
    front = ("a", "b")
    draws = [sample_parent(front, scores, random.Random(7)) for _ in range(1)]
    assert draws == [sample_parent(front, scores, random.Random(7))]


def test_a_mutation_that_drops_a_placeholder_is_refused_before_any_tokens_are_spent():
    """A dropped hole renders an EMPTY evidence section -- a silent ablation nobody declared,
    which `render()` cannot even raise on, because the hole is gone rather than unresolved."""
    parent = Candidate(text="ask about {{question}} given {{evidence}}")
    ok, why = accept_mutation(parent, "ask about {{question}}")
    assert not ok and "placeholders changed" in why
    ok, why = accept_mutation(parent, "consider {{question}} in light of {{evidence}}")
    assert ok, why


def test_a_mutation_that_would_announce_the_budget_is_refused():
    parent = Candidate(text="ask about {{question}}")
    ok, why = accept_mutation(parent, "ask about {{budget}}")
    assert not ok
    assert "placeholders changed" in why or "forbidden" in why


def test_a_mutation_that_would_break_prompt_parity_is_refused():
    """MEASURED, not hypothetical. The 2026-09-01 rung-0 search had no length constraint and
    produced 25 scored candidates of which exactly ONE fit the parity cap -- the seed. Its
    winner reached 522 tokens against a cap of 440, and a length control showed its entire
    advantage was reproducible by padding the SEED with inert text (winner - lenctl = +0.003,
    p=0.96 at n=36). A search that may grow the prompt optimises length, and length is what
    `promptlib.PARITY_FAMILIES` exists to hold fixed.
    """
    parent = Candidate(text="ask about {{question}}")
    long = "ask about {{question}} " + " ".join(["padding"] * 400)
    ok, why = accept_mutation(parent, long, max_tokens=50)
    assert not ok and "parity" in why, why
    ok, why = accept_mutation(parent, "consider {{question}} carefully", max_tokens=50)
    assert ok, why
    # with no cap declared, length is not a reason to refuse -- old behaviour is preserved
    assert accept_mutation(parent, long)[0]


def test_rung0_takes_n_tasks_from_EACH_suite_not_the_first_n_of_a_grouped_file(tmp_path):
    """MEASURED. `write_task_ids` sorts by (suite, task), so the file is GROUPED -- every
    musique id, then every wiki2 id. The rung-0 CLI sliced `keys[: n_tasks * len(suites)]`,
    which is `n_tasks` per suite only if the file INTERLEAVES them. It does not, so the
    2026-09-02 search declared `--suite musique --suite wiki2 --n-tasks 12` and actually
    trained on 24 musique and 0 wiki2 -- while its manifest recorded both suites. The number
    was fine; the provenance was a lie, and that is the failure this repo exists to prevent.
    """
    from pi_run.cmd_train import _slice_per_suite

    keys = [SimpleNamespace(suite_id="musique", task_id=f"m{i}") for i in range(60)]
    keys += [SimpleNamespace(suite_id="wiki2", task_id=f"w{i}") for i in range(60)]

    train = _slice_per_suite(keys, ("musique", "wiki2"), 0, 12)
    assert Counter(k.suite_id for k in train) == {"musique": 12, "wiki2": 12}, (
        "a grouped file must still yield n_tasks PER SUITE"
    )
    val = _slice_per_suite(keys, ("musique", "wiki2"), 12, 6)
    assert Counter(k.suite_id for k in val) == {"musique": 6, "wiki2": 6}
    assert not ({k.task_id for k in train} & {k.task_id for k in val}), "train/val must be disjoint"

    # one suite short of what was asked for is a refusal, not a silent shortfall
    thin = [SimpleNamespace(suite_id="musique", task_id=f"m{i}") for i in range(3)]
    with pytest.raises(ValueError, match="musique"):
        _slice_per_suite(thin, ("musique",), 0, 12)


def test_the_parity_budget_is_derived_from_the_family_and_can_be_disabled():
    """ "family" must equal what the family actually permits, not min*1.10. INCLUDES makes those
    different: self_inquire embeds inquirer_prompted, so growing the base tightens its own cap
    from 480 to 440 -- the gap every rejected rung-0 candidate sat in."""
    from pinq import promptlib
    from pinq_train.rung0_gepa.config import Rung0Config
    from pinq_train.rung0_gepa.search import _budget_from

    assert _budget_from(Rung0Config()) == promptlib.parity_budget("inquirer_prompted", "inquirer")
    assert _budget_from(Rung0Config(max_prompt_tokens=None)) is None
    assert _budget_from(Rung0Config(max_prompt_tokens=600)) == 600
    naive = int(min(promptlib.tokens(n) for n in promptlib.PARITY_FAMILIES["inquirer"]) * 1.10)
    assert _budget_from(Rung0Config()) < naive, "INCLUDES must tighten the cap below min*1.10"


def test_an_empty_proposal_is_refused_because_that_is_what_a_token_overrun_returns():
    parent = Candidate(text="ask about {{question}}")
    ok, why = accept_mutation(parent, "   ")
    assert not ok and "empty proposal" in why
    assert not accept_mutation(parent, parent.text)[0]


def test_the_search_finds_the_better_prompt_and_is_deterministic_in_its_seed():
    """A fake evaluator and a fake reflector: no server, no tokens, no network."""
    cfg = Rung0Config(
        suites=("musique",), n_tasks=3, n_val_tasks=0, n_candidates=2, n_generations=2, minibatch=2
    )
    tasks = tuple(TaskKey("musique", f"t{i}") for i in range(3))

    def evaluate(cand: Candidate, key: TaskKey) -> TaskResult:
        # Longer prompts score higher, so the search has something monotone to find.
        return TaskResult(key=str(key), reward=len(cand.text) / 100.0, trace=f"{key} ok")

    counter = {"n": 0}

    def reflect(parent: Candidate, traces) -> str:
        """Append a note, keeping every placeholder. Monotone, so the search has a target."""
        counter["n"] += 1
        assert traces, "the reflector must be shown the parent's worst traces"
        return f"{parent.text} note{counter['n']}"

    a = run_search(
        cfg,
        seed_prompt="base {{question}}",
        tasks=tasks,
        evaluate=evaluate,
        reflect=reflect,
        on_event=lambda _: None,
    )
    assert a.winner_mean > a.baseline_mean
    assert a.n_evaluations == 3 * (1 + 2 * 2)
    assert a.generations == 2
    assert "{{question}}" in a.winner.text  # the placeholder survived every mutation


# --------------------------------------------------------------------------- rung 3


def test_rung3_refuses_without_a_written_go():
    with pytest.raises(Rung3NotStarted, match="written go"):
        assert_go(written_go=False, today=date(2026, 9, 1))


def test_rung3_refuses_after_its_kill_date():
    assert_go(written_go=True, today=KILL_DATE)
    with pytest.raises(Rung3NotStarted, match="kill date"):
        assert_go(written_go=True, today=date(2026, 9, 13))


def test_rung3_train_says_plainly_that_it_is_not_implemented():
    """An empty function would imply "nothing to do here". This says there is no trainer.

    The gate in front of the scaffold is date-dependent (`assert_go` refuses after KILL_DATE,
    2026-09-12), and this test used to rely on the wall clock being before it: from 2026-09-13
    it failed with Rung3NotStarted, which is the GATE speaking, not the scaffold. The belief
    "today is before the kill date" was never what this test guards, so the date is now
    explicit: on the last permitted day, a written go reaches the scaffold and the scaffold
    says it is a scaffold. `test_rung3_refuses_after_its_kill_date` covers the day after.
    """
    cfg = Rung3Config(base_model="m", adapter="a")
    with pytest.raises(Rung3NotValidated, match="scaffold"):
        grpo_train(cfg, written_go=True, today=KILL_DATE)


def test_the_rollout_server_contract_matches_the_wire_types():
    """A trainer wired to endpoints the server does not serve fails at 3 a.m., not in review."""
    assert set(ROLLOUT_SERVER_CONTRACT) == set(Health(ok=True).endpoints)


def _resp(eid: str, potential, turns, stop="policy_stop") -> ScoreResponse:
    return ScoreResponse(
        episode_id=eid,
        split="train",
        q_terminal=potential[-1],
        potential=tuple(potential),
        turns=tuple(TurnScore(turn_idx=i, n_retrieved=n, n_new=n) for i, n in enumerate(turns)),
        stop_reason=stop,
        n_ret=float(len(turns)),
    )


def test_group_advantages_are_centred_on_the_group_not_on_a_critic():
    group = [
        _resp("a", [0.0, 0.5, 1.0], [1, 1]),
        _resp("b", [0.0, 0.2, 0.4], [1, 1]),
        _resp("c", [0.0, 0.1, 0.2], [1, 1]),
    ]
    ga = group_rewards(group)
    assert len(ga.advantages) == 3
    assert sum(ga.advantages) == pytest.approx(0.0, abs=1e-9)
    assert ga.advantages[0] == max(ga.advantages)


def test_a_group_of_one_has_no_baseline_and_is_refused():
    with pytest.raises(ValueError, match="group IS the baseline"):
        group_rewards([_resp("a", [0.0, 1.0], [1])])


def test_dr_grpo_does_not_divide_by_the_group_std_by_default():
    group = [_resp("a", [0.0, 1.0], [1]), _resp("b", [0.0, 0.0], [1])]
    plain = group_rewards(group)
    scaled = group_rewards(group, normalize_by_std=True)
    assert plain.advantages != scaled.advantages
    assert Rung3Config(base_model="m", adapter="a").normalize_advantage_by_std is False


def test_the_monotonicity_gate_is_decided_in_advance():
    assert is_monotone([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    assert not is_monotone([0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    assert not is_monotone([1.0])  # too short to say anything


# --------------------------------------------------------------------------- rung0 baseline
#
# `baseline_mean` was the seed's mean over the TRAIN tasks while `winner_mean` came from the
# disjoint VAL slice. `cmd_train` prints them as "mean X (baseline Y)" and `write_winner`
# serialises both into rung0.manifest.json -- the provenance record for the prompt the
# untrained, trained and frontier families all share. The headline "the optimised prompt beats
# the seed" was a TASK-SET contrast wearing a prompt contrast's label.


def _difficulty_only_evaluator(hard_reward=1.0, easy_reward=9.0):
    """Reward depends ONLY on the task, never on the candidate. True prompt gain is exactly 0."""
    from pinq_train.rung0_gepa.search import TaskResult

    def evaluate(candidate, key):
        r = hard_reward if key.task_id.startswith("hard") else easy_reward
        return TaskResult(key=str(key), reward=r, trace="t")

    return evaluate


def test_rung0_measures_the_seed_on_the_same_tasks_as_the_winner():
    """THE TEST THAT COULD NOT HAVE PASSED BEFORE. With a reward that depends only on the task,
    any reported gain is a lie about the prompt. The old code reported +8.0."""
    from pinq_train.rung0_gepa.config import Rung0Config
    from pinq_train.rung0_gepa.search import TaskKey, run_search

    seed = "P {{question}} {{instructions}} {{evidence}} {{draft}} {{history}} {{user_channel}}"
    hard = [TaskKey(suite_id="musique", task_id=f"hard{i}", seed=0) for i in range(2)]
    easy = [TaskKey(suite_id="musique", task_id=f"easy{i}", seed=0) for i in range(2)]

    cfg = Rung0Config(
        suites=("musique",), n_tasks=2, n_val_tasks=2, n_candidates=2, n_generations=1
    )
    res = run_search(
        cfg,
        seed_prompt=seed,
        tasks=hard,
        val_tasks=easy,
        evaluate=_difficulty_only_evaluator(),
        reflect=lambda *a, **k: seed,  # every mutation is the parent, so nothing can improve
        on_event=lambda _e: None,
    )
    assert res.winner_mean == pytest.approx(res.baseline_mean), (
        f"reward depends only on the task, so the true prompt gain is 0; got "
        f"winner={res.winner_mean} baseline={res.baseline_mean} "
        f"(a gain of {res.winner_mean - res.baseline_mean:+.4f} from a search that changed nothing)"
    )
    assert res.baseline_mean == pytest.approx(9.0), (
        "the baseline must be the seed's HELD-OUT mean (9.0), not its train mean (1.0)"
    )


def test_the_seed_is_scored_on_held_out_even_when_it_is_off_the_pareto_front():
    """`val` covers only the front, and a seed beaten by every mutant is not on it -- so the
    baseline has to be scored explicitly rather than looked up. This is the case where the old
    code could not have reported the right number even in principle."""
    from pinq_train.rung0_gepa.config import Rung0Config
    from pinq_train.rung0_gepa.search import TaskKey, TaskResult, run_search

    seed = "P {{question}} {{instructions}} {{evidence}} {{draft}} {{history}} {{user_channel}}"
    hard = [TaskKey(suite_id="musique", task_id=f"hard{i}", seed=0) for i in range(2)]
    easy = [TaskKey(suite_id="musique", task_id=f"easy{i}", seed=0) for i in range(2)]

    def evaluate(candidate, key):
        # A REAL prompt effect (+0.1 per mutation) on top of a task-difficulty offset.
        base = 1.0 if key.task_id.startswith("hard") else 5.0
        return TaskResult(key=str(key), reward=base + 0.1 * candidate.text.count("MUT"), trace="t")

    n = {"i": 0}

    def reflect(*_a, **_k):
        n["i"] += 1
        return seed + " MUT" * n["i"]

    cfg = Rung0Config(
        suites=("musique",), n_tasks=2, n_val_tasks=2, n_candidates=2, n_generations=2
    )
    res = run_search(
        cfg,
        seed_prompt=seed,
        tasks=hard,
        val_tasks=easy,
        evaluate=evaluate,
        reflect=reflect,
        on_event=lambda _e: None,
    )
    # The seed's held-out mean is 5.0 whatever the front contains.
    assert res.baseline_mean == pytest.approx(5.0), res.baseline_mean
    gain = res.winner_mean - res.baseline_mean
    assert 0.0 <= gain <= 1.0, f"a +0.1-per-mutation effect cannot yield {gain:+.2f}"


def test_the_cost_estimate_prices_the_seeds_held_out_pass():
    """The extra pass is real money. A quoted cost that omits it is a number the operator
    budgets against and then overruns."""
    from pinq_train.rung0_gepa.config import Rung0Config

    a = Rung0Config(suites=("musique",), n_tasks=4, n_val_tasks=0, n_candidates=2, n_generations=2)
    b = Rung0Config(suites=("musique",), n_tasks=4, n_val_tasks=3, n_candidates=2, n_generations=2)
    held_out = 3 * 1 * b.n_seeds
    assert b.n_rollouts - a.n_rollouts == held_out * b.n_candidates + held_out


def test_the_caller_slices_val_tasks_to_the_number_it_priced():
    """It used to take the entire remainder of the task-id file, so the pre-spend estimate was
    wrong by however many ids that file happened to carry.

    Asserts the COUNT rather than the source text. The original grepped cmd_train for the
    literal slice expression, which pinned one implementation rather than the guarantee -- so
    fixing the grouped-file bug (`_slice_per_suite`) broke it while preserving exactly what it
    was protecting. `n_rollouts` prices `n_val_tasks * len(suites) * n_seeds` held-out passes,
    and that is what this now checks.
    """
    from pi_run.cmd_train import _slice_per_suite

    keys = [SimpleNamespace(suite_id="musique", task_id=f"m{i}") for i in range(50)]
    keys += [SimpleNamespace(suite_id="wiki2", task_id=f"w{i}") for i in range(50)]
    n_tasks, n_val, suites = 12, 6, ("musique", "wiki2")

    val = _slice_per_suite(keys, suites, n_tasks, n_val)
    assert len(val) == n_val * len(suites), "priced for n_val per suite; got a different count"
    assert len(val) < len(keys) - n_tasks * len(suites), "must not take the whole remainder"
    train = _slice_per_suite(keys, suites, 0, n_tasks)
    assert not ({k.task_id for k in train} & {k.task_id for k in val})


# --------------------------------------------------------------------------- the gate that wasn't
#
# `dataset_report`'s own docstring says a 90%-STOP dataset "will train a policy that stops
# immediately -- and it will do so while every per-example loss looks healthy". It computed
# `stop_share`, printed it, and acted on nothing. And the diversity gate was guarded by
# `if questions else None` -- the very condition `assert_diverse` exists to refuse.


def _rows(n_stop: int, questions: list[str], suite: str = "musique", per_task: int = 5):
    """Rows carrying a `task_id`, because every real exported row does and the diversity gate
    now reads it: it measures variation WITHIN a task, which is where state-dependence shows.
    A fixture with no task id puts every question under one task and no longer represents
    anything the exporter produces."""
    import json as _json

    rows = []
    for i in range(n_stop):
        rows.append(
            {
                "suite_id": suite,
                "task_id": f"t{i // per_task}",
                "is_stop": True,
                "action_json": _json.dumps({"kind": "stop"}),
            }
        )
    for i, q in enumerate(questions):
        rows.append(
            {
                "suite_id": suite,
                "task_id": f"t{i // per_task}",
                "is_stop": False,
                "action_json": _json.dumps({"kind": "ask", "question": q}),
            }
        )
    return rows


def test_a_fully_collapsed_dataset_no_longer_skips_the_diversity_gate():
    """`assert_diverse` refuses an empty sample on purpose -- "an empty sample passes every
    diversity test ever written". Guarding the call with `if questions` meant a 100%-STOP
    dataset skipped the gate entirely and reported distinct_3: None. The gate was unreachable
    precisely in the state it guards against."""
    from pinq_train.rung1_sft.collapse import ModeCollapse
    from pinq_train.rung1_sft.train import dataset_report

    with pytest.raises(ModeCollapse, match="no questions to measure"):
        dataset_report(_rows(50, []))


def test_too_few_questions_to_measure_is_refused_rather_than_scored():
    """A collapsing dataset loses its questions FIRST, so the sample the gate runs on gets
    smallest exactly as the thing it guards against arrives. Same rule as SIGMA_J_MIN_PAIRS."""
    from pinq_train.rung1_sft.collapse import MIN_QUESTIONS_TO_MEASURE, ModeCollapse
    from pinq_train.rung1_sft.train import dataset_report

    few = [f"what is the value of {i}?" for i in range(3)]
    with pytest.raises(ModeCollapse, match="below"):
        dataset_report(_rows(2, few))
    assert MIN_QUESTIONS_TO_MEASURE >= 20


def test_a_stop_dominated_dataset_is_refused_not_merely_reported():
    """A number that is computed, printed and never acted on documents the failure instead of
    preventing it."""
    from pinq_train.rung1_sft.collapse import STOP_SHARE_CEILING, ModeCollapse
    from pinq_train.rung1_sft.train import dataset_report

    diverse = [f"what year did event {i} happen in city {i}?" for i in range(40)]
    with pytest.raises(ModeCollapse, match="STOP"):
        dataset_report(_rows(400, diverse))  # ~91% stop
    assert 0.5 < STOP_SHARE_CEILING < 1.0


def test_a_healthy_dataset_still_passes():
    """The gates must not make a good dataset unusable, or they get removed."""
    from pinq_train.rung1_sft.train import dataset_report

    # GENUINELY VARIED, not 60 slot-fills of one template. The previous fixture was
    # `f"what year did event {i} happen in city {i}?"` for i in range(60) -- which shares
    # almost every trigram and scores 0.579 within a task, right on top of the measured
    # semi-collapsed baseline (0.603: ten canned templates emitted regardless of state). It
    # passed only because the pooled gate could not tell templated text from varied text.
    _stems = [
        "who founded {}",
        "when did {} open",
        "where is {} located",
        "which company acquired {}",
        "what is the population of {}",
    ]
    _subjects = "Arclight Bellhaven Corvid Dunmore Everly Fairwood Glenmoor Hollis".split()
    diverse = [
        _stems[i % len(_stems)].format(_subjects[(i // len(_stems)) % len(_subjects)])
        + f" in {1900 + i}?"
        for i in range(60)
    ]
    rep = dataset_report(_rows(20, diverse))
    assert rep["n_questions"] == 60
    assert rep["stop_share"] == pytest.approx(20 / 80)
    assert rep["distinct_3"] is not None


def test_the_train_extra_declares_every_module_the_rungs_import():
    """`[train]` read ["torch", "vllm", "trl", "peft", "dspy-ai"] and OMITTED `transformers`
    (imported by both trainers, declared only in [mine]) and `datasets` (imported by rung 2,
    declared only in [suites]) -- so `pip install -e ".[train]"` produced an environment in
    which BOTH trainers still ImportError. It also declared `dspy-ai`, which nothing in src/
    imports: rung 0 implements GEPA itself.

    Asserted against the IMPORTS, so adding a new one to a trainer without declaring it fails
    here rather than on a rented GPU."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    declared = set()
    for line in (root / "pyproject.toml").read_text().splitlines():
        if line.startswith("train "):
            declared = {m.group(1) for m in re.finditer(r'"([A-Za-z0-9_.-]+)[><=]', line)}
            break
    assert declared, "the [train] extra must exist"

    imported = set()
    # merge.py is scanned too: it is the only other module that reaches for transformers and
    # peft, and it does so on the rented box where a missing declaration costs an hour.
    for f in ("rung1_sft/train.py", "rung2_dpo/train.py", "merge.py"):
        for line in (root / "src/pinq_train" / f).read_text().splitlines():
            m = re.match(r"\s*(?:from|import)\s+([a-z_]+)", line)
            if m and m.group(1) in {"torch", "transformers", "peft", "trl", "datasets", "vllm"}:
                imported.add(m.group(1))

    missing = sorted(imported - declared - {"vllm"})  # vllm is rung 3, in [train-grpo]
    assert not missing, f"[train] does not declare: {missing}"


def test_the_extras_map_points_at_an_extra_that_would_install_the_module():
    """`pinq.extras` names the extra in its ImportError. Naming one that does not carry the
    module sends the reader to a command that will not fix their error."""
    import pathlib

    from pinq.extras import EXTRA_OF

    root = pathlib.Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text()
    for module in ("torch", "transformers", "peft", "trl", "datasets"):
        extra = EXTRA_OF[module]
        line = next(
            x for x in text.splitlines() if x.startswith(f"{extra} ") or x.startswith(f"{extra}=")
        )
        assert module.replace("_", "-") in line or module in line, (
            f"extras.py sends a missing {module!r} to [{extra}], which does not carry it"
        )


def test_rung2_refuses_an_empty_pairs_file():
    """`preflight` returned a clean report with `n_pairs: 0` and exited 0. Rung 1 refuses the
    analogous case loudly -- "an empty sample passes every diversity test ever written" -- and
    rung 2 did not. With `--train --acknowledge-untested` on a GPU box it would have run
    DPOTrainer over `Dataset.from_list([])` and written rung2.manifest.json as if a checkpoint
    existed: a rented-GPU failure mode, invisible on a laptop where `--train` cannot run.

    `n_pairs` is 0 by construction today -- a pair needs TWO candidates at one state, recorded
    runs hold one action per state, and no candidate sampler exists -- so this fires on the
    repository's actual state, not a hypothetical one."""
    import pytest

    from pinq_train.rung2_dpo import DPOConfig, NoPairs, preflight

    # preflight validates first, so the config has to be a legal one for NoPairs to be what
    # fires; the default reference (rung 1's adapter, loaded frozen) needs only the adapter.
    cfg = DPOConfig(base_model="Qwen/Qwen2.5-7B-Instruct", adapter="artifacts/rung1")
    with pytest.raises(NoPairs, match="0 preference pairs"):
        preflight(cfg, [])


def test_split_disagreement_can_no_longer_fire():
    """Its docstring used to open "THERE ARE TWO BUCKET FUNCTIONS IN THIS REPOSITORY AND THEY DO
    NOT AGREE" and quote a ~40% recall loss. There is now ONE definition -- `pinq.splitting` --
    and both `pi_run.manifest` and `pinq_train.split` delegate to it, so the runner's stamp and
    the exporter's re-check cannot disagree."""
    from pi_run.manifest import split_of as manifest_split
    from pinq_train.split import split_of as export_split

    for suite in ("musique", "strategyqa", "wiki2"):
        for task in ("2hop__1_2", "3hop1__a_b_c", "4hop2__w_x_y_z", "plain-id"):
            assert manifest_split(suite, task) == export_split(suite, task), (suite, task)
