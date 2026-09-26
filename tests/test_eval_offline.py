"""Offline dev metrics, with the model replaced by a script.

WHY THE MODEL IS A CALLABLE AND NOT A CHECKPOINT. The three metrics in
`pinq_train.eval_offline` are decision rules, and a decision rule is testable only when the
thing it decides over is under the test's control. With a real checkpoint these tests would
assert "the number did not move", which is what a regression test does AFTER someone has
decided the number was right; nothing would say whether a perfect stopper actually scores
1.0, or whether the length normalisation in `stop_confusion` does anything at all.

So the model is `logprob(prompt_ids, completion_ids) -> float` and the tokenizer is a
whitespace split. Every test below scripts the log-probs so that the correct answer is known
by construction, and then asserts the function finds it.

THE LENGTH TEST IS THE ONE THAT MATTERS. `STOP_ACTION_JSON` is short and an ASK is long, so a
rule that argmaxes the SUM of completion log-probs prefers STOP on nearly every state for a
reason that has nothing to do with stopping. `test_a_short_stop_does_not_win_on_brevity_alone`
constructs a state where the two rules disagree and asserts BOTH directions, so the test
fails if the normalisation is removed AND fails if the case stops discriminating.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Sequence

import pytest

from pinq.actions import STOP_ACTION_JSON, ask_action_json

# --------------------------------------------------------------------------- the fake stack


def _render(state_text: str, action_json: str):
    """A whitespace tokenizer whose ids are recoverable, so a scripted model can read them.

    Ids are `hash`-free on purpose: the fake log-prob functions below decide by DECODING the
    completion, which is only honest if the encoding is injective on the strings in play.
    """
    from pinq_train.eval_offline import Rendered

    prompt = state_text.split()
    completion = action_json.split()
    words = prompt + completion
    ids = [_WORDS.setdefault(w, len(_WORDS)) for w in words]
    return Rendered(
        prompt_ids=tuple(ids[: len(prompt)]),
        completion_ids=tuple(ids[len(prompt) :]),
    )


_WORDS: dict[str, int] = {}


def _decode(ids: Sequence[int]) -> str:
    back = {v: k for k, v in _WORDS.items()}
    return " ".join(back[i] for i in ids)


def _is_stop(completion_ids: Sequence[int]) -> bool:
    return _decode(completion_ids) == " ".join(STOP_ACTION_JSON.split())


def _row(task_id: str, action_json: str, *, done_before, state: str = "s") -> dict:
    return {
        "suite_id": "musique",
        "task_id": task_id,
        "state_text": f"{state} for {task_id}",
        "action_json": action_json,
        "done_before": done_before,
        "split": "dev",
    }


ASK = ask_action_json("who directed it", "need the director")


# --------------------------------------------------------------------------- stop_confusion


def test_a_perfect_stopper_scores_one_on_both_cells() -> None:
    """The calibration point. If this is not 1.0/1.0 the metric cannot be read at all."""
    from pinq_train.eval_offline import stop_confusion

    rows = [
        _row("t1", ASK, done_before=True),
        _row("t2", ASK, done_before=True),
        _row("t3", ASK, done_before=False),
        _row("t4", ASK, done_before=False),
    ]
    done_states = {r["state_text"] for r in rows if r["done_before"]}

    def perfect(prompt_ids, completion_ids) -> float:
        state_is_done = _decode(prompt_ids) in done_states
        want_stop = _is_stop(completion_ids)
        # the right action gets the better PER-TOKEN score, whatever its length
        per_token = -0.1 if state_is_done == want_stop else -9.0
        return per_token * len(completion_ids)

    out = stop_confusion(rows, render=_render, logprob=perfect)
    assert out["p_stop_given_done"] == 1.0
    assert out["p_ask_given_not_done"] == 1.0
    assert out["n_done"] == 2 and out["n_not_done"] == 2


def test_an_always_stop_policy_scores_one_and_zero() -> None:
    """The failure I.8 names: a checkpoint buys P(STOP|done) by stopping everywhere, and the
    second cell is the only thing that says so."""
    from pinq_train.eval_offline import stop_confusion

    rows = [
        _row("t1", ASK, done_before=True),
        _row("t2", ASK, done_before=False),
        _row("t3", ASK, done_before=False),
    ]

    def always_stop(prompt_ids, completion_ids) -> float:
        per_token = -0.1 if _is_stop(completion_ids) else -9.0
        return per_token * len(completion_ids)

    out = stop_confusion(rows, render=_render, logprob=always_stop)
    assert out["p_stop_given_done"] == 1.0
    assert out["p_ask_given_not_done"] == 0.0


def test_a_short_stop_does_not_win_on_brevity_alone() -> None:
    """CONSTRUCTED so the two rules disagree, and both directions are asserted.

    STOP renders to 7 tokens scored at -0.5 each (sum -3.5); the ASK renders to 20 tokens at
    -0.3 each (sum -6.0). Argmax over the SUM picks STOP -- purely because it is shorter, since
    the model likes every ASK token strictly better. Argmax over the PER-TOKEN mean picks ASK.
    The state is not done, so ASK is the correct answer and the shipped rule must find it.
    """
    from pinq_train.eval_offline import stop_confusion

    long_ask = ask_action_json(" ".join(f"w{i}" for i in range(12)), "r")
    row = _row("t1", long_ask, done_before=False)
    n_stop = len(STOP_ACTION_JSON.split())
    n_ask = len(long_ask.split())
    assert n_ask > n_stop, "the case only discriminates while the ASK is the longer string"

    def brevity_trap(prompt_ids, completion_ids) -> float:
        per_token = -0.5 if _is_stop(completion_ids) else -0.3
        return per_token * len(completion_ids)

    # the case is real: the sum rule genuinely prefers STOP here
    stop_sum = brevity_trap((), _render(row["state_text"], STOP_ACTION_JSON).completion_ids)
    ask_sum = brevity_trap((), _render(row["state_text"], long_ask).completion_ids)
    assert stop_sum > ask_sum, "the brevity trap did not trap: the sum rule already prefers ASK"

    out = stop_confusion([row], render=_render, logprob=brevity_trap)
    assert out["p_ask_given_not_done"] == 1.0, (
        "the length-normalised rule picked STOP, i.e. it argmaxed the sum: a 7-token STOP "
        "beat a 20-token ASK the model liked better at every single token"
    )


def test_a_stop_row_without_a_reference_ask_is_skipped_and_counted() -> None:
    """A STOP row has no ASK string of its own, so the comparison is undefined. Skipping it
    silently would shrink the denominator with nothing saying by how much."""
    from pinq_train.eval_offline import stop_confusion

    rows = [_row("t1", STOP_ACTION_JSON, done_before=True)]
    out = stop_confusion(rows, render=_render, logprob=lambda p, c: -1.0 * len(c))
    assert out["n_done"] == 0 and out["n_skipped_no_reference"] == 1

    out2 = stop_confusion(
        rows,
        render=_render,
        logprob=lambda p, c: -0.1 * len(c) if _is_stop(c) else -9.0,
        reference_ask=ASK,
    )
    assert out2["n_done"] == 1 and out2["p_stop_given_done"] == 1.0


def test_an_unlabelled_row_is_skipped_rather_than_called_not_done() -> None:
    """`done_before` is None for 'unmeasured', never False. Reading None as not-done would
    move every unlabelled row into the second cell's denominator."""
    from pinq_train.eval_offline import stop_confusion

    rows = [_row("t1", ASK, done_before=None)]
    out = stop_confusion(rows, render=_render, logprob=lambda p, c: -1.0 * len(c))
    assert out["n_done"] == 0 and out["n_not_done"] == 0
    assert out["n_skipped_no_label"] == 1


# --------------------------------------------------------------------------- sft_nll


def test_sft_nll_is_token_weighted_and_example_weighted_separately() -> None:
    """Two rows of unequal length, hand-computed. A checkpoint that gets LONGER and one that
    gets WORSE move these two numbers differently, which is why both are reported."""
    from pinq_train.eval_offline import sft_nll

    short = ask_action_json("a", "")  # some number of tokens
    long = ask_action_json(" ".join(f"w{i}" for i in range(10)), "")
    rows = [_row("t1", short, done_before=None), _row("t2", long, done_before=None)]
    n1 = len(short.split())
    n2 = len(long.split())
    assert n1 != n2

    out = sft_nll(rows, render=_render, logprob=lambda p, c: -2.0 * len(c))
    assert out["n"] == 2
    assert out["nll_per_token"] == pytest.approx(2.0)
    assert out["nll_per_example"] == pytest.approx((2.0 * n1 + 2.0 * n2) / 2)


# --------------------------------------------------------------------------- pair_accuracy


def _pair(task_id: str, chosen: str, rejected: str, *, kind="ask_ask", decided_by="rule") -> dict:
    return {
        "suite_id": "musique",
        "task_id": task_id,
        "state_text": f"s for {task_id}",
        "chosen_json": chosen,
        "rejected_json": rejected,
        "pair_kind": kind,
        "decided_by": decided_by,
        "split": "dev",
    }


def _prefers(wanted: str):
    """A model that scores one exact completion string above everything else."""
    target = " ".join(wanted.split())

    def fn(prompt_ids, completion_ids) -> float:
        return -1.0 if _decode(completion_ids) == target else -5.0

    return fn


def test_pair_accuracy_is_one_when_the_fake_prefers_every_chosen() -> None:
    from pinq_train.eval_offline import pair_accuracy

    good = ask_action_json("the good one", "")
    bad = ask_action_json("the bad one", "")
    pairs = [_pair("t1", good, bad), _pair("t2", good, bad, kind="ask_stop")]
    out = pair_accuracy(pairs, render=_render, logprob=_prefers(good))
    assert out["acc"] == 1.0
    assert out["n"] == 2
    assert out["acc_by_kind"] == {"ask_ask": 1.0, "ask_stop": 1.0}
    assert out["mean_margin"] > 0


def test_pair_accuracy_is_zero_when_the_preference_is_reversed() -> None:
    """0.5 is 'learned nothing'; 0.0 is 'learned the opposite'. A metric that cannot tell them
    apart cannot support I.9's Tier A reading."""
    from pinq_train.eval_offline import pair_accuracy

    good = ask_action_json("the good one", "")
    bad = ask_action_json("the bad one", "")
    pairs = [_pair("t1", good, bad), _pair("t2", good, bad)]
    out = pair_accuracy(pairs, render=_render, logprob=_prefers(bad))
    assert out["acc"] == 0.0
    assert out["mean_margin"] < 0


def test_pair_accuracy_splits_by_kind_and_by_deciding_key() -> None:
    """I.8 asks for both breakdowns because ask_stop pairs carry a ~60-character length
    asymmetry no guard can remove: a pooled number hides which population moved."""
    from pinq_train.eval_offline import pair_accuracy

    good = ask_action_json("the good one", "")
    bad = ask_action_json("the bad one", "")
    pairs = [
        _pair("t1", good, bad, kind="ask_ask", decided_by="rule"),
        _pair("t2", bad, good, kind="ask_stop", decided_by="rater"),
    ]
    out = pair_accuracy(pairs, render=_render, logprob=_prefers(good))
    assert out["acc"] == 0.5
    assert out["acc_by_kind"] == {"ask_ask": 1.0, "ask_stop": 0.0}
    assert out["acc_by_decided_by"] == {"rule": 1.0, "rater": 0.0}


# ------------------------------------ the length-sign split (plan v4 sec 4 / rev-2 sec 4)
#
# WHAT THESE PIN. `acc` and `acc_by_kind` both read 0.5 on a policy that ordered every pair by
# question length: it gets the chosen-longer half right and the chosen-shorter half wrong, and
# the two cancel. The SIGN of the chosen-minus-rejected question length is the only breakdown
# that separates that policy from one that learned nothing, so the tests below script one model
# that reads ONLY length and one that reads ONLY the label, hand them THE SAME PAIRS, and assert
# the split tells them apart.

_LONG_QUESTIONS = (
    "which of the two directors of the film was born first",
    "what year did the studio that produced the film stop trading",
    "who founded the company that later bought the studio outright",
    "in which country was the screenwriter of the sequel educated",
)
_SHORT_QUESTIONS = ("who directed it", "what year", "which studio", "born where")


def _length_sign_pairs() -> list[dict]:
    """Two pairs whose chosen side is the LONGER question and two whose chosen side is the
    SHORTER one. All eight questions are distinct, so the label-reading model below can be keyed
    on the chosen strings alone and cannot be accidentally reading length."""
    pairs = [
        _pair(
            f"long{i}",
            ask_action_json(_LONG_QUESTIONS[i], ""),
            ask_action_json(_SHORT_QUESTIONS[i], ""),
        )
        for i in (0, 1)
    ]
    pairs += [
        _pair(
            f"short{i}",
            ask_action_json(_SHORT_QUESTIONS[i], ""),
            ask_action_json(_LONG_QUESTIONS[i], ""),
        )
        for i in (2, 3)
    ]
    assert all(len(long) > len(short) for long, short in zip(_LONG_QUESTIONS, _SHORT_QUESTIONS)), (
        "the two question banks stopped differing in length, so nothing below discriminates"
    )
    return pairs


def _prefers_the_longer_completion(prompt_ids, completion_ids) -> float:
    """The failure mode the split exists to name: an ordering that reads nothing but length."""
    return float(len(_decode(completion_ids)))


def _prefers_each_pairs_chosen(pairs):
    """Length-blind and perfect by construction: it recognises each pair's chosen string."""
    wanted = {" ".join(str(p["chosen_json"]).split()) for p in pairs}
    assert len(wanted) == len(pairs), (
        "a chosen string is shared between pairs; the fake is ambiguous"
    )

    def fn(prompt_ids, completion_ids) -> float:
        return -1.0 if _decode(completion_ids) in wanted else -5.0

    return fn


def test_a_length_following_policy_scores_one_and_zero_across_the_length_sign() -> None:
    """The diagnostic's calibration point, and the reason the pooled number cannot be read.

    This model orders every pair by length and NOTHING else. `acc` is 0.5 -- which is also what
    "learned nothing" reads -- and `acc_by_kind` is 0.5 too, because every pair here is ask_ask.
    Only the sign split says which 0.5 this is.
    """
    from pinq_train.eval_offline import pair_accuracy

    pairs = _length_sign_pairs()
    out = pair_accuracy(pairs, render=_render, logprob=_prefers_the_longer_completion)

    assert out["acc"] == 0.5, "the pooled number must be the uninformative one, or nothing is shown"
    assert out["acc_by_kind"] == {"ask_ask": 0.5}
    assert out["acc_by_len_sign"]["chosen_longer"] == {"acc": 1.0, "n": 2}
    assert out["acc_by_len_sign"]["chosen_shorter"] == {"acc": 0.0, "n": 2}
    assert out["acc_by_len_sign"]["equal"] == {"acc": None, "n": 0}
    assert out["len_sign_gap"] == 1.0


def test_a_length_blind_perfect_policy_scores_one_on_both_length_signs() -> None:
    """THE SAME PAIRS, a model that reads the label instead of the length. If this did not read
    1.0/1.0 the split would be measuring the pairs' length asymmetry rather than the policy's."""
    from pinq_train.eval_offline import pair_accuracy

    pairs = _length_sign_pairs()
    out = pair_accuracy(pairs, render=_render, logprob=_prefers_each_pairs_chosen(pairs))

    assert out["acc"] == 1.0
    assert out["acc_by_len_sign"]["chosen_longer"] == {"acc": 1.0, "n": 2}
    assert out["acc_by_len_sign"]["chosen_shorter"] == {"acc": 1.0, "n": 2}
    assert out["len_sign_gap"] == 0.0


def test_equal_length_questions_land_in_equal_and_leave_the_gap_unmeasured() -> None:
    """A tie is its own cell, not rounded into a sign. And with one side of the gap empty the gap
    is `None` -- "not measured" -- rather than 0.0, which would read as "measured, and no length
    exploitation" on a pair set that could not test for it at all."""
    from pinq_train.eval_offline import pair_accuracy

    chosen = ask_action_json("who directed it", "")
    rejected = ask_action_json("what studio was", "")
    assert len("who directed it") == len("what studio was"), "the tie case stopped being a tie"

    out = pair_accuracy([_pair("t1", chosen, rejected)], render=_render, logprob=_prefers(chosen))
    assert out["acc"] == 1.0
    assert out["acc_by_len_sign"]["equal"] == {"acc": 1.0, "n": 1}
    assert out["acc_by_len_sign"]["chosen_longer"] == {"acc": None, "n": 0}
    assert out["acc_by_len_sign"]["chosen_shorter"] == {"acc": None, "n": 0}
    assert out["len_sign_gap"] is None


def test_ask_stop_pairs_are_excluded_from_the_length_split() -> None:
    """A STOP is 18 bytes and every ASK is longer, so an ask_stop pair lands on one side of the
    split BY CONSTRUCTION -- that says what the two actions ARE, not what the policy learned.
    They stay in `acc` and `acc_by_kind`, where the asymmetry is already named, and are kept out
    of a split they would otherwise dominate."""
    from pinq_train.eval_offline import pair_accuracy

    ask = ask_action_json("which of the two directors of the film was born first", "")
    pairs = [
        _pair("t1", ask, STOP_ACTION_JSON, kind="ask_stop"),
        _pair("t2", STOP_ACTION_JSON, ask, kind="ask_stop_synth"),
        _pair("t3", ask, ask_action_json("who", ""), kind="ask_ask"),
    ]
    out = pair_accuracy(pairs, render=_render, logprob=_prefers(ask))

    assert out["n"] == 3, "the STOP pairs must still be scored; only the SPLIT excludes them"
    assert out["acc_by_kind"] == {"ask_ask": 1.0, "ask_stop": 1.0, "ask_stop_synth": 0.0}
    sign = out["acc_by_len_sign"]
    assert sum(cell["n"] for cell in sign.values()) == 1, sign
    assert sign["chosen_longer"] == {"acc": 1.0, "n": 1}
    assert sign["chosen_shorter"] == {"acc": None, "n": 0}


def test_the_length_sign_is_recomputed_from_the_questions_not_read_off_len_delta() -> None:
    """`len_delta` is an `abs(...)`: always >= 0, carrying no sign at all. An implementation that
    read its magnitude would call every pair chosen_longer and the diagnostic would be a constant.

    The second pair pins the other half: the sign is measured on the parsed QUESTION -- the same
    thing `export.dataset.question_len_delta` measures and the export-side length guard capped.
    There the chosen side's whole `action_json` is the far LONGER string because its rationale is
    long, while its question is the shorter one, so a raw-bytes implementation files it under
    `chosen_longer`.
    """
    from pinq_train.eval_offline import pair_accuracy

    short_q = ask_action_json("who directed it", "")
    long_q = ask_action_json("which of the two directors of the film was born first", "")
    misreported = _pair("t1", short_q, long_q) | {"len_delta": 38}
    assert misreported["len_delta"] > 0, "the trap only springs while len_delta is non-zero"

    fat_rationale = ask_action_json("who directed it", "b" * 200)
    lean_rationale = ask_action_json("which of the two directors of the film was born first", "")
    assert len(fat_rationale) > len(lean_rationale), "the raw-bytes trap did not spring"
    by_rationale = _pair("t2", fat_rationale, lean_rationale)

    out = pair_accuracy([misreported, by_rationale], render=_render, logprob=_prefers(short_q))
    assert out["acc_by_len_sign"]["chosen_shorter"] == {"acc": 0.5, "n": 2}
    assert out["acc_by_len_sign"]["chosen_longer"] == {"acc": None, "n": 0}


# --------------------------------------------------------------------------- the torch path


def test_the_landed_chat_renderer_is_build_chat_masked_example_and_not_a_second_copy() -> None:
    """`test_the_torch_factory_names_track_a_rather_than_silently_using_the_raw_path` used to
    live here. It pinned the REFUSAL that stood in for track A: until `build_chat_masked_example`
    existed, a `chat_template=True` request had to raise rather than silently fall back to the
    raw concatenation. Track A landed permanently in this tree (see `git log -- src/pinq_train/
    rung1_sft/mask.py`), so that test's `if has_chat: pytest.skip(...)` now fires on every
    checkout from here forward -- this machine, CI, the cluster -- because none of them branch
    back to one without the function. A skip that can never again do anything else is not
    coverage, so the test was deleted rather than left to advertise a check it could not perform.

    The branch it used to guard -- the one that actually renders the ids a dev NLL is computed
    over -- has its replacement here and below, and had it since track A landed: this test and
    `test_enable_thinking_is_forwarded_rather_than_hardcoded_false` assert identity against the
    real builder, which is strictly more than "does not raise" ever proved.

    WHAT WOULD GO WRONG AND HOW IT WOULD LOOK. `torch_logprob_fn` renders through
    `build_chat_masked_example`; rung 1 trains through the same function. If the two ever stop
    agreeing -- a re-implemented split, a dropped `enable_thinking`, `n_prompt` read as a count of
    prompt TOKENS somewhere and as an index somewhere else -- the dev NLL is computed over a
    different token sequence than the training NLL. It stays finite, it stays plausible, it stays
    comparable across checkpoints, and it selects the wrong one. Nothing raises.

    So this asserts identity against the builder itself rather than against a golden number: a
    golden would have to be updated whenever the template changes, and updating it is exactly how
    a divergence gets blessed.

    NO TORCH. `torch_logprob_fn` imports it lazily inside the LOGPROB half; the RENDER half is
    pure, which is what makes this testable at all on a machine with no GPU (and, here, no torch
    installed). Only the render half is exercised.
    """
    from tests.test_chat_template_identity import ACTION, STATE, FakeChatTokenizer

    from pinq_train.eval_offline import Rendered, torch_logprob_fn
    from pinq_train.rung1_sft.mask import build_chat_masked_example

    tok = FakeChatTokenizer()
    render, _logprob = torch_logprob_fn(object(), tok, chat_template=True)
    got = render(STATE, ACTION)

    want = build_chat_masked_example(FakeChatTokenizer(), STATE, ACTION, enable_thinking=False)

    assert isinstance(got, Rendered)
    assert got.prompt_ids + got.completion_ids == tuple(want.input_ids), (
        "the offline evaluator rendered a different token sequence than the trainer did for the "
        "same (state, action); the dev NLL would not be the training NLL's counterpart"
    )
    assert len(got.prompt_ids) == want.n_prompt, (
        "the split point moved: the log-prob sum would start at the wrong position and supervise "
        "part of the chat header, or miss the action's first token"
    )
    assert got.completion_ids, "a zero-length completion makes the per-token mean undefined"

    # `enable_thinking=False` reaches the template. Asserted on the tokenizer's OWN record of
    # what it was called with, because a default that is merely never overridden and a default
    # that is actually forwarded are the same thing until the caller wants the other value.
    assert tok.calls, "apply_chat_template was never called; some other path rendered these ids"
    assert all(c["enable_thinking"] is False for c in tok.calls), tok.calls


def test_enable_thinking_is_forwarded_rather_than_hardcoded_false() -> None:
    """The parameter is plumbed, not decorative.

    The test above would pass on a renderer that ignored `enable_thinking` and wrote `False`
    itself, because `False` is the default on both sides. A Qwen3 checkpoint trained with the
    thinking block open would then be scored without it, and the failure is the same silent-NLL
    one -- so the non-default value is what proves the wire exists.
    """
    from tests.test_chat_template_identity import ACTION, STATE, FakeChatTokenizer

    from pinq_train.eval_offline import torch_logprob_fn
    from pinq_train.rung1_sft.mask import build_chat_masked_example

    tok = FakeChatTokenizer()
    render, _ = torch_logprob_fn(object(), tok, chat_template=True, enable_thinking=True)
    got = render(STATE, ACTION)

    assert all(c["enable_thinking"] is True for c in tok.calls), tok.calls
    want = build_chat_masked_example(FakeChatTokenizer(), STATE, ACTION, enable_thinking=True)
    assert got.prompt_ids + got.completion_ids == tuple(want.input_ids)
    assert len(got.prompt_ids) == want.n_prompt

    # And the two settings are actually distinguishable on this tokenizer, or neither assertion
    # above could fail.
    off = build_chat_masked_example(FakeChatTokenizer(), STATE, ACTION, enable_thinking=False)
    assert tuple(off.input_ids) != tuple(want.input_ids), (
        "the fake template renders both settings identically, so these tests cannot fail"
    )


# --------------------------------------------------------------------------- the CLI refusal


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_eval_offline_refuses_a_file_whose_rows_are_not_dev(tmp_path, capsys) -> None:
    """The mirror of the trainers' HeldOutDataset refusal. `pi train eval-offline` reports the
    number that SELECTS a checkpoint; computing it on train rows would select on the training
    set and nothing in the JSON would say so."""
    from pi_run.cli import build_parser

    sft = tmp_path / "sft.dev.jsonl"
    pairs = tmp_path / "pairs.dev.jsonl"
    _write_jsonl(sft, [_row("t1", ASK, done_before=True) | {"split": "train"}])
    _write_jsonl(pairs, [])

    args = build_parser().parse_args(
        [
            "train",
            "eval-offline",
            "--checkpoint",
            str(tmp_path / "ckpt"),
            "--dev-sft",
            str(sft),
            "--dev-pairs",
            str(pairs),
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert args.fn(args) == 2
    err = capsys.readouterr().err
    assert "split" in err and "train" in err


def _install_a_fake_trainer_stack(monkeypatch, logprob) -> None:
    """A `transformers` and a `torch`, neither of which is installed in the gate venv.

    The Tier A JSON is the artifact the reviewer applies the length kill rule to, so "the block
    reaches the FILE" has to be assertable on a machine with no GPU stack. Only the two
    `from_pretrained` calls and the GPU count probe are faked; the (render, logprob) pair is the
    scripted one, so the numbers written out are the ones the metric actually computed.
    """
    tr = types.ModuleType("transformers")
    tr.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: object())
    tr.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: types.SimpleNamespace(eval=lambda: object())
    )
    torch = types.ModuleType("torch")
    # device_count(), not is_available(): eval_offline_load_kwargs() (job 778746's fix) counts
    # visible GPUs so it can tell one from more-than-one, same as base_model_load_kwargs.
    torch.cuda = types.SimpleNamespace(device_count=lambda: 0)
    monkeypatch.setitem(sys.modules, "transformers", tr)
    monkeypatch.setitem(sys.modules, "torch", torch)

    from pinq_train import eval_offline

    monkeypatch.setattr(eval_offline, "torch_logprob_fn", lambda *a, **k: (_render, logprob))


def test_eval_offline_writes_the_length_sign_split_into_the_tier_a_json(tmp_path, monkeypatch):
    """The block has to survive the trip to disk, not merely exist in the return value.

    `pair_accuracy`'s dict is embedded whole by `cmd_train_eval_offline`, so a field can only be
    lost between here and the reviewer by someone re-keying that dict -- and the JSON file, not
    the Python object, is what the kill rule is read off.
    """
    from pi_run.cli import build_parser

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "tokenizer_config.json").write_text("{}")

    sft = tmp_path / "sft.dev.jsonl"
    pairs = tmp_path / "pairs.dev.jsonl"
    _write_jsonl(sft, [_row("t1", ASK, done_before=True), _row("t2", ASK, done_before=False)])
    _write_jsonl(pairs, _length_sign_pairs())

    _install_a_fake_trainer_stack(monkeypatch, _prefers_the_longer_completion)
    out_path = tmp_path / "out.json"
    args = build_parser().parse_args(
        [
            "train",
            "eval-offline",
            "--checkpoint",
            str(ckpt),
            "--dev-sft",
            str(sft),
            "--dev-pairs",
            str(pairs),
            "--out",
            str(out_path),
        ]
    )
    assert args.fn(args) == 0

    written = json.loads(out_path.read_text())["pair_accuracy"]
    assert written["acc_by_len_sign"] == {
        "chosen_longer": {"acc": 1.0, "n": 2},
        "chosen_shorter": {"acc": 0.0, "n": 2},
        "equal": {"acc": None, "n": 0},
    }
    assert written["len_sign_gap"] == 1.0
    # and nothing the runbook already documents was displaced by it
    assert {"acc", "acc_by_kind", "acc_by_decided_by", "mean_margin", "n"} <= set(written)


# --------------------------------------------------- where the tokenizer for a checkpoint is


def test_a_lora_adapter_directory_resolves_its_tokenizer_from_the_base_model(tmp_path):
    """`--checkpoint` is documented as "a local model/adapter directory" and the adapter half
    did not work.

    MEASURED on the Mac smoke run, transformers 5.17.0, against rung 1's output directory:
    `AutoModelForCausalLM.from_pretrained(adapter_dir)` SUCCEEDS -- transformers 5.x reads
    `adapter_config.json`, resolves `base_model_name_or_path` and applies the LoRA -- but
    `AutoTokenizer.from_pretrained(adapter_dir)` raises

        ValueError: Couldn't instantiate the backend tokenizer from one of: ...

    because peft's `save_pretrained` writes no tokenizer files. So Tier A, the free
    deterministic check that is supposed to choose between rung 1 and rung 2 before any
    rollout is bought, could only ever be run on a merged full model -- and merging is what
    the dtype finding in `pinq_train.merge` shows is not free.

    A LoRA cannot change the tokenizer, so the base's is the right one by construction; the
    only question was where to look for it, and the adapter records the answer itself.
    """
    from pi_run.cmd_train import tokenizer_source

    adapter = tmp_path / "rung1"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"a")
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-0.6B", "peft_type": "LORA"})
    )
    assert tokenizer_source(adapter) == "Qwen/Qwen3-0.6B"


def test_a_full_model_directory_is_its_own_tokenizer_source(tmp_path):
    """The merged case, unchanged: `pi train merge` saves the base's tokenizer beside the
    weights precisely so the directory is self-contained."""
    from pi_run.cmd_train import tokenizer_source

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    (merged / "tokenizer.json").write_text("{}")
    (merged / "tokenizer_config.json").write_text("{}")
    assert tokenizer_source(merged) == str(merged)


def test_a_directory_with_neither_is_refused_by_name(tmp_path):
    """Not "fall back to some default tokenizer". Scoring a checkpoint with the wrong
    tokenizer produces a plausible NLL over token ids that mean something else, and no
    assertion downstream could see it."""
    from pi_run.cmd_train import tokenizer_source

    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="tokenizer"):
        tokenizer_source(empty)


def test_eval_offline_keeps_the_cuda_pin_with_exactly_one_visible_gpu():
    """`pi train eval-offline` loaded the checkpoint with a bare `from_pretrained(path)`;
    under transformers 5.x that resolves to `torch.get_default_device()` (cpu) and
    `torch.get_default_dtype()` (fp32), so the Tier A pass of job 766502 would have taken
    hours per checkpoint on the CPU of an A100 node (measured 2026-09-14 by the checkpoint
    evaluator, which had to override the torch defaults to keep up). PINNED BY VALUE: with
    exactly one visible GPU the result is exactly what every 8B-and-smaller run of record has
    used -- `device_map="cuda"` -- unchanged by job 778746's fix below, since a future edit to
    the >1 branch must not be able to quietly move this one too."""
    from pi_run.cmd_train import eval_offline_load_kwargs

    assert eval_offline_load_kwargs(n_visible_gpus=1) == {
        "device_map": "cuda",
        "dtype": "bfloat16",
    }


def test_eval_offline_has_no_cuda_kwargs_with_zero_visible_gpus():
    """PINNED BY VALUE: zero visible GPUs (the Mac smoke path, no CUDA device) returns `{}`
    exactly as before job 778746 -- the library's own defaults stand (fp32 on MPS/CPU)."""
    from pi_run.cmd_train import eval_offline_load_kwargs

    assert eval_offline_load_kwargs(n_visible_gpus=0) == {}


def test_eval_offline_spreads_across_every_visible_gpu_above_one():
    """MEASURED 2026-09-16, cluster job 778746 (rung1-32b-headline): Qwen3-32B trained fine on
    2 GPUs (`base_model_load_kwargs` -> `device_map="auto"`), but the OLD
    `eval_offline_load_kwargs` returned `device_map="cuda"` unconditionally whenever CUDA was
    merely present, pinning the whole model onto device 0 regardless of how many cards were
    visible -- nvidia-smi recorded one card at 76,815 MiB and the other at 4 MiB -- and
    `pinq_train.eval_offline._logprob`'s `model(batch).logits.float()` then died with CUDA
    OOM on the one card that held the entire model, while the LSF job went on to hold both
    GPUs idle for hours. Two (or more) visible GPUs must get accelerate's `device_map="auto"`,
    mirroring training's own `base_model_load_kwargs`, with the dtype still STATED rather than
    left to the library -- `device_map="auto"` computes its split FROM the dtype, and an
    unstated dtype on a bf16 checkpoint upcasts to fp32 before splitting, doubling the memory
    the split has to fit (the same reasoning `base_model_load_kwargs`'s own docstring gives)."""
    from pi_run.cmd_train import eval_offline_load_kwargs

    for n in (2, 3):
        assert eval_offline_load_kwargs(n_visible_gpus=n) == {
            "device_map": "auto",
            "dtype": "bfloat16",
        }, n


def test_eval_offline_load_kwargs_does_not_import_torch_at_module_import_time():
    """A previous commit deliberately kept `torch` out of this module's top-level imports --
    `make gate` and the main venv that runs `pi suites audit` have no torch installed, and
    must never need it just to import `pi_run.cmd_train`. `n_visible_gpus` defaulting to
    `None` and importing torch ONLY inside the function body (only when the caller has not
    already counted the devices itself, as the real `cmd_train_eval_offline` call site now
    does via `torch.cuda.device_count()`) is what preserves that. Run in a fresh subprocess
    because a torch import anywhere earlier in THIS test process (several fixtures above use
    `pytest.importorskip("torch", ...)`) would otherwise already be sitting in `sys.modules`
    and hide a regression here."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", "import pi_run.cmd_train, sys; assert 'torch' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
