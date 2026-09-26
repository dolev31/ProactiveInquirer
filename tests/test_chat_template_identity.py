"""The trainer and the server must tokenize the same bytes.

WHY THIS FILE EXISTS. Rung 1 trains on `state_text` concatenated with `action_json` and nothing
else. vLLM serves the same policy through `/v1/chat/completions`, which renders the message list
through the model's chat template: role headers, an end-of-turn token, and (on Qwen3) an optional
thinking block. Those two strings are not the same string. Train on one and serve the other and
every dev NLL in the paper is measured on a prompt the policy never sees at inference -- a gap
that produces no error, no warning and no failing assertion anywhere else in this repository.

So the templated path is tested for the four properties that make the two identical:

1. the prompt rendering is a STRICT PREFIX of the full rendering (otherwise the mask is off by
   however many tokens the template inserted at the join, and the loss supervises the wrong
   positions while every count still looks right);
2. the supervised span is exactly `full - prompt`;
3. it ends at the TEMPLATE's end-of-turn id, not at `eos_token_id` -- those are different tokens
   on some models, and appending our own eos would teach the policy to emit a token the server's
   stop criteria do not recognise;
4. `enable_thinking=False` reaches BOTH renderings -- passing it to one is worse than passing it
   to neither, because then the prompt and the target disagree about whether a `<think>` block is
   coming.

The fifth property -- that the ids equal the ids a real vLLM produces -- cannot be asserted from a
laptop with no GPU and no server. `test_serving_tokens_equal_training_tokens` is the check that
closes it, and it is skipped unless `PI_VLLM_URL` names a running server. Plan I.14: run it before
spending anything.
"""

from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import pytest

from pinq_train.rung1_sft.mask import (
    IGNORE_INDEX,
    OverLength,
    TemplateNotPrefix,
    build_chat_masked_example,
    build_masked_example,
)
from pinq_train.rung1_sft.train import SFTConfig, build_examples

GOLDEN = Path(__file__).parent / "fixtures" / "train" / "golden_episode.json"

STATE = "who directed the film and when"
ACTION = '{"action": "ASK", "question": "Who is the spouse?"}'


class FakeChatTokenizer:
    """A chat template with the shape every Qwen/Llama template has, and nothing more.

    `eos_token_id` is deliberately a DIFFERENT id from the end-of-turn token. On Qwen3 chat they
    happen to coincide; making them coincide here would make property (3) above untestable, and a
    test that cannot fail is not a test.
    """

    IM_START = 3
    IM_END = 7  # the template's end-of-turn id
    NL = 5
    THINK = 9
    eos_token_id = 2  # NOT IM_END, on purpose

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [(zlib.crc32(t.encode()) % 30_000) + 100 for t in text.split()]
        return ([1] + ids) if add_special_tokens else ids

    def _turn(self, role: str, content: str, *, enable_thinking: bool) -> list[int]:
        ids = [self.IM_START, *self.encode(role, False), self.NL]
        if role == "assistant" and enable_thinking:
            ids.append(self.THINK)
        return [*ids, *self.encode(content, False), self.IM_END]

    def apply_chat_template(
        self,
        conversation,
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        enable_thinking: bool = False,
        **kwargs,
    ):
        self.calls.append(
            {
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "enable_thinking": enable_thinking,
            }
        )
        ids: list[int] = []
        for i, m in enumerate(conversation):
            if i:
                ids.append(self.NL)
            ids += self._turn(str(m["role"]), str(m["content"]), enable_thinking=enable_thinking)
        if add_generation_prompt:
            ids += [self.NL, self.IM_START, *self.encode("assistant", False), self.NL]
            if enable_thinking:
                ids.append(self.THINK)
        return ids if tokenize else " ".join(str(x) for x in ids)


class DesyncedChatTokenizer(FakeChatTokenizer):
    """A template whose generation prompt is NOT what the assistant turn starts with.

    This is a real failure mode, not an invented one: a template that writes
    `<|im_start|>assistant\\n` when asked for a generation prompt and `<|im_start|>assistant` when
    rendering a recorded assistant turn differs by one token, the prompt stops being a prefix, and
    a mask built from `len(prompt_ids)` silently supervises one token of the header.
    """

    def _turn(self, role, content, *, enable_thinking):
        ids = super()._turn(role, content, enable_thinking=enable_thinking)
        if role == "assistant":
            ids.pop(2)  # the newline after the role
        return ids


class BatchEncodingChatTokenizer(FakeChatTokenizer):
    """A tokenizer whose `apply_chat_template(tokenize=True)` returns a MAPPING, not a list.

    THIS IS transformers >= 5.0, NOT A HYPOTHETICAL. `apply_chat_template`'s `return_dict`
    parameter defaults to False in 4.x and to True in 5.x, so on transformers 5.17.0 both calls
    in `build_chat_masked_example` come back as a `BatchEncoding` with keys `input_ids` and
    `attention_mask`. The old code wrapped each call in `list(...)`, which on a mapping yields
    the KEY NAMES -- `['input_ids', 'attention_mask']` -- for the prompt and for the full
    rendering alike.

    The near-miss is the reason this test exists rather than a smoke run's traceback: the
    prefix check PASSES (two identical two-element key lists), and only the `n_action <= 0`
    branch catches it, with a message about the assistant turn that names the wrong cause. Had
    the two renderings differed in their key sets by even one entry, the mask would have been
    built over strings and the failure would have been an unrelated exception deeper in torch.
    """

    def apply_chat_template(self, conversation, **kwargs):
        ids = super().apply_chat_template(conversation, **kwargs)
        if not isinstance(ids, list):  # tokenize=False still returns the string
            return ids
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class FakeTokenizer:
    """The raw path's tokenizer, byte-identical in behaviour to `test_train_rungs.FakeTokenizer`."""

    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [(zlib.crc32(t.encode()) % 30_000) + 10 for t in text.split()]
        return ([1] + ids) if add_special_tokens else ids


# ------------------------------------------------------------------ the prefix, and its refusal


def test_the_prompt_rendering_is_a_strict_prefix_of_the_full_rendering():
    tok = FakeChatTokenizer()
    ex = build_chat_masked_example(tok, STATE, ACTION)

    msgs = [{"role": "user", "content": STATE}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    full = tok.apply_chat_template(
        [*msgs, {"role": "assistant", "content": ACTION}],
        add_generation_prompt=False,
        tokenize=True,
    )
    assert list(ex.input_ids) == full
    assert full[: len(prompt)] == prompt
    assert len(full) > len(prompt), "a prefix that is the whole thing supervises nothing"
    assert ex.n_prompt == len(prompt)


def test_a_template_whose_generation_prompt_is_not_a_prefix_is_refused():
    with pytest.raises(TemplateNotPrefix, match="prefix"):
        build_chat_masked_example(DesyncedChatTokenizer(), STATE, ACTION)


def test_a_tokenizer_that_returns_a_batch_encoding_is_read_for_its_ids():
    """transformers 5.x returns a mapping from `apply_chat_template(tokenize=True)`.

    MEASURED, on transformers 5.17.0 + Qwen/Qwen3-0.6B: every one of the 300 smoke rows raised
    `TemplateNotPrefix("the assistant turn added no tokens")`, because `list(BatchEncoding)`
    is `['input_ids', 'attention_mask']` on both halves and the two lengths are therefore
    equal. Rung 1, the dev NLL and the STOP 2x2 all route through this function, so the whole
    templated path was dead against the library the training venv actually has.

    The assertion is identity with the list-returning tokenizer, not merely "it did not
    raise": a fix that read the mapping but lost the flattening would still produce a masked
    example, and nothing downstream would see the difference.
    """
    listy = build_chat_masked_example(FakeChatTokenizer(), STATE, ACTION)
    dicty = build_chat_masked_example(BatchEncodingChatTokenizer(), STATE, ACTION)

    assert dicty.input_ids == listy.input_ids
    assert dicty.labels == listy.labels
    assert dicty.n_prompt == listy.n_prompt
    assert dicty.n_action == listy.n_action
    assert dicty.n_supervised == dicty.n_action > 0


# ------------------------------------------------------------------------- the supervised span


def test_the_supervised_span_is_exactly_the_full_rendering_minus_the_prompt():
    tok = FakeChatTokenizer()
    ex = build_chat_masked_example(tok, STATE, ACTION)

    assert set(ex.labels[: ex.n_prompt]) == {IGNORE_INDEX}
    assert list(ex.labels[ex.n_prompt :]) == list(ex.input_ids[ex.n_prompt :])
    assert ex.n_supervised == ex.n_action == len(ex.input_ids) - ex.n_prompt
    ex.validate()


def test_the_last_supervised_token_is_the_templates_end_of_turn_not_the_eos():
    """We append nothing. Whatever the template ended the assistant turn with is the last token
    the policy is taught to emit, because that is the token the server's stop criteria watch."""
    tok = FakeChatTokenizer()
    ex = build_chat_masked_example(tok, STATE, ACTION)

    assert ex.input_ids[-1] == FakeChatTokenizer.IM_END
    assert ex.labels[-1] == FakeChatTokenizer.IM_END
    assert tok.eos_token_id != FakeChatTokenizer.IM_END, "the fixture must keep the two distinct"
    assert tok.eos_token_id not in ex.input_ids, "we must not have appended our own eos"


# ------------------------------------------------------------------------------- the think flag


def test_thinking_is_disabled_in_both_renderings():
    tok = FakeChatTokenizer()
    ex = build_chat_masked_example(tok, STATE, ACTION, enable_thinking=False)

    assert len(tok.calls) == 2, "the prompt and the full sequence are two renderings"
    assert [c["enable_thinking"] for c in tok.calls] == [False, False]
    assert [c["add_generation_prompt"] for c in tok.calls] == [True, False]
    assert [c["tokenize"] for c in tok.calls] == [True, True]
    assert FakeChatTokenizer.THINK not in ex.input_ids


def test_thinking_when_asked_for_reaches_both_renderings_too():
    """The flag is forwarded, not hard-coded off: a caller who turns it on must get it in BOTH
    renderings, or the prompt and the target disagree about whether a think block is coming."""
    tok = FakeChatTokenizer()
    ex = build_chat_masked_example(tok, STATE, ACTION, enable_thinking=True)

    assert [c["enable_thinking"] for c in tok.calls] == [True, True]
    assert FakeChatTokenizer.THINK in ex.input_ids


# ------------------------------------------------------------------------------ the raw path
#
# `build_masked_example` must stay byte-identical: two golden tests in test_train_rungs.py pin it,
# and the templated path is an ADDITION beside it rather than a replacement of it. Pinned here as
# well, on the same recorded decision point, so the person editing mask.py sees it in the file
# they are reading.


def test_the_raw_path_is_unchanged_on_the_golden_episode():
    ep = json.loads(GOLDEN.read_text())
    tok = FakeTokenizer()

    ex = build_masked_example(tok, ep["state_text"], ep["action_json"], append_eos=False)
    assert (ex.n_prompt, ex.n_action) == (469, 17)
    assert ex.n_supervised == 17
    assert set(ex.labels[:469]) == {IGNORE_INDEX}
    assert list(ex.labels[469:]) == list(ex.input_ids[469:])

    with_eos = build_masked_example(tok, ep["state_text"], ep["action_json"], append_eos=True)
    assert (with_eos.n_prompt, with_eos.n_action) == (469, 18)
    assert with_eos.labels[-1] == tok.eos_token_id
    assert with_eos.weight == 1.0, "the appended field must not disturb the raw path's default"


# ------------------------------------------------------------------- over-length, and the packer


def test_an_over_length_templated_row_is_refused_rather_than_truncated():
    """Left truncation cuts the chat HEADER -- `<|im_start|>user` -- which produces a row the
    server can never produce. The whole point of the template is that the two agree, so the only
    honest response to a row that does not fit is to refuse it and count it."""
    tok = FakeChatTokenizer()
    rows = [{"state_text": STATE, "action_json": ACTION}]
    with pytest.raises(OverLength, match="chat header"):
        build_examples(tok, rows, max_seq_len=8, chat_template=True)

    built = build_examples(tok, rows, max_seq_len=8, chat_template=True, refuse_over_length=False)
    assert built.n_over_length == 1 and built.examples == ()


def test_pack_sequences_and_chat_template_cannot_both_be_on():
    cfg = SFTConfig(base_model="m", tau=0.05, sigma_j=0.01, chat_template=True, pack_sequences=True)
    with pytest.raises(ValueError, match="pack"):
        cfg.validate()
    SFTConfig(
        base_model="m", tau=0.05, sigma_j=0.01, pack_sequences=True, chat_template=False
    ).validate()


# --------------------------------------------------------------------------- the real thing


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("PI_VLLM_URL"),
    reason="needs a running vLLM: set PI_VLLM_URL, e.g. http://127.0.0.1:8000",
)
def test_serving_tokens_equal_training_tokens():
    """The only check that the trainer's bytes are the SERVER's bytes. Everything above is a
    property of a fake template; this is the one that would catch a real template we got wrong.

    Run it on the rented box against the same `--served-model-name` the sweep will use, BEFORE
    spending anything (plan I.14)."""
    import urllib.request

    from transformers import AutoTokenizer

    base = os.environ["PI_VLLM_URL"].rstrip("/")
    model = os.environ.get("PI_VLLM_MODEL", "qwen3-8b-base")
    tok = AutoTokenizer.from_pretrained(os.environ.get("PI_VLLM_TOKENIZER", "Qwen/Qwen3-8B"))

    ex = build_chat_masked_example(tok, STATE, ACTION, enable_thinking=False)

    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": STATE}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/tokenize", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        served = json.loads(resp.read())["tokens"]

    assert served == list(ex.input_ids[: ex.n_prompt]), (
        "the server renders a different prompt than the trainer trained on. Every dev NLL would "
        "be measured on a string the policy never sees. Fix the template or the flags before "
        "any paid rollout."
    )
