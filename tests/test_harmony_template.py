"""gpt-oss speaks harmony, not ChatML, and the difference is in the supervised span.

WHY THIS FILE EXISTS, BESIDE `test_chat_template_identity.py`. That file pins the four
properties of the Qwen/Llama (ChatML) path. gpt-oss renders a DIFFERENT shape, and three of
its differences are the kind that produce no error:

1. the assistant turn opens with a CHANNEL header -- `<|channel|>final<|message|>` -- which the
   generation prompt does NOT contain. The model emits it itself at inference, so it belongs in
   the supervised span; but it means the span is not "the action" the way it is on Qwen, and a
   test that asserts `n_action == len(tokenizer(action_json))` would be wrong about gpt-oss
   without being wrong about anything else;
2. harmony has a SECOND channel, `analysis`, which is the reasoning trace. It is absent from the
   default rendering of `{"role": "assistant", "content": ...}` -- but it appears the moment an
   assistant message carries a `thinking` key, and then it lands INSIDE the supervised span with
   no length check that would notice. Supervising it would train the policy to emit a reasoning
   trace the Inquirer's parser does not read and the paper never measures;
3. `enable_thinking` -- the Qwen kwarg `build_chat_masked_example` forwards -- is SILENTLY
   IGNORED by the harmony template (MEASURED: the rendering with it and the rendering without it
   are the same ids). A flag that is quietly ignored is how a "we disabled thinking" claim
   survives a run in which nothing was disabled. The harmony knob is `reasoning_effort`, and it
   changes the SYSTEM line rather than the assistant turn.

MEASURED against the real tokenizer (transformers 5.17.0, `openai/gpt-oss-20b`, 20 rows of
`data/rl/dev/sft.dev.sample.jsonl`): the prefix property holds 20/20 as-is; the delta is
`<|channel|>final<|message|>` + the action JSON + `<|return|>` (200005, 17196, 200008 ... 200002)
on every row; no `analysis` channel is emitted. So the prefix is not what needed fixing. What
needed fixing is that nothing asserted the other three, and that the Qwen kwarg was being
forwarded into a template that drops it.

The real-tokenizer checks at the bottom need a 27 MB download and are skipped unless
`PI_HARMONY_TOKENIZER` names a model, in the same way `test_serving_tokens_equal_training_tokens`
needs `PI_VLLM_URL`.
"""

from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import pytest

from pinq_train.rung1_sft.mask import (
    IGNORE_INDEX,
    SupervisesReasoning,
    build_chat_masked_example,
    template_family,
)
from pinq_train.rung1_sft.train import SFTConfig, resolve_lora_targets

STATE = "who directed the film and when"
ACTION = '{"action": "ASK", "question": "Who is the spouse?"}'

DEV_SAMPLE = Path(__file__).parents[1] / "data" / "rl" / "dev" / "sft.dev.sample.jsonl"


# ------------------------------------------------------------------------------- the fixture


class FakeHarmonyTokenizer:
    """The harmony shape, at the token ids the real `openai/gpt-oss-20b` tokenizer uses.

    The special ids are the REAL ones (measured: `<|start|>` 200006, `<|end|>` 200007,
    `<|return|>` 200002, `<|channel|>` 200005, `<|message|>` 200008) so that a reader comparing
    this fixture to the probe output is comparing the same numbers. The content ids are a hash,
    as in `FakeChatTokenizer`.

    `eos_token_id` IS `<|return|>` here, because on the real tokenizer it is (200002) -- unlike
    the ChatML fixture, which keeps them distinct on purpose. The property that we append nothing
    is tested on the ChatML fixture, where it can fail; here the two coincide in reality and
    pretending otherwise would be a fixture that tests a model nobody ships.
    """

    START = 200006
    END = 200007
    RETURN = 200002
    CHANNEL = 200005
    MESSAGE = 200008
    eos_token_id = 200002

    chat_template = (
        "{{- '<|start|>system<|message|>' }}{{- 'Reasoning: ' + reasoning_effort }}"
        "{{- '<|channel|>final<|message|>' }}"  # the marker `template_family` looks for
    )

    SPECIALS = {
        START: "<|start|>",
        END: "<|end|>",
        RETURN: "<|return|>",
        CHANNEL: "<|channel|>",
        MESSAGE: "<|message|>",
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        # `decode` must be the inverse of `encode`, or the fixture tests a tokenizer that loses
        # the very word -- `final`, `analysis` -- the channel guard reads. A lossy decode here
        # made the guard fire on a correct span, which is how this line came to exist.
        self.vocab: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        out = []
        for t in text.split():
            i = (zlib.crc32(t.encode()) % 30_000) + 100
            self.vocab[i] = t
            out.append(i)
        return out

    def decode(self, ids) -> str:
        return "".join(self.SPECIALS.get(int(i), self.vocab.get(int(i), "?")) for i in ids)

    def apply_chat_template(
        self,
        conversation,
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        self.calls.append(
            {"add_generation_prompt": add_generation_prompt, "tokenize": tokenize, **kwargs}
        )
        effort = str(kwargs.get("reasoning_effort", "medium"))
        ids: list[int] = [self.START, *self.encode("system"), self.MESSAGE, *self.encode(effort)]
        ids += [self.END]
        for m in conversation:
            role = str(m["role"])
            if role != "assistant":
                ids += [self.START, *self.encode(role), self.MESSAGE]
                ids += [*self.encode(str(m["content"])), self.END]
                continue
            if m.get("thinking"):
                ids += [self.START, *self.encode("assistant"), self.CHANNEL]
                ids += [*self.encode("analysis"), self.MESSAGE]
                ids += [*self.encode(str(m["thinking"])), self.END]
            ids += [self.START, *self.encode("assistant"), self.CHANNEL]
            ids += [*self.encode("final"), self.MESSAGE]
            ids += [*self.encode(str(m["content"])), self.RETURN]
        if add_generation_prompt:
            ids += [self.START, *self.encode("assistant")]
        return ids if tokenize else " ".join(str(x) for x in ids)


class ThinkingHarmonyTokenizer(FakeHarmonyTokenizer):
    """A harmony template that opens an analysis channel whether or not it was asked to.

    Not invented: this is what the real template does for an assistant message carrying a
    `thinking` key (MEASURED -- `{"role":"assistant","content":"there","thinking":"SECRET_COT"}`
    renders `<|channel|>analysis<|message|>SECRET_COT<|end|>` before the final channel). The
    failure mode it stands for is a future caller who adds rationales to the assistant message
    and thereby supervises a reasoning trace, with no length check anywhere that would notice.
    """

    def apply_chat_template(self, conversation, **kwargs):
        conv = [
            {**m, "thinking": "the user wants a question"} if m["role"] == "assistant" else m
            for m in conversation
        ]
        return super().apply_chat_template(conv, **kwargs)


# --------------------------------------------------------------------------- the family helper


def test_template_family_names_harmony_from_the_channel_marker():
    assert template_family(FakeHarmonyTokenizer()) == "harmony"


def test_template_family_names_chatml_for_a_qwen_style_template():
    from test_chat_template_identity import FakeChatTokenizer

    tok = FakeChatTokenizer()
    tok.chat_template = "{%- for m in messages %}{{- '<|im_start|>' + m.role }}{%- endfor %}"
    assert template_family(tok) == "chatml"


def test_a_tokenizer_with_no_template_at_all_is_chatml():
    """The default must be the path that has always run. A tokenizer that exposes no template
    -- a fake, an old third-party class -- must not silently acquire harmony's kwargs."""

    class NoTemplate:
        def apply_chat_template(self, conversation, **kwargs):
            return [1, 2, 3]

        eos_token_id = 2

    assert template_family(NoTemplate()) == "chatml"


def test_a_dict_valued_chat_template_is_searched_not_stringified_blindly():
    """`chat_template` may be a dict of named templates (`{"default": ..., "tool_use": ...}`).
    Reading `.startswith` or `in` on the dict itself would search the KEYS and find nothing."""

    tok = FakeHarmonyTokenizer()
    tok.chat_template = {"default": FakeHarmonyTokenizer.chat_template, "tool_use": "x"}
    assert template_family(tok) == "harmony"


# ------------------------------------------------------------------------ the supervised span


def test_a_harmony_row_supervises_the_final_channel_header_the_action_and_the_return():
    """The channel header is IN the span on purpose: the generation prompt stops at
    `<|start|>assistant`, so at inference the policy must emit `<|channel|>final<|message|>`
    itself. Masking it out would train a policy that never opens a channel, which the harmony
    parser reads as a malformed turn."""
    tok = FakeHarmonyTokenizer()
    ex = build_chat_masked_example(tok, STATE, ACTION)

    span = list(ex.input_ids[ex.n_prompt :])
    assert span[0] == FakeHarmonyTokenizer.CHANNEL
    assert span[-1] == FakeHarmonyTokenizer.RETURN
    assert span[1:3] == [*tok.encode("final"), FakeHarmonyTokenizer.MESSAGE][:2]
    assert span[3:-1] == tok.encode(ACTION)
    assert set(ex.labels[: ex.n_prompt]) == {IGNORE_INDEX}
    assert list(ex.labels[ex.n_prompt :]) == span
    assert ex.n_supervised == ex.n_action == len(span)
    ex.validate()


def test_the_harmony_prompt_is_a_strict_prefix():
    tok = FakeHarmonyTokenizer()
    ex = build_chat_masked_example(tok, STATE, ACTION)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": STATE}], add_generation_prompt=True, tokenize=True
    )
    assert list(ex.input_ids[: ex.n_prompt]) == prompt
    assert ex.n_prompt == len(prompt) and len(ex.input_ids) > len(prompt)


def test_an_analysis_channel_inside_the_supervised_span_is_refused():
    """The invariant the paper rests on: a reasoning channel is never supervised. Nothing else
    in the pipeline would see it -- `n_supervised == n_action` still holds, the prefix still
    holds, and the only symptom is a checkpoint that emits chain-of-thought at serve time."""
    with pytest.raises(SupervisesReasoning, match="analysis"):
        build_chat_masked_example(ThinkingHarmonyTokenizer(), STATE, ACTION)


# ------------------------------------------------------------------------------- the kwargs


def test_the_qwen_thinking_kwarg_is_not_forwarded_to_a_harmony_template():
    """MEASURED on `openai/gpt-oss-20b`: `apply_chat_template(..., enable_thinking=False)` returns
    ids identical to the call without it. The template has no `enable_thinking` variable at all.
    Forwarding it therefore does nothing except let a caller believe thinking was disabled."""
    tok = FakeHarmonyTokenizer()
    build_chat_masked_example(tok, STATE, ACTION, enable_thinking=False)
    assert len(tok.calls) == 2
    assert all("enable_thinking" not in c for c in tok.calls)


def test_enable_thinking_true_against_harmony_is_refused_not_dropped():
    """Dropping the DEFAULT (False) is right -- it asks for nothing. Dropping an explicit True is
    not: the caller asked for a thinking block, the template would render none, and the run would
    report a config nobody could have obtained. Refused for the same reason `reasoning_effort` is
    refused on ChatML -- in the other direction."""
    with pytest.raises(ValueError, match="enable_thinking"):
        build_chat_masked_example(FakeHarmonyTokenizer(), STATE, ACTION, enable_thinking=True)


def test_reasoning_effort_when_set_reaches_both_renderings():
    tok = FakeHarmonyTokenizer()
    build_chat_masked_example(tok, STATE, ACTION, reasoning_effort="low")
    assert [c.get("reasoning_effort") for c in tok.calls] == ["low", "low"]


def test_reasoning_effort_unset_is_not_passed_at_all():
    """`None` means "the template's own default", which is what the SERVER will use when the
    gateway sends no `chat_template_kwargs`. Passing the string "None" or guessing "medium" here
    would make the trainer's system line differ from the server's."""
    tok = FakeHarmonyTokenizer()
    build_chat_masked_example(tok, STATE, ACTION)
    assert all("reasoning_effort" not in c for c in tok.calls)


def test_reasoning_effort_is_refused_on_a_chatml_template():
    """A knob that silently does nothing is the bug this whole file is about. If a caller asks
    for a reasoning effort against a Qwen tokenizer, that is a configuration error, not a no-op."""
    from test_chat_template_identity import FakeChatTokenizer

    with pytest.raises(ValueError, match="reasoning_effort"):
        build_chat_masked_example(FakeChatTokenizer(), STATE, ACTION, reasoning_effort="low")


def test_reasoning_effort_is_in_the_config_sha():
    """If it is not in `cfg.sha`, two checkpoints trained at two efforts are one identity and no
    table can tell them apart."""
    base = dict(base_model="m", tau=0.05, sigma_j=0.01)
    a = SFTConfig(**base)
    b = SFTConfig(**base, reasoning_effort="low")
    assert a.reasoning_effort is None
    assert a.sha != b.sha


# --------------------------------------------------------------- the ChatML path is untouched


def test_qwen_rows_are_byte_identical_to_before():
    """The golden lock. These three numbers were computed on HEAD before this change; the
    harmony work is an ADDITION beside the ChatML path, never a modification of it."""
    from test_chat_template_identity import FakeChatTokenizer

    ex = build_chat_masked_example(FakeChatTokenizer(), STATE, ACTION)
    assert (ex.n_prompt, ex.n_action, len(ex.input_ids)) == (14, 8, 22)
    assert zlib.crc32(repr(list(ex.input_ids)).encode()) == 4278892001
    assert zlib.crc32(repr(list(ex.labels)).encode()) == 1204763839


# ------------------------------------------------------------------------------ LoRA targets


def test_lora_targets_for_gpt_oss_are_the_attention_projections_only():
    """MEASURED (peft 0.20.0, `GptOssForCausalLM`): the experts are FUSED 3-D parameters on a
    `GptOssExperts` module -- `mlp.experts.gate_up_proj (32, 2880, 5760)`,
    `mlp.experts.down_proj (32, 2880, 2880)` -- not `nn.Linear` submodules. No module in the
    model is named `gate_proj`, `up_proj` or `down_proj`. peft matches on MODULE names, so the
    Qwen set attaches adapters to q/k/v/o and SILENTLY DROPS the other three: `get_peft_model`
    with the Qwen set and with the attention-only set produce the same 8 `lora_A` tensors and no
    warning. The checkpoint is therefore fine either way; the recorded config is not. It would
    claim the MLP was adapted, and an ablation of "full LoRA vs attention-only LoRA" on gpt-oss
    would compare two identical runs and report a null result that is an artefact of the drop.
    """
    qwen = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    assert resolve_lora_targets("GptOssForCausalLM", qwen) == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )


def test_the_router_is_never_a_lora_target():
    """`mlp.router` IS a module (`GptOssTopKRouter`) and so peft COULD adapt it. It is excluded
    deliberately: LoRA on the router changes which experts fire, which is a different
    intervention from LoRA on the attention, and mixing the two into one arm makes the arm
    uninterpretable."""
    assert "router" not in resolve_lora_targets("GptOssForCausalLM", ("q_proj", "router"))


def test_lora_targets_for_every_other_architecture_are_the_configured_set_unchanged():
    qwen = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    for arch in ("Qwen3ForCausalLM", "LlamaForCausalLM", "", "Qwen2MoeForCausalLM"):
        assert resolve_lora_targets(arch, qwen) == qwen


def test_an_explicitly_narrowed_target_set_is_never_widened():
    """The helper only ever REMOVES what cannot match. A caller who asked for q/v only on gpt-oss
    gets q/v, not the full attention block."""
    assert resolve_lora_targets("GptOssForCausalLM", ("q_proj", "v_proj")) == ("q_proj", "v_proj")


def test_a_gpt_oss_target_set_that_would_end_up_empty_is_refused():
    """Silently training an adapter with no adapted module is a run that reports a checkpoint and
    a loss curve and has learned nothing."""
    with pytest.raises(ValueError, match="no module"):
        resolve_lora_targets("GptOssForCausalLM", ("gate_proj", "up_proj", "down_proj"))


# ---------------------------------------------------------------------------- the real thing


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("PI_HARMONY_TOKENIZER"),
    reason="needs the real tokenizer: set PI_HARMONY_TOKENIZER=openai/gpt-oss-20b",
)
def test_the_real_harmony_tokenizer_on_twenty_real_rows():
    """The only check that the fixture above is the model. Run in `.venv-train`:

        PI_HARMONY_TOKENIZER=openai/gpt-oss-20b .venv-train/bin/python -m pytest \\
            tests/test_harmony_template.py -k real_harmony -q
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.environ["PI_HARMONY_TOKENIZER"])
    assert template_family(tok) == "harmony"

    rows = [json.loads(line) for line in DEV_SAMPLE.read_text().splitlines() if line.strip()][:20]
    assert len(rows) == 20

    for r in rows:
        ex = build_chat_masked_example(tok, str(r["state_text"]), str(r["action_json"]))
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": str(r["state_text"])}],
            add_generation_prompt=True,
            tokenize=True,
        )
        prompt = list(prompt["input_ids"]) if hasattr(prompt, "keys") else list(prompt)
        assert list(ex.input_ids[: ex.n_prompt]) == prompt, "prefix broken on a real row"

        span = tok.decode(list(ex.input_ids[ex.n_prompt :]))
        assert span == "<|channel|>final<|message|>" + str(r["action_json"]) + "<|return|>", span
        assert "<|channel|>analysis" not in span
        assert list(ex.labels[ex.n_prompt :]) == list(ex.input_ids[ex.n_prompt :])
        assert set(ex.labels[: ex.n_prompt]) == {IGNORE_INDEX}


# ------------------------------------------------------------------------------ the CLI knob
#
# `reasoning_effort` in `cfg.sha` that no command line and no grid file can set is the failure
# `_resolve_options` names in its own docstring, one layer up: "a value that silently does not
# apply, under a config sha that says it did." `enable_thinking` has that shape on main and is
# left alone; the new field must not be added with it.


def _rung1_cfg(monkeypatch, tmp_path, *flags):
    """The rung-1 CLI helper from `test_train_cli`, reused rather than re-implemented.

    A second copy of "how a rung-1 command line becomes an SFTConfig" is a second thing to keep
    in step with the parser; when the two drift, the one that is wrong still passes.
    """
    from test_train_cli import _rung1_cfg as impl

    return impl(monkeypatch, tmp_path, *flags)


def test_the_reasoning_effort_flag_reaches_the_rung1_config(monkeypatch, tmp_path, capsys):
    cfg = _rung1_cfg(monkeypatch, tmp_path, "--reasoning-effort", "low")
    capsys.readouterr()
    assert cfg.reasoning_effort == "low"


def test_reasoning_effort_defaults_to_none(monkeypatch, tmp_path, capsys):
    """`None` is "the template's own default", which is what every Qwen run so far rendered and
    what the server renders when the gateway sends no `chat_template_kwargs`."""
    cfg = _rung1_cfg(monkeypatch, tmp_path)
    capsys.readouterr()
    assert cfg.reasoning_effort is None


def test_an_optional_knob_at_its_absent_default_does_not_move_the_rung1_config_sha():
    """THE IDENTITY CHANGES ONLY WHEN THE TRAINING BYTES CHANGE.

    The previous version of this test pinned the opposite belief: that adding `reasoning_effort`
    MUST move every rung-1 sha (44eee2b7... -> 9cb0f33f...). That belief was wrong for the
    reason it stated itself -- at the default `None` the code path renders exactly the bytes it
    rendered before -- and its cost was real: four rung-1 runs already trained (the pooled 8B,
    the headline 8B, its seeds) would no longer join their recorded `config_sha`. So a knob at
    its absent default is omitted from the identity (`SHA_OMIT_WHEN_NONE`), and a knob that is
    SET enters it like any other field, because then the system line does change.
    """
    cfg = SFTConfig(base_model="Qwen/Qwen3-8B", tau=0.05, sigma_j=0.01)
    assert cfg.reasoning_effort is None
    assert cfg.sha == "44eee2b7d306bdb27ae7bc3b9626c847dfb523f1f0d8b5c21bae4f83b1fdcc08"
    low = SFTConfig(base_model="Qwen/Qwen3-8B", tau=0.05, sigma_j=0.01, reasoning_effort="low")
    assert low.sha != cfg.sha
    assert (
        low.sha
        != SFTConfig(
            base_model="Qwen/Qwen3-8B", tau=0.05, sigma_j=0.01, reasoning_effort="high"
        ).sha
    )


def test_a_grid_file_may_set_the_reasoning_effort(monkeypatch, tmp_path, capsys):
    conf = tmp_path / "effort.json"
    conf.write_text(json.dumps({"reasoning_effort": "high"}) + "\n")
    cfg = _rung1_cfg(monkeypatch, tmp_path, "--config", str(conf))
    capsys.readouterr()
    assert cfg.reasoning_effort == "high"
