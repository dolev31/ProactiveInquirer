"""The loss mask, and the acceptance rule that decides what is worth imitating.

THE MASK IS THE WHOLE RUNG. An SFT example here is a long rendered state -- a task, an
evidence block that can run to thousands of tokens, a question/answer history -- followed by a
short JSON action. The state was written by the harness and the Drafter; only the action was
written by the Inquirer. If the loss is taken over the whole sequence, better than 95% of the
gradient is the model learning to reproduce retrieved documents, the checkpoint drifts toward
a document language model, and the evaluation shows a policy that got *worse* at deciding for
reasons no metric in the paper can name. Masking to the Inquirer's own tokens is not a
efficiency tweak; it is the difference between training the policy and training a parrot.

`tests/test_train_rungs.py::test_loss_mask_sums_to_the_inquirer_json_length` asserts exactly
that on a golden episode recorded by this repository: the number of supervised positions
equals the tokenizer length of the Inquirer's JSON, and not one token more.

WHY THE TOKENIZER IS A PROTOCOL. `transformers` lives behind the `[train]` extra and there is
no GPU on the development machine, so a test that needed a real tokenizer would be a test that
never runs. `Tokenizer` below is the two-method surface this module actually uses; the real
`AutoTokenizer` satisfies it, and so does a deterministic fake. That is what makes everything
except the `.train()` call testable without a GPU.

THE ACCEPTANCE RULE. Rejection-sampling SFT keeps the argmax candidate at a state only if its
value clears the judge's noise floor: `phi_max > tau + 1.5 * sigma_J * sqrt(2)`. The sqrt(2)
is there because the quantity being thresholded is a DIFFERENCE of two judged quantities and
therefore carries two independent noise draws. Below the floor the honest target is STOP --
"this question was not measurably better than not asking" is exactly what a tie should teach,
and labelling the tie a win teaches noise with a confident face.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

# The margin multiplier. 1.5 rather than 2 because the rule is a TRAINING filter, not an
# inference: a one-sided 1.5-sigma screen keeps roughly the top decile of ties out of the
# dataset while still admitting enough accepted states to train on. The reported statistics
# use the full 2-sigma noise floor; these two numbers are different on purpose.
MARGIN_SIGMAS = 1.5

# Two independent judge draws enter a difference of judged values.
NOISE_SQRT2 = math.sqrt(2.0)

# The label value that means "no loss here". torch's cross_entropy ignore_index default.
IGNORE_INDEX = -100


class Tokenizer(Protocol):
    """The two things this module needs from a tokenizer. `AutoTokenizer` satisfies it."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    @property
    def eos_token_id(self) -> int | None: ...


def margin_threshold(tau: float, sigma_j: float) -> float:
    """`tau + 1.5 * sigma_J * sqrt(2)`: the value an accepted candidate must clear.

    `tau` is the formalism's own stopping threshold (`RewardWeights.tau`, the 35th percentile
    of pilot phi_LOO) and `sigma_j` is the judge's measured standard deviation. Both are
    MEASURED inputs; there is deliberately no default for either, because a default would let
    a threshold that was never estimated decide what a checkpoint imitates.
    """
    if sigma_j < 0:
        raise ValueError(f"sigma_j={sigma_j}: a standard deviation cannot be negative")
    return tau + MARGIN_SIGMAS * sigma_j * NOISE_SQRT2


@dataclass(frozen=True, slots=True)
class MaskedExample:
    """One SFT row: the full sequence, and the labels with everything but the action masked."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    n_prompt: int
    n_action: int
    # The exporter's per-row weight, 1/sqrt(rows in that task and kind) normalised to mean 1.
    # APPENDED with a default so every existing construction is unchanged and the two golden mask
    # tests stay byte-identical. 1.0 means "unweighted", which is also what a row that predates
    # the field gets -- and `dataset_report` counts those, because a loss that silently treats an
    # unknown weight as 1.0 is a loss nobody can audit.
    weight: float = 1.0

    @property
    def n_supervised(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE_INDEX)

    def validate(self) -> None:
        if len(self.input_ids) != len(self.labels):
            raise ValueError(
                f"{len(self.input_ids)} input ids but {len(self.labels)} labels: a shifted "
                "mask supervises the wrong tokens and nothing downstream can see it."
            )
        if self.n_supervised != self.n_action:
            raise ValueError(
                f"{self.n_supervised} supervised positions but the action is {self.n_action} "
                "tokens. The mask is the rung; a mismatch here trains on the evidence block."
            )


def build_masked_example(
    tok: Tokenizer,
    state_text: str,
    action_json: str,
    *,
    append_eos: bool = True,
) -> MaskedExample:
    """Concatenate state + action, and supervise ONLY the action.

    `add_special_tokens=False` on both halves: a BOS inserted in the middle of a concatenated
    sequence is a token the model never sees at inference, and a second BOS at the join is the
    classic way a masked-SFT pipeline learns an off-by-one.

    The EOS is supervised when present, because "stop emitting" is part of the action the
    policy must learn -- a policy that never emits EOS produces JSON followed by improvised
    commentary, which the parser scores as malformed.
    """
    prompt_ids = list(tok.encode(state_text, add_special_tokens=False))
    action_ids = list(tok.encode(action_json, add_special_tokens=False))
    eos = tok.eos_token_id if append_eos else None
    if eos is not None:
        action_ids = action_ids + [int(eos)]
    input_ids = prompt_ids + action_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(action_ids)
    ex = MaskedExample(
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        n_prompt=len(prompt_ids),
        n_action=len(action_ids),
    )
    ex.validate()
    return ex


class ChatTokenizer(Protocol):
    """A tokenizer that can render a message list the way the SERVER will render it.

    Two members, because two are all this module uses. The real `AutoTokenizer` satisfies it and
    so does a deterministic fake, which is what keeps the templated path testable on a laptop with
    neither a GPU nor `transformers` installed.
    """

    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, str]],
        *,
        add_generation_prompt: bool = ...,
        tokenize: bool = ...,
        **kwargs: Any,
    ) -> list[int] | str: ...

    @property
    def eos_token_id(self) -> int | None: ...


def _template_ids(rendered: Any) -> list[int]:
    """The token ids out of whatever `apply_chat_template(tokenize=True)` returned.

    IT RETURNS TWO DIFFERENT THINGS DEPENDING ON THE INSTALLED transformers. Its `return_dict`
    parameter defaults to False on 4.x (a flat `list[int]`) and to True on 5.x (a `BatchEncoding`
    mapping with `input_ids` and `attention_mask`). `list(...)` on the second yields the KEY
    NAMES, so both halves of `build_chat_masked_example` came back as
    `['input_ids', 'attention_mask']`: equal lengths, and every row raised "the assistant turn
    added no tokens". MEASURED on transformers 5.17.0 + Qwen/Qwen3-0.6B: 300 of 300 smoke rows.

    Normalised here rather than by passing `return_dict=False`, because that kwarg does not
    exist on 4.x-era third-party tokenizers and the fakes this module is tested against accept
    `**kwargs` silently -- a flag that is quietly ignored is how this class of bug returns.

    The BATCHED shape is refused rather than unwrapped. A `[[...]]` from a future default would
    make `len()` 1 for the prompt and 1 for the full rendering, which lands in the same
    equal-lengths hole this function exists to climb out of; there is no reading of a batch of
    one that is safe to guess at.
    """
    if isinstance(rendered, Mapping) or hasattr(rendered, "keys"):
        ids = rendered["input_ids"]
    else:
        ids = rendered
    out = list(ids)
    if out and not isinstance(out[0], int):
        raise TemplateNotPrefix(
            f"apply_chat_template(tokenize=True) yielded {type(out[0]).__name__} rather than "
            f"int (first element {out[0]!r}). A mask built from the LENGTH of that is a mask "
            "over something that is not a token sequence, and every count in the report would "
            "still look right."
        )
    return out


class TemplateNotPrefix(ValueError):
    """The generation prompt is not a prefix of the full rendering.

    This is refused rather than worked around. The mask is built from `len(prompt_ids)`; if the
    template inserted, moved or dropped a token at the join, that length indexes into the wrong
    place and the loss supervises part of the header while `n_supervised` still equals `n_action`
    and every count in the report still looks right. There is no downstream check that would see
    it -- so it is caught here, where it is a one-line comparison, or it is never caught at all.
    """


class SupervisesReasoning(ValueError):
    """A reasoning channel landed inside the supervised span.

    Harmony (gpt-oss) assistant turns carry CHANNELS: `analysis` is the reasoning trace,
    `commentary` is tool preamble, `final` is the answer. Only `final` is the Inquirer's action.
    A row whose assistant message carries a `thinking` key renders
    `<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...`,
    and every check in this module still passes: the prompt is still a prefix, `n_supervised`
    still equals `n_action`, the length is still plausible. The only symptom is a checkpoint that
    learned to emit chain-of-thought the Inquirer's parser does not read -- discovered, if at all,
    as an unexplained drop in parse rate weeks later.
    """


class OverLength(ValueError):
    """A templated example does not fit the context, and must not be truncated into one.

    `truncate_left` drops tokens from the FRONT, which on a templated sequence is the chat header
    (`<|im_start|>user`). A row missing its header is a row the server can never produce, so
    training on it reintroduces exactly the train/serve gap the template exists to close. The
    honest options are to refuse the row or to raise `max_seq_len`; silently reshaping it is not
    one of them.
    """


# The marker that separates the two chat formats this repository trains against. ChatML
# (Qwen, Llama) wraps turns in `<|im_start|>role ... <|im_end|>`; harmony (gpt-oss) wraps them
# in `<|start|>role<|channel|>NAME<|message|> ... <|end|>` and the channel is the whole
# difference -- it is what makes "the assistant turn" and "the action" two different spans.
# Detected from the TEMPLATE TEXT rather than from the model name, because the name is a string
# a caller chose (`--base-model /proj/pinq/user/hf/...`) and the template is what will actually
# render. A local path, a fine-tuned copy or a renamed served model all keep their template.
HARMONY_CHANNEL_MARKER = "<|channel|>"
CHATML = "chatml"
HARMONY = "harmony"


def template_family(tok: Any) -> str:
    """`"harmony"` or `"chatml"`, from the tokenizer's own chat template.

    PURE, and defaults to `chatml` -- the path that has always run. A tokenizer that exposes no
    template (a fake, an older third-party class) must not silently acquire harmony's kwargs and
    harmony's channel guard; it gets exactly the behaviour it had before this function existed.

    `chat_template` may be a `dict` of named templates (`{"default": ..., "tool_use": ...}`).
    Testing `marker in that_dict` searches the KEYS and quietly reports `chatml` for a harmony
    model, so the values are joined first.
    """
    tpl = getattr(tok, "chat_template", None)
    if isinstance(tpl, Mapping):
        tpl = "\n".join(str(v) for v in tpl.values())
    return HARMONY if HARMONY_CHANNEL_MARKER in str(tpl or "") else CHATML


def _harmony_template_kwargs(
    *, enable_thinking: bool, reasoning_effort: str | None
) -> dict[str, Any]:
    """The kwargs a harmony template actually reads. Pure, so the refusals are testable.

    `enable_thinking` IS NOT ONE OF THEM. MEASURED on `openai/gpt-oss-20b` (transformers 5.17.0):
    the template contains no `enable_thinking` variable, and `apply_chat_template` with it
    returns ids identical to the call without it. Forwarding it is therefore not harmless -- it
    is how a caller comes to believe thinking was disabled on a run where nothing was. So the
    default (False) is dropped, and an explicit request for True is REFUSED rather than ignored.

    `reasoning_effort` is harmony's knob, and it changes the SYSTEM line ("Reasoning: medium"),
    not the assistant turn. `None` means "whatever the template defaults to", which is what the
    server will also use when the gateway sends no `chat_template_kwargs` -- so `None` is passed
    by omitting the kwarg, never by sending the string "None" or by guessing "medium" here.
    """
    if enable_thinking:
        raise ValueError(
            "enable_thinking=True against a harmony (gpt-oss) template. That template has no "
            "such variable and would drop it silently, leaving a run that claims a thinking "
            "block it never rendered. Harmony's knob is reasoning_effort."
        )
    return {} if reasoning_effort is None else {"reasoning_effort": str(reasoning_effort)}


def _refuse_supervised_reasoning(tok: Any, span: Sequence[int]) -> None:
    """The supervised span must open the FINAL channel and open no other.

    Structural rather than textual: a good span decodes to
    `<|channel|>final<|message|>{action}<|return|>` -- exactly one channel marker, at position 0,
    named `final`. An analysis channel puts a second marker in the span (or moves the first one's
    name), and that is the only signal there is; the lengths and the prefix stay correct.

    `decode` is required rather than probed for. A harmony tokenizer without `decode` is not a
    tokenizer, and skipping the check when the method is missing is how a guard comes to be
    disabled on exactly the tokenizer it was written for.
    """
    decode = getattr(tok, "decode", None)
    if decode is None:
        raise SupervisesReasoning(
            "a harmony tokenizer with no `decode`: the channel guard cannot run, and an "
            "unrunnable guard must not pass by default."
        )
    text = str(decode(list(span)))
    n = text.count(HARMONY_CHANNEL_MARKER)
    head = text.split(HARMONY_CHANNEL_MARKER, 1)[-1].split("<|message|>", 1)[0]
    if n != 1 or not text.startswith(HARMONY_CHANNEL_MARKER) or head != "final":
        raise SupervisesReasoning(
            f"the supervised span opens {n} channel(s), the first named {head!r}, and the span "
            f"starts with {text[:40]!r}. Only the `final` channel is the Inquirer's action; an "
            "`analysis` (or `commentary`) channel in this span trains the policy to emit a "
            "reasoning trace that nothing downstream parses, and no length or prefix check "
            "would ever notice."
        )


def build_chat_masked_example(
    tok: ChatTokenizer,
    state_text: str,
    action_json: str,
    *,
    enable_thinking: bool = False,
    reasoning_effort: str | None = None,
) -> MaskedExample:
    """Render state and action THROUGH THE CHAT TEMPLATE, and supervise only the action.

    THE POINT IS BYTE IDENTITY WITH THE SERVER. `build_masked_example` concatenates two raw
    strings; vLLM's `/v1/chat/completions` renders role headers, an end-of-turn token and (on
    Qwen3) an optional thinking block around them. Training on the first and serving the second
    means every dev NLL is measured on a prompt the policy never sees -- a discrepancy that
    produces no error and no failing test anywhere else. `tests/test_chat_template_identity.py`
    pins the four properties that make the two the same string, and its `PI_VLLM_URL` integration
    test compares these ids against a real server's `/tokenize` before any paid rollout.

    NOTHING IS APPENDED. The last supervised token is whatever end-of-turn id the template emitted
    -- not `eos_token_id`, which on some models is a different token. Appending our own would
    teach the policy to emit a token the server's stop criteria do not watch for.

    `enable_thinking` goes to BOTH renderings. Forwarding it to one only is worse than forwarding
    it to neither: the prompt would then promise a thinking block that the target does not open.

    TWO FAMILIES, ONE MASK. `template_family` decides which kwargs the template actually reads,
    because the two have different reasoning knobs and forwarding the wrong one is a silent no-op
    rather than an error (see `_harmony_template_kwargs`). Everything after the kwargs -- the
    prefix check, the mask, `MaskedExample` -- is identical for both; only harmony gets the extra
    channel guard, because only harmony HAS channels.

    MEASURED on `openai/gpt-oss-20b` (transformers 5.17.0), 20 rows of `sft.dev.sample.jsonl`:
    the prefix holds 20/20 as-is, the prompt ends `...<|end|><|start|>assistant`, and the span is
    `<|channel|>final<|message|>` + the action + `<|return|>`. The channel header is INSIDE the
    span on purpose: the generation prompt stops before it, so the policy must emit it itself.
    """
    family = template_family(tok)
    if family == HARMONY:
        kwargs = _harmony_template_kwargs(
            enable_thinking=enable_thinking, reasoning_effort=reasoning_effort
        )
    elif reasoning_effort is not None:
        raise ValueError(
            f"reasoning_effort={reasoning_effort!r} against a {family} template, which has no "
            "such variable and would drop it. A knob that silently does nothing is worse than "
            "an absent one: it makes a run look configured. Use enable_thinking on ChatML."
        )
    else:
        kwargs = {"enable_thinking": enable_thinking}
    msgs = [{"role": "user", "content": state_text}]
    prompt_ids = _template_ids(
        tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            **kwargs,
        )
    )
    full_ids = _template_ids(
        tok.apply_chat_template(
            [*msgs, {"role": "assistant", "content": action_json}],
            add_generation_prompt=False,
            tokenize=True,
            **kwargs,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise TemplateNotPrefix(
            f"the {len(prompt_ids)}-token generation prompt is not a prefix of the "
            f"{len(full_ids)}-token full rendering. The mask is built from the prompt's LENGTH, "
            "so a template that rewrites the join would supervise part of the chat header while "
            "every count in the report still looks correct. Fix the template or the flags; do "
            "not mask around it."
        )
    n_action = len(full_ids) - len(prompt_ids)
    if n_action <= 0:
        raise TemplateNotPrefix(
            "the assistant turn added no tokens: the full rendering equals the prompt. There is "
            "nothing to supervise, and a row with zero supervised positions trains on nothing "
            "while still counting as an example."
        )
    if family == HARMONY:
        _refuse_supervised_reasoning(tok, full_ids[len(prompt_ids) :])
    ex = MaskedExample(
        input_ids=tuple(full_ids),
        labels=tuple([IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]),
        n_prompt=len(prompt_ids),
        n_action=n_action,
    )
    ex.validate()
    return ex


def truncate_left(ex: MaskedExample, max_len: int) -> MaskedExample:
    """Drop tokens from the FRONT of the state when a sequence exceeds the context.

    Left truncation, never right: the action lives at the right-hand end, and truncating there
    removes the only supervised tokens in the example while leaving a row that still trains.
    The evidence block's head is the least load-bearing part of the state -- units are sorted
    by uid, so no position in it is privileged.
    """
    if max_len <= 0:
        raise ValueError(f"max_len={max_len}: must be positive")
    if len(ex.input_ids) <= max_len:
        return ex
    if ex.n_action > max_len:
        raise ValueError(
            f"the action alone is {ex.n_action} tokens, over a {max_len}-token window. "
            "Truncating it would supervise half a JSON object."
        )
    cut = len(ex.input_ids) - max_len
    return MaskedExample(
        input_ids=ex.input_ids[cut:],
        labels=ex.labels[cut:],
        n_prompt=ex.n_prompt - cut,
        n_action=ex.n_action,
    )


def pack(examples: Sequence[MaskedExample], max_len: int) -> list[tuple[list[int], list[int]]]:
    """Greedy sequence packing into `max_len` windows. Returns (input_ids, labels) pairs.

    Packing is first-fit in ARRIVAL order, not best-fit: reordering examples to fill windows
    tighter would make the epoch's example order depend on length, and length correlates with
    evidence size, which correlates with task difficulty. A cheaper epoch is not worth a
    curriculum nobody chose.
    """
    out: list[tuple[list[int], list[int]]] = []
    ids: list[int] = []
    labs: list[int] = []
    for ex in examples:
        if len(ex.input_ids) > max_len:
            ex = truncate_left(ex, max_len)
        if ids and len(ids) + len(ex.input_ids) > max_len:
            out.append((ids, labs))
            ids, labs = [], []
        ids.extend(ex.input_ids)
        labs.extend(ex.labels)
    if ids:
        out.append((ids, labs))
    return out
