"""Rung 1 computes the loss only where the loss is supervised.

THE SIZE OF THE THING. `_weighted_ce` materialised `[T, V]` logits three times -- the slice, the
`view`, and cross_entropy's own copy -- with V = 151,669 for Qwen3 and T up to `max_seq_len`.
At T = 5,120 one such tensor is 3.1 GB in fp32; the model's own `ForCausalLMLoss` adds a fourth
by calling `logits.float()` on the full sequence before it shifts. None of it is needed: the
mask this rung exists to apply leaves exactly the ACTION supervised, the action is the TAIL of
the sequence, and every logit before it is multiplied by zero.

`logits_to_keep` (transformers >= 4.45) makes the model slice the hidden states before the LM
head, so the `[T, V]` tensor is never built at all. The saving is the ratio of the prompt to the
action, which on these rows is roughly 40:1.

WHY THIS IS A CORRECTNESS TEST AND NOT A PERFORMANCE ONE. A kept span one position too short
silently drops the first supervised token from the loss -- no error, a slightly different
checkpoint, and a number nobody can trace. So the tail is derived from the LABELS rather than
from a config field, the alignment is derived from the logits the model actually RETURNED
rather than from what was asked for, and a shape that does not line up raises instead of
broadcasting. `test_the_tail_loss_equals_the_full_logits_loss` is the one that would have
caught an off-by-one; the rest keep the fallback honest.
"""

from __future__ import annotations

import pytest

from pinq_train.rung1_sft.mask import IGNORE_INDEX
from pinq_train.rung1_sft.train import (
    accepts_logits_to_keep,
    forward_for_loss,
    logits_to_keep_for,
    weighted_ce,
)

M = IGNORE_INDEX


# ------------------------------------------------------------------ how much of the tail


def test_the_kept_span_is_the_longest_action_in_the_batch_plus_one():
    """Plus one because the loss is SHIFTED: the logit that predicts the first supervised label
    sits one position before it. Keeping `n_action` exactly would drop that token."""
    labels = [
        [M, M, M, M, M, M, M, 7, 8, 9],  # 3 supervised
        [M, M, M, M, M, 4, 5, 6, 7, 8],  # 5 supervised -- the longest
    ]
    assert logits_to_keep_for(labels) == 6


def test_the_span_is_read_off_the_labels_not_off_a_length_field():
    """A row whose mask is not a clean tail (a template that supervises two spans) still has to
    be covered, and the earliest supervised position is what covers it."""
    assert logits_to_keep_for([[M, 1, M, M, 4, M]]) == 6


def test_a_fully_supervised_row_is_capped_at_the_sequence_length():
    """`logits_to_keep=T+1` is not an error in transformers -- `[-k:]` just returns everything --
    but asking for more positions than exist is a claim about the tensor that is false."""
    assert logits_to_keep_for([[1, 2, 3]]) == 3


def test_a_batch_with_nothing_supervised_takes_the_full_path():
    """Nothing to keep is not "keep nothing"; it is a batch this optimisation has no opinion
    about, and the full path is the one whose behaviour is already pinned."""
    assert logits_to_keep_for([[M, M, M]]) is None


# ------------------------------------------------------------------ what the model is told


class _Model:
    """A model whose forward signature is the thing under test."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.seen: dict = {}

    def forward(self, input_ids=None, attention_mask=None, labels=None, logits_to_keep=0):
        raise AssertionError("never called directly")

    def __call__(self, **kw):
        self.seen = kw
        return object()


class _NoKwarg(_Model):
    def forward(self, input_ids=None, attention_mask=None, labels=None):  # noqa: D102
        raise AssertionError("never called directly")


class _OnlyVarKwargs(_Model):
    def forward(self, input_ids=None, **kwargs):  # noqa: D102
        raise AssertionError("never called directly")


def _inputs():
    return {"input_ids": [[1, 2, 3, 4]], "attention_mask": [[1, 1, 1, 1]], "labels": [[M, M, 3, 4]]}


def test_a_model_that_names_the_kwarg_is_given_the_tail_and_no_labels():
    """The labels are dropped on purpose. `ForCausalLMLoss` shifts the FULL label sequence
    against whatever logits it was handed, so a tail plus full labels is a shape error -- and
    dropping them also removes the `logits.float()` copy of the full `[T, V]` block, which is
    the largest single allocation of the step and was always thrown away here."""
    m = _Model("named")
    forward_for_loss(m, _inputs())
    assert m.seen["logits_to_keep"] == 3
    assert "labels" not in m.seen
    assert m.seen["input_ids"] == [[1, 2, 3, 4]] and m.seen["attention_mask"] == [[1, 1, 1, 1]]


def test_a_model_whose_forward_does_not_take_it_gets_the_call_it_always_got():
    """`[train]` allows transformers >= 4.42 and the kwarg arrives in 4.45. The old call must
    stay reachable rather than become a TypeError on the rented box."""
    m = _NoKwarg("absent")
    assert accepts_logits_to_keep(m) is False
    forward_for_loss(m, _inputs())
    assert "logits_to_keep" not in m.seen
    assert m.seen["labels"] == [[M, M, 3, 4]], "the full path passes labels, as it always did"


def test_a_wrapper_that_forwards_kwargs_counts_as_accepting_it():
    """`PeftModelForCausalLM.forward` names input_ids, labels and `**kwargs`, and passes the
    rest down to the base model -- so a probe for the NAME alone would disable this on every
    LoRA run, which is every run this rung makes."""
    m = _OnlyVarKwargs("varkw")
    assert accepts_logits_to_keep(m) is True
    forward_for_loss(m, _inputs())
    assert m.seen["logits_to_keep"] == 3


def test_a_batch_with_nothing_supervised_is_not_asked_for_a_tail():
    m = _Model("named")
    forward_for_loss(m, {"input_ids": [[1, 2]], "labels": [[M, M]]})
    assert "logits_to_keep" not in m.seen


# ------------------------------------------------------------------ the number it produces


def _torch():
    return pytest.importorskip("torch", reason="the loss is torch; the gate venv has none")


def _batch(torch, *, n_action=(3, 5), seq=10, vocab=7):
    torch.manual_seed(0)
    logits = torch.randn(len(n_action), seq, vocab, dtype=torch.float64)
    labels = torch.full((len(n_action), seq), IGNORE_INDEX, dtype=torch.long)
    for i, n in enumerate(n_action):
        labels[i, seq - n :] = torch.randint(0, vocab, (n,))
    return logits, labels, torch.tensor([1.0, 3.0], dtype=torch.float64)


def _ce(logits, labels, weights):
    return weighted_ce(
        logits, labels, weights, per_device_batch=1, grad_accum=16, num_items_in_batch=None
    )


def test_the_tail_loss_equals_the_full_logits_loss():
    """THE PROPERTY. Two examples, two different action lengths, so the kept span is right for
    one of them and generous for the other -- which is the case an off-by-one would survive on
    a single-example batch."""
    torch = _torch()
    logits, labels, w = _batch(torch)
    keep = logits_to_keep_for(labels)
    assert keep == 6

    full = _ce(logits, labels, w)
    tail = _ce(logits[:, -keep:, :], labels, w)
    assert abs(float(full) - float(tail)) < 1e-6, f"full {float(full)!r} vs tail {float(tail)!r}"


def test_one_position_short_is_a_different_number_not_an_error():
    """WHY THE +1 IS TESTED RATHER THAN ARGUED. Dropping it does not raise; it silently trains
    on a loss missing its first supervised token per example."""
    torch = _torch()
    logits, labels, w = _batch(torch)
    short = _ce(logits[:, -(logits_to_keep_for(labels) - 1) :, :], labels, w)
    assert abs(float(short) - float(_ce(logits, labels, w))) > 1e-6


def test_a_tail_longer_than_the_labels_is_refused_rather_than_broadcast():
    """A model that returns more positions than the labels can name is a mis-wired call, and
    silently aligning it would put the loss on the wrong tokens."""
    torch = _torch()
    logits, labels, w = _batch(torch)
    with pytest.raises(ValueError, match="do not line up"):
        _ce(torch.cat([logits, logits[:, :2, :]], dim=1), labels, w)


def test_a_model_that_accepts_the_kwarg_and_ignores_it_is_merely_not_optimised():
    """The alignment is derived from the logits the model RETURNED, not from what was asked
    for, so a wrapper that swallows the kwarg costs memory and nothing else."""
    torch = _torch()
    logits, labels, w = _batch(torch)

    class _Ignores:
        def forward(self, input_ids=None, labels=None, **kwargs):
            raise AssertionError("never called directly")

        def __call__(self, **kw):
            return type("Out", (), {"logits": logits})()

    out = forward_for_loss(_Ignores(), {"input_ids": labels, "labels": labels})
    assert float(_ce(out.logits, labels, w)) == float(_ce(logits, labels, w))
