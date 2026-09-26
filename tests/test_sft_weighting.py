"""The exporter weighted the rows. This is the test that the trainer honours the weights.

WHY A WEIGHT AT ALL. Three musique tasks contribute 107 ASK states each and most tasks contribute
two or three. Unweighted, the policy spends most of its gradient learning those three tasks'
phrasing; the alternative the exporter rejected -- a per-task cap -- deleted 65% of the ASK rows,
every one of them a state no kept row represented. So `sample_weight` is 1/sqrt(rows in that task
and kind), normalised to mean 1 within the kind.

WHY THE ARITHMETIC IS TESTED HERE AND NOT ON THE GPU. `sum(w*m)/sum(w)` does NOT decompose across
micro-batches, and at the shipped hyperparameters (batch 1 x accumulation 16) each micro-batch
holds exactly one example -- so a naive per-micro-batch weighted mean computes `w*m/w = m` and the
weights cancel EXACTLY. That is a silent no-op: the loss curve is unchanged, every count is
unchanged, and the checkpoint is simply not the one the plan describes.
`test_sixteen_accumulated_micro_batches_equal_one_batch_of_sixteen` is the test that fails when
that happens, and it runs on a laptop because the two functions it exercises are pure.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from pinq.actions import STOP_ACTION_JSON, ask_action_json
from pinq_train.rung1_sft.mask import MaskedExample
from pinq_train.rung1_sft.train import (
    SFTConfig,
    accum_scale,
    build_examples,
    dataset_report,
    preflight,
    row_weight,
    weighted_loss,
)

SFT_JSONL = Path(__file__).resolve().parents[1] / "data" / "rl" / "sft.jsonl"


class FakeTokenizer:
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [len(t) for t in text.split()]
        return ([1] + ids) if add_special_tokens else ids


# ----------------------------------------------------------------------- the pure arithmetic


def test_weighted_loss_is_the_weight_normalised_mean():
    # 2*1.0 + 1*4.0 == 6.0, over weights summing to 3 -> 2.0. Computed by hand, not by the code
    # under test: a test that recomputes the implementation cannot disagree with it.
    assert weighted_loss([1.0, 4.0], [2.0, 1.0]) == pytest.approx(2.0)
    assert weighted_loss([3.0], [7.5]) == pytest.approx(3.0)


def test_weighted_loss_equals_the_plain_mean_when_every_weight_is_one():
    means = [0.5, 1.5, 2.5, 3.5]
    assert weighted_loss(means, [1.0] * 4) == pytest.approx(sum(means) / 4)


def test_weighted_loss_refuses_a_misaligned_zip_and_a_non_positive_total():
    with pytest.raises(ValueError, match="misaligned|weights"):
        weighted_loss([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="empty batch|no examples"):
        weighted_loss([], [])
    with pytest.raises(ValueError, match="sign of the gradient|sum to"):
        weighted_loss([1.0, 2.0], [1.0, -1.0])


def test_sixteen_accumulated_micro_batches_equal_one_batch_of_sixteen():
    """THE test. Sixteen accumulated micro-batches of one must equal one batch of sixteen.

    Both regimes of `num_items_in_batch` are checked, because they differ in whether the HF
    Trainer divides by `gradient_accumulation_steps` and getting that backwards scales the whole
    loss by 16 -- which looks like a learning-rate problem and is not one.
    """
    means = [0.1 * (i + 1) for i in range(16)]
    # Mean exactly 1, the way the exporter normalises. Eight heavy rows, eight light ones.
    weights = [1.5] * 8 + [0.5] * 8
    assert sum(weights) == pytest.approx(16.0)

    one_batch = weighted_loss(means, weights)

    # num_items_in_batch PRESENT: the Trainer does not divide, so the scale carries N_window.
    accumulated = sum(
        weighted_loss([m], [w])
        * accum_scale([w], per_device_batch=1, grad_accum=16, num_items_in_batch=4096)
        for m, w in zip(means, weights)
    )
    assert accumulated == pytest.approx(one_batch)

    # num_items_in_batch ABSENT: the Trainer divides by grad_accum, so we must not.
    raw = sum(
        weighted_loss([m], [w])
        * accum_scale([w], per_device_batch=1, grad_accum=16, num_items_in_batch=None)
        for m, w in zip(means, weights)
    )
    assert raw / 16 == pytest.approx(one_batch)


def test_the_weights_do_not_cancel_at_a_micro_batch_of_one():
    """The regression guard for the silent no-op. If `accum_scale` ever returns 1.0 regardless of
    the weight, the accumulated loss collapses to the UNWEIGHTED mean and nothing else notices."""
    means = [0.0] * 8 + [1.0] * 8
    heavy_on_the_ones = [0.5] * 8 + [1.5] * 8
    accumulated = sum(
        weighted_loss([m], [w])
        * accum_scale([w], per_device_batch=1, grad_accum=16, num_items_in_batch=1)
        for m, w in zip(means, heavy_on_the_ones)
    )
    assert accumulated == pytest.approx(0.75)
    assert accumulated != pytest.approx(sum(means) / 16), "the weights cancelled: 0.75 != 0.5"


def test_accum_scale_refuses_a_degenerate_batch_shape():
    with pytest.raises(ValueError, match=">= 1"):
        accum_scale([1.0], per_device_batch=0, grad_accum=16, num_items_in_batch=None)


# --------------------------------------------------------------------- the default, and its count


def test_a_row_without_a_sample_weight_defaults_to_one_and_is_counted():
    """1.0 is a DEFAULT, not a measurement. `accum_scale`'s correctness rests on the dataset's
    mean weight being 1, so rows that were defaulted are the one way it quietly stops holding --
    which makes an uncounted default worse than a missing one."""
    rows = [
        {"state_text": "a b c", "action_json": '{"x":1}', "sample_weight": 0.5},
        {"state_text": "d e f", "action_json": '{"x":2}'},
    ]
    built = build_examples(FakeTokenizer(), rows, max_seq_len=64)
    assert [e.weight for e in built.examples] == [0.5, 1.0]

    # Varied ASK rows, because `dataset_report` runs the diversity gate FIRST: an all-STOP
    # fixture raises `ModeCollapse` before the count is ever reached, and the count is what this
    # test is about.
    stems = ["who founded {}", "when did {} open", "where is {} located", "which firm bought {}"]
    subjects = "Arclight Bellhaven Corvid Dunmore Everly Fairwood Glenmoor Hollis".split()
    dataset = []
    for i in range(40):
        row = {
            "suite_id": "musique",
            "task_id": f"t{i // 4}",
            "is_stop": False,
            "action_json": json.dumps(
                {"action": "ASK", "question": f"{stems[i % 4].format(subjects[(i // 4) % 8])}?"}
            ),
        }
        if i >= 3:  # the first three predate the field
            row["sample_weight"] = 1.0
        dataset.append(row)
    rep = dataset_report(dataset)
    assert rep["n_unweighted_rows"] == 3
    assert rep["n_questions"] == 40


def test_an_explicit_null_weight_is_defaulted_rather_than_crashing():
    """`row.get("sample_weight", 1.0)` returns None when the key EXISTS and is null, and
    `float(None)` raises. That is a TypeError on row 30,000 of a 44,624-row file, three hours into
    a rented GPU -- the most expensive place in this project to discover a one-line bug."""
    rows = [{"state_text": "a b", "action_json": '{"x":1}', "sample_weight": None}]
    built = build_examples(FakeTokenizer(), rows, max_seq_len=64)
    assert [e.weight for e in built.examples] == [1.0]


def test_use_sample_weight_off_trains_every_row_at_one():
    rows = [{"state_text": "a b c", "action_json": '{"x":1}', "sample_weight": 0.25}]
    built = build_examples(FakeTokenizer(), rows, max_seq_len=64, use_sample_weight=False)
    assert [e.weight for e in built.examples] == [1.0]


def test_the_masked_example_weight_is_appended_and_defaults_to_one():
    ex = MaskedExample(input_ids=(1, 2), labels=(-100, 2), n_prompt=1, n_action=1)
    assert ex.weight == 1.0
    assert dataclasses.replace(ex, weight=0.5).weight == 0.5


# ------------------------------------------------------------------------------- the config sha


@pytest.mark.parametrize(
    "field,value",
    [
        ("chat_template", False),
        ("enable_thinking", True),
        ("pack_sequences", True),
        ("use_sample_weight", False),
        ("bf16", False),
        ("gradient_checkpointing", False),
        ("include_arms", ("inquirer_prompted", "inquirer_may_ask_user")),
    ],
)
def test_every_new_config_field_changes_the_config_sha(field, value):
    """`cfg.sha` is the checkpoint's identity. A field that changes what is trained and does not
    change the sha makes two different checkpoints indistinguishable in the manifest -- which is
    rule 1 of this repository failing quietly rather than loudly."""
    base = SFTConfig(base_model="m", tau=0.05, sigma_j=0.01)
    assert getattr(base, field) != value, f"{field}: the fixture must actually flip it"
    assert dataclasses.replace(base, **{field: value}).sha != base.sha


def test_two_identical_configs_agree_on_the_sha():
    a = SFTConfig(base_model="m", tau=0.05, sigma_j=0.01)
    b = SFTConfig(base_model="m", tau=0.05, sigma_j=0.01)
    assert a.sha == b.sha


def test_the_defaults_are_the_plans():
    cfg = SFTConfig(base_model="m", tau=0.05, sigma_j=0.01)
    assert cfg.chat_template is True and cfg.enable_thinking is False
    assert cfg.pack_sequences is False and cfg.use_sample_weight is True
    assert cfg.bf16 is True and cfg.gradient_checkpointing is True
    assert cfg.include_arms == ("inquirer_prompted",)


# --------------------------------------------------------------- the real file, when it is here


@pytest.mark.skipif(not SFT_JSONL.exists(), reason="no exported SFT dataset in this checkout")
def test_the_exported_dataset_has_mean_weight_one():
    """`accum_scale` normalises the accumulation window by `per_device_batch * grad_accum` instead
    of by the window's own weight sum, and that substitution is only valid because the exporter
    normalises `sample_weight` to mean 1. This is the measurement that licenses it.

    Read STREAMING: the file is 415 MB and `read_jsonl` would load all of it."""
    n = 0
    total = 0.0
    with SFT_JSONL.open() as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            total += float(json.loads(line).get("sample_weight", 1.0))
    assert n > 0
    assert abs(total / n - 1.0) < 1e-6, f"mean weight {total / n!r} over {n} rows"


# --------------------------------------------------- the STOP multiplier, and what it is for
#
# THE MEASUREMENT THAT ASKS FOR IT. Every rung-1 SFT checkpoint stops far earlier than the
# prompted base when it is run live -- mean asks 3.0-3.5 against 6.25 on musique dev, 1.3-1.5
# against 5.6 on strategyqa -- and loses evidence coverage doing it. Lowering the STOP SHARE OF
# THE FILE from 68% to 39% did not move it, which rules out "there are too many STOP rows" as
# stated and leaves "the STOP rows carry too much of the GRADIENT" unmeasured. `stop_weight` is
# the direct lever on the second: it multiplies every STOP row's `sample_weight`, so the arm is
# one number and the number is in `cfg.sha`.
#
# WHICH FIELD SAYS A ROW IS A STOP. The ACTION BYTES, through `pinq.actions.action_kind_of` --
# the same call the exporter's `_is_stop_row` makes, and for the same reason: the bytes are what
# the mask supervises, so a weight keyed on anything else can scale the loss of a row whose
# target is an ASK. `label_rule` is "" on every row written before the stop rule existed and
# `is_stop` is a flag that is set beside the bytes rather than derived from them. MEASURED on
# `data/rl/sft.jsonl` (43,837 rows): all three agree on all 43,837, so this is a choice about
# which field can never drift, not a disagreement about today's corpus.


def _stop_row(weight: float = 1.0) -> dict:
    """An exported STOP row. It carries `is_stop` and `label_rule` because every real one does
    (they are dataclass fields on `export.dataset.Example`), and because the report prints a
    count from each predicate: a fixture that omitted them would make the two agree vacuously."""
    return {
        "state_text": "a b c",
        "action_json": STOP_ACTION_JSON,
        "is_stop": True,
        "label_rule": "stop_done",
        "sample_weight": weight,
    }


def _ask_row(i: int, weight: float = 1.0) -> dict:
    return {
        "state_text": "a b c",
        "action_json": ask_action_json(f"who founded body {i} in {i}"),
        "is_stop": False,
        "label_rule": "ask_clears_floor",
        "sample_weight": weight,
    }


def test_a_stop_weight_scales_stop_rows_and_leaves_ask_rows_alone():
    rows = [_stop_row(1.2), _ask_row(1, 1.2), _stop_row(0.5), _ask_row(2, 0.5)]
    built = build_examples(FakeTokenizer(), rows, max_seq_len=64, stop_weight=0.25)
    assert [e.weight for e in built.examples] == pytest.approx([0.3, 1.2, 0.125, 0.5])

    # None is the absent default and must change nothing at all.
    same = build_examples(FakeTokenizer(), rows, max_seq_len=64, stop_weight=None)
    assert [e.weight for e in same.examples] == pytest.approx([1.2, 1.2, 0.5, 0.5])


def test_which_field_says_a_row_is_a_stop_is_the_action_bytes():
    """The bytes are what the mask supervises. A row that CLAIMS to be a STOP in a metadata
    field while its action is an ASK would otherwise have the ASK it teaches down-weighted."""
    lying_ask = {
        "state_text": "a b c",
        "action_json": ask_action_json("who founded it"),
        "is_stop": True,
        "label_rule": "stop_done",
        "sample_weight": 1.0,
    }
    # A STOP written before `label_rule` existed: "" there, and no flag at all.
    legacy_stop = {"state_text": "a b c", "action_json": STOP_ACTION_JSON, "sample_weight": 1.0}
    built = build_examples(
        FakeTokenizer(), [lying_ask, legacy_stop], max_seq_len=64, stop_weight=0.5
    )
    assert [e.weight for e in built.examples] == pytest.approx([1.0, 0.5])


def test_row_weight_is_the_one_place_the_two_multipliers_meet():
    """`preflight` reports the share of the loss the STOP rows take; `build_examples` decides
    it. Both go through `row_weight`, so the number in the manifest is the number trained."""
    assert row_weight(_stop_row(0.8), use_sample_weight=True, stop_weight=0.5) == pytest.approx(0.4)
    assert row_weight(_stop_row(0.8), use_sample_weight=True, stop_weight=None) == pytest.approx(
        0.8
    )
    # `use_sample_weight=False` is "train every row at 1.0", and that includes the STOP rows:
    # the trainer does not even pass a weight column in that mode.
    assert row_weight(_stop_row(0.8), use_sample_weight=False, stop_weight=0.5) == 1.0
    # An explicit null is a row the exporter could not weight -- defaulted, then multiplied.
    null = {"action_json": STOP_ACTION_JSON, "sample_weight": None}
    assert row_weight(null, use_sample_weight=True, stop_weight=0.5) == pytest.approx(0.5)


def test_sixteen_micro_batches_still_equal_one_batch_of_sixteen_under_a_stop_weight():
    """The accumulation-window normalisation has to survive the multiplier.

    THE EXISTING EQUALITY TEST CHOSE ITS WEIGHTS TO SUM TO EXACTLY 16, which is what lets it
    compare against a bare `weighted_loss`. A stop weight is precisely the thing that breaks
    that: the window's weight sum is no longer `per_device_batch * grad_accum`. So the
    invariant is stated where it is actually true -- the WINDOW loss `sum(w*m)/N_window` is the
    same number whether it is taken in sixteen pieces or in one -- and the fixture asserts that
    the sum really did move, or the test would be passing for the old reason.
    """
    rows = [_stop_row() for _ in range(8)] + [_ask_row(i) for i in range(8)]
    weights = [row_weight(r, use_sample_weight=True, stop_weight=0.5) for r in rows]
    means = [0.1 * (i + 1) for i in range(16)]
    assert sum(weights) == pytest.approx(12.0), "8 STOP at 0.5 + 8 ASK at 1.0"

    # ONE step over all sixteen: per_device_batch 16, grad_accum 1.
    one_step = weighted_loss(means, weights) * accum_scale(
        weights, per_device_batch=16, grad_accum=1, num_items_in_batch=4096
    )
    assert one_step == pytest.approx(sum(w * m for w, m in zip(weights, means)) / 16)

    accumulated = sum(
        weighted_loss([m], [w])
        * accum_scale([w], per_device_batch=1, grad_accum=16, num_items_in_batch=4096)
        for m, w in zip(means, weights)
    )
    assert accumulated == pytest.approx(one_step)

    # num_items_in_batch ABSENT: the Trainer divides by grad_accum, so we must not.
    raw = sum(
        weighted_loss([m], [w])
        * accum_scale([w], per_device_batch=1, grad_accum=16, num_items_in_batch=None)
        for m, w in zip(means, weights)
    )
    assert raw / 16 == pytest.approx(one_step)


def test_a_stop_weight_at_its_absent_default_does_not_move_the_config_sha():
    """An optional knob at its absent default renders the bytes the older code rendered, so it
    must not unjoin the rung-1 runs already trained. The constant is the one
    `tests/test_harmony_template.py` pins for the same reason."""
    cfg = SFTConfig(base_model="Qwen/Qwen3-8B", tau=0.05, sigma_j=0.01)
    assert cfg.stop_weight is None
    assert cfg.sha == "44eee2b7d306bdb27ae7bc3b9626c847dfb523f1f0d8b5c21bae4f83b1fdcc08"


def test_a_set_stop_weight_is_part_of_the_run_identity():
    """Two checkpoints trained at two stop weights are two experiments. If the sha did not
    move, the ablation's two arms would join one row in every table that keys on it."""
    base = SFTConfig(base_model="Qwen/Qwen3-8B", tau=0.05, sigma_j=0.01)
    half = dataclasses.replace(base, stop_weight=0.5)
    quarter = dataclasses.replace(base, stop_weight=0.25)
    assert half.sha != base.sha
    assert quarter.sha != base.sha
    assert half.sha != quarter.sha
    # 1.0 is a no-op ARITHMETICALLY and a different run DELIBERATELY: it says the knob was
    # considered and set, which the manifest has to be able to report.
    assert dataclasses.replace(base, stop_weight=1.0).sha != base.sha


def test_a_stop_weight_that_could_not_reach_the_loss_is_refused():
    base = SFTConfig(base_model="m", tau=0.05, sigma_j=0.01)
    dataclasses.replace(base, stop_weight=0.5).validate()
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="stop_weight"):
            dataclasses.replace(base, stop_weight=bad).validate()
    # `use_sample_weight=False` selects the stock Trainer, which is handed no weight column at
    # all: the knob would be in the sha and in the manifest and in nothing else.
    with pytest.raises(ValueError, match="stop_weight"):
        dataclasses.replace(base, stop_weight=0.5, use_sample_weight=False).validate()
    # A packed window is a concatenation of rows and trains at 1.0 by construction.
    with pytest.raises(ValueError, match="stop_weight"):
        dataclasses.replace(
            base, stop_weight=0.5, pack_sequences=True, chat_template=False
        ).validate()


def _preflight_rows() -> list[dict]:
    """36 ASK rows and 24 STOP rows, every one at weight 1.0, so the shares are exact.

    36 questions clears `MIN_QUESTIONS_TO_MEASURE` (30) and 24/60 = 40% STOP clears the 80%
    ceiling, so the two `dataset_report` gates pass and what is left is the arithmetic.
    """
    return [_ask_row(i) for i in range(36)] + [_stop_row() for _ in range(24)]


def test_preflight_reports_the_stop_share_of_the_loss_after_weighting():
    """`stop_share` is a property of the FILE; the paper's claim is about the GRADIENT. At
    stop_weight 0.5 the STOP rows are 40% of the rows and 25% of the loss, and the second
    number is the one the ablation is about -- so it is measured and recorded, not inferred."""
    rows = _preflight_rows()
    cfg = SFTConfig(base_model="m", tau=0.05, sigma_j=0.01, stop_weight=0.5)
    rep = preflight(cfg, rows)
    assert rep["stop_share"] == pytest.approx(24 / 60)
    # The two predicates, side by side: `n_stop` is the `is_stop` flag, `n_stop_by_action_json`
    # is the bytes the mask supervises. The weighting uses the second.
    assert rep["n_stop"] == rep["n_stop_by_action_json"] == 24
    assert rep["stop_weight"] == 0.5
    # 24 STOP x 0.5 = 12 against 36 ASK x 1.0; 12 / 48, by hand.
    assert rep["stop_share_of_loss"] == pytest.approx(0.25)
    # `accum_scale` normalises the window by N_window because the dataset's mean weight is 1.
    # The multiplier is exactly what makes that stop being true, so the new mean is reported
    # beside the share rather than left for a reader to derive: 48 / 60.
    assert rep["mean_row_weight"] == pytest.approx(0.8)


def test_without_the_knob_the_stop_share_of_the_loss_is_the_stop_share_of_the_file():
    """The null arm's number, and the reason the knob is legible: the exporter normalises
    `sample_weight` to mean 1 WITHIN each kind, so at stop_weight None the share of the loss
    is the share of the rows. MEASURED on data/rl/sft.jsonl: 0.6784679608549855 of rows and
    0.6784679608549308 of the loss."""
    rep = preflight(SFTConfig(base_model="m", tau=0.05, sigma_j=0.01), _preflight_rows())
    assert rep["stop_weight"] is None
    assert rep["stop_share_of_loss"] == pytest.approx(rep["stop_share"])
    assert rep["mean_row_weight"] == pytest.approx(1.0)


def test_the_manifest_records_the_stop_weight_and_the_share_it_bought(tmp_path, monkeypatch):
    """A number without provenance is not a result. The checkpoint this ablation produces has
    to be able to say, from its own directory, what the STOP rows were worth in its gradient.

    The doubles are `tests/test_resume_from_checkpoint.py`'s: the real `train()` over fake
    torch/peft/transformers, which is the only way the manifest writer runs on a laptop.
    """
    from pinq_train.rung1_sft.train import train as rung1_train
    from test_resume_from_checkpoint import _FakeTokenizer, _install_fake_train_libraries

    rows = [
        _ask_row(i) | {"suite_id": "musique", "task_id": f"t{i // 2}", "split": "train"}
        for i in range(36)
    ] + [
        _stop_row() | {"suite_id": "musique", "task_id": f"t{i // 2}", "split": "train"}
        for i in range(24)
    ]
    ds = tmp_path / "sft.jsonl"
    ds.write_text("".join(json.dumps(r) + "\n" for r in rows))

    _install_fake_train_libraries(monkeypatch)
    out = rung1_train(
        SFTConfig(
            base_model="Qwen/Qwen3-8B",
            dataset=str(ds),
            out_dir=str(tmp_path / "rung1"),
            tau=0.05,
            sigma_j=0.01,
            chat_template=False,
            stop_weight=0.5,
        ),
        acknowledge_untested=True,
        tokenizer=_FakeTokenizer(),
    )
    assert out["stop_share_of_loss"] == pytest.approx(0.25)
    man = json.loads((tmp_path / "rung1" / "rung1.manifest.json").read_text())
    assert man["config"]["stop_weight"] == 0.5
    assert man["dataset"]["stop_weight"] == 0.5
    assert man["dataset"]["stop_share"] == pytest.approx(24 / 60)
    assert man["dataset"]["stop_share_of_loss"] == pytest.approx(0.25)
