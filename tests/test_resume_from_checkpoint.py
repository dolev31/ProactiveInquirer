"""A PREEMPTED RUNG RESUMES FROM ITS LAST CHECKPOINT INSTEAD OF STEP 0.

THE FAILURE. Both trainers called `trainer.train()` with no arguments
(`rung1_sft/train.py:716`, `rung2_dpo/train.py:1046`), so HF Trainer's own
`resume_from_checkpoint` was unreachable from anywhere in this repository. Rung 1 already WROTE
`checkpoint-*/` -- `save_strategy="epoch"` has been set since the trainer was written -- so the
state to resume from was on disk and nothing ever read it. A cluster job killed mid-epoch was
resubmitted with the same command and paid for every step again, silently: the second run's
loss curve looks exactly like a first run's, because it is one.

WHY IT MATTERS HERE RATHER THAN AS A CONVENIENCE. docs/HPC_RUNBOOK.md 7 states the etiquette
the GPU reservation asks for, and "make every long job resumable" is one of its clauses:
the reservation is shared with the rest of `grp_res_gpu` and a job may be preempted at any
time. Rung 1 is ~180 GPU-h. "Resubmit the same command" has to mean continue.

NOTHING HERE TOUCHES A GPU, which is the point. `torch`, `peft`, `trl` and `transformers` live
behind the `[train]` extra and are not installed in the gate venv, so the one line of each rung
that a laptop cannot execute is exactly the line nothing pinned. The doubles below record what
the trainer was passed; `latest_checkpoint` is pure and needs no double at all. The rung-2 twin
of the trainer test lives in `tests/test_rung2_reference_policy.py`, beside the TRL doubles it
already owns.

WHY `resume` IS NOT IN `cfg.sha` AND `save_steps` IS. `cfg.sha` is the checkpoint's identity:
what was trained, on what data, at which hyperparameters. Whether THIS invocation picked up
where a preempted one stopped is a property of the execution, not of the thing produced -- and
in the sha it would give a resumed run a different identity from the run it is continuing, so
a preemption would split one experiment into two that no table could rejoin. `save_steps` is
the other way round: it decides how often `checkpoint-N/` is written, so it changes what the
output directory contains and how much a preemption costs, and two cadences are two runs.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import types
import zlib
from pathlib import Path

import pytest

from pinq_train.resume import latest_checkpoint
from pinq_train.rung1_sft.train import SFTConfig
from pinq_train.rung2_dpo.train import DPOConfig

# --------------------------------------------------------------------- the pure helper


def _ckpt(root: Path, step: int, *, state: bool = True) -> Path:
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    if state:
        (d / "trainer_state.json").write_text(json.dumps({"global_step": step}) + "\n")
    return d


def test_an_output_directory_with_no_checkpoint_resumes_nothing(tmp_path):
    """The first submission of a job. `None` is what `Trainer.train` wants for "start fresh",
    so the default path and the resume path are the same call."""
    assert latest_checkpoint(str(tmp_path)) is None
    assert latest_checkpoint(str(tmp_path / "never-created")) is None


def test_one_checkpoint_is_the_one_resumed(tmp_path):
    d = _ckpt(tmp_path, 200)
    assert latest_checkpoint(str(tmp_path)) == str(d)


def test_the_highest_step_wins_and_not_the_lexicographic_one(tmp_path):
    """`sorted()` over the NAMES puts `checkpoint-1000` before `checkpoint-200`, so a string
    sort resumes from the older one and silently replays 800 steps while reporting a resume."""
    for step in (200, 400, 1000):
        _ckpt(tmp_path, step)
    assert latest_checkpoint(str(tmp_path)) == str(tmp_path / "checkpoint-1000")


def test_a_directory_without_trainer_state_is_not_a_checkpoint(tmp_path):
    """A job killed between `mkdir` and the state write leaves weights that nothing can place
    in the schedule; handing that directory to `resume_from_checkpoint` raises AFTER the base
    model has loaded, which on a rented card is the expensive place to find out."""
    _ckpt(tmp_path, 200)
    _ckpt(tmp_path, 400, state=False)
    assert latest_checkpoint(str(tmp_path)) == str(tmp_path / "checkpoint-200")


def test_names_that_are_not_checkpoint_n_are_ignored(tmp_path):
    """Every one of these carries a `trainer_state.json`, so what is being tested is the NAME
    and not the state file. `artifacts/rung1` also holds the adapter and the manifest."""
    for name in ("runs", "checkpoint-", "checkpoint-abc", "checkpoint-1-2", "checkpoint-12x"):
        d = tmp_path / name
        d.mkdir()
        (d / "trainer_state.json").write_text("{}\n")
    (tmp_path / "adapter_model.safetensors").write_text("")
    assert latest_checkpoint(str(tmp_path)) is None


def test_a_file_named_like_a_checkpoint_is_not_a_checkpoint(tmp_path):
    (tmp_path / "checkpoint-200").write_text("")
    assert latest_checkpoint(str(tmp_path)) is None


# ------------------------------------------------------------------- what the sha covers


def _sft(**kw) -> SFTConfig:
    return SFTConfig(base_model="Qwen/Qwen3-8B", tau=0.05, sigma_j=0.01, **kw)


def _dpo(**kw) -> DPOConfig:
    return DPOConfig(base_model="Qwen/Qwen3-8B", adapter="artifacts/rung1", **kw)


@pytest.mark.parametrize("cfg", [_sft(), _dpo()], ids=["rung1", "rung2"])
def test_resuming_is_an_execution_property_and_stays_out_of_the_sha(cfg):
    """A resumed run and the run it continues are ONE experiment. If `resume` moved the sha,
    a preemption would rename the run halfway through and the checkpoint would be attributable
    to neither half."""
    assert cfg.resume is True, "the default has to be on, or the flag protects nobody"
    assert dataclasses.replace(cfg, resume=False).sha == cfg.sha


@pytest.mark.parametrize("cfg", [_sft(), _dpo()], ids=["rung1", "rung2"])
def test_the_checkpoint_cadence_is_part_of_the_run_identity(cfg):
    """200 steps is ~3,200 rows at the shipped effective batch (1 x 16), which is the most a
    preemption can cost. Change the cadence and the output directory holds different things."""
    assert cfg.save_steps == 200
    assert dataclasses.replace(cfg, save_steps=50).sha != cfg.sha


# ------------------------------------------------- rung 1: what the trainer was actually told

_NOT_PASSED = "<train() was called with no checkpoint argument>"

_ARG_FIELDS = (
    "output_dir",
    "learning_rate",
    "num_train_epochs",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "warmup_ratio",
    "seed",
    "logging_steps",
    "save_strategy",
    "save_steps",
    "save_total_limit",
    "report_to",
    "bf16",
    "gradient_checkpointing",
    "group_by_length",
    "remove_unused_columns",
)
# A REAL dataclass, because `_build_trainer` spells its kwargs from
# `dataclasses.fields(TrainingArguments)` -- a stub that accepted anything would let the
# version probe pass names the installed library does not declare.
_TrainingArguments = dataclasses.make_dataclass(
    "TrainingArguments", [(n, object, dataclasses.field(default=None)) for n in _ARG_FIELDS]
)


class _FakeTokenizer:
    """The two-method surface `mask.Tokenizer` declares. Copied in spirit from
    `tests/test_train_rungs.py`: crc32 rather than `hash()`, whose salt changes per process."""

    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [(zlib.crc32(t.encode()) % 30_000) + 10 for t in text.split()]
        return ([1] + ids) if add_special_tokens else ids


class _FakeModel:
    def __init__(self, seen: dict) -> None:
        self._seen = seen

    def save_pretrained(self, out) -> None:
        self._seen["saved"] = str(out)

    def tie_weights(self) -> None:
        """No-op. The real call ties `lm_head.weight` to the input embedding on a
        `tie_word_embeddings=True` architecture (`train()`'s belt-and-suspenders call, added
        alongside `lora_capacity`); this double has no tensors to tie, and every test in this
        file runs the single-card path where the answer to the resume/curriculum/weighting
        questions those tests ask does not depend on it."""

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        """`peft`'s own return shape (trainable, all_param), stubbed rather than measured: this
        double is not wrapped by a real `get_peft_model` (`pf.get_peft_model` below is the
        identity function), so there is no real LoRA adapter here to count. `0, 0` is chosen
        so nothing this file asserts could mistake it for a measured figure -- the actual
        counting rule is cross-checked against real `peft` in
        `tests/test_model_placement.py`, gated behind `.venv-train`."""
        return 0, 0


def _install_fake_train_libraries(monkeypatch) -> dict:
    """`torch`, `peft` and `transformers` as doubles that record every call rung 1 makes."""
    seen: dict = {"train": [], "saved": None, "args": None}

    th = types.ModuleType("torch")
    th.long = "long"
    th.float = "float"
    th.tensor = staticmethod(lambda data, dtype=None: data)
    th.ones = staticmethod(lambda n, dtype=None: [1] * n)
    # `train()` asks the machine how many cards it was given before it loads the base
    # (`base_model_load_kwargs`, tests/test_model_placement.py). ZERO is this machine's truth,
    # and it is the answer that keeps these tests on the single-card path: the double's
    # `from_pretrained` still takes exactly one argument, so a placement kwarg leaking onto the
    # one-GPU load would fail here rather than change how a finished arm was loaded.
    cuda = types.ModuleType("torch.cuda")
    cuda.device_count = staticmethod(lambda: 0)
    th.cuda = cuda
    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")
    data.Dataset = type("Dataset", (), {})
    # `SequentialSampler(ds)` iterates `range(len(ds))`. The list is that, and it is here
    # because the depth curriculum's trainer overrides `_get_train_sampler` to return one --
    # see `tests/test_curriculum.py`.
    data.SequentialSampler = staticmethod(lambda ds: list(range(len(ds))))
    utils.data = data
    th.utils = utils

    class _Trainer:
        def __init__(self, *, model=None, args=None, train_dataset=None, **kw) -> None:
            self.model, self.args = model, args
            # `train_dataset` ON THE INSTANCE, as the real Trainer stores it: an override of
            # `_get_train_sampler` reads it there, and a double that dropped it would make
            # that override raise here and pass on a GPU.
            self.train_dataset = train_dataset
            seen["args"] = args
            seen["trainer"] = self
            seen["train_dataset"] = train_dataset
            seen["n_rows"] = len(train_dataset)

        def train(self, resume_from_checkpoint=_NOT_PASSED) -> None:
            seen["train"].append(resume_from_checkpoint)

    tr = types.ModuleType("transformers")
    tr.AutoTokenizer = type("AutoTokenizer", (), {"from_pretrained": staticmethod(lambda n: n)})
    tr.AutoModelForCausalLM = type(
        "AutoModelForCausalLM",
        (),
        {"from_pretrained": staticmethod(lambda n: _FakeModel(seen))},
    )
    tr.TrainingArguments = _TrainingArguments
    tr.Trainer = _Trainer

    pf = types.ModuleType("peft")
    pf.LoraConfig = type("LoraConfig", (), {"__init__": lambda self, **kw: None})
    pf.get_peft_model = staticmethod(lambda model, cfg: model)

    for name, mod in (
        ("torch", th),
        ("torch.utils", utils),
        ("torch.utils.data", data),
        ("transformers", tr),
        ("peft", pf),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return seen


def _dataset(tmp_path: Path, n: int = 30) -> Path:
    """A dataset the REAL `preflight` accepts: 30 questions is its floor to measure diversity
    at all, and two per task is what makes the within-task statistic defined."""
    from pinq.actions import ask_action_json

    rows = [
        {
            "suite_id": "musique",
            "task_id": f"t{i // 2}",
            "split": "train",
            "state_text": f"TASK: settle claim {i}. EVIDENCE RETRIEVED SO FAR: none.",
            "action_json": ask_action_json(f"which {i} author revised the {i} charter?"),
            "sample_weight": 1.0,
        }
        for i in range(n)
    ]
    p = tmp_path / "sft.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _run_rung1(tmp_path: Path, monkeypatch, **over) -> tuple[dict, dict]:
    """The real `rung1.train()` over the doubles. Returns (what it returned, what was seen).

    NOT `from pinq_train.rung1_sft import train`: the package rebinds that name to the
    FUNCTION, so the attribute path would hand back `train()` rather than the module.
    """
    from pinq_train.rung1_sft.train import train as rung1_train

    seen = _install_fake_train_libraries(monkeypatch)
    cfg = SFTConfig(
        base_model="Qwen/Qwen3-8B",
        dataset=str(_dataset(tmp_path)),
        out_dir=str(tmp_path / "rung1"),
        tau=0.05,
        sigma_j=0.01,
        # The raw-concatenation path: the chat template needs a tokenizer that can render one,
        # and what is under test here is the call, not the rendering.
        chat_template=False,
        **over,
    )
    out = rung1_train(cfg, acknowledge_untested=True, tokenizer=_FakeTokenizer())
    return out, seen


def test_rung1_resumes_from_the_checkpoint_on_disk(tmp_path, monkeypatch):
    """THE BUG, named. With `checkpoint-400/` beside the adapter, a resubmitted job must
    continue from step 400; `trainer.train()` with no argument restarts from 0 and nothing in
    the artifact says which of the two happened."""
    ckpt = _ckpt(tmp_path / "rung1", 400)
    out, seen = _run_rung1(tmp_path, monkeypatch)
    assert seen["train"] == [str(ckpt)]
    assert out["resumed_from"] == str(ckpt)


def test_rung1_starts_fresh_when_resume_is_off(tmp_path, monkeypatch):
    """`None` is passed EXPLICITLY rather than by omission. "The trainer was told to start
    fresh" and "the trainer was never told anything" are different facts, and only the first
    one is a decision this repository made."""
    _ckpt(tmp_path / "rung1", 400)
    out, seen = _run_rung1(tmp_path, monkeypatch, resume=False)
    assert seen["train"] == [None]
    assert out["resumed_from"] is None


def test_rung1_records_in_its_manifest_which_checkpoint_it_continued(tmp_path, monkeypatch):
    """A resumed run and a fresh one produce the same adapter file and the same cfg.sha. The
    manifest is the only place that can say which one this directory holds."""
    ckpt = _ckpt(tmp_path / "rung1", 400)
    _run_rung1(tmp_path, monkeypatch)
    man = json.loads((tmp_path / "rung1" / "rung1.manifest.json").read_text())
    assert man["resumed_from"] == str(ckpt)
    assert man["config"]["resume"] is True and man["config"]["save_steps"] == 200


def test_rung1_records_a_null_when_there_was_nothing_to_resume(tmp_path, monkeypatch):
    _run_rung1(tmp_path, monkeypatch)
    man = json.loads((tmp_path / "rung1" / "rung1.manifest.json").read_text())
    assert man["resumed_from"] is None


def test_rung1_asks_the_trainer_for_the_step_cadence_it_configured(tmp_path, monkeypatch):
    """`save_strategy="epoch"` writes one checkpoint per epoch, and rung 1 runs 2 epochs over
    ~44k rows: a preemption at 95% of epoch 1 loses everything. The cadence is what bounds the
    loss, so it has to reach `TrainingArguments` and not just the config."""
    _, seen = _run_rung1(tmp_path, monkeypatch, save_steps=50)
    assert seen["args"].save_strategy == "steps"
    assert seen["args"].save_steps == 50
    assert seen["args"].save_total_limit == 2
