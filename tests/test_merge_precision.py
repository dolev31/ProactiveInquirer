"""What dtype the merge runs in, and why that is a correctness question and not a preference.

THE CLAIM THIS PROTECTS. `merge.py` exists so that "rung 2's reference policy IS rung 1" has
exactly one meaning: rung 1 is folded into the weights, the only adapter left is the one DPO
trains, and `merged_base_sha` names the weights. Every word of that is true of the *file* and
false of the *distribution* if the fold rounds the update away.

MEASURED, on the Mac smoke run (Qwen3-0.6B, rung-1 LoRA r=32 after 6 optimiser steps):

  * `Qwen/Qwen3-0.6B` ships `torch_dtype: "bfloat16"`, and transformers 5.x makes
    `from_pretrained` default to the CHECKPOINT's dtype rather than float32 (4.x upcast). So
    `_hf_loaders` loaded bf16, `merge_and_unload()` folded in bf16, and `save_pretrained` wrote
    bf16.
  * mean |LoRA delta| was 2.819e-05 against a typical |W| of 2.363e-02, whose bf16 ulp is
    9.229e-05 -- the update is roughly a THIRD of one representable step of the weight it is
    being added to.
  * 44.7% of `sum|delta|` did not survive: 6862.69 of 12413.99. 307,036,664 weights had a
    non-zero fp32 delta that became exactly zero.
  * `test_reference_logprobs_equal_a_freshly_loaded_merged_model` -- the integration test
    written for precisely this property, and never run until PI_SMOKE_MODEL was set -- failed
    at 2.74e-02 against its 1e-04 tolerance.

AND IT IS STORAGE, NOT ARITHMETIC. Folding in fp32 and then storing bf16 keeps the same 55.3%,
to the digit. So doing the matmul in higher precision fixes nothing; the merged weights have to
be STORED in a dtype whose ulp is below the update. That is why the default here is float32 and
why it is a parameter rather than a constant: the cost is real (a merged 8B base is ~32 GB
rather than ~16 GB) and a run whose update is large relative to a bf16 ulp may legitimately
choose otherwise -- but it must choose, and the manifest must say which.
"""

from __future__ import annotations

import json
import sys
import types

from pinq_train import merge


class _FakeModel:
    def __init__(self, name: str) -> None:
        self.name = name

    def merge_and_unload(self):
        return self

    def save_pretrained(self, out: str) -> None:
        (__import__("pathlib").Path(out) / "model.safetensors").write_bytes(b"weights")


def _install_fake_libraries(monkeypatch) -> dict:
    """A `transformers` and a `peft` whose `from_pretrained` record their kwargs.

    The gate venv has neither library, and this check has to run there: the whole failure was a
    default that only shows up when a real `from_pretrained` is called.
    """
    seen: dict = {"model_kwargs": None, "tokenizer_kwargs": None}

    def model_from_pretrained(name, **kwargs):
        seen["model_kwargs"] = kwargs
        return _FakeModel(name)

    def tokenizer_from_pretrained(name, **kwargs):
        seen["tokenizer_kwargs"] = kwargs
        return _FakeModel(name)

    tr = types.ModuleType("transformers")
    tr.AutoModelForCausalLM = type(
        "AutoModelForCausalLM", (), {"from_pretrained": model_from_pretrained}
    )
    tr.AutoTokenizer = type("AutoTokenizer", (), {"from_pretrained": tokenizer_from_pretrained})
    pf = types.ModuleType("peft")
    pf.PeftModel = type("PeftModel", (), {"from_pretrained": staticmethod(lambda m, d: m)})
    monkeypatch.setitem(sys.modules, "transformers", tr)
    monkeypatch.setitem(sys.modules, "peft", pf)
    return seen


def test_the_base_is_loaded_at_an_explicit_dtype_not_the_checkpoints_own(monkeypatch):
    """The defect, stated as the absence of a kwarg. With no dtype argument, transformers 5.x
    loads the checkpoint's dtype -- bfloat16 for every Qwen3 -- and the fold is then a fold into
    bf16 weights, which loses any update below one ulp."""
    seen = _install_fake_libraries(monkeypatch)
    merge._hf_loaders(merge.MERGE_DTYPE).model("Qwen/Qwen3-0.6B")
    assert seen["model_kwargs"], "from_pretrained was called with no dtype at all"
    assert seen["model_kwargs"].get("dtype") == "float32"


def test_float32_is_the_default_because_a_bf16_ulp_is_larger_than_the_update():
    """Not a style preference. 2.819e-05 of update against a 9.229e-05 ulp is the measurement;
    see this module's docstring."""
    assert merge.MERGE_DTYPE == "float32"


def test_the_manifest_records_which_dtype_the_merge_ran_in(tmp_path, monkeypatch):
    """A merge that rounded the update away and one that did not produce different
    distributions under the same `output_sha` shape. The run must be able to say which it was,
    or "the reference policy was rung 1" is again a claim nothing can check."""
    _install_fake_libraries(monkeypatch)
    out = tmp_path / "merged"
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"a")
    (adapter / "adapter_config.json").write_text("{}")

    merge.merge_adapter("Qwen/Qwen3-0.6B", str(adapter), str(out))
    man = json.loads((out / merge.MANIFEST).read_text())
    assert man["dtype"] == "float32"


def test_a_caller_may_choose_another_dtype_and_the_manifest_follows(tmp_path, monkeypatch):
    """The cost is real -- a merged 8B base is ~32 GB at fp32 -- so bf16 stays reachable. What
    is refused is choosing it by accident."""
    seen = _install_fake_libraries(monkeypatch)
    out = tmp_path / "merged"
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"a")

    merge.merge_adapter("m", str(adapter), str(out), dtype="bfloat16")
    assert seen["model_kwargs"].get("dtype") == "bfloat16"
    assert json.loads((out / merge.MANIFEST).read_text())["dtype"] == "bfloat16"
