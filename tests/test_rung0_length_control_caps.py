"""`scripts/rung0_length_control.py` used to retype `Rung0Config`'s own `budget_cap=8,
max_turns=16` field defaults as two literals in its own `Rung0Config(...)` construction, so
the search's own cap and the length-control harness's replay of it could silently diverge if
either default ever moved. The fix reads `dataclasses.fields(gepa.Rung0Config)` instead.

WHY THIS TEST EXECS A SOURCE SLICE INSTEAD OF IMPORTING THE SCRIPT. The script has no
`if __name__ == "__main__":` guard and no functions -- importing it runs the WHOLE thing
top to bottom, including a live `SeamClient`/`SeamEvaluator` network loop that scores real
prompts against a real seam server. That is exactly the GPU/cluster/LLM spend this repository
must not trigger from a test. The construction of `cfg = gepa.Rung0Config(...)` itself makes
no network call and depends on nothing but `dataclasses` and `gepa.Rung0Config` -- so this
test extracts just that slice of the script's source (up to, but not including, the first
line that touches `sys.argv`-derived task keys) and `exec`s it in an isolated namespace,
never reaching the argv-dependent paths or the evaluation loop.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

SCRIPT = Path("scripts/rung0_length_control.py")


def _cap_slice(source: str) -> str:
    """From the caps' construction through to (not including) the first argv-dependent line.

    Two markers, because the CURRENT script reads its caps through a `_rung0_defaults` dict
    comprehension that a pre-fix script does not have; starting at whichever is present keeps
    this helper usable for red-evidence checks against the old, retyped-literal source too.
    """
    marker = "_rung0_defaults = {f.name"
    start = source.index(marker) if marker in source else source.index("\ncfg = gepa.Rung0Config(")
    end = source.index("\nkeys = _read_task_keys", start)
    return source[start:end]


def _build_cfg(source: str):
    import pinq_train.rung0_gepa as gepa

    ns = {"dataclasses": dataclasses, "gepa": gepa}
    exec(compile(_cap_slice(source), "<rung0_length_control-cap-slice>", "exec"), ns)
    return ns["cfg"]


def test_the_no_override_caps_match_rung0configs_own_defaults():
    """Behaviour-preserving: the harness must replay the search's own cap, unchanged."""
    cfg = _build_cfg(SCRIPT.read_text())
    assert cfg.budget_cap == 8
    assert cfg.max_turns == 16


def test_a_changed_rung0config_default_reaches_the_harness(monkeypatch):
    """Move `Rung0Config`'s own field defaults and confirm the harness's `cfg` follows, the
    same proof used for the three other launcher sites in this migration."""
    import pinq_train.rung0_gepa as gepa

    monkeypatch.setattr(gepa.Rung0Config.__dataclass_fields__["budget_cap"], "default", 24)
    monkeypatch.setattr(gepa.Rung0Config.__dataclass_fields__["max_turns"], "default", 41)

    cfg = _build_cfg(SCRIPT.read_text())
    assert cfg.budget_cap == 24
    assert cfg.max_turns == 41
