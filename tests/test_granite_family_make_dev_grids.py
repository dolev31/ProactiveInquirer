"""scripts/granite_family/make_dev_grids.py: the per-seed dev-gate grid derivation.

Written against the REAL conf/grids/dev_select_<suite>.yaml files rather than a fixture,
because the whole point of the script is a text substitution over those specific files, and a
synthetic copy would not catch a future edit to their `name:` or `arms:` lines that changes the
exact string this script matches on -- which is precisely the failure `--check` exists to
surface, so the test exercises the same entry point a CI run would.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "granite_family"))

import make_dev_grids as G  # noqa: E402


def test_planned_outputs_cover_every_suite_seed_and_the_base() -> None:
    plan = G.planned_outputs()
    names = {p.stem for p in plan}
    for suite in G.SUITES:
        for seed in G.SEEDS:
            assert f"dev_select_{suite}_granite_s{seed}" in names
        assert f"dev_select_{suite}_granite_base" in names
    assert len(plan) == len(G.SUITES) * (len(G.SEEDS) + 1) == 12


def test_seed_variant_changes_only_the_name_line() -> None:
    src = (G.GRIDS / "dev_select_musique.yaml").read_text()
    out = G.derive(src, "dev_select_musique", "dev_select_musique_granite_s0", prompted_base=False)
    src_lines = src.splitlines()
    out_lines = out.splitlines()
    assert len(src_lines) == len(out_lines)
    diffs = [i for i, (a, b) in enumerate(zip(src_lines, out_lines)) if a != b]
    assert diffs == [0]
    assert out_lines[0] == "name: dev_select_musique_granite_s0"


def test_base_variant_changes_the_name_line_and_the_arms_line_only() -> None:
    src = (G.GRIDS / "dev_select_musique.yaml").read_text()
    out = G.derive(src, "dev_select_musique", "dev_select_musique_granite_base", prompted_base=True)
    src_lines = src.splitlines()
    out_lines = out.splitlines()
    diffs = [i for i, (a, b) in enumerate(zip(src_lines, out_lines)) if a != b]
    assert diffs == [0, src_lines.index("  - inquirer_trained")]
    assert "  - inquirer_prompted" in out_lines
    assert "  - inquirer_trained" not in out_lines


def test_generated_files_on_disk_match_the_plan_exactly() -> None:
    # This is what `--check` runs; asserted directly so a failure here names the file, not just
    # a process exit code.
    plan = G.planned_outputs()
    for path, text in plan.items():
        assert path.exists(), f"{path} was never written -- run make_dev_grids.py"
        assert path.read_text() == text, f"{path} is stale relative to its source grid"
