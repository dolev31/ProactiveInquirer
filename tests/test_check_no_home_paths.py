"""`scripts/check_no_home_paths.sh` must scan the paper sources too.

A committed absolute home path is how a research repo stops being reproducible for anyone but
its author, and this script is the gate every session runs. Its `--include` list named
`py toml yaml yml md cfg sh` and NOT `tex` -- so the entire paper, which six lanes are writing
concurrently, was outside the one check that would have caught it. Nothing to clean up: the
same grep with `--include='*.tex'` added is clean across both the committed tree (14 .tex) and
the live working tree (32 .tex), measured 2026-09-17 before the extension was added, so this is
preventive.

WHY A FULL-SCRIPT FIXTURE. This script has no functions to pull out by name, so the
extract-a-function harness (`tests/test_hpc_eval_checkpoints.py::_extract`) does not apply. It
is a flat script that `cd`s to its own parent and greps `.`, which makes it directly runnable
against a small tree -- the other technique that file already uses for shell (`eck_cluster`),
plus a static read of its own text.

EVERY HOME PATH BELOW IS ASSEMBLED FROM PIECES, never written literally: this file is a `.py`
and is therefore itself scanned by the very check under test. The existing suite does the same
(see tests/test_convlog_scrub_fails_closed.py).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_no_home_paths.sh"

# Assembled, so this file does not trip the check it is testing.
HOME_PATH = "/" + "Users/" + "testperson" + "/PycharmProjects/x/y.json"


def _tree(tmp_path: Path) -> Path:
    """A repo-shaped tree the script can `cd ..` into and grep."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, root / "scripts" / SCRIPT.name)
    (root / "paper").mkdir()
    return root


def _run(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(root / "scripts" / SCRIPT.name)], capture_output=True, text=True
    )


def test_a_home_path_in_a_paper_source_is_caught(tmp_path):
    """The hole, stated as the case it lets through: a `.tex` the gate never opened."""
    root = _tree(tmp_path)
    (root / "paper" / "results.tex").write_text(
        "\\begin{document}\nThe table is generated from " + HOME_PATH + "\n\\end{document}\n"
    )
    r = _run(root)
    assert r.returncode == 1, f"a home path in a .tex slipped the gate: {r.stdout + r.stderr!r}"
    assert "results.tex" in r.stdout + r.stderr, "the offending file must be named"


def test_a_clean_paper_source_still_passes(tmp_path):
    """Adding an extension must not make the gate fire on the paper as it actually is."""
    root = _tree(tmp_path)
    (root / "paper" / "results.tex").write_text(
        "\\begin{document}\nRead from \\texttt{artifacts/gate/8b2-t20/}, a repo-relative path."
        "\n\\end{document}\n"
    )
    r = _run(root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "check_no_home_paths: clean" in r.stdout


def test_the_extensions_it_already_covered_are_still_covered(tmp_path):
    """A control on the change being additive: the include list gained one entry and lost none."""
    for name in ("a.py", "b.md", "c.sh", "d.yaml", "e.toml", "f.cfg", "g.yml"):
        root = _tree(tmp_path / name)
        (root / name).write_text("path = " + HOME_PATH + "\n")
        r = _run(root)
        assert r.returncode == 1, f"{name} stopped being scanned: {r.stdout + r.stderr!r}"


def test_the_include_list_names_tex(tmp_path):
    """Static read, so the reason survives a future reformat of the command."""
    assert "--include='*.tex'" in SCRIPT.read_text()
