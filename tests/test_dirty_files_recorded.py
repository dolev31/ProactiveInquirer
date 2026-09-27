"""`dirty=True` says a run is unreproducible but never says WHY, so it can never be cleared.

A run is stamped dirty when `git status --porcelain --untracked-files=no` prints anything --
ANY uncommitted tracked file in the repository, including one belonging to work the run never
imports. Measured on this repo: 6,754 of 8,811 ok musique runs came out dev- because a
concurrent session had five files open in `pi_eval` and `pi_run`, three of which a rollout
worker provably cannot import (contract 1 forbids it, and the process split enforces it).

THE FLAG IS RIGHT AND UNARGUABLE AS RECORDED, which is the problem. `code_version` names a
commit, `dirty` says "plus some edits", and nothing says which. So there is no evidence that
could later clear the run -- and the comparison that would settle it empirically cannot exist
either: a dev- run and a clean run never share a code_version, because committing is exactly
what flips one into the other. Verified: 0 such pairs across 10,719 runs.

Recording the file list changes that. A future run whose diff touched only `tests/` or only
gold-side code can be adjudicated from the manifest instead of being excluded forever. It does
NOT launder the existing runs -- their diff was never written down and is gone.
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

from pi_run.manifest import build_manifest, git_info, manifest_to_dict
from pinq.ids import SEMANTIC_FIELDS
from pinq.types import RunManifest

# One run's worth of non-provenance fields, so every assertion below differs only in the
# thing under test. Shaped after tests/test_tau2_fork_wiring.py::_manifest.
_MANIFEST_KW = dict(
    suite_id="tau2_retail",
    task_id="55",
    arm_id="inquirer_prompted",
    policy_id="inquirer_prompted",
    seed=0,
    corpus_hash="c",
    budget_cap=16,
    max_turns=16,
    word_cap=180,
    code_version="abc123",
)


def test_git_info_reports_which_files_are_dirty() -> None:
    g = git_info(".")
    assert hasattr(g, "dirty_files"), "dirty is recorded without saying what made it dirty"
    assert isinstance(g.dirty_files, tuple)


def test_a_clean_tree_lists_nothing() -> None:
    g = git_info(".")
    if not g.dirty:
        assert g.dirty_files == ()


def test_the_list_is_consistent_with_the_flag() -> None:
    """A dirty flag with an empty file list, or the reverse, would be worse than either alone:
    it would look like evidence while carrying none."""
    g = git_info(".")
    assert bool(g.dirty) == bool(g.dirty_files)


def test_paths_are_repo_relative_and_sorted() -> None:
    """Sorted so two runs of one tree produce the same list, and the manifest stays diffable."""
    g = git_info(".")
    assert list(g.dirty_files) == sorted(g.dirty_files)
    assert not any(p.startswith("/") for p in g.dirty_files)


def test_a_broken_git_is_still_dirty_with_no_list() -> None:
    """ "An absent or broken git is not fatal but IS dirty" -- and it can name no files, which
    is the one case where an empty list beside a dirty flag is correct."""
    g = git_info("/nonexistent-path-for-this-test")
    assert g.dirty is True
    assert g.sha == "unknown"
    assert g.dirty_files == ()


# --------------------------------------------------------------------------- the recorded NAMES
#
# The list above is only worth having if the names in it are the names of real files. It was
# not: `git status --porcelain` puts a two-character status field before a space, so an
# UNSTAGED modification is " M path" with a LEADING SPACE, and `git_info` applied `.strip()`
# to the WHOLE output before slicing each line at a fixed `line[3:]`. That strip ate the
# leading space of the FIRST line only, leaving "M path" -- two characters before the path
# where the slice expected three -- so the first dirty file lost its first character.
#
# Measured on this repo before the fix, one unstaged edit each to paper/sections/mechanism.tex
# and src/pi_run/cli.py:
#
#     porcelain first line: ' M paper/sections/mechanism.tex'
#     dirty_files:          ('aper/sections/mechanism.tex', 'src/pi_run/cli.py')
#
# `aper/...` is not a path in this repository. Under CONTRIBUTING.md rule 1 that makes it a
# provenance defect, not a formatting one: it is what `dirty_sweep_refusal` prints to the
# operator whose campaign was just refused.
#
# WHAT IT DID *NOT* REACH, measured 2026-09-17 over 109,503 manifests under runs/: 9,094 carry
# dirty=True and 0 carry a non-empty `dirty_files`, because `manifest_to_dict` wrote the key
# from `getattr(m, "dirty_files", ())` and `RunManifest` had no such field (with `slots=True`,
# so it could never acquire one). The wrong names therefore never got onto disk, and the
# adjudication this module's docstring promises had never once worked -- for a reason unrelated
# to the parse. Both halves are repaired now; see the section at the bottom of this file.
#
# HOW TO RECOGNISE A RECORD FROM BEFORE EITHER FIX, since nothing is backfilled:
#   * `"dirty": true` with `"dirty_files": []` -- or with no such key at all -- is an OLD
#     record. It is NOT a clean tree and NOT a tree whose diff was empty; it is a run whose
#     diff was never written down. Nothing can recover it: `code_version` names the commit the
#     edits sat on top of, and a dev- run never shares a code_version with a clean one.
#   * the corrupted names live only in operator-facing text (refusal blocks in campaign logs),
#     never in a run record, and only ever the FIRST line of such a block, and only when that
#     file's status was an unstaged modification.


def _repo_with_every_status_kind(root: Path) -> None:
    """A repo whose porcelain output is multi-line and starts with a leading-space status.

    Covers the five kinds this repository actually produces. Measured output of
    `git status --porcelain --untracked-files=no` in exactly this tree:

        ' M a_unstaged.txt\\nM  b_staged.txt\\nA  c_added.txt\\nD  d_deleted.txt\\n'
        'R  e_old.txt -> e_new.txt\\n'

    The unstaged modification sorts first ON PURPOSE: it is the one status whose line begins
    with a space, and the whole-output strip could only corrupt the first line.
    """

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(root), "-c", "commit.gpgsign=false", *args],
            check=True,
            capture_output=True,
            text=True,
        )

    git("init", "-q", ".")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    # Distinct bodies: identical contents make git's rename detection pair the added file
    # with the deleted one, and the fixture would then test a different set of statuses.
    (root / "a_unstaged.txt").write_text("unstaged one\n")
    (root / "b_staged.txt").write_text("staged two\n")
    (root / "d_deleted.txt").write_text("deleted four\n")
    (root / "e_old.txt").write_text(
        "renamed five, a distinctive body so rename detection pairs it\n"
    )
    git("add", "-A")
    git("commit", "-qm", "init")
    with (root / "a_unstaged.txt").open("a") as fh:  # " M " -- leading space, first line
        fh.write("edited\n")
    with (root / "b_staged.txt").open("a") as fh:  # "M  "
        fh.write("edited\n")
    git("add", "b_staged.txt")
    (root / "c_added.txt").write_text("added three\n")  # "A  "
    git("add", "c_added.txt")
    git("rm", "-q", "d_deleted.txt")  # "D  "
    git("mv", "e_old.txt", "e_new.txt")  # "R  e_old.txt -> e_new.txt"


def test_the_recorded_names_are_the_real_paths_for_every_status_kind(tmp_path) -> None:
    """The names, not just the count -- and the first one especially.

    A RENAME CONTRIBUTES BOTH PATHS. The point of the list is that a run can be adjudicated
    from its own manifest ("this diff touched only tests/"), and a file renamed OUT of
    `src/pi_eval/` touched `src/pi_eval/` -- recording only the destination would make that
    diff read as if the gold side was never involved, which is the exact class of wrong
    answer this field exists to prevent. Both paths are real paths git itself reported; only
    one of them still exists on disk, which is already true of a deletion.
    """
    _repo_with_every_status_kind(tmp_path)
    g = git_info(str(tmp_path))
    assert g.dirty is True
    assert g.dirty_files == (
        "a_unstaged.txt",
        "b_staged.txt",
        "c_added.txt",
        "d_deleted.txt",
        "e_new.txt",
        "e_old.txt",
    ), g.dirty_files


def test_the_parse_reads_fields_not_offsets() -> None:
    """The unit test under the end-to-end one, on the exact bytes git produced.

    Measured with `git status --porcelain -z --untracked-files=no` on the fixture above (git
    2.50.1). The line-oriented form of the same tree, for comparison, was

        ' M a_unstaged.txt\\nM  b_staged.txt\\nA  c_added.txt\\nD  d_deleted.txt\\n'
        'R  e_old.txt -> e_new.txt\\n'

    which is where the leading space on the first record, and the arrow that made one entry
    out of two paths, both came from.
    """
    from pi_run.manifest import _dirty_paths

    measured = (
        " M a_unstaged.txt\0M  b_staged.txt\0A  c_added.txt\0"
        "D  d_deleted.txt\0R  e_new.txt\0e_old.txt\0"
    )
    assert _dirty_paths(measured) == (
        "a_unstaged.txt",
        "b_staged.txt",
        "c_added.txt",
        "d_deleted.txt",
        "e_new.txt",
        "e_old.txt",
    )


def test_the_trailing_separator_does_not_become_an_entry() -> None:
    """Dropping the whole-output `.strip()` is not enough on its own: git terminates every
    record, so the last separator would otherwise yield an empty path -- an entry naming no
    file, which counts as a dirty file in `len(gi.dirty_files)` and prints as a blank line in
    the refusal."""
    from pi_run.manifest import _dirty_paths

    assert _dirty_paths("") == ()
    assert _dirty_paths(" M a.txt\0") == ("a.txt",)
    assert _dirty_paths(" M a.txt\0\0") == ("a.txt",)


def test_paths_the_line_oriented_form_would_have_quoted_come_back_raw() -> None:
    """The second corruption in the same expression, and the reason the argv gained `-z`
    rather than the slice losing one character.

    Measured on a probe repo, line-oriented porcelain for these three files:

        ' M "has space.txt"'
        ' M "unicode_caf\\303\\251.txt"'
        ' M "weird\\\\"quote.txt"'

    Every one of those strings names a file that does not exist. `-z` emits the real bytes.
    """
    from pi_run.manifest import _dirty_paths

    measured = ' M has space.txt\0 M unicode_café.txt\0 M weird"quote.txt\0'
    assert _dirty_paths(measured) == ("has space.txt", "unicode_café.txt", 'weird"quote.txt')


def _committed_repo(root: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(root), "-c", "commit.gpgsign=false", *args],
            check=True,
            capture_output=True,
            text=True,
        )

    git("init", "-q", ".")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    (root / "f.txt").write_text("x\n")
    git("add", "-A")
    git("commit", "-qm", "init")


def test_a_clean_tree_is_still_not_dirty_and_still_names_nothing(tmp_path) -> None:
    """The invariant the name fix must not touch. `RunManifest.dirty` feeds the `dev-` prefix
    straight into `run_id`, so a shift in this boolean would reinterpret runs already stamped
    -- a far worse outcome than the wrong names it replaces."""
    _committed_repo(tmp_path)
    g = git_info(str(tmp_path))
    assert g.dirty is False
    assert g.dirty_files == ()


def test_a_dirty_tree_is_still_dirty(tmp_path) -> None:
    _committed_repo(tmp_path)
    (tmp_path / "f.txt").write_text("x\ny\n")
    g = git_info(str(tmp_path))
    assert g.dirty is True
    assert g.dirty_files == ("f.txt",)


# ------------------------------------------------------------- and onto the manifest at last
#
# The list reached `GitInfo` and `dirty_sweep_refusal` and stopped there. `manifest_to_dict`
# wrote the key from `getattr(m, "dirty_files", ())` against a `RunManifest` that has no such
# field and `slots=True`, so the key was `[]` on every manifest ever written: measured
# 2026-09-17 over 109,503 manifests under runs/, 9,094 of them dirty=True, 0 with a name.
#
# So the adjudication the module docstring above promises -- deciding whether a dev- run's
# dirtiness disqualified it (source changed) or was irrelevant (a paper file changed) -- could
# not be made from any record. That distinction is the whole reason the field exists.


def test_a_manifest_cannot_be_given_the_file_list_at_all() -> None:
    """THE DEFECT, on the surface that existed before the field did.

    `manifest_to_dict` emits a `dirty_files` key, so a reader would reasonably believe a
    manifest can carry one. It cannot: the field is absent from the dataclass, and `slots=True`
    means it can never be set after the fact either. Both halves are asserted, because either
    one alone reads as a naming quibble rather than an empty provenance record.
    """
    names = {f.name for f in dataclasses.fields(RunManifest)}
    assert "dirty_files" in names, "manifest_to_dict writes a key no field can fill"
    m = build_manifest(**_MANIFEST_KW, dirty=True)
    assert dataclasses.replace(m, dirty_files=("src/pi_run/cli.py",)).dirty_files == (
        "src/pi_run/cli.py",
    )
    assert manifest_to_dict(m)["dirty_files"] == []


def test_a_manifest_from_a_dirty_tree_names_the_real_files(tmp_path) -> None:
    """The coordinator's endpoint: the run record itself names them, not just the refusal.

    Driven from a real repository rather than a literal, so what lands on the manifest is
    whatever `git_info` actually recovered -- the two halves cannot drift apart and agree.
    """
    _repo_with_every_status_kind(tmp_path)
    g = git_info(str(tmp_path))
    m = build_manifest(**_MANIFEST_KW, dirty=g.dirty, dirty_files=g.dirty_files)
    assert m.dirty is True
    assert m.dirty_files == (
        "a_unstaged.txt",
        "b_staged.txt",
        "c_added.txt",
        "d_deleted.txt",
        "e_new.txt",
        "e_old.txt",
    )
    # `pi compact` and every reader downstream see the dict, never the dataclass.
    assert manifest_to_dict(m)["dirty_files"] == [
        "a_unstaged.txt",
        "b_staged.txt",
        "c_added.txt",
        "d_deleted.txt",
        "e_new.txt",
        "e_old.txt",
    ]


def test_a_manifest_from_a_clean_tree_carries_an_empty_tuple(tmp_path) -> None:
    _committed_repo(tmp_path)
    g = git_info(str(tmp_path))
    m = build_manifest(**_MANIFEST_KW, dirty=g.dirty, dirty_files=g.dirty_files)
    assert m.dirty is False
    assert m.dirty_files == ()
    assert manifest_to_dict(m)["dirty_files"] == []


def test_the_file_list_is_outside_run_identity() -> None:
    """THE CONSTRAINT. Two manifests differing only in `dirty_files` are ONE run.

    If this field entered `semantic_hash`, every finished dev- run would be renamed by the
    files that happened to be open when it launched -- and a dirty run's id would depend on a
    CONCURRENT session's editor state, which is how 6,754 musique runs came to be dev- in the
    first place. `tests/test_run_id_stability.py` holds the other half of this proof: it
    recomputes seven ids already on disk, and would fail if the hash rule moved at all.
    """
    plain = build_manifest(**_MANIFEST_KW, dirty=True)
    listed = build_manifest(**_MANIFEST_KW, dirty=True, dirty_files=("src/pi_run/cli.py",))
    other = build_manifest(**_MANIFEST_KW, dirty=True, dirty_files=("paper/main.tex",))
    assert plain.semantic_hash == listed.semantic_hash == other.semantic_hash
    assert plain.run_id == listed.run_id == other.run_id
    assert "dirty_files" not in SEMANTIC_FIELDS


# ------------------------------------------------------------------ and down to every worker
#
# A field on `RunManifest` that no launcher fills is the defect this commit is repairing, one
# layer up. `git_info` is called ONCE in the parent (see UnitSpec's docstring: forking git
# 30,000 times turns a 4-hour sweep into a 6-hour one), so the list has to travel on the spec.


def test_the_plan_stamps_the_file_list_on_every_spec() -> None:
    """`sweep.plan` is the one factory `pi run` builds units with. Driven, not grepped."""
    from pi_run import sweep

    specs = sweep.plan(
        suite_id="musique",
        corpus_dir="/tmp/corpus",
        task_ids=["t1", "t2"],
        arm_ids=["self_ask"],
        seeds=[0, 1],
        runs_root="/tmp/runs",
        cache_root="/tmp/cache",
        code_version="abc123",
        dirty=True,
        dirty_files=("paper/sections/mechanism.tex", "src/pi_run/cli.py"),
    )
    assert len(specs) == 4
    assert all(
        s.dirty_files == ("paper/sections/mechanism.tex", "src/pi_run/cli.py") for s in specs
    ), [s.dirty_files for s in specs]
    # Provenance, not identity: the reassembly key must not have grown a field.
    assert len({s.key for s in specs}) == 4


def test_a_spec_that_names_no_files_is_the_default_not_a_crash() -> None:
    """The git-failed case, and every caller that predates the parameter."""
    from pi_run.worker import UnitSpec

    s = UnitSpec(
        suite_id="musique",
        corpus_dir="",
        task_id="t1",
        arm_id="self_ask",
        seed=0,
        runs_root="/tmp/runs",
        cache_root="/tmp/cache",
    )
    assert s.dirty_files == ()


def _manifest_kwargs_at(fn) -> set[str]:
    """The keyword names a function's own `build_manifest(...)` call passes."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_manifest"
        ):
            return {kw.arg for kw in node.keywords if kw.arg}
    raise AssertionError(f"{fn.__qualname__} does not call build_manifest at all")


def test_both_runners_stamp_the_same_git_provenance() -> None:
    """`run_tau2_unit` IS A HAND COPY of `run_unit`, and every run in this project is tau2.

    On 2026-09-07/08 four fields `run_unit` honoured were found silently dropped in the copy:
    the foreign-trace prefix, the branch identity, the sampling temperature and the fork
    itself. A provenance field stamped only in `run_unit` is stamped nowhere that matters, and
    the symptom is never an error -- it is a manifest that says `dirty: true` and names nothing.

    Read off the AST rather than the text, so `dirty_files` appearing anywhere in the function
    for any other reason cannot satisfy it: the assertion is that it is passed AS A KEYWORD to
    `build_manifest`. A real tau2 unit needs an environment, a user simulator and an LLM, none
    of which a test has; what the two runners hand the builder is decided before any of that.
    """
    from pi_run.stages.tau2_runner import run_tau2_unit
    from pi_run.worker import run_unit

    git_fields = {"code_version", "dirty", "dirty_files"}
    for fn in (run_unit, run_tau2_unit):
        assert git_fields <= _manifest_kwargs_at(fn), (
            f"{fn.__qualname__} drops {git_fields - _manifest_kwargs_at(fn)}"
        )
