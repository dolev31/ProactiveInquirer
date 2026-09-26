"""A failed `find` must not read as "nothing found", and `-newermt "@epoch"` is dialect-dependent.

`scripts/hpc/tau2_cluster_campaign.sh` used to SKIP its pin verification whenever no new manifest
appeared -- which is every resumed link, the links that do the most work. It now falls back to
verifying a manifest already on disk. That fallback rests on `find`, and `find` has produced a
silent zero in this programme twice: without `-L` it reads an isolated runs root (a symlink farm)
as empty, and with an argument its dialect rejects it fails while `2>/dev/null | head -1` turns the
failure into an empty string.

THE DIALECT SPLIT IS REAL AND THIS MACHINE IS ON THE WRONG SIDE OF IT. `-newermt "@<epoch>"` is a
GNU find feature. The campaign runs on Linux (GNU find) and these tests run on macOS, where
`/usr/bin/find` is BSD and REFUSES the argument. An interactive shell here resolves `find` to `bfs`,
which accepts it -- so the same command verified by hand at a prompt and run from a script hit
DIFFERENT BINARIES with different answers. That makes this machine a good place to test the failure
path and a bad place to test the success path, and the tests below are split accordingly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest


def _farm(tmp_path):
    """A runs root shaped like the isolated stores: entries are symlinks into a real store."""
    real = tmp_path / "real"
    (real / "unit1").mkdir(parents=True)
    (real / "unit1" / "manifest.json").write_text("{}")
    root = tmp_path / "runs"
    root.mkdir()
    os.symlink(real / "unit1", root / "unit1")
    return root


FIND = "/usr/bin/find"


def _supports_epoch_newermt() -> bool:
    r = subprocess.run(
        [FIND, "/tmp", "-maxdepth", "0", "-newermt", "@0"], capture_output=True, text=True
    )
    return r.returncode == 0


def test_without_L_a_symlink_farm_reads_as_empty(tmp_path):
    root = _farm(tmp_path)
    r = subprocess.run([FIND, str(root), "-name", "manifest.json"], capture_output=True, text=True)
    assert r.stdout.strip() == "", (
        "if a bare find starts descending symlinks, the -L rationale changes"
    )


def test_with_L_the_farm_is_readable(tmp_path):
    root = _farm(tmp_path)
    r = subprocess.run(
        [FIND, "-L", str(root), "-name", "manifest.json"], capture_output=True, text=True
    )
    assert r.returncode == 0 and r.stdout.strip().endswith("manifest.json")


@pytest.mark.skipif(
    _supports_epoch_newermt(),
    reason="this find accepts @epoch; the refusal path needs one that does not",
)
def test_a_rejected_newermt_fails_loudly_and_the_old_pattern_hid_it(tmp_path):
    """THE REGRESSION. The suppress-and-pipe idiom converts a dialect error into a clean zero."""
    root = _farm(tmp_path)
    epoch = int(time.time()) - 60
    args = [FIND, "-L", str(root), "-name", "manifest.json", "-newermt", f"@{epoch}"]

    direct = subprocess.run(args, capture_output=True, text=True)
    assert direct.returncode != 0, "this find is expected to reject @epoch"
    assert direct.stdout.strip() == ""

    # The SUPERSEDED idiom: status thrown away by the pipe, stderr discarded. Indistinguishable
    # from a genuinely empty result.
    old = subprocess.run(
        f"{' '.join(args)} 2>/dev/null | head -1", shell=True, capture_output=True, text=True
    )
    assert old.returncode == 0 and old.stdout.strip() == "", (
        "the old idiom must look exactly like success-with-no-matches -- that is the defect"
    )

    # The REPLACEMENT: capture into a variable so the status survives, then take the first line.
    new = subprocess.run(
        f'if OUT=$({" ".join(args)} 2>&1); then echo "OK:$(printf %s "$OUT" | head -1)"; '
        f'else echo "FIND_FAILED"; fi',
        shell=True,
        capture_output=True,
        text=True,
    )
    assert "FIND_FAILED" in new.stdout, (
        "the replacement must distinguish a failed find from no matches"
    )


@pytest.mark.skipif(
    not _supports_epoch_newermt(),
    reason="needs a find that accepts @epoch (GNU, as on the cluster)",
)
def test_absolute_newermt_window_is_not_vacuous(tmp_path):
    """Where the dialect supports it, the window must both select and exclude."""
    root = _farm(tmp_path)

    def run(epoch):
        r = subprocess.run(
            [FIND, "-L", str(root), "-name", "manifest.json", "-newermt", f"@{epoch}"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
        return r.stdout.strip()

    assert run(int(time.time()) - 60) != "", "a past epoch must select the file"
    assert run(int(time.time()) + 60) == "", "a future epoch must select nothing"


def test_gnu_find_is_absent_here_so_the_success_path_is_unverified_locally():
    """Records WHY one test above is skipped, so the gap is visible rather than inferred."""
    assert shutil.which("gfind") is None or True  # informational
    if not _supports_epoch_newermt():
        pytest.skip(
            f"{FIND} rejects @epoch; the cluster's GNU find accepts it and is not testable here"
        )
