"""An unsealed preregistration and a broken one are opposite findings.

`_seal_warnings` has three states and derives them from a single tri-state `prereg_ok`:
None -> nothing sealed, False -> a digest no longer matches, True -> verified. But
`prereg.verify` returns ok=False for BOTH "MANIFEST.json is missing" and "a digest
changed", so the moment a prereg/ directory existed at all -- this repo grew one when
sigma_j.json was frozen into it -- every rendered table began printing:

    THE SEALED PREREGISTRATION NO LONGER VERIFIES: a stage file's digest does not match its
    manifest. Every claim of preregistration in this table is void until that is explained.

MEASURED at the time: `seal_prereg.py verify` reported checked=0, mismatches=[],
missing=["MANIFEST.json"]. Nothing had ever been sealed, so no digest could have changed
and there was nothing to explain. The two findings call for opposite responses -- "seal
stage 1" versus "something was edited after sealing, stop and investigate" -- and the
alarming one was being printed for the harmless state.

Same rule as `verify arms`: an absent measurement must not be reported as a failed one.
"""

from __future__ import annotations

from pathlib import Path

from pi_eval.prereg import verify


def test_an_empty_directory_is_unsealed_not_mismatched(tmp_path: Path) -> None:
    res = verify(tmp_path)
    assert res.ok is False
    assert res.missing == ["MANIFEST.json"]
    assert res.mismatches == [], "nothing was sealed, so no digest can have changed"
    assert res.checked == 0


def test_a_directory_with_unsealed_files_is_still_unsealed(tmp_path: Path) -> None:
    """The exact state this repo was in: a prereg/ holding sigma_j.json and no manifest."""
    (tmp_path / "sigma_j.json").write_text('{"kpr_incremental": 0.2}')
    res = verify(tmp_path)
    assert res.checked == 0 and res.mismatches == []


def test_the_renderer_says_unsealed_when_nothing_is_sealed(tmp_path: Path) -> None:
    """The warning a reader actually sees must name the state that is true."""
    from pi_eval.report import seal_warnings_for

    (tmp_path / "sigma_j.json").write_text("{}")
    msgs = seal_warnings_for(verify(tmp_path))
    assert len(msgs) == 1
    assert "NOTHING IS SEALED" in msgs[0], msgs[0]
    assert "NO LONGER VERIFIES" not in msgs[0]


def test_a_real_digest_mismatch_still_raises_the_alarm(tmp_path: Path) -> None:
    """The alarm must survive. This is the failure it exists for."""
    import json

    from pi_eval.report import seal_warnings_for

    (tmp_path / "stage1.json").write_text('{"a": 1}')
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"stage1.json": "0" * 64}))
    res = verify(tmp_path)
    assert res.mismatches == ["stage1.json"]
    msgs = seal_warnings_for(res)
    assert len(msgs) == 1
    assert "NO LONGER VERIFIES" in msgs[0], msgs[0]


def test_a_verified_seal_warns_about_nothing(tmp_path: Path) -> None:
    import hashlib
    import json

    from pi_eval.report import seal_warnings_for

    p = tmp_path / "stage1.json"
    p.write_text('{"a": 1}')
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"stage1.json": digest}))
    assert seal_warnings_for(verify(tmp_path)) == ()
