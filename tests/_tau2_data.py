"""Skip a test that needs tau2's data -- unless the run REQUIRES it, and then fail.

A skip prints as the same grey word as a pass, so a staged tree that silently lacks the tau2
data would report every real-data test here as fine. `PINQ_REQUIRE_TAU2_DATA=1` (set by the HPC
wrapper) turns an unreachable tau2 -- the package, or the domain's `db.json` under
`TAU2_DATA_DIR` -- into a FAILURE. Read at call time, not import time, so a test can set it.

The skip message is `available()`'s own reason, unchanged, which is what
`tests/_skip_gates.py` matches against.
"""

from __future__ import annotations

import os

import pytest

REQUIRE_ENV = "PINQ_REQUIRE_TAU2_DATA"


def require_tau2_data(domain: str) -> None:
    """Return if tau2 and `domain`'s data are reachable; otherwise skip, or fail if required."""
    from pinq_adapters.tau2._probe import available, domain_data_dir

    ok, why = available()
    if ok:
        try:
            if not (domain_data_dir(domain) / "db.json").is_file():
                ok, why = False, f"no db.json under {domain_data_dir(domain)}"
        except Exception as exc:  # noqa: BLE001 - the reason is the message, not the type
            ok, why = False, f"{type(exc).__name__}: {exc}"
    if ok:
        return
    if os.environ.get(REQUIRE_ENV) == "1":
        pytest.fail(f"{REQUIRE_ENV}=1 and tau2 {domain} data is unreachable: {why}")
    pytest.skip(why)
