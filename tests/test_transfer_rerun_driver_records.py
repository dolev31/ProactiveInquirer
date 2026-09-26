"""The transfer reader reads what the extra-pin driver itself writes (RULES amendment 11).

End to end on lane D's `run_extra` (scripts/tau2_rerun/run_extra_pin.py, b3a81d8), through lane D's
own fixtures -- a staged git tree and fake units -- so the ARM_LAUNCH.json, the job directory and
the questioner probe record read here are the driver's bytes, not a restatement of them. Kept apart
from test_transfer_rerun.py because lane D's `job` fixture must be imported by that name.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "tau2_concordance"))
import transfer_rerun as tr  # noqa: E402

# lane D's own fixtures: a staged git tree and the driver's real run_extra with fake units
from test_tau2_rerun_extra_pin import _run as _drv_run  # noqa: E402
from test_tau2_rerun_extra_pin import drv as DRV  # noqa: E402
from test_tau2_rerun_extra_pin import job, staged  # noqa: E402, F401

G120 = DRV.EXTRA_PIN
CAMPAIGN = DRV.CAMPAIGN_SHA
C = "inquirer_prompted"


def test_the_reader_reads_the_records_the_driver_itself_writes(job):  # noqa: F811
    """End to end on `run_extra` itself: the reader finds the job the driver staged, reads the
    ARM_LAUNCH.json it wrote and the questioner probe record it copied, and accepts them at the
    staged tree's own sha -- and refuses them at the real campaign sha, which that tree is not."""
    assert _drv_run(job) == 0
    runs_root = Path(job.a.runs_root)
    arm, suite = runs_root.parent, job.a.suite
    store = runs_root / tr.comparator_store_dir(G120)
    named = {
        json.loads(p.read_text())["pins"]["inquirer"]["model_id"]
        for p in store.glob("*/manifest.json")
    }
    assert named == {G120}
    jobs = tr.comparator_job_dirs(arm, [suite])
    assert [d.name for d in jobs] == ["gptoss120b-t"]
    emit = tr.Emit(None)
    kw = dict(records=[], suites=[suite], pin=G120, c_arm=C, legacy=False, disabled=[], emit=emit)
    planned, launch = tr.load_arm_launch(
        [str(arm)], code_version=job.sha, campaign_sha=job.sha, **kw
    )
    (rec,) = launch
    assert (rec["regime"], rec["driver_sha256"], rec["harness_sha"]) == (
        DRV.GATEWAY_ONLY,
        DRV.driver_sha256(),
        job.sha,
    )
    assert sum(len(v) for v in planned.values()) == 204
    tr.check_comparator_jobs(
        jobs, launch_by_suite={suite: rec}, pin=G120, campaign_sha=job.sha, emit=emit
    )
    q = tr.check_questioner_probe(
        [G120], [], emit, paths=[], job_dirs=jobs, code_version=job.sha, campaign_sha=job.sha
    )
    assert q["questioner_probe"]["max_tokens"] == {G120: job.max_tokens}
    with pytest.raises(tr.Refusal, match="harness_sha"):
        tr.load_arm_launch([str(arm)], code_version=job.sha, campaign_sha=CAMPAIGN, **kw)
