# Repo root and scratch dir come from the environment: a committed absolute home path is
# how a research repo stops being reproducible for anyone but its author.

import collections
import json
import os
import pathlib

PI_REPO = os.environ.get("PI_REPO", os.getcwd()).rstrip("/")


out = []
for line in open("/private/tmp/mech-ks-idl/control_manifests.txt"):
    p = pathlib.Path(line.strip())
    m = json.load(open(p))
    st = {}
    sp = p.parent / "status.json"
    if sp.exists():
        st = json.load(open(sp))
    inq = (m.get("pins") or {}).get("inquirer") or {}
    out.append(
        dict(
            run_id=m["run_id"],
            arm=m["arm_id"],
            suite=m["suite_id"],
            task=m["task_id"],
            seed=m["seed"],
            split=m["split"],
            cv=m["code_version"],
            dirty=m.get("dirty"),
            cap=m.get("budget_cap"),
            grid=m.get("grid_name"),
            pin=m.get("model_pin_hash"),
            tmpl=m.get("template_id"),
            exploratory=m.get("exploratory"),
            gold_exposed=m.get("gold_exposed"),
            canary_hit=m.get("canary_hit"),
            firewall_ok=m.get("firewall_ok"),
            cf=m.get("counterfactual_kind"),
            status=st.get("status"),
            n_asks=st.get("n_asks"),
            retr=st.get("retrieval_calls"),
            rt=st.get("reconciled_tokens"),
            rd=st.get("reconciled_docs"),
            is_dev=m.get("is_dev_run", st.get("is_dev_run")),
            pilot=m.get("pilot_flag"),
            inq_model=inq.get("model_id"),
            inq_adapter=inq.get("adapter_sha"),
            dirn=str(p.parent),
        )
    )
json.dump(out, open("/private/tmp/mech-ks-idl/controls_disk.json", "w"))
c = collections.Counter(
    (r["arm"], r["cv"][:7], r["suite"], r["split"], r["cap"], r["status"]) for r in out
)
for k, v in sorted(c.items()):
    print(k, v)
