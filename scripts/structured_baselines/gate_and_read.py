"""Gate the clean structured-baselines grid, then read its contrasts -- and REFUSE if a gate fails.

WHY THIS IS A SCRIPT AND NOT A SESSION OF COMMANDS. The first attempt at this grid produced three
invocations at three code_versions, one arm spanning all three and another spanning two, so
"latest wins" selected a MIXTURE rather than a population. A second attempt failed 400/400 units at
$0 against a proxy with no gateway base URL. Neither failure was visible in the numbers that came
out of them: a pooled contrast across code_versions still prints, and a 400/400 failure still
leaves an arm with an arm_id. So the gates below run BEFORE any contrast and the script exits
non-zero rather than printing a number.

WHAT EACH GATE WOULD CATCH, since a gate that cannot fail is not a gate:

  one code_version        the mixture that sank the first attempt. Two versions is a refusal, not
                          a footnote, because 221 of 1,200 cells disagreed across commits once.
  one base_url_sha        base_url_sha sits inside model_pin_hash inside run_id, so an arm on a
                          different endpoint is a different experiment even with zero errors. The
                          first attempt measured par2_rag twice, once per endpoint.
  distinct inquirer prompts  an arm whose prompt slot is never read passes every other check and
                          produces a clean, meaningless null. par2_rag names its prompts after the
                          policy (par2_rag_plan, par2_rag_refine) and not after the role, so this
                          compares the SET of inquirer-ish keys rather than looking for one name.
  identical frozen prompts   the other half of the same check: if drafter or answerer prompts differ
                          between arms then more than the inquirer differs and the contrast is not
                          the one claimed.
  two arms at minimum     every arm-comparing gate is guarded by len(present) >= 2 and would SKIP
                          on a single-arm store, returning no failures for a population that
                          cannot support a contrast at all. Refused before those guards run.
  no dev- prefix          a dirty tree stamps runs training-only.
  ok == distinct cells    a sweep that spans a commit re-runs finished units under new run_ids,
                          and the ok counter counts each as progress. A peer burned ~79 units
                          producing zero new cells while reporting them as advancement.
  paired intersection     NOT balance. A paired contrast reads on the intersection by construction,
                          so one stray error costs one task. What is fatal is an intersection much
                          smaller than the arms, meaning the survivors were selected -- the first
                          attempt lost 171 of 400 strategyqa units on ONE arm and none on the other.

Usage:
    python scripts/structured_baselines/gate_and_read.py <runs_root> [--parquet DIR]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib


class MissingRoot(RuntimeError):
    """The runs root itself is absent. Distinct from a root that exists and holds nothing."""


ARMS = ("inquirer_prompted", "par2_rag", "inquirer_trained")
FROZEN_PROMPTS = ("answerer", "answerer_frozen", "drafter_draft", "drafter_resolve")


def load(runs_root: pathlib.Path) -> list[dict]:
    """Read every manifest under runs_root, following symlinks.

    An isolated store is a symlink farm, so a plain rglob that does not follow links reads it as
    empty -- which looks exactly like a campaign that never ran.
    """
    if not runs_root.is_dir():
        # A missing root is the commonest cause of an empty read and it is almost always a typo or
        # a stale path, not a failed campaign. Raising FileNotFoundError out of os.listdir buries
        # that under a traceback AND exits 1, which a caller checking only for non-zero cannot
        # distinguish from "the data failed the gates" at exit 2. Distinct exit code, named cause.
        raise MissingRoot(f"runs root does not exist or is not a directory: {runs_root}")
    out = []
    for d in sorted(runs_root.iterdir()):
        mf = d / "manifest.json"
        if not mf.exists():
            continue
        try:
            m = json.loads(mf.read_text())
        except Exception:
            continue
        st = d / "status.json"
        m["_status"] = None
        if st.exists():
            try:
                m["_status"] = json.loads(st.read_text()).get("status")
            except Exception:
                m["_status"] = "torn"
        out.append(m)
    return out


def gates(ms: list[dict]) -> list[str]:
    """Return a list of failures. Empty means every gate passed."""
    fail: list[str] = []
    ok = [m for m in ms if m["_status"] == "ok"]
    if not ok:
        return ["no units with status ok"]

    cvs = {m.get("code_version", "?")[:12] for m in ok}
    if len(cvs) != 1:
        fail.append(f"{len(cvs)} code_versions among ok units: {sorted(cvs)}")

    urls: set[str] = set()
    for m in ok:
        for pin in (m.get("pins") or {}).values():
            urls.add(str(pin.get("base_url_sha"))[:12])
    if len(urls) != 1:
        fail.append(f"{len(urls)} distinct base_url_sha across arms and roles: {sorted(urls)}")

    dev = [m for m in ok if str(m.get("run_id", "")).startswith("dev-")]
    if dev:
        fail.append(f"{len(dev)} dev- prefixed run ids")

    # ok runs must equal DISTINCT cells. A peer's sweep spanned its own guard-fix commits, which
    # moved code_version and therefore every run_id, so resume stopped recognising finished units
    # and about 79 units produced ZERO new cells while the ok counter counted every one of them as
    # progress. At one code_version resume overwrites in place and the two are equal by
    # construction, so this is cheap; it earns its place by failing loudly if they ever diverge,
    # which is the only signal that a completion count is measuring re-runs.
    cells = {(m.get("suite_id"), m.get("task_id"), m.get("seed"), m.get("arm_id")) for m in ok}
    if len(cells) != len(ok):
        fail.append(
            f"{len(ok)} ok runs but only {len(cells)} distinct cells: "
            f"{len(ok) - len(cells)} are re-runs being counted as progress"
        )

    per_arm = collections.Counter(m.get("arm_id") for m in ok)
    present = [a for a in ARMS if per_arm.get(a)]
    # FEWER THAN TWO ARMS IS A REFUSAL, and this is the hole a peer's equivalent had. Every check
    # below that compares arms is guarded by len(present) >= 2, so they SKIP silently on a
    # single-arm store and the function returns "no failures" -- a population that cannot support
    # any contrast certifying as ready for one. Worse than the empty case, because 80 ok runs at
    # one code_version and one base_url_sha look healthy. Their version printed a 2x2 of zeros
    # whose prompted_only=0 was byte-identical to a real result where the comparator never won: a
    # true zero attached to a different question.
    if len(present) < 2:
        fail.append(
            f"only {len(present)} arm present ({present or 'none of ' + str(list(ARMS))}): "
            "no contrast is possible, and every arm-comparing gate below would skip silently"
        )
        return fail
    # Balance is not the test. A paired contrast reads on the INTERSECTION by construction, so a
    # stray errored unit costs one task and is not a defect. What matters is whether the
    # intersection is much smaller than the arms, which means the surviving tasks were selected --
    # and selection on the treatment side is what sank the first attempt at this grid, where one
    # arm lost 171 of 400 strategyqa units and the other lost none.
    tasks: dict[str, set[tuple[str, str, int]]] = {}
    for a in present:
        tasks[a] = {
            (str(m.get("suite_id")), str(m.get("task_id")), int(m.get("seed", -1)))
            for m in ok
            if m.get("arm_id") == a
        }
    if len(present) >= 2:
        inter = set.intersection(*(tasks[a] for a in present))
        # Against the LARGEST arm, not the smallest. Comparing to the smallest made this gate
        # unable to fire at all: if one arm loses half its units, the intersection equals that
        # arm's survivors, so the ratio is 1.00 no matter how severe the selection. A test that
        # dropped 40 of 80 units on one arm reported "40 of 40" and passed. The largest arm is the
        # population the grid intended, so it is the right denominator.
        largest = max(len(tasks[a]) for a in present)
        print(f"  paired intersection: {len(inter)} of {largest} in the largest arm")
        if largest and len(inter) < 0.95 * largest:
            fail.append(
                f"intersection {len(inter)} is under 95% of the largest arm {largest}: "
                f"the surviving tasks are selected, per-arm counts {dict(per_arm)}"
            )

    # Two arms must differ on the inquirer side in SOMETHING: the prompt or the weights. An arm
    # whose prompt slot is never read, on the same weights, is the same arm under two names. The
    # trained questioner runs the prompted template on its own weights, so identical prompts on
    # DIFFERENT inquirer models are the design, not a defect (the first version of this gate refused
    # that pair on the live grid). Frozen prompts must be IDENTICAL.
    inq: dict[str, set[str]] = {}
    inq_models: dict[str, set[str]] = {}
    frozen: dict[str, dict[str, str]] = {}
    for a in present:
        keys: set[str] = set()
        models: set[str] = set()
        fz: dict[str, str] = {}
        for m in ok:
            if m.get("arm_id") != a:
                continue
            models.add(str(((m.get("pins") or {}).get("inquirer") or {}).get("model_id")))
            for k, v in (m.get("prompt_hashes") or {}).items():
                if k in FROZEN_PROMPTS:
                    fz[k] = v
                elif k not in ("fragment_user_channel_placebo",):
                    keys.add(f"{k}={v[:12]}")
        inq[a] = keys
        inq_models[a] = models
        frozen[a] = fz
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            if inq[a] and inq[a] == inq[b] and inq_models[a] == inq_models[b]:
                fail.append(
                    f"{a} and {b} record IDENTICAL inquirer prompts on identical inquirer "
                    f"weights {sorted(inq_models[a])}: {sorted(inq[a])}"
                )
            shared = set(frozen[a]) & set(frozen[b])
            diff = {k for k in shared if frozen[a][k] != frozen[b][k]}
            if diff:
                fail.append(f"{a} and {b} differ on FROZEN prompts {sorted(diff)}")
    return fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root", type=pathlib.Path)
    args = ap.parse_args()

    try:
        ms = load(args.runs_root)
    except MissingRoot as exc:
        print(f"REFUSING: {exc}")
        return 3
    print(f"manifests: {len(ms)}")
    by = collections.Counter((m.get("arm_id"), m.get("suite_id"), m["_status"]) for m in ms)
    for k in sorted(by, key=str):
        print(f"  {str(k):58s} {by[k]}")

    fail = gates(ms)
    print()
    if fail:
        print("GATES FAILED, no contrast will be read:")
        for f in fail:
            print(f"  - {f}")
        return 2
    print("gates passed: one code_version, one base_url_sha, distinct inquirer prompts,")
    print("identical frozen prompts, zero dev- prefixes, unselected paired intersection.")
    print()
    print("NOT DONE HERE, deliberately, and the next step has a trap in it.")
    print()
    print("1. Score first. The contrast needs a scored store and these rollouts ran with")
    print("   PI_GOLD_ROOT unset, which is the firewall working. Scoring exports it.")
    print("2. Read the contrast through scripts/matched_cost.py, so verify_instrument refuses")
    print("   unless the reconstructed ladder reproduces what the stored scorer wrote. A ladder")
    print("   built for the occasion is not the instrument that produced the paper's figures.")
    print("3. USE THE SYMMETRIC RULE. matched_cost.contrast() is asymmetric by construction: it")
    print("   adds an offset to the arm passed as `trained` and reads the comparator at")
    print("   min(k, its own asks), so it truncates the COMPARATOR only. Table 1 was moved off")
    print("   that rule in September because it flattered the treatment on exactly the pairs")
    print("   where the comparator is the shorter side. par2_rag asks about as many questions as")
    print("   inquirer_prompted, so the two rules will nearly agree here -- which is the danger,")
    print("   because agreement is not a check and the rule still has to be the reported one.")
    print("   The committed symmetric implementation is scripts/plan_metrics_completed/")
    print("   symmetric_completed.py, which locks against the published asymmetric value first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
