#!/usr/bin/env python
"""Generate and seal a preregistration stage.

Deliberately a SEPARATE command from the sweep runner, run by a human on a date, because
sealing is a commitment and commitments should not be a side effect of running an experiment.

  python scripts/seal_prereg.py draft            # print stage 1, seal nothing
  python scripts/seal_prereg.py seal-stage1      # write-once, then `git tag prereg-stage1`
  python scripts/seal_prereg.py seal-stage2 ...  # pilot-derived constants
  python scripts/seal_prereg.py verify           # recompute every digest
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi_eval.prereg import (  # noqa: E402
    PreregStage2,
    PreregTampered,
    check_live_constants,
    default_stage1,
    live_constants,
    seal,
    verify,
)

ROOT = Path(__file__).resolve().parents[1] / "prereg"


def _load_exclusions() -> dict[str, list[str]]:
    """Read builder-emitted exclusion lists. Never invents ids, and never silently finds none.

    THIS READ A PATH AND A SHAPE THE BUILDER HAS NEVER PRODUCED. Two independent mismatches,
    both hidden by an `except Exception: pass`:

      * PATH. `musique_build` writes `<gold>/graphs/<suite>/excluded_<split>.json`; this globbed
        `data/gold/*/excluded.json`, which matches neither the directory depth nor the filename.
        Measured: the glob returned [] while
        data/gold/graphs/musique/excluded_train.json sat on disk.
      * SHAPE. The builder writes a dict -- {split, exclusion_list, n_blocked_questions,
        n_excluded_tasks, excluded_task_ids}. This did `sorted(json.loads(...))`, which over a
        dict returns its KEYS, so a path fix alone would have frozen
        `["excluded_task_ids", "exclusion_list", ...]` as the list of excluded task ids.

    So `_load_exclusions()` returned `{}` on every invocation, and `seal-stage1` would have
    frozen `excluded_task_ids: {}` -- a preregistration formally declaring no contamination
    control -- while the builder had already identified 24 leaked musique tasks and written
    them down. Nothing would have been wrong in any table; the GUARANTEE would simply not
    exist, which is worse, because it is the guarantee the paper cites.

    A malformed file now raises. An exclusion list that cannot be read is the one case where
    proceeding quietly produces exactly the document this function exists to prevent.
    """
    out: dict[str, list[str]] = {}
    gold = Path(__file__).resolve().parents[1] / "data" / "gold" / "graphs"
    for p in sorted(gold.glob("*/excluded_*.json")):
        payload = json.loads(p.read_text())  # deliberately unguarded
        ids = payload["excluded_task_ids"]
        if not isinstance(ids, list):
            raise TypeError(f"{p}: excluded_task_ids is {type(ids).__name__}, expected list")
        out.setdefault(p.parent.name, []).extend(str(i) for i in ids)
    return {suite: sorted(set(ids)) for suite, ids in sorted(out.items())}


class BurnedPilotsMissing(RuntimeError):
    """A pilot ran on tasks the seal declares were never piloted."""


def _load_burned_pilots() -> dict[str, list[str]]:
    """Task ids a PILOT already ran on, read from the compacted runs.

    A pilot burns its tasks: the design was chosen after seeing them, so re-using them in
    the confirmatory set makes the confirmatory test partly a re-test of the data that
    picked the hypothesis. `burned_pilot_ids` is the field that records this, and it was
    never produced -- `default_stage1` was called with exclusions only, so every seal
    declared `{}`: a formal statement that no pilot was ever run.

    Reads `pilot_flag = TRUE`, which `pi_run.compact` sets from the grid's `pilot:` key.
    Absent parquet -> empty, and `_check_burned_pilots` is what turns that into a refusal
    when a pilot really did run.
    """
    root = Path(__file__).resolve().parents[1] / "scores" / "parquet" / "runs.parquet"
    if not root.exists():
        return {}
    try:
        import duckdb
    except ModuleNotFoundError:
        return {}
    rows = (
        duckdb.connect()
        .execute(
            "SELECT suite_id, task_id FROM read_parquet(?) WHERE pilot_flag = TRUE",
            [str(root)],
        )
        .fetchall()
    )
    out: dict[str, list[str]] = {}
    for suite, task in rows:
        out.setdefault(str(suite), []).append(str(task))
    return {suite: sorted(set(ids)) for suite, ids in sorted(out.items())}


def _pilot_suites() -> set[str]:
    """Suites whose GRID declares a pilot, independent of whether it has run yet."""
    import yaml

    out: set[str] = set()
    for p in sorted((Path(__file__).resolve().parents[1] / "conf" / "grids").glob("*.yaml")):
        g = yaml.safe_load(p.read_text()) or {}
        if g.get("pilot"):
            out.update(str(s) for s in (g.get("suites") or []))
    return out


def _load_grid_hashes() -> dict[str, str]:
    """grid name -> `Grid.source_sha256`, read through the loader the runner uses.

    Through `grids.load` rather than hashing the file here, so the sealed value is the same
    string a run records in its manifest. Two independent hashes of one file would be a
    comparison that can fail for reasons that are not drift.
    """
    from pi_run.grids import load

    out: dict[str, str] = {}
    for path in sorted((Path(__file__).resolve().parents[1] / "conf" / "grids").glob("*.yaml")):
        try:
            g = load(path)
        except Exception:  # a malformed grid is a separate failure; do not mask it as drift
            continue
        out[g.name] = g.source_sha256
    return dict(sorted(out.items()))


def _pilot_slices() -> dict[str, set[str]]:
    """suite -> the exact task ids each PILOT grid selects.

    Read from the grid the same way the runner reads it (`n_tasks` after `task_offset` and
    `split`), so the answer is the tasks a pilot would actually have run rather than "the
    suite appears somewhere".
    """
    import yaml

    from pi_run.cli import _resolve_corpus, _select_task_ids
    from pi_run.worker import load_suite

    root = Path(__file__).resolve().parents[1]
    out: dict[str, set[str]] = {}
    for path in sorted((root / "conf" / "grids").glob("*.yaml")):
        g = yaml.safe_load(path.read_text()) or {}
        if not g.get("pilot"):
            continue
        for suite_id in g.get("suites") or ():
            try:
                suite = load_suite(str(suite_id), str(_resolve_corpus(root, str(suite_id), None)))
            # SystemExit, NOT just Exception. `_resolve_corpus` raises SystemExit when the
            # corpus is missing, and SystemExit derives from BaseException -- so the intent
            # stated in the old `except Exception` ("a suite whose corpus is absent cannot have
            # been piloted") was not what the code did: a checkout without data/ crashed the
            # sealer instead of skipping the suite.
            except (Exception, SystemExit):
                continue
            ids = _select_task_ids(
                suite,
                str(suite_id),
                n=int(g.get("n_tasks") or 0),
                split=g.get("split"),
                offset=int(g.get("task_offset") or 0),
            )
            out.setdefault(str(suite_id), set()).update(ids)
    return out


def _piloted_tasks() -> dict[str, set[str]]:
    """suite -> task ids that have a run a PILOT produced, i.e. `pilot_flag = TRUE`.

    THE FLAG, NOT THE PRESENCE OF ROWS. This read "any compacted run, whatever grid produced
    it", and that is a different question with a different answer. MEASURED on this
    repository: `conf/grids/tier1_pilot.yaml` takes musique at task_offset 200 for 120 tasks
    and declares no split, so its slice straddles the wall (73 train / 29 test / 18 dev) --
    and those 120 ids carry 3,830 runs from OTHER grids (latent_trainset, growth_parents_*,
    dev_select_musique). None of them is a pilot; `pilot_flag = TRUE` selects 0 of them. The
    old signal therefore refused to seal over a pilot that has never run, and the fix is to
    ask the question the flag answers.

    `pilot_flag` is set by `pi_run.compact` from the grid's `pilot:` key, and it rides inside
    `semantic_hash`, so it cannot be retro-fitted onto a run that was not one.
    """
    root = Path(__file__).resolve().parents[1] / "scores" / "parquet" / "runs.parquet"
    if not root.exists():
        return {}
    try:
        import duckdb
    except ModuleNotFoundError:
        return {}
    rows = (
        duckdb.connect()
        .execute(
            "SELECT DISTINCT suite_id, task_id FROM read_parquet(?) WHERE pilot_flag = TRUE",
            [str(root)],
        )
        .fetchall()
    )
    out: dict[str, set[str]] = {}
    for suite, task in rows:
        out.setdefault(str(suite), set()).add(str(task))
    return out


def _pilots_that_ran(
    *, pilot_slices: Mapping[str, set[str]], piloted: Mapping[str, set[str]]
) -> set[str]:
    """Suites where a pilot grid's OWN TASKS carry a PILOT run, so burned ids must be present.

    Three signals have been wrong here, and all three failed silently:

      * `pilot_suites & set(burned)` is empty exactly when `burned` is empty, so the guard
        could never fire in the one case it exists for.
      * "the suite has any runs at all" fires on this repo today: musique carries runs from
        sweeps that were not the pilot.
      * "the pilot slice's TASKS have any runs at all" fires too, and that is what blocked the
        drafted seal: 3,830 runs from three other grids sit on those 120 musique ids.

    WHAT THIS COSTS, STATED. `burned` is read from the same `pilot_flag = TRUE` rows, so a
    suite that reaches this set almost always has burned ids and the refusal below is close to
    a tautology -- it is now a COHERENCE check between two reads of the pilot flag rather than
    a contamination detector. That is the honest shape of it, and it is why the seal no longer
    rests on this function: `_check_trained_arm_not_on_test` is the assertion that can
    actually catch a checkpoint already evaluated on the confirmatory population.
    """
    return {
        suite
        for suite, tasks in pilot_slices.items()
        if tasks and (tasks & piloted.get(suite, set()))
    }


class TrainedArmAlreadyOnTest(RuntimeError):
    """A trained arm was already run on a task the splitter calls `test`, before the seal."""


def _trained_runs_on_confirmatory_tasks(*, runs_parquet: Path | None = None) -> list[dict]:
    """Runs of a TRAINED arm sitting on a held-out task, recomputed from the ids. Empty is
    the good case.

    WHY THE SEAL NEEDS THIS AND THE PILOT GUARD CANNOT GIVE IT. The seal is what AUTHORISES
    the trained arm's comparison on the confirmatory population. A trained-arm run that
    already exists on one of those tasks was produced outside that authorisation -- a
    selection sweep, a dev gate, a sanity check -- which means the checkpoint has already been
    looked at there, and the "confirmatory" test is partly a re-test of the data that chose
    the policy. Such runs carry no `pilot_flag` and sit on tasks no pilot grid selects, so
    `_pilots_that_ran` is structurally blind to them.

    RECOMPUTED, NOT READ OFF `r.split`. The stamped column records what the grid believed; the
    question is what the splitter says about the id, and a disagreement between the two is
    itself the finding.

    THE TEMPLATE IS PASSED AS THE TEMPLATE. `split_of(suite, task_id, template_id)` hashes on
    the template where a suite mints one -- musique does, the sorted hop ids -- so the
    two-argument call buckets a different thing, and on a FIXED-split suite it raises outright
    rather than answering. A first pass without it produced a false alarm.

    An absent parquet RAISES. Nothing to read is not evidence of nothing to find, and a seal
    that passed because the file was missing would be the worst of both.
    """
    p = Path(runs_parquet or (Path(__file__).resolve().parents[1] / "scores/parquet/runs.parquet"))
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist, so the trained-arm scan cannot run. Run `pi compact` first; "
            "sealing without this scan is sealing without knowing whether the checkpoint has "
            "already been evaluated on the confirmatory tasks."
        )
    import duckdb

    from pi_eval.report import TRAINED_ARMS
    from pinq.splitting import split_of

    arms = ", ".join(f"'{a}'" for a in sorted(TRAINED_ARMS))
    rows = (
        duckdb.connect()
        .execute(
            "SELECT DISTINCT run_id, suite_id, task_id, arm_id, template_id "
            f"FROM read_parquet(?) WHERE arm_id IN ({arms}) ORDER BY 1",
            [str(p)],
        )
        .fetchall()
    )
    out: list[dict] = []
    for run_id, suite, task, arm, template in rows:
        recomputed = split_of(str(suite), str(task), str(template or "") or None)
        if recomputed == "test":
            out.append(
                {
                    "run_id": str(run_id),
                    "suite_id": str(suite),
                    "task_id": str(task),
                    "arm_id": str(arm),
                    "template_id": str(template or ""),
                    "recomputed_split": recomputed,
                }
            )
    return out


def _check_trained_arm_not_on_test(violations: Sequence[Mapping[str, object]]) -> None:
    """REFUSE LOUDLY. Sealing over these would preregister a comparison that has already
    happened."""
    if not violations:
        return
    shown = ", ".join(
        f"{v['run_id']} ({v['suite_id']}/{v['task_id']}, {v['arm_id']})" for v in violations[:5]
    )
    raise TrainedArmAlreadyOnTest(
        f"{len(violations)} trained-arm run(s) already exist on tasks the splitter calls "
        f"`test`: {shown}{' ...' if len(violations) > 5 else ''}. The seal is what authorises "
        "that comparison, so a run that precedes it means the checkpoint was already looked "
        "at on the confirmatory population. Either exclude those run ids in stage 1 "
        "(`excluded_run_ids`) with the reason written down, or do not claim those tasks as "
        "confirmatory."
    )


def _check_burned_pilots(burned: Mapping[str, Sequence[str]], pilot_ran: Iterable[str]) -> None:
    """REFUSE LOUDLY rather than seal a false negative declaration.

    An empty list is not a neutral default here: it is the claim "no pilot touched these
    tasks", which the paper's contamination guarantee cites directly. If a pilot grid ran
    and produced no burned ids, something is wrong with the compaction, not with the pilot.
    """
    for suite in sorted(pilot_ran):
        if not burned.get(suite):
            raise BurnedPilotsMissing(
                f"{suite}: a pilot grid ran but burned_pilot_ids is empty. Sealing now would "
                f"declare that no pilot ever touched these tasks. Run `pi compact` so "
                f"pilot_flag reaches runs.parquet, or remove the pilot grid for {suite}."
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["draft", "seal-stage1", "seal-stage2", "verify"])
    ap.add_argument("--sigma-j", type=float)
    ap.add_argument("--theta-nli", type=float, default=0.85)
    ap.add_argument("--discoverability-threshold", type=float, default=0.60)
    ap.add_argument("--word-cap", type=int, default=120)
    args = ap.parse_args()

    if args.cmd == "verify":
        r = verify(ROOT)
        out = dataclasses.asdict(r)
        # A SEALED CONSTANT NOTHING COMPARES AGAINST IS A DECORATION. `PreregStage2` is defined,
        # sealed, and read by nothing in src/, so an edit to DWR_WEIGHTS after sealing would
        # diverge from the preregistration with no signal anywhere. Digests catch a changed
        # PREREG; this catches changed CODE.
        #
        # BOTH STAGES. This read stage2.json only, while stage 1 now seals the kill-switch
        # equivalence MARGINS and the resample count -- the margins decide whether a switch
        # fires and the resample count decides what every interval means, so a document
        # checked on its weights and not on those is checked on the cheaper half.
        drift: list[str] = []
        for stage in ("stage1", "stage2"):
            p = ROOT / f"{stage}.json"
            if p.exists():
                drift += [f"{stage}: {d}" for d in check_live_constants(json.loads(p.read_text()))]
        out["constant_drift"] = drift
        print(json.dumps(out, indent=2))
        if r.unsealed:
            print(
                f"UNSEALED in the prereg directory: {r.unsealed}. These carry no digest, so an "
                "edit to them is invisible to `verify`. sigma_j.json gets here because "
                "measure_sigma_j.py --write bypasses seal(); run the next seal-stage to bring "
                "it under the manifest.",
                file=sys.stderr,
            )
        return 0 if (r.ok and not drift and not r.unsealed) else 1

    burned = _load_burned_pilots()
    if args.cmd in ("seal-stage1", "draft"):
        _check_burned_pilots(
            burned,
            _pilots_that_ran(pilot_slices=_pilot_slices(), piloted=_piloted_tasks()),
        )
        # THE SHARPER ONE. See `_trained_runs_on_confirmatory_tasks`: the pilot guard cannot
        # see a checkpoint that was already evaluated on the confirmatory tasks, because such
        # runs carry no pilot_flag and sit on tasks no pilot grid selects.
        _check_trained_arm_not_on_test(_trained_runs_on_confirmatory_tasks())
    s1 = default_stage1(
        excluded_task_ids=_load_exclusions(),
        burned_pilot_ids=burned,
        grid_hashes=_load_grid_hashes(),
    )

    if args.cmd == "draft":
        print(json.dumps(dataclasses.asdict(s1), indent=2, sort_keys=True))
        print(
            f"\n# confirmatory tests: {len(s1.primary)} primary + "
            f"{len(s1.secondary_family)} secondary = "
            f"{len(s1.primary) + len(s1.secondary_family)}",
            file=sys.stderr,
        )
        return 0

    if args.cmd == "seal-stage1":
        try:
            p = seal(ROOT, "stage1", s1)
        except (FileExistsError, PreregTampered) as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        print(f"sealed {p}\nNow: git add prereg && git commit && git tag prereg-stage1")
        return 0

    if args.sigma_j is None:
        print(
            "seal-stage2 needs --sigma-j, measured by test-retest. It cannot be guessed: "
            "it is the gate on what may be claimed at all.",
            file=sys.stderr,
        )
        return 2
    # FROM THE CODE, NOT RETYPED. These were literals here and had drifted: this froze
    # dwr_weights {"0":0.1,...,"3":0.4} while pi_eval.score.DWR_WEIGHTS is
    # {0:0.5,...,5:2.0} -- different values and two fewer depths -- so the preregistration
    # would have frozen a weighting no reported `dwr` has ever been computed with.
    live = live_constants()
    s2 = PreregStage2(
        sigma_j=args.sigma_j,
        tau=2 * args.sigma_j,  # derived, never chosen
        dwr_weights=live["dwr_weights"],  # type: ignore[arg-type]
        theta_nli=args.theta_nli,
        discoverability_threshold=args.discoverability_threshold,
        word_cap=args.word_cap,
        # The frontier grid is DATA-DEPENDENT: report.py builds it as
        # log2_grid(1.0, max(observed_spend, 2.0), FRONTIER_GRID_POINTS), so a fixed tuple here
        # would preregister a grid the report never uses. What is preregisterable is the number
        # of points and the log2 spacing rule, and that is what is frozen.
        budget_grid=tuple(range(1, int(live["frontier_grid_points"]) + 1)),  # type: ignore[call-overload]
        notes=(
            "tau = 2*sigma_J by construction; no effect below 2*sigma_J/sqrt(n) is reportable. "
            "dwr_weights are read from pi_eval.score.DWR_WEIGHTS at seal time and re-checked by "
            "`verify`. budget_grid records the NUMBER of log2-spaced frontier points; the "
            "endpoints are data-dependent (report.py: log2_grid(1, max(spend, 2), n))."
        ),
    )
    try:
        p = seal(ROOT, "stage2", s2)
    except (FileExistsError, PreregTampered) as exc:
        # A clean nonzero status, not a traceback. Re-running a seal is an ordinary operator
        # mistake and the refusal is the designed behaviour, so it should read as a refusal.
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(f"sealed {p}\nNow: git add prereg && git commit && git tag prereg-stage2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
