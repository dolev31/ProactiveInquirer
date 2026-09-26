"""Preregistration: write-once, git-tagged, and mechanically checked.

WHY THIS EXISTS. The full design runs ~5 non-baseline arms x 5 budgets x ~9 metrics x 3
suites, and the stratifications multiply that again -- on the order of 2,000 comparisons.
Under the global null at alpha=0.05 you expect ~100 "significant" results and
P(at least one false positive) is essentially 1. Without a commitment made BEFORE the data,
a reviewer is entitled to read the abstract as a selection from a lottery, and they would be
right.

So: ONE primary endpoint per suite (uncorrected), ONE declared secondary family (BH-FDR at
q=0.05), and everything else exploratory -- point estimate and CI, NO p-value, labelled as
such. Total confirmatory tests: 14.

TWO SEALS, because a single seal would be a lie. Constants that do not exist until a pilot
has run (sigma_J, tau, the DWR weights, the discoverability threshold) cannot honestly be
committed on day one, and pretending otherwise is worse than staging it:
  stage 1 -- endpoints, arms, exclusion ids, stopping rules, the burned pilot ids
  stage 2 -- pilot-derived constants, sealed before the first Inquirer arm of the real run

Integrity is checked by content hash, not by trust: `verify()` recomputes the sha256 of every
sealed artifact and compares it to MANIFEST.json.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

Stage = Literal["stage1", "stage2"]

# The pooling wildcard. An endpoint on POOLED is declared over every suite at once, so
# `classify(s1, <any suite>, metric)` resolves it. Kept here rather than in report.py because
# the endpoint declarations live here and a second copy of the sentinel is a second thing to
# drift.
POOLED = "*"

# The test vocabulary, spelled ONCE. report.py compares against these constants rather than
# against its own private "mcnemar"/"signflip" spellings, which is what made the two endpoint
# declarations able to disagree in the first place.
MCNEMAR_EXACT = "mcnemar_exact"
SIGN_FLIP_PERMUTATION = "sign_flip_permutation"
# THE FORK-PAIR TEST, which had no name. A fork endpoint pairs runs on
# (foreign_trace_sha, foreign_prefix_k, seed) and its outcome is a COUNT of follow-up turns:
# there is no cluster to permute and no binary to McNemar, and `pi_eval.fork_report` already
# computes its p-value with an exact two-sided sign test over the informative pairs. Without a
# vocabulary entry a declaration would have had to invent a spelling that report.py cannot
# resolve -- the exact shape of the bug this block of constants exists to prevent.
SIGN_TEST_EXACT = "sign_test_exact"

# THE RESAMPLE COUNT, and the one place it is written down.
#
# It was written down twice and the two disagreed. The drafted claim_rule sealed
# "task-clustered BCa 95% intervals at 1,000 resamples, seed 0"; `report.Agg` defaulted to
# n_boot=10,000 and `pi report --n-boot` to 10,000. So the number the document froze and the
# number every rendered table was computed at were different numbers, and nothing compared
# them: `pi report --n-boot 1000 --seed 0` had to be typed by hand, forever, by everyone.
#
# THE SEALED VALUE IS THE SOURCE and the report follows it, because a preregistration that
# tracks the code is not a preregistration. Stated plainly, because it is a real cost: at
# 1,000 resamples each 2.5% tail of a BCa interval rests on 25 draws, so the interval ENDS are
# visibly grainier than at 10,000 while the point estimate is unaffected. Raising it is a
# one-line change HERE, and `check_live_constants` then reports every sealed document that
# still names the old number instead of letting the two drift apart again.
CI_RESAMPLES = 1000

# The synthetic suite is the ZERO-TOKEN calibration suite: its arms are the LLM-free reference
# components (pinq_expt.fakes), not the LLM arms every other suite uses.
SYNTH_TREATMENT = "fake_chain"
SYNTH_COMPARATOR = "fake_drafter_only"

# "directional" is a preregistered test that RUNS and COUNTS for multiplicity but is not
# powered to confirm: it is reported as a direction with a CI. tau2 is the case -- MDE 22.1pp
# at n=97, simulated through the test that actually runs. The label changes what may be
# CONCLUDED, never what is computed: confirmatory_count, is_confirmatory and classify all key
# off membership in the primary/secondary tuples, not off this field.
Claim = Literal["confirmatory", "directional", "calibration"]


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One declared comparison: WHICH metric, on WHICH suite, against WHICH arm, by WHICH test.

    `metric` is the name a scorer actually emits into `scores.parquet`, not a prose label for
    the comparison. That is enforced by a test (`test_every_prereg_metric_resolves_to_an
    emitter`), and it is the whole reason this dataclass is the single source of truth: the
    previous arrangement had this module naming the drgym endpoint `kpr_incremental` while
    report.py queried `keypoint_recall`, with nothing on either side able to notice.
    """

    suite_id: str
    metric: str
    contrast: tuple[str, str]  # (treatment, comparator)
    test: str
    cluster_by: str | None
    rationale: str
    claim: Claim = "confirmatory"
    note: str = ""  # short enough to print in a table cell; `rationale` is for the seal
    # WHICH suites a POOLED endpoint pools. Empty for a suite-specific endpoint.
    #
    # Sealed as a LITERAL, never derived at read time: a pooled endpoint whose denominator
    # is "whatever suites happen to be on disk" silently changes its own claim when a suite
    # is added or a run fails. Measured before this existed: pooled `evidence_coverage`
    # drew 69 of 252 rows (27.4%) from `synth`, the CALIBRATION suite -- closed-form
    # metrics, zero tokens -- into a confirmatory endpoint.
    pool_suites: tuple[str, ...] = ()
    # What happens if this endpoint CANNOT run. Written now, because "we dropped the suite
    # that needed a key we never got" is a different sentence after you have seen whether
    # the other suites worked.
    contingency: str = ""

    @property
    def treatment(self) -> str:
        return self.contrast[0]

    @property
    def comparator(self) -> str:
        return self.contrast[1]

    @property
    def key(self) -> str:
        return f"{self.suite_id}:{self.metric}:{self.contrast[0]}-vs-{self.contrast[1]}"

    def covers(self, suite_id: str, metric: str) -> bool:
        """A POOLED endpoint covers every suite; a suite-specific one covers only its own."""
        return metric == self.metric and self.suite_id in (POOLED, suite_id)


@dataclass(frozen=True, slots=True)
class PreregStage1:
    primary: tuple[Endpoint, ...]
    secondary: tuple[Endpoint, ...]
    arms: tuple[str, ...]
    kill_switches: Mapping[str, str]
    excluded_task_ids: Mapping[str, tuple[str, ...]]
    burned_pilot_ids: Mapping[str, tuple[str, ...]]
    seeds: tuple[int, ...]
    stopping_rule: str
    calibration: tuple[Endpoint, ...] = ()
    # grid name -> `Grid.source_sha256`, the exact bytes of the grid file this
    # preregistration commits to running. Computed and printed since the loader was
    # written, and it entered no sealed record: recording it on a RUN says which grid the
    # run came from, but sealing it answers the question a reader of a frozen document
    # actually has -- did the grid change after the commitment was made?
    grid_hashes: Mapping[str, str] = field(default_factory=dict)
    fdr_q: float = 0.05
    alpha: float = 0.05
    # THE MULTIPLICITY RULE, sealed before any number exists. Stating it afterwards is
    # indistinguishable from choosing whichever rule the results happen to satisfy.
    claim_rule: str = ""
    # FIXED, never len(what actually ran). An endpoint that did not run is a NON-REJECTION,
    # not a smaller family: deriving the family size from completed cells makes every
    # failed or skipped run silently raise the power of the ones that survived.
    fdr_family_size: int = 0
    # How the qualitative cases get PICKED. Selecting them after seeing the numbers is how a
    # qualitative section becomes cherry-picking with extra steps.
    qualitative_rule: str = ""
    # THE EQUIVALENCE MARGINS, given a sealed home. `KILL_SWITCH_MARGINS` was a module constant
    # that no sealed digest covered, while the margins are what decides whether a kill switch
    # FIRES -- so an edit to them after sealing would change every three-way verdict with no
    # signal anywhere, and the drafted document had to smuggle them into the consequence
    # STRINGS to record them at all. `check_live_constants` compares this field against the
    # running code, exactly as it does for the stage-2 constants.
    kill_switch_margins: Mapping[str, float] = field(default_factory=dict)
    # The resample count `claim_rule` names, as a NUMBER a checker can compare. See
    # CI_RESAMPLES: the prose said 1,000 while the renderer defaulted to 10,000.
    ci_resamples: int = CI_RESAMPLES
    # suite_id -> confirmatory | exploratory | calibration. WHAT A NUMBER FROM THIS SUITE MAY
    # BE CALLED, which is a decision and not a consequence of the endpoint list: a document can
    # define a test on a suite and still refuse it a p-value (tau2's primary is declared
    # DIRECTIONAL in its own rationale), and it can authorise one whose endpoints are sealed
    # elsewhere (the retail fork pair). See SUITE_ROLES.
    suite_roles: Mapping[str, str] = field(default_factory=dict)
    # RUN ids the analysis must drop, with the reason, keyed on `run_id` and NOTHING ELSE.
    # `excluded_task_ids` and `burned_pilot_ids` key on the TASK, which is right for
    # contamination -- a burned task is burned in every arm -- and wrong for a duplicate: the
    # run of record shares (suite, task, arm, seed) with the orphan it supersedes, so a
    # cell-keyed exclusion deletes both and leaves the cell empty, silently.
    excluded_run_ids: Mapping[str, str] = field(default_factory=dict)

    @property
    def secondary_family(self) -> tuple[str, ...]:
        """The BH-FDR family, as endpoint keys.

        DERIVED, never stored: a stored copy alongside `secondary` is two declarations of the
        same family, which is the exact bug this module was rewritten to remove.
        """
        return tuple(e.key for e in self.secondary)


@dataclass(frozen=True, slots=True)
class PreregStage2:
    sigma_j: float
    tau: float
    dwr_weights: Mapping[str, float]
    theta_nli: float
    discoverability_threshold: float
    word_cap: int
    budget_grid: tuple[int, ...]
    notes: str = ""


def live_constants() -> dict[str, object]:
    """The values the SCORER actually uses, for the fields stage 2 freezes.

    ONE SOURCE OF TRUTH. `seal-stage2` used to retype these as literals, and they had drifted:
    it froze dwr_weights {"0":0.1,"1":0.2,"2":0.3,"3":0.4} while `pi_eval.score.DWR_WEIGHTS` is
    {0:0.5, 1:1.0, 2:1.5, 3:2.0, 4:2.0, 5:2.0} -- different values AND two fewer depths. The
    preregistration would have frozen a weighting no reported number has ever been computed
    with, for `dwr`, a reported metric.

    Imported lazily: `prereg` is imported by report paths that must not pay for `score`.
    """
    from .report import FRONTIER_GRID_POINTS
    from .score import DWR_WEIGHTS

    return {
        "dwr_weights": {str(k): float(v) for k, v in sorted(DWR_WEIGHTS.items())},
        "frontier_grid_points": int(FRONTIER_GRID_POINTS),
        # STAGE 1's comparable constants. They are here rather than in a second checker
        # because "the sealed document and the running code disagree" is one question, and
        # asking it twice is how the two checkers come to cover different fields.
        "kill_switch_margins": {str(k): float(v) for k, v in sorted(KILL_SWITCH_MARGINS.items())},
        "ci_resamples": int(CI_RESAMPLES),
    }


def check_live_constants(sealed: Mapping[str, object]) -> list[str]:
    """Fields where a SEALED document and the running code disagree. Empty is the good case.

    Takes EITHER stage's payload: it compares only the keys the payload actually carries, so a
    stage-1 file is checked on its margins and its resample count and a stage-2 file on its
    weights, with no branch and no second function to fall behind.

    A sealed constant nothing compares against is a decoration. `PreregStage2` is defined,
    sealed and read by nothing in src/, so an edit to DWR_WEIGHTS after sealing would silently
    diverge from the preregistration with no signal anywhere -- and `KILL_SWITCH_MARGINS`, the
    thing that decides whether a kill switch fires at all, was not in any sealed document.
    """
    live = live_constants()
    out = []
    for k, v in live.items():
        if k in sealed and sealed[k] != v:
            out.append(f"{k}: sealed {sealed[k]!r} != code {v!r}")
    return out


# The test vocabulary, resolved to the code that runs it. A declared `Endpoint.test` that names
# no implementation is a committed test that cannot be executed, which is indistinguishable in
# a sealed document from one that was quietly never run.
def test_function(name: str):
    """The callable a declared `Endpoint.test` names. KeyError on a spelling nobody wrote.

    Imported lazily and per name: `fork_report` and `stats.inference` are cheap, but `prereg`
    is imported by report paths that must not pay for either.
    """
    if name == SIGN_TEST_EXACT:
        from .fork_report import sign_test_p

        return sign_test_p
    from .stats.inference import mcnemar_exact, sign_flip_p

    return {MCNEMAR_EXACT: mcnemar_exact, SIGN_FLIP_PERMUTATION: sign_flip_p}[name]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def seal(root: Path, stage: Stage, payload: object) -> Path:
    """Write a stage file plus its sidecar digest, and refuse to overwrite a sealed one."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{stage}.json"
    if path.exists():
        raise FileExistsError(
            f"{path} is already sealed. Preregistration is write-once by design: "
            "if it genuinely must change, file an amendment as a new stage file and say so "
            "in the paper."
        )
    path.write_text(json.dumps(asdict(payload), indent=2, sort_keys=True) + "\n")
    (root / f"{stage}.json.sha256").write_text(_sha(path) + "\n")
    _rebuild_manifest(root)
    return path


class PreregTampered(RuntimeError):
    """A file already in the manifest no longer hashes to its recorded digest."""


def _rebuild_manifest(root: Path) -> None:
    """APPEND-ONLY. An existing entry is never recomputed, and a changed file is refused.

    This used to rebuild every entry from whatever was on disk. `seal()` refuses to overwrite
    a stage FILE, so stage1.json could not be re-sealed -- but nothing stopped it being edited,
    and the next ordinary `seal()` call silently re-hashed it into the manifest. Reproduced end
    to end on the normal workflow (seal stage 1, run the pilot, seal stage 2):

        after sealing stage1        ok=True
        after editing stage1.json   ok=False  mismatches=['stage1.json']   <- verify works
        after sealing stage2        ok=True                                <- and is undone

    with the tampered text still on disk. A silently edited preregistration is the one failure
    mode `verify` names as invalidating the whole multiplicity defence, and the routine act of
    sealing the next stage was what cleared the evidence.

    So the manifest only ever GAINS entries. Sealing a later stage cannot re-bless an earlier
    one, and a mismatch found here raises instead of being written away -- because at this point
    the process is about to append a digest that would make the tamper permanent and invisible.
    """
    path = root / "MANIFEST.json"
    entries: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}
    for f in sorted(root.glob("*.json")):
        if f.name == "MANIFEST.json":
            continue
        digest = _sha(f)
        if f.name in entries:
            if entries[f.name] != digest:
                raise PreregTampered(
                    f"{f.name} no longer matches its sealed digest "
                    f"(manifest {entries[f.name][:12]}, on disk {digest[:12]}). Refusing to "
                    "update the manifest: doing so would re-bless an edited preregistration and "
                    "make `verify` report ok. Restore the file from git, or file an amendment as "
                    "a NEW stage file and say so in the paper."
                )
            continue
        entries[f.name] = digest
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")


@dataclass
class VerifyResult:
    ok: bool
    checked: int
    mismatches: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    # JSON files sitting in the sealed directory that NO manifest entry covers.
    #
    # `scripts/measure_sigma_j.py --write` writes prereg/sigma_j.json directly rather than
    # through `seal()`, so the published sigma_J -- the constant that decides what may be
    # claimed at all -- lives inside the sealed directory, is not write-once, has no digest,
    # and `verify` reported ok=True regardless because it only ever walked the manifest.
    #
    # It is picked up by the NEXT seal (the manifest gains any file it does not yet name), so
    # the exposure is the window between writing it and sealing the following stage -- and
    # forever, if it is written after the last seal. Reported rather than fatal, because that
    # intermediate state is a legitimate step in the Gate 4 procedure; the CLI exits nonzero on
    # it, since by the time `verify` is run for the record everything should be covered.
    unsealed: list[str] = field(default_factory=list)


def verify(root: Path) -> VerifyResult:
    """Recompute every digest. A silently edited prereg is the one failure mode that would
    invalidate the whole multiplicity defence, so this is checked, never assumed."""
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.exists():
        return VerifyResult(False, 0, missing=["MANIFEST.json"])
    manifest = json.loads(manifest_path.read_text())
    res = VerifyResult(True, 0)
    for name, digest in sorted(manifest.items()):
        p = root / name
        if not p.exists():
            res.missing.append(name)
            res.ok = False
            continue
        res.checked += 1
        if _sha(p) != digest:
            res.mismatches.append(name)
            res.ok = False
    # Anything in the sealed directory that no manifest entry covers -- see VerifyResult.unsealed.
    res.unsealed = sorted(
        f.name for f in root.glob("*.json") if f.name != "MANIFEST.json" and f.name not in manifest
    )
    return res


SIGMA_J_FILE = "sigma_j.json"
SIGMA_J_PROVENANCE_FILE = "sigma_j.provenance.json"


def sigma_j_judge(start: str | Path) -> str:
    """The judge model sigma_J was MEASURED on, from the provenance sidecar. "" if unknown.

    `load_sigma_j` deliberately returns only `{metric: float}` -- a stray non-numeric entry
    would be read as a sigma of 1.0 and silently raise the floor on every judge metric -- so
    the instrument's identity lives in a sidecar. Nothing compared it to the judge that
    actually produced the scores, which means a dead band measured on one model could gate
    effects produced by another. `pi_eval.score` now checks.
    """
    p = Path(start).resolve()
    for cand in (p, *p.parents):
        f = cand / "prereg" / SIGMA_J_PROVENANCE_FILE
        if f.exists():
            try:
                raw = json.loads(f.read_text())
            except ValueError:
                return ""
            return str(raw.get("judge_model", "")) if isinstance(raw, dict) else ""
    return ""


# The preregistered floor on what a judge-derived effect may claim. Measured by test-retest
# (scripts/measure_sigma_j.py), sealed into stage 2, and never guessed: the minimum design is
# 200 identical + 200 paraphrase retest pairs, which is what makes the estimate a measurement
# rather than an anecdote.
SIGMA_J_MIN_PAIRS = 200


def load_sigma_j(start: str | Path) -> dict[str, float]:
    """metric name -> sigma_J, from a frozen `prereg/sigma_j.json` at or above `start`.

    ABSENT ON PURPOSE UNTIL MEASURED. A judge-derived metric with no pinned sigma_J has a
    noise floor of +inf, so `pi_eval.report` flags every effect on it `below_noise_floor` --
    which is exactly the preregistration ("sigma_J published and frozen before any Inquirer
    arm runs"). The conservative direction is to flag, never to claim.

    NON-NUMERIC AND BOOLEAN ENTRIES ARE IGNORED. `isinstance(True, int)` is True in Python, so
    a stray flag in this file would otherwise be read as a sigma_J of 1.0 and silently raise
    the floor on every judge metric. Provenance belongs in the sidecar, not in this map.
    """
    p = Path(start).resolve()
    for cand in (p, *p.parents):
        f = cand / "prereg" / SIGMA_J_FILE
        if f.exists():
            try:
                raw = json.loads(f.read_text())
            except ValueError:
                return {}
            if not isinstance(raw, dict):
                return {}
            return {
                str(k): float(v)
                for k, v in raw.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
    return {}


def confirmatory_count(s1: PreregStage1) -> int:
    """Primary + secondary. The CALIBRATION endpoints are deliberately not counted: they
    carry no confirmatory claim, so counting them would inflate the multiplicity denominator
    with tests nobody is allowed to cite."""
    return len(s1.primary) + len(s1.secondary)


def is_confirmatory(s1: PreregStage1, suite_id: str, metric: str) -> bool:
    return any(e.covers(suite_id, metric) for e in (*s1.primary, *s1.secondary))


def classify(s1: PreregStage1, suite_id: str, metric: str) -> str:
    for label, endpoints in (
        ("primary", s1.primary),
        ("secondary", s1.secondary),
        ("calibration", s1.calibration),
    ):
        if any(e.covers(suite_id, metric) for e in endpoints):
            return label
    return "exploratory"


def excluded(s1: PreregStage1, suite_id: str, task_id: str) -> bool:
    """Pilot ids are BURNED: data used to choose a threshold cannot also test it."""
    return task_id in s1.excluded_task_ids.get(suite_id, ()) or task_id in s1.burned_pilot_ids.get(
        suite_id, ()
    )


# RUN ids the analysis must drop, and why. Kept in code for the reason `default_stage1` is:
# the sealed JSON is generated rather than hand-typed, so a change to this list arrives as a
# reviewable diff instead of as an edit to a frozen document.
#
# THE N6 ORPHANS. The `tier1_trained_qa_teacher` sweep's agent was killed and relaunched after
# a mid-campaign commit, so 19 musique / inquirer_prompted units were re-minted at
# code_version 3295544 and re-spent ($0.2406). The fc1def5 runs of the same cells are the ones
# of record (artifacts/n6/run_ids.inquirer_prompted.txt, 1,200 ids, orphans already removed).
#
# KEYED ON run_id, NOT ON THE CELL. The kept run shares (suite, task, arm, seed) with the
# orphan -- that is what makes them duplicates -- so a (suite, task) exclusion would delete
# the run of record too and leave the cell empty. A missing cell is not an error anywhere in
# the report path, so that failure would have been silent.
_N6_ORPHAN_REASON = "duplicate cell from a sweep restart; superseded run kept"
EXCLUDED_RUN_IDS: Mapping[str, str] = {
    rid: _N6_ORPHAN_REASON
    for rid in (
        "4ab7e54494ad9e183a5b0aed9793a7ed",
        "6f11f274d00f38172680246a03a7f261",
        "cb6ca1a8973958e6b059738ef800f688",
        "97cb5d8b2a098c00bbb0c7c74e9f87e0",
        "9e0ae9dd7849c3cca25036d0cfb4385e",
        "c0d691b73bd216daab961bdc3c347442",
        "4a2f6753c6923633c5ad79c9760b01b7",
        "2dd9ed2328e11ccb05acd829dac6896d",
        "9f4ad3b5afcb06ce2abd4873576edfde",
        "a6f717a218f240362d00674c33c1651f",
        "18aa1ed2d72d63115bbeea64986d0887",
        "2a3213ef3fce0cdabdf2132523b45e83",
        "7b7a4b0c53a94893ad14ae873568a99e",
        "ee12df8fa44ae670ab07556f4893c83c",
        "f82d589b306ae6f8f998fe33dbcdc65c",
        "227f7c71fef85bcd40aad318ccde6772",
        "d220c0c73c9c88274d5b857f136b20f4",
        "ab64dd6856f895003a98c7933127d0fb",
        "4eb232494b9041e0167d8a04576daf14",
    )
}


def excluded_runs(root: Path) -> tuple[frozenset[str], str]:
    """(run ids the SEALED stage 1 excludes, a note). Empty when nothing is sealed.

    Separate from `excluded_pairs` because the two answer different questions and must not be
    collapsed: a burned or contaminated TASK is excluded in every arm, while an excluded RUN is
    one row of a cell whose other rows are the measurement.
    """
    f = Path(root) / "stage1.json"
    if not f.exists():
        return frozenset(), "no sealed stage 1; exclusions not applied"
    d = json.loads(f.read_text())
    ids = frozenset(str(r) for r in (d.get("excluded_run_ids") or {}))
    return ids, f"{len(ids)} run(s) excluded by the sealed stage 1"


def excluded_pairs(root: Path) -> tuple[frozenset[tuple[str, str]], str]:
    """((suite_id, task_id) the SEALED stage 1 excludes, a note). Empty when nothing is sealed.

    THE SEALED DOCUMENT HAD NO READER. `excluded()` had zero callers outside its own
    definition, and `stage1.json` was read by nothing in src/ -- so the exclusion list was
    written, digested, frozen, and then had no effect on any analysis.

    For `excluded_task_ids` that was belt-and-braces rather than a hole: musique_build drops a
    contaminated task from the CORPUS, so it never becomes a task, never runs and never scores.
    Verified on the built corpus -- 0 of the 24 declared ids are present. But the guarantee then
    rests entirely on the corpus that happens to be on disk, and the sealed statement about it
    is decorative.

    For `burned_pilot_ids` it is a genuine hole. `ELIGIBLE` filters `pilot_flag = FALSE`, which
    drops the pilot RUN -- but the rule is about the TASK: if tau was chosen by looking at task
    T, then T's confirmatory run is contaminated too, in every arm. Nothing burned the task.

    Returns pairs rather than a predicate so the caller can put it in SQL and in provenance.
    """
    f = Path(root) / "stage1.json"
    if not f.exists():
        return frozenset(), "no sealed stage 1; exclusions not applied"
    d = json.loads(f.read_text())
    pairs = set()
    n_burned = 0
    for key in ("excluded_task_ids", "burned_pilot_ids"):
        for suite, ids in (d.get(key) or {}).items():
            for t in ids:
                pairs.add((str(suite), str(t)))
                n_burned += key == "burned_pilot_ids"
    return frozenset(pairs), (
        f"{len(pairs)} (suite, task) pairs excluded by the sealed stage 1 "
        f"({n_burned} burned pilot tasks)"
    )


# (suite, metric) pairs the suite's GOLD structurally cannot produce, with what was counted.
# Not "no rows yet" -- that is an unrun cell. These are cells that can never fill, so pooling
# them would add pairs whose difference is identically zero: no signal, and an inflated n.
NOT_COMPUTABLE: Mapping[tuple[str, str], str] = {
    ("drgym", "cad_ge2"): (
        "0 of 16,156 drgym gold nodes sit at depth >= 2 (all 16,156 are depth 0), so no "
        "depth-weighted coverage exists to measure."
    ),
    ("tau2", "cad_ge2"): (
        "0 of 5,790 tau2 gold nodes sit at depth >= 2 (1,804 at depth 0, 349 at depth 1, "
        "and 3,637 carry no depth at all)."
    ),
    ("wiki2", "cad_ge2"): (
        "0 of 31,120 wiki2 gold nodes sit at depth >= 2 (19,164 at depth 0, 11,956 at 1)."
    ),
    ("drgym", "evidence_coverage"): (
        "0 of 16,156 drgym gold nodes carry gold_ev_uids: the suite's gold has no evidence "
        "spans, so |E n gold| has no denominator."
    ),
    ("drgym", "frontier_auc"): (
        "the frontier is a curve of evidence coverage against spend, and drgym gold carries "
        "0 evidence spans across all 16,156 nodes."
    ),
}


def not_computable(suite_id: str, metric: str) -> str:
    """The stated reason this cell can never fill, or "" when it can."""
    return NOT_COMPUTABLE.get((suite_id, metric), "")


# Seeds are declared PER SUITE, because a single global tuple cannot be true of a design
# whose suites differ by an order of magnitude in cost. Measured against conf/grids/*.yaml:
# the sealed tuple was (0, 1) while tau2 and musique grids run (0, 1, 2) and the drgym grid
# runs (0,) -- two of the three PRIMARY suites contradicted the seal.
SEEDS_BY_SUITE: Mapping[str, tuple[int, ...]] = {
    "musique": (0, 1, 2),  # tier1_confirmatory (0,1); tier1_trained adds seed 2
    "tau2": (0, 1, 2),  # tier2_confirmatory
    "drgym": (0,),  # tier3_drgym: judge calls make a second seed the most expensive one
    "strategyqa": (0, 1),  # tier1_confirmatory
    "wiki2": (0, 1),  # tier1_confirmatory
    "synth": (0, 1),  # calibration; zero tokens, so seeds are free
}


def seeds_for(suite_id: str) -> tuple[int, ...]:
    return SEEDS_BY_SUITE.get(suite_id, ())


TREATMENT = "inquirer_prompted"
ANCESTOR = "self_inquire"  # the closest published ancestor, not a strawman
BASELINE = "drafter_only"

# ------------------------------------------------------------------ THE ENDPOINT DECLARATIONS
#
# THIS IS THE ONLY PLACE AN ENDPOINT IS DECLARED. `pi_eval.report` imports these tuples and
# converts them into its `Contrast` objects; it declares none of its own. Before this, the two
# modules held independent lists that named the SAME endpoints with DIFFERENT metric strings
# (`tau_reward` vs `task_success`, `token_f1` vs `answer_token_f1`, `kpr_incremental` vs
# `keypoint_recall`) and nothing compared them, so the sealed preregistration could commit to
# one endpoint while the paper's table reported another. Two tests now close that hole:
# every metric named here must resolve to something `pi_eval.score` actually emits, and
# report.py's tables must be exactly these endpoints.

PRIMARY: tuple[Endpoint, ...] = (
    Endpoint(
        "tau2",
        # THE VISIBLE DIFF THE OLD COMMENT PROMISED. This line read `task_success` -- every
        # required need resolved -- because "the plan called this `tau_reward` (DB + user-DB
        # hash equality); NOTHING EMITS THAT TODAY, and a preregistered endpoint naming a
        # metric no scorer writes is not an endpoint. If the hash-equality reward is later
        # emitted it lands as a separate named metric and this line changes in a visible diff."
        #
        # It is emitted now. `pi_run.stages.tau2_runner` drives the Orchestrator and grades the
        # transcript with upstream's own EnvironmentEvaluator, and
        # `pi verify tau2 --replay-gold` reproduces the gold DB hash on 97 of 97 tasks -- the
        # first validation this reward has ever had against the benchmark.
        #
        # `task_success` was the wrong endpoint for tau2 and not merely a placeholder: it is a
        # RESOLUTION-COVERAGE proxy computed from the gold need graph, so the tau2 primary row
        # reported how much of the knowledge base the policy had covered, on a benchmark whose
        # own definition of success is whether the customer's account ended up in the right
        # state. An agent can resolve every need and change nothing.
        #
        # DENOMINATOR, STATED: 88 of the 97 tasks carry a DB reward basis (87 DB + 1
        # DB/NL_ASSERTION). The 9 ACTION-basis tasks are graded but WITHHELD, because an ENV
        # evaluator hands them a free 1.0 that is not a measurement. `task_success` remains a
        # secondary on this suite, where a coverage proxy is what it honestly is.
        "tau_reward",
        (TREATMENT, ANCESTOR),
        MCNEMAR_EXACT,
        "template_id",
        "Binary and judge-free: agent-DB and user-DB hash equality against a gold environment, "
        "computed by upstream's own evaluator over the transcript the Orchestrator produced. "
        "Clustered at template_id because tau2's 97 tasks derive from ~18 banking scenarios "
        "and resampling tasks would understate the SE. n=88 of 97: the ACTION-basis tasks are "
        "withheld rather than credited with the evaluator's free 1.0. "
        "DIRECTIONAL, NOT POWERED: MDE 22.1pp at 80% power, simulated through the "
        "cluster-level permutation that pi_eval.report actually runs "
        "(scripts/power_analysis.py). The 12.6pp this document once claimed is the textbook "
        "McNemar normal approximation at ~20 discordant pairs, and at that lift the real "
        "power is 42% -- optimistic by a factor of 1.8 in a document meant to be frozen. "
        "Reported as a direction with a CI; a null here is not evidence of no effect.",
        claim="directional",
    ),
    Endpoint(
        "musique",
        "answer_token_f1",
        (TREATMENT, ANCESTOR),
        SIGN_FLIP_PERMUTATION,
        None,
        "Graded, gold-answer-based, judge-free. Paired against the closest ancestor rather "
        "than a strawman.",
    ),
    Endpoint(
        "drgym",
        "kpr_incremental",
        (TREATMENT, ANCESTOR),
        SIGN_FLIP_PERMUTATION,
        None,
        "Incremental key-point recall against benchmark-author gold, both-order paired "
        "judging. Incremental rather than raw so that restating one key point cannot "
        "outscore covering many; see pi_eval.metrics.kpr_incremental. Chosen over holistic "
        "preference, which is what length bias attacks hardest.",
        contingency=(
            "DRGYM_API_KEY is not held today. If it is not obtained before the run freeze "
            "this endpoint is NOT RUN and is reported as NOT RUN in every table -- never "
            "silently dropped, never backfilled with a suite that was easier to reach, and "
            "never replaced by a different metric on a different corpus. Because claims are "
            "PER-SUITE, the other two suites are unaffected and no claim is transferred to "
            "them; the abstract states that the third suite was not run and why. An "
            "unobtained key is a missing measurement, not a negative result."
        ),
    ),
)

# NOT CONFIRMATORY, and declared here so it cannot be invented downstream. The synthetic suite
# has closed-form metrics and zero tokens; its contrast checks the pipeline, not the thesis.
CALIBRATION: tuple[Endpoint, ...] = (
    Endpoint(
        "synth",
        "evidence_coverage",
        (SYNTH_TREATMENT, SYNTH_COMPARATOR),
        SIGN_FLIP_PERMUTATION,
        None,
        "Closed-form calibration suite. Every value is analytically known, so a deviation "
        "here is a bug in the harness rather than a finding about a system.",
        claim="calibration",
        note="closed-form calibration suite; carries no confirmatory claim",
    ),
)

# ONE declared family. BH-FDR at q=0.05 across exactly these eleven and nothing else.
SECONDARY: tuple[Endpoint, ...] = (
    Endpoint(
        "musique",
        "answer_token_f1",
        (TREATMENT, BASELINE),
        SIGN_FLIP_PERMUTATION,
        None,
        "Gain over the no-inquiry baseline on the judge-free suite.",
    ),
    Endpoint(
        "tau2",
        # The SAME metric as the primary, for the same reason. Two contrasts on one suite
        # measured with two different metrics is not a ladder: P1 would ask whether inquiry
        # beats self-inquiry at changing the customer's account, and S2 would ask whether it
        # beats no-inquiry at covering the knowledge base, and the two could disagree without
        # anyone being able to say what that meant. `task_success` stays available as an
        # exploratory diagnostic, where a coverage proxy is honestly what it is.
        "tau_reward",
        (TREATMENT, BASELINE),
        MCNEMAR_EXACT,
        "template_id",
        "Gain over the no-inquiry baseline on the binary suite, in tau2's own terms: DB-hash "
        "equality rather than need coverage.",
    ),
    Endpoint(
        "drgym",
        "kpr_incremental",
        (TREATMENT, BASELINE),
        SIGN_FLIP_PERMUTATION,
        None,
        "Gain over the no-inquiry baseline on the open-corpus suite. Same metric as the "
        "drgym primary on purpose: a family member that silently changed the metric would "
        "answer a different question than the endpoint it is paired with.",
    ),
    Endpoint(
        POOLED,
        "evidence_coverage",
        (TREATMENT, "inquirer_noevidence"),
        SIGN_FLIP_PERMUTATION,
        None,
        "Vertical proactivity: does inquiry buy anything once evidence is removed?",
        pool_suites=("musique", "strategyqa", "wiki2", "tau2"),
    ),
    Endpoint(
        POOLED,
        "evidence_coverage",
        (TREATMENT, "parallel_replay"),
        SIGN_FLIP_PERMUTATION,
        None,
        "Sequencing: does the dependency order buy anything over the same questions asked at once?",
        pool_suites=("musique", "strategyqa", "wiki2", "tau2"),
    ),
    Endpoint(
        POOLED,
        "evidence_coverage",
        (TREATMENT, "verbosity"),
        SIGN_FLIP_PERMUTATION,
        None,
        "Length control: the null explanation for any gain is that the answer got longer.",
        pool_suites=("musique", "strategyqa", "wiki2", "tau2"),
    ),
    Endpoint(
        POOLED,
        "evidence_coverage",
        (TREATMENT, "compute_matched"),
        SIGN_FLIP_PERMUTATION,
        None,
        "Compute control: the same token budget spent without inquiry.",
        pool_suites=("musique", "strategyqa", "wiki2", "tau2"),
    ),
    Endpoint(
        POOLED,
        "rnr_resolve",
        (TREATMENT, BASELINE),
        SIGN_FLIP_PERMUTATION,
        None,
        "Required-need recall at the RESOLVE rung: discovery, not answer quality.",
        pool_suites=("musique", "strategyqa", "wiki2", "tau2", "drgym"),
    ),
    Endpoint(
        POOLED,
        "cad_ge2",
        (TREATMENT, BASELINE),
        SIGN_FLIP_PERMUTATION,
        None,
        "Coverage at depth >= 2, |V_d|-weighted: the vertical claim in one number.",
        pool_suites=("musique", "strategyqa"),
    ),
    Endpoint(
        POOLED,
        "frontier_auc",
        (TREATMENT, BASELINE),
        SIGN_FLIP_PERMUTATION,
        None,
        "Area under the Q-versus-spend frontier. Always reported with the curve.",
        pool_suites=("musique", "strategyqa", "wiki2", "tau2"),
    ),
    Endpoint(
        "tau2",
        "unlock_and_invoke",
        (TREATMENT, ANCESTOR),
        MCNEMAR_EXACT,
        "template_id",
        "The discoverable-tool prerequisite edge: binary, objective, environment-verified.",
    ),
    # ---- THE ADDITIONAL PUBLISHED ANCESTORS, and the recall companion that keeps the
    # comparison honest.
    #
    # COSTS ZERO ROLLOUTS: self_ask and ircot already run in tier1_confirmatory at
    # n=200 x 2 seeds on musique, on a judge-free metric. Without these declarations the
    # ancestor claim has no preregistered test and no rendered interval anywhere.
    #
    # WHY EVERY ONE HAS A RECALL TWIN. Measured on 164 musique runs:
    # Spearman(answer_n_words, answer_token_f1) is -0.677 pooled and -0.934 WITHIN
    # inquirer_prompted alone, and the arms differ by up to 7 mean words. Against ircot the
    # recall means are TIED at 0.500 while F1 reads 0.433 to 0.359 -- the whole F1 gap is
    # precision, i.e. concision. Declaring F1 alone would let "we beat the ancestor" mean
    # "we were shorter". Recall is the half padding cannot manufacture, and it is
    # preregistered rather than merely reported so the answer cannot be chosen afterwards.
    Endpoint(
        "musique",
        "answer_token_f1",
        (TREATMENT, "self_ask"),
        SIGN_FLIP_PERMUTATION,
        None,
        "Published ancestor: self-ask decomposition. Already run at n=200 x 2 seeds.",
    ),
    Endpoint(
        "musique",
        "answer_token_recall",
        (TREATMENT, "self_ask"),
        SIGN_FLIP_PERMUTATION,
        None,
        "Length-robust twin of the self_ask contrast: discovery rather than concision.",
    ),
    Endpoint(
        "musique",
        "answer_token_f1",
        (TREATMENT, "ircot"),
        SIGN_FLIP_PERMUTATION,
        None,
        "Published ancestor: interleaved retrieval and chain-of-thought.",
    ),
    Endpoint(
        "musique",
        "answer_token_recall",
        (TREATMENT, "ircot"),
        SIGN_FLIP_PERMUTATION,
        None,
        "THE ONE THAT DECIDES THE ANCESTOR CLAIM: on pilot data the recall means tie at "
        "0.500 while F1 differs by 0.074, so an F1-only ancestor claim would be a claim "
        "about answer length.",
    ),
    Endpoint(
        "musique",
        "answer_token_recall",
        (TREATMENT, ANCESTOR),
        SIGN_FLIP_PERMUTATION,
        None,
        "Decomposes the musique PRIMARY. If the primary's effect survives here it is about "
        "finding the answer; if it lives only in precision it is about stating it briefly.",
    ),
    Endpoint(
        "musique",
        "answer_token_recall",
        (TREATMENT, BASELINE),
        SIGN_FLIP_PERMUTATION,
        None,
        "The headline baseline contrast, length-controlled. This is where the pilot evidence "
        "is strongest -- recall 0.500 against 0.000, on a quantity padding cannot inflate -- "
        "and it also survives restricting to well-formed answers (drafter_only stays 0.000 "
        "on the 10 of 14 that are not a serialized tool call).",
    ),
)


# THE KILL SWITCHES, AND THE ONE PLACE THEY ARE WRITTEN DOWN.
#
# There used to be two lists. `pi_eval.report.KILLSWITCH_RULES` carried five; this function
# built four, silently omitting `random_q`. Sealing stage 1 would therefore have frozen a
# preregistration missing the comparator whose consequence is that the thesis does not
# survive, while `pi report killswitch` went on printing five rows -- so the report would have
# shown a decision the sealed document never committed to, which is the exact shape of the
# thing preregistration exists to prevent. The report now derives from this.
#
# Each entry is a comparator and the CONSEQUENCE, written before the data existed. A
# consequence that reads "investigate further" is not a consequence.
# THE EQUIVALENCE MARGIN, WITHOUT WHICH "MATCHES" IS JUST A SMALL SAMPLE.
#
# A kill switch fires when the treatment MATCHES a comparator, and the consequences below run
# up to "STOP; no reframe survives". That is ACCEPTING THE NULL, and a confidence interval
# covering zero is not evidence of equivalence -- it is equally consistent with "no effect" and
# with "an effect this pilot is too small to resolve". Simulated at the pilot's actual n=120 on
# `evidence_coverage`, through `paired_difference` itself:
#
#     true effect      P(CI covers zero) at sd=0.20      at sd=0.30
#          0.00                     94%                       94%
#          +0.02                    77%                       87%
#          +0.04                    38%                       65%
#          +0.08                     0%                       14%
#
# So a REAL two-point gain would have fired the switch about four times in five, and a
# four-point gain between a third and two thirds of the time. The pilot was overwhelmingly
# likely to stop a paper whose effect was genuine but modest.
#
# `margin` is the largest difference that still counts as "the same" for THIS comparator --
# a scientific judgement, not a statistical one, which is why it is written down per switch and
# sealed rather than derived. The verdict becomes three-way: EQUIVALENT (the whole CI inside
# +/- margin) fires the rule; SEPARATED wins; INCONCLUSIVE means the pilot could not decide and
# is NOT a licence to stop. See `report.killswitch_table`.
KillSwitch = tuple[str, float]

# Margins are in the units of `report.KILLSWITCH_METRIC` (evidence_coverage, a [0,1] recall).
# 0.05 is the default: half of the 10pp headroom Gate 2 requires of the oracle arm, so a
# comparator within 5pp of the treatment is "the same" against a mechanism that must be worth
# at least twice that to be worth a paper. verbosity is TIGHTER (0.03) because its consequence
# is the unconditional STOP and a false stop there is unrecoverable.
#
# THESE MARGINS MAKE THE PLANNED PILOT UNDERSIZED, AND THAT IS THE POINT OF WRITING THEM DOWN.
# At paired sd 0.20 the 95% CI half-width is 1.96*sd/sqrt(n):
#
#     n = 120 (the planned pilot)   CI width 0.072  -> CANNOT fit inside +/-0.03
#     n = 400                       CI width 0.039  -> fits
#
# So `verbosity` -- the switch whose consequence is "STOP; no reframe survives" -- is
# STRUCTURALLY UNDECIDABLE at n=120 whatever the data say, and the honest verdict there is
# INCONCLUSIVE rather than a stop. Simulated P(EQUIVALENT | true effect 0), sd 0.20:
#
#     margin 0.03:  n=120  0%    n=400  67%
#     margin 0.05:  n=120  60%   n=400 100%   ... but 77% FALSE stop on a true +2pt effect
#
# The 0.05 rows are the reason not to simply loosen the margin: a wide window buys decisiveness
# by calling a real two-point gain "the same". Reproduce with `scripts/power_analysis.py`.
# Choosing between "raise the pilot n" and "accept an INCONCLUSIVE kill-switch day" is a
# research decision, and it is now an informed one instead of an implicit one.
KILL_SWITCH_MARGINS: Mapping[str, float] = {
    "verbosity": 0.03,
    "self_inquire": 0.05,
    "inquirer_noevidence": 0.05,
    # Not a claim margin: the expectation is EXACT equality, so any resolvable difference is
    # a harness defect. Kept at the default so the TOST machinery still renders a row.
    "parallel_replay": 0.05,
    "inquirer_depth1": 0.05,
    "random_q": 0.05,
    "compute_matched": 0.05,
}

KILL_SWITCHES: Mapping[str, str] = {
    "verbosity": (
        "If verbosity matches inquirer_prompted the effect is answer length, not inquiry. "
        "STOP; no reframe survives."
    ),
    "self_inquire": (
        "If it matches, the contribution is the protocol, not the two-agent split. Pull the "
        "tau2 pilot forward and reframe."
    ),
    "inquirer_noevidence": (
        "If it matches, vertical proactivity does not exist; the paper becomes a "
        "horizontal-facet result."
    ),
    # RE-POINTED. This read "If it matches, sequencing buys nothing and the graph has no
    # dependency structure; drop the graph framing" -- a conclusion that does not follow
    # from this arm, and a switch that fires with CERTAINTY.
    #
    # `ParallelReplayInquirer` emits the recorded run's own question strings without looking
    # at what the previous one returned. The same strings retrieve the same uids, and
    # `draft()` and the Answerer are pure in the evidence subset hash, so the answer text is
    # identical too. The delta on any function of the evidence set is exactly zero BEFORE
    # any data exists -- proven LLM-free on synth by
    # tests/test_prereg.py::test_parallel_replay_is_input_identical_to_the_arm_it_replays,
    # and observed on 2 of 2 paired musique cells (identical subset_hash and identical
    # answer_evidence_hash). A kill switch that cannot fail to fire is not evidence.
    "parallel_replay": (
        "A DETERMINISM check on the harness, not a sequencing counterfactual. The delta is "
        "identically zero by construction, so a MATCH confirms purity and says nothing about "
        "the graph. A NON-ZERO delta is the finding: it indicts the harness -- a Drafter that "
        "is not pure in (view, subset_hash, seed), a retriever with hidden state, or a "
        "leaking cache -- and every phi_LOO, prefix-ladder and stop-test number is void "
        "until it is explained."
    ),
    # The arm that CAN vary the sequencing channel: it enumerates its questions from x alone
    # in a single call, so none of them saw an answer AND the strings genuinely differ from
    # the treatment's. This is where "drop the graph framing" now hangs.
    "inquirer_depth1": (
        "If it matches, questions that never saw an answer do as well as questions that did: "
        "sequencing buys nothing, there is no dependency structure to exploit, and the graph "
        "framing goes. This is the sequencing switch; parallel_replay is not."
    ),
    "random_q": (
        "If it matches, the gain is retrieval VOLUME rather than question content: the "
        "Inquirer is not selecting, and the thesis does not survive it."
    ),
    # ADDED. `compute_matched` was a declared SECONDARY comparator -- the pooled
    # evidence_coverage endpoint `(inquirer_prompted, compute_matched)`, "the same token budget
    # spent without inquiry" -- with no kill switch, i.e. a preregistered test whose outcome
    # had no preregistered consequence. That is the same hole `random_q` was in, and it is the
    # comparator that separates the paper's claim from a claim about spend.
    "compute_matched": (
        "If it matches, the gain is the COMPUTE BUDGET rather than the inquiry: the same "
        "tokens spent without asking anything do as well. STOP; a protocol that is "
        "indistinguishable from spending more is not the contribution."
    ),
}


# THE SUITE ROLES, declared rather than derived, and the one place they are written down.
#
# A role is not a property of the endpoint list. `tau2` HAS a primary endpoint here whose own
# rationale says "DIRECTIONAL, NOT POWERED ... a null here is not evidence of no effect", and
# `tau2_retail`'s fork pair is confirmatory in the trained programme whose endpoints are sealed
# in its own stage file. Reading the role off the endpoints therefore said the opposite of the
# endpoint's text in one direction and blocked the registry from describing the running
# programme in the other.
#
# These are the roles the stage-1 draft names (artifacts/prereg_draft/, rev 2) and the roles
# `pi_run.suites.REGISTRY` declares; `pi suites audit` is what holds the two equal, and
# `tests/test_suites_roles.py` pins the set. Read against
# plans/2026-09-15-execution-plan-v4.md: D2 makes tau2 retail/airline/banking zero-shot
# TRANSFER and stops any tau2 row training -- it does not forbid a preregistered test on a
# transfer suite, and §3 phase 2.1 seals "primaries per suite (the ordered pair; matched-cost
# coverage on QA; follow-ups + tau_reward on forks)", which is the retail fork pair by name.
# D8 keeps rung 3 dead, which touches no suite here.
#
# CHANGING A ROLE IS AN AMENDMENT, not an edit: it moves what a number in the paper is allowed
# to be called. Every change carries its date in the registry's note and a row in docs/SUITES.md.
SUITE_ROLES: Mapping[str, str] = {
    # the QA family: the headline, matched-cost coverage
    "musique": "confirmatory",
    "strategyqa": "confirmatory",
    "wiki2": "confirmatory",
    # the fork family. Its endpoints are NOT in this document -- they are sealed in the trained
    # programme's stage file -- which is why the role has to be said out loud here.
    "tau2_retail": "confirmatory",
    # the zero-token reference suite, whose every value is known in closed form
    "synth": "calibration",
    # EXPLORATORY BY DECISION: a point estimate and a task-clustered CI, never a p-value.
    # tau2 (banking) is at the success floor with zero legal fork points and its endpoint was
    # already declared directional; drgym is eval-only and judge-derived, so a confirmatory row
    # would spend multiplicity on a transfer claim that must first clear the sigma_J floor.
    "tau2": "exploratory",
    "drgym": "exploratory",
}


def default_stage1(
    *,
    excluded_task_ids: Mapping[str, Sequence[str]] | None = None,
    burned_pilot_ids: Mapping[str, Sequence[str]] | None = None,
    grid_hashes: Mapping[str, str] | None = None,
    excluded_run_ids: Mapping[str, str] | None = None,
) -> PreregStage1:
    """The design's committed endpoints. Kept in code so the sealed JSON is generated, not
    hand-typed, and so a change to it shows up as a code diff."""
    return PreregStage1(
        primary=PRIMARY,
        secondary=SECONDARY,
        calibration=CALIBRATION,
        arms=(
            "drafter_only",
            "compute_matched",
            "query_expansion",
            "verbosity",
            "self_ask",
            "ircot",
            "self_inquire",
            "checklist",
            "inquirer_prompted",
            "inquirer_trained",
            "inquirer_noevidence",
            "inquirer_depth1",
            "random_q",
            "parallel_replay",
            "inquirer_may_ask_user",
            "gold_evidence",
            "oracle_vreq",
        ),
        kill_switches=dict(KILL_SWITCHES),
        kill_switch_margins=dict(KILL_SWITCH_MARGINS),
        suite_roles=dict(SUITE_ROLES),
        ci_resamples=CI_RESAMPLES,
        excluded_task_ids={k: tuple(v) for k, v in (excluded_task_ids or {}).items()},
        burned_pilot_ids={k: tuple(v) for k, v in (burned_pilot_ids or {}).items()},
        excluded_run_ids=dict(sorted({**EXCLUDED_RUN_IDS, **(excluded_run_ids or {})}.items())),
        grid_hashes=dict(sorted((grid_hashes or {}).items())),
        # The union across suites, kept for the fields and consumers that expect one tuple.
        # `seeds_for(suite_id)` is the authority; see SEEDS_BY_SUITE for why per-suite.
        seeds=tuple(sorted({s for v in SEEDS_BY_SUITE.values() for s in v})),
        claim_rule=(
            "PER-SUITE. We test three benchmarks (tau2, musique, drgym) and report each "
            "separately; there is no pooled headline claim across suites and no claim is "
            "conditioned on another suite's result. alpha is therefore UNCORRECTED across "
            "suites, which is only correct because the abstract names every suite it was "
            "run on -- including the ones that fail. Within a suite, the secondary family "
            "is BH-FDR controlled at q=0.05 over a FIXED family of fdr_family_size tests. "
            "A POOLED endpoint pools only the suites named in its pool_suites and reports "
            "per-suite n as a column, so its denominator is visible rather than implied. "
            f"Intervals are task-clustered BCa at {CI_RESAMPLES} resamples, seed 0, which is "
            "`pi_eval.prereg.CI_RESAMPLES` and is what `report.Agg` and `pi report --n-boot` "
            "default to: the sealed number and the rendered number are ONE number, not two."
        ),
        fdr_family_size=len(SECONDARY),
        qualitative_rule=(
            "Qualitative cases are drawn by a seeded sample from the PREREGISTERED strata "
            "(one where the treatment wins, one where it loses, one tie) using seed 0 over "
            "task_id order, before any transcript is read. Cases are never swapped after "
            "reading. If a drawn case is unusable (empty transcript, harness error) the "
            "next id in seeded order replaces it and the substitution is reported."
        ),
        stopping_rule=(
            "A gate slipping more than 48h means the level below it is the paper. "
            "No new arms after the run freeze."
        ),
    )


def grid_drift(sealed: Mapping[str, str], current: Mapping[str, str]) -> tuple[str, ...]:
    """Sealed grids whose bytes no longer match, or that have gone missing. Sorted.

    A grid ADDED after sealing is not drift: it commits to nothing and is simply not
    preregistered. A grid that changed or vanished is, because the sealed document says
    those exact bytes are what will run.
    """
    return tuple(sorted(name for name, sha in sealed.items() if current.get(name) != sha))


def all_endpoints(s1: PreregStage1 | None = None) -> tuple[Endpoint, ...]:
    """Every declared endpoint, confirmatory and calibration alike, in table order."""
    s1 = s1 or default_stage1()
    return (*s1.calibration, *s1.primary, *s1.secondary)


def endpoint_suites(s1: Mapping[str, object] | PreregStage1 | None = None) -> frozenset[str]:
    """Every suite a stage 1 names in an endpoint, with POOLED resolved to its pool.

    TAKES A SEALED PAYLOAD, not only a live object. `pi suites audit` cross-checks the
    registry's `endpoint_status` against the preregistration, and the trained programme seals
    its own stage file, so the question has to be answerable from a document on disk where
    plain dicts are all there is.

    `POOLED` never appears in the result: an endpoint declared over the wildcard is declared
    over the suites in its `pool_suites` and over nothing else, and leaking the sentinel into
    a suite set would make every containment test quietly wrong.
    """
    if s1 is None:
        s1 = default_stage1()
    if isinstance(s1, PreregStage1):
        groups: Sequence[Sequence[object]] = (s1.calibration, s1.primary, s1.secondary)
    else:
        groups = tuple(
            tuple(s1.get(k) or ()) for k in ("calibration", "primary", "secondary")
        )  # a sealed payload
    out: set[str] = set()
    for group in groups:
        for e in group:
            if isinstance(e, Mapping):
                suite, pool = str(e.get("suite_id") or ""), tuple(e.get("pool_suites") or ())
            else:
                suite, pool = e.suite_id, tuple(getattr(e, "pool_suites", ()) or ())
            out.add(suite)
            out.update(str(p) for p in pool)
    out.discard(POOLED)
    out.discard("")
    return frozenset(out)


def confirmatory_suites(s1: Mapping[str, object] | PreregStage1 | None = None) -> frozenset[str]:
    """Every suite whose rows this stage 1 allows to carry a p-value.

    DECLARED, NOT DERIVED, and that distinction is the whole of this function. "Which suites
    carry a p-value" was read off the endpoint tuples, which conflates two different
    statements: that a document DEFINES a test on a suite, and that it AUTHORISES a
    confirmatory claim from it. They come apart in both directions and did:

      * `tau2` has a primary endpoint here and that endpoint's own rationale already says
        DIRECTIONAL, NOT POWERED -- "reported as a direction with a CI; a null here is not
        evidence of no effect". Deriving its role from the endpoint's existence therefore
        contradicted the endpoint's own text.
      * `tau2_retail`'s fork pair is confirmatory in the trained programme, whose endpoints
        are sealed in its OWN stage file. Deriving the role from this document's endpoints
        meant the registry could not describe the programme the paper is running without the
        audit calling it a broken promise.

    So the role is a field. A payload that carries `suite_roles` is read from it; one written
    before the field existed (the drafted stage 1 is one) falls back to its endpoint suites,
    because a checker that cannot read the document it was added to agree with is no checker.
    """
    if s1 is None:
        s1 = default_stage1()
    roles: Mapping[str, str] = (
        s1.suite_roles if isinstance(s1, PreregStage1) else (s1.get("suite_roles") or {})  # type: ignore[assignment]
    )
    if not roles:
        return endpoint_suites(s1)
    # `calibration` is in the set for the same reason it always was: the audit checks the
    # registry's `confirmatory` and `exploratory` statuses and leaves `calibration` alone, and
    # dropping synth here would change nothing except make the two sets stop matching by luck.
    return frozenset(s for s, role in roles.items() if role in ("confirmatory", "calibration"))


def endpoint_metrics(s1: PreregStage1 | None = None) -> tuple[str, ...]:
    """The distinct metric names the preregistration commits to.

    Every one of these must be something `pi_eval.score` emits — asserted by
    tests/test_prereg.py. A preregistered endpoint naming a metric no scorer writes reads as
    a committed test that was quietly never run.
    """
    return tuple(sorted({e.metric for e in all_endpoints(s1)}))
