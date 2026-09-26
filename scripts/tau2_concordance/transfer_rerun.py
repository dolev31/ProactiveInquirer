"""The pre-declared reader for the tau2 transfer rerun: trained seeds 1 and 2 against the base, prompted.

THE RULES ARE NOT DECIDED HERE. They are `artifacts/tau2_rerun_20260923/RULES.md`, fixed before any
rerun unit ran. Each rule is a module constant or a CLI flag, so an amendment is a one-line change
with its own commit, and the `--out` JSON records the constants it ran under.

WHAT IT REUSES, AND WHY IT COPIES NOTHING. A hand copy of run-unit logic has already cost this
repository a silent four-field divergence, and three instruments in this directory once re-keyed pairs
by hand and paired the wrong fork points. So:

  * pairing is `pi_eval.fork_report._pair_forks`, called once per (suite, prompt variant, trained
    seed) -- `fork_paired.py`'s partition by variant, with the trained seed added because two
    trained models at one fork key would otherwise be refused as a duplicate arm;
  * the PRIMARY (RULES amendment 7) is follow-up user turns after the fork point,
    `pi_eval.fork_report.follow_ups`; the CO-PRIMARY GUARD is task success, `fork_paired.succeeded`
    (`native.tau_reward > 0`, written by `pinq_adapters.tau2.actuator.attach_reward`), read by the
    same machinery and non-inferior at SUCCESS_NI_MARGIN; the HEADLINE combines the two;
  * the sign test is `pi_eval.fork_report.sign_test_p`, follow-ups are `fork_report.follow_ups`,
    and READ / WRITE are `fork_paired.typed_calls` over `conf/tau2/tool_types.json`;
  * the interval is `pi_eval.stats.inference.cluster_bootstrap` over one unit per TASK -- the call
    `paired_difference(..., clusters=None)` makes, which is the path Table 1 takes through
    `scripts/matched_cost.py`.

THE UNIT IS THE TASK. One task carries several fork points and each fork point several run seeds,
so a pair count is not an n. The pair deltas are averaged within a task and the task mean is the
observation. Pair-level counts are printed and labelled NOT QUOTABLE. The seeds-pooled line
combines the two trained seeds WITHIN a task before the bootstrap, so tasks are weighted equally
and training-seed variance is not in the interval (SEED_POOLING; the other rule is printed
whenever the two differ).

THE GATE (RULES amendment 1). DECIDED means every 50k interval, at seeds 101/202/303, excludes 0.
The task sign test, its n-0 floor and the Holm-adjusted p are a reported column, and a
disagreement is printed as `criteria: DISAGREE`. Every interval is printed beside its
informative-task split (`6-1 of 25, 18 tied`); the printed interval is 10000 resamples at seed 0.

A SECOND COMPARATOR (RULES amendment 11). `--comparator-store ROOT --comparator-model PIN` reads
the comparator from its own root (`<root>/<suite>/<pin>/`, its own `--comparator-launch-record`)
and pairs the trained seeds against it: the same endpoints, rules and format, labelled by its
CONTRAST ("trained vs prompted gpt-oss-120b"), in a Holm family of its own. Every provenance
refusal reads both stores together. The prompted 8B base, read from the main source, is set
against it as a descriptive block with no claim. A gateway questioner is not thinking-probed; its
`--questioner-probe-record` stands in, and the share of its calls at the token cap is printed.

NOTHING ABSENT IS DROPPED SILENTLY. An arm missing at a fork point and a unit that did not finish
are counted per arm and variant, and the primary is re-read under both worst-case imputations. A
selection rule can select absence, so a reading that only looks at complete pairs can look clean
over a population that lost exactly the units that would have changed it.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob as globmod
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE))

import extract_records  # noqa: E402
import fork_paired  # noqa: E402
import gold_objects  # noqa: E402

from pi_eval.fork_report import _pair_forks, follow_ups, sign_test_p  # noqa: E402
from pi_eval.stats.inference import cluster_bootstrap  # noqa: E402
from pinq_train.export.dataset import is_paraphrase  # noqa: E402

# ============================================================================ the declared design
RULES_DOC = "artifacts/tau2_rerun_20260923/RULES.md"
EXIT_REFUSED = 3
VALIDATION_STAMP = "VALIDATION ONLY — NOT A RESULT"
PILOT_STAMP = "PILOT — NOT A RESULT"

TREATMENT_ARM = "inquirer_trained"
CONTROL_ARM = "inquirer_prompted"
#: the same base weights, prompted. Checked against every comparator run's questioner pin.
COMPARATOR_MODEL = "qwen3-8b-base"
#: (label, questioner model_id, expected weight-sha256 prefix). Unregistered on purpose: a first
#: registry row would re-label finished runs that pinned these names. Served by LSF 910128.
TRAINED_SEEDS: tuple[tuple[str, str, str], ...] = (
    ("s1", "qwen3-8b-dpo-stacked-notdone-both-s1", "c33d3e6e0b7fd666"),
    ("s2", "qwen3-8b-dpo-stacked-notdone-both-s2", "8e8603830b153535"),
)
SUITES: tuple[str, ...] = ("tau2_retail", "tau2_airline")
#: `UnitSpec.prompt_variant`, i.e. status.json key[9]. Never pooled.
VARIANTS: tuple[str, ...] = ("tau2_base", "tau2_stop")
CAP = 16

#: RULES amendment 7 (the user's design, fixed before any rerun data). The PRIMARY is follow-up
#: user turns after the fork point (`pi_eval.fork_report.follow_ups`: n_user_turns minus
#: n_prefix_user_turns), trained minus comparator; fewer is fewer human turns. Task success is the
#: CO-PRIMARY GUARD, read by the same machinery, and non-inferiority at SUCCESS_NI_MARGIN: it passes
#: only if EVERY deciding (50k) lower bound, at every bootstrap seed, is above the margin.
PRIMARY_ENDPOINT = "follow-up user turns after the fork point"
GUARD_ENDPOINT = "task success"
PRIMARY_FIELDS = ("n_user_turns", "n_prefix_user_turns")
SUCCESS_NI_MARGIN = -0.10
HEADLINE_YES = "reduces the human in the loop: YES"
HEADLINE_NO = "reduces the human in the loop: NO"
#: hand-offs to a human agent, per unit (status field from the enforcement lane)
HANDOFF_FIELD = "n_transfer_to_human"
#: retrieval k must be identical across arms: UnitSpec k on every manifest
MANIFEST_K_FIELD = "k"
RETRIEVAL_K = 5

# ============================================================================ where each field lives
#: status.json `key` is `pi_run.worker.UnitSpec.key`; these are its positions.
KEY_SUITE, KEY_TASK, KEY_ARM, KEY_SEED, KEY_TRACE, KEY_PREFIX_K, KEY_VARIANT = 0, 1, 2, 3, 7, 8, 9
#: the primary endpoint's field; `fork_paired.succeeded` reads it as `> 0`.
REWARD_PATH: tuple[str, ...] = ("native", "tau_reward")
#: manifest.json as embedded by `extract_records.py`.
MANIFEST_FIELD = "_manifest"
PINS_FIELD = "pins"
PIN_MODEL = "model_id"
PIN_BASE_URL_SHA = "base_url_sha"
PIN_ADAPTER_SHA = "adapter_sha"
QUESTIONER_ROLE = "inquirer"
CODE_VERSION_FIELD = "code_version"
DIRTY_FIELD = "dirty"
MANIFEST_CAP_FIELD = "budget_cap"
#: added by the enforcement lane; read from status.json, then from manifest.json.
BUDGET_ENFORCEMENT_FIELD = "budget_enforcement"
#: RULES amendment 7: separate budgets -- the asks ledger (spent.retrieval_calls, budget_cap) and
#: the gate's own tool budget (n_gate_charged, tool_call_cap). Any other regime refuses.
BUDGET_ENFORCEMENT_REQUIRED = "refuse_and_tell_split"
TOOL_CALL_CAP_FIELD = "tool_call_cap"
POST_HOC_OVERRUN_PATH: tuple[str, ...] = ("spent", "post_hoc_budget_overrun")
SPENT_RETRIEVAL_PATH: tuple[str, ...] = ("spent", "retrieval_calls")
#: calls refused in-dialogue by the cap (enforcement lane ba48ab0). Absent prints NOT RECORDED,
#: never zero; on an ok run it must equal len(refused_budget_calls).
N_REFUSED_BUDGET_FIELD = "n_refused_budget"
REFUSED_CALLS_FIELD = "refused_budget_calls"
#: ba48ab0 writes the cap on the status as well as the manifest; both must be the declared cap.
STATUS_CAP_FIELD = "budget_cap"
#: THE BUDGET IS ON LIVE SPEND (RULES amendment 2): asks + executed non-GENERIC tool calls after
#: the fork point; the inherited prefix is free, so n_env_calls (prefix + live) may exceed the cap
#: and is never refused on. Field names from the enforcement lane; reconcile against its sha.
N_PREFIX_ENV_CALLS_FIELD = "n_prefix_env_calls"
N_LIVE_ENV_CALLS_FIELD = "n_live_env_calls"
N_LIVE_GENERIC_FIELD = "n_live_env_calls_generic"
N_GATE_CHARGED_FIELD = "n_gate_charged"
#: the asks the ledger holds beside the gate's charge. On the withdrawn retail records
#: spent.retrieval_calls == n_asks + n_env_calls exactly on 345 of 408 runs (the rest: failed,
#: uncharged calls), so the asks term is `n_asks` unless the enforcement lane names another.
GATE_ASKS_FIELD = "n_asks"
#: tool types the gate neither charges nor gates (calculate, transfer_to_human_agents).
UNCHARGED_TOOL_TYPES = frozenset({"GENERIC"})
#: 1ae52cb: the exempt list the run used, which must equal the map's GENERIC set for its domain
#: (the reader recounts from the MAP, never from this list); the transcript census's verdict on
#: the gate; the census's live non-GENERIC answered calls; asks the loop rejected, which are in
#: n_asks but never charged (a ledger counter: absent means none were recorded).
BUDGET_EXEMPT_FIELD = "budget_exempt_tools"
GATE_CENSUS_AGREES_FIELD = "gate_census_agrees"
N_LIVE_NONGENERIC_OK_FIELD = "n_live_env_calls_nongeneric_ok"
REJECTED_ASKS_PATH: tuple[str, ...] = ("spent", "rejected_user_asks")
#: status.json keeps ONLY THE LAST rollout's stop_reason (tau2_runner.merge_trajectories) and
#: turns.jsonl holds ask turns only, so rollout 1's stop and asks are recorded nowhere.
STOP_REASON_FIELD = "stop_reason"
POLICY_STOP = "policy_stop"
#: RULES amendment 4. Live non-GENERIC calls EXECUTED (answered, error or not), from the census.
N_LIVE_NONGENERIC_FIELD = "n_live_env_calls_nongeneric"
#: the enforcement lane's rollout-1 and first-live-call fields, on every gated status. A missing
#: one refuses on a gated run; NOT RECORDED is for legacy populations only.
#: spent_before_first_live_call is the ledger's spend when the first live tool call executed, and
#: null when none executed (first_live_call_executed false).
ROLLOUT1_STOP_FIELD = "rollout1_stop_reason"
ROLLOUT1_ASKS_FIELD = "rollout1_n_asks"
FIRST_LIVE_EXECUTED_FIELD = "first_live_call_executed"
SPENT_BEFORE_FIRST_CALL_FIELD = "spent_before_first_live_call"
#: "the asks reached the ask cap before the first tool call" is exact from those fields; the old
#: status derivation is kept only as a labelled fallback for a population that lacks them. Under
#: separate budgets it is a DESCRIPTIVE stopping measure, never a refusal: the asks reaching their
#: own cap blocks no tool call (TOOL_CALLS_BLOCKED_BY_ASKS), and a leak is caught by the refused-
#: call-below-tool-cap check.
STARVED_EXACT = "exact"
STARVED_FALLBACK = "fallback: (spent - n_gate_charged) >= cap and n_gate_charged == 0"
TOOL_CALLS_BLOCKED_BY_ASKS = "0 by design (separate budgets)"
#: amendment 4's prose aid, made DESCRIPTIVE after 8cdf0a1: its wording rule was about the
#: comparator exhausting the SHARED budget on questions, and amendment 7 removed the shared budget.
#: The aid prints at every share, beside every primary line, and says the rule is superseded.
WORDING_AID_TAIL = (
    "(descriptive; amendment 4's shared-budget wording rule is superseded by amendment 7's "
    "separate budgets)"
)
A4_STARVED = "asks reached the ask cap before the first tool call"
A4_ROLLOUT1_STOP = "rollout-1 STOP (rollout1_stop_reason == policy_stop)"
A4_ZERO_NONGENERIC = "zero executed non-GENERIC calls"
#: RULES amendment 6: QUOTED on every unit -- spent_before_first_live_call where a live call
#: executed, else the total spend (censored). Restricting the paired difference to pairs where both
#: arms ACTED would condition on a post-treatment variable, which amendment 4 forbids; that version
#: is printed only as a labelled descriptive line with no interval.
A4_CALLS_BEFORE_FIRST = "calls spent before the first executed tool call"
CALLS_QUOTED_LABEL = "defined on all units (censored at total spend where no live call executed)"
CALLS_DESCRIPTIVE_LABEL = "conditional on both arms acting — descriptive, not quoted"
DEFINED, UNDEFINED, NOT_RECORDED = "defined", "undefined", "not recorded"
ENV_CALL_SEQ = "_env_call_seq"
#: per ask, the record uids the harness recorded it retrieving (extract_records, from turns.jsonl)
ASK_UIDS_FIELD = "_ask_retrieved_uids"
#: RULES amendment 7's secondaries, signed off with conditions by the paper lane. Never called
#: "precision". A gold object is a pre-dialogue record a gold action names (gold_objects.py).
A7_HIT = "gold-object hit rate per ask"
A7_RECALL = "gold-object recall"
A7_ASKS = "asks per unit"
#: RULES amendment 9: where the tool cap bound. Printed beside the success guard, per arm and as a
#: paired task-level difference (10k/seed 0, descriptive): a success gap must be readable against
#: the units on which the 16-call tool cap refused a call.
A9_REFUSED_UNITS = "units with a refused tool call (n_refused_budget > 0)"
#: RULES amendment 8: EXPLORATORY descriptive secondaries, declared after one interim look at about
#: 170 units and labelled exploratory on every line and in the JSON, whatever they show. Per arm,
#: with paired task-level differences on the primary's OWN pairs (printed interval only, in no Holm
#: family); a unit with no ask is undefined, its pair dropped and counted. "Asks whose target is the
#: user" is omitted: 0 by construction in these closed-channel arms. The near-exact guard is the
#: repo's pair-paraphrase guard, IMPORTED (`is_paraphrase` above), never restated here.
EXPLORATORY_LABEL = "EXPLORATORY (declared in amendment 8 after one interim look)"
AQ_RETRIEVE = "share of asks retrieving >= 1 record"
AQ_EXACT = "exact repeat rate"
AQ_NEAR = "near-exact repeat rate"
AQ_MEASURES = (AQ_RETRIEVE, AQ_EXACT, AQ_NEAR)
AQ_DEFINITIONS = {
    AQ_RETRIEVE: "the unit's asks whose recorded retrieved_uids hold >= 1 record, over its asks",
    AQ_EXACT: "the unit's asks whose text, lowercased and whitespace-normalised, equals an earlier "
    "ask's in the unit, over its asks",
    AQ_NEAR: "the unit's asks that pinq_train.export.dataset.is_paraphrase (token-set Jaccard >= "
    "0.5 with >= 4 shared content tokens) matches to an earlier ask in the unit, over its asks; "
    "the shared-token floor means a short exact repeat need not be a near-exact one",
}
AQ_OMITTED = "asks whose target is the user: 0 by construction in these arms (amendment 8)"
ASK_QUESTIONS_FIELD = "_ask_questions"
#: RULES amendment 11: a prompted gpt-oss-120b questioner arm -- `inquirer_prompted` pinned to a
#: GATEWAY model -- run at the same code_version in its OWN ROOT with its own launch record. The
#: comparator is selected by arm AND pin, never by arm alone: the 8B base shares its arm_id. The
#: prompted 8B base (--reference-model, from the main source) set against it is descriptive only.
REFERENCE_ARM = "reference"
REFERENCE_LABEL = "descriptive, no claim"
#: the thinking probe probes hosted_vllm routes; a gateway pin is not thinking-probed and carries
#: the record `run_extra_pin.py questioner-probe` writes instead (its `questioner_probe` dict plus
#: `_probe_main`'s fields), one per job. max_tokens there is the EFFECTIVE budget sent (content x
#: reasoning headroom), the same basis as a call row's tok_completion.
QPROBE_GLOB = "questioner_probe*.json"
QPROBE_MODEL = "model"
QPROBE_MAX_TOKENS = "max_tokens"
QPROBE_COMPLETION = "completion_tokens"
QPROBE_REASONING = "reasoning_tokens"
QPROBE_HARNESS = "harness_sha"
QPROBE_CONTENT_CHARS = "content_chars"
LSB_FIELDS = ("LSB_JOBID", "LSB_JOBINDEX")
#: RULES amendment 11, aligned with lane D's driver (scripts/tau2_rerun/run_extra_pin.py, b3a81d8,
#: sha256 bd8665e9). The pin, its store directory, the regime string and the campaign sha are the
#: DRIVER'S OWN, imported from that file (`extra_pin_driver`), never restated here. Per suite the
#: arm's root `<arm-root>/<suite>/` holds ARM_LAUNCH.json, the units under the driver's STORE_DIR,
#: and each job's records under `_jobs/<label>/`: JOB.json beside the job's own questioner probe.
DRIVER_PATH = _HERE.parent / "tau2_rerun" / "run_extra_pin.py"
ARM_LAUNCH_NAME = "ARM_LAUNCH.json"
JOBS_DIR = "_jobs"
JOB_RECORD = "JOB.json"
JOB_QPROBE = "questioner_probe.json"
#: a questioner call is AT THE TOKEN CAP when its completion tokens (the provider's
#: completion_tokens, reasoning included) reach TOKEN_CAP_FRACTION of the questioner's max_tokens.
#: max_tokens is on no call row (the client grows it by a reasoning headroom), so it is read from
#: the questioner probe record, which ran at it. Exact arithmetic: a float 0.9 must not decide a
#: boundary call. Above TOKEN_CAP_LIMITATION across the arm it is a limitation of the arm.
TOKEN_CAP_FRACTION = Fraction(9, 10)
TOKEN_CAP_LIMITATION = 0.05
TOKEN_CAP_LIMITATION_TEXT = "limitation of the arm (amendment 11)"
INQUIRER_CALLS_FIELD = "_inquirer_calls"
A7_DEFINITION = (
    "gold-object hit rate per ask = the share of a unit's asks whose retrieved record uids "
    "intersect the task's gold objects (undefined where the arm asked nothing: the pair is "
    "dropped, never imputed); gold-object recall = the share of the task's gold objects any ask "
    "touched; both undefined where the task has no pre-dialogue gold object. Read together, never "
    "alone, beside asks per unit (amendment 7, paper-lane sign-off with conditions)."
)

#: the launcher's store layout (0212fe0): one runs sub-root per questioner pin, per suite.
STORE_LAYOUT = "<root>/<suite>/<pin>/"
STORE_SUITE = "_store_suite"
STORE_PIN = "_store_pin"
#: the launcher's LAUNCH.json (0212fe0 `launch_record`): `suite`, and `plan.units`, each unit
#: {suite, pin, arm_id, variant, seed, task_id, k, trace_sha}.
LAUNCH_SUITE = "suite"
LAUNCH_UNITS: tuple[str, ...] = ("plan", "units")
ENV_CALLS_FROM_OUTCOME = "_env_calls_from_outcome"
ENV_TOOL_COUNTS = "_env_tool_counts"
ENV_REQUESTOR_COUNTS = "_env_requestor_counts"

#: `scripts/hpc/serve.sh` SERVED.<job>.json
SERVE_SCHEMA = "pinq.serve.v1"
SERVE_ADAPTERS = "adapters"
SERVE_SHA = "sha256"
SERVE_GLOB = "SERVED.*.json"
SERVE_FIELDS_KEPT = ("job", "host", "port", "written_at", "base_model", "served_base")

#: the thinking-probe guard's record. Assumed shape, to reconcile with that lane:
#: {"verdict": "PASS"|"REFUSE", "proxy_url": str, "routes": [{"model": str,
#:  "chat_template_kwargs": {...}, "completion_tokens": int, "thinking_detected": bool}]}
PROBE_VERDICT = "verdict"
PROBE_PASS = "PASS"
PROBE_PROXY_URL = "proxy_url"
PROBE_ROUTES = "routes"
PROBE_ROUTE_MODEL = "model"
PROBE_ROUTE_KWARGS = "chat_template_kwargs"
PROBE_ROUTE_THINKING = "thinking_detected"
PROBE_CONFIG_SHA = "config_sha256"
PROBE_CONFIG_PATH = "config_path"
PROBE_GLOB = "thinking_probe.json"
#: RULES amendment 10: from link 2 on, each job's proxy config is GENERATED -- the tracked
#: conf/serving/litellm.tau2.yaml plus gateway deployments identical but for
#: `api_key: os.environ/LITELLM_API_KEY_PARALLEL_{1,2,3}` -- so one population carries several
#: config_sha256 values. Each job archives its config NEXT TO its probe record; the archive is named
#: by the basename of the probe's `config_path` (found by NAME, then its sha checked: finding it by
#: sha would make check (a) true by construction).
LOCAL_ROUTE_PREFIX = "hosted_vllm/"
GATEWAY_ROUTE_PREFIX = "openai/"
#: the fields of a hosted_vllm entry compared byte-for-byte, as canonical JSON (check b)
LOCAL_ENTRY_FIELDS = ("model", "api_base", "api_key", "extra_body")
ENV_KEY = re.compile(r"os\.environ/[A-Za-z_][A-Za-z0-9_]*")
CONFIG_CHECKS = {
    "a": "each archived config (next to its probe record, named by the basename of its "
    "config_path) hashes to that record's config_sha256",
    "b": "hosted_vllm entries byte-identical across the configs (model_name, model, api_base, "
    "api_key, extra_body)",
    "c": "entries that differ are gateway (openai/*) entries differing only in api_key, each "
    "os.environ/<NAME>",
    "d": "litellm_settings, general_settings and every other section but model_list identical",
}

# ============================================================================ the statistics
#: the verdict ladder (RULES amendment 1): DECIDED iff every interval at the largest count excludes 0.
BOOT_RESAMPLES: tuple[int, ...] = (1_000, 10_000, 50_000)
BOOT_SEEDS: tuple[int, ...] = (101, 202, 303)
#: the interval that is PRINTED beside the split (amendment 1): n_boot, seed -- Table 1's.
PRINTED_INTERVAL: tuple[int, int] = (10_000, 0)
ALPHA = 0.05
#: How the two training seeds meet inside a task before the bootstrap. `pair_mean` (default): every
#: seed's pairs in one task mean -- the arithmetic of `scripts/seed_identity/table1_by_seed.py::
#: pooled_symmetric`, so a seed weighs by its (defined) pairs. `seed_mean`: each seed's task mean,
#: then their mean. Identical when every seed has the same pairs in a task.
SEED_POOLINGS = ("seed_mean", "pair_mean")
#: Table 1's code decides (table1_by_seed.py:75-83, pooled_symmetric): every training seed's pairs
#: go into ONE per-task mean. seed_mean is printed beside it whenever the two differ.
SEED_POOLING = "pair_mean"
INTERVAL_STATEMENT = (
    "BCa interval over tasks (pi_eval.stats.inference.cluster_bootstrap, one unit per task): it "
    "weights tasks equally and excludes training-seed variance, because the training seeds are "
    "combined within the task before the bootstrap."
)
#: a bootstrap bound closer to 0 than this touches zero (float noise at a mass point).
ZERO_TOL = 1e-9
#: the declared primary family. Variants are never pooled, so "suite x seed" per variant is two
#: tests per (suite, seed), and the default family holds all of them.
HOLM_FAMILIES = ("suite_variant_seed", "suite_seed_within_variant")
HOLM_FAMILY = HOLM_FAMILIES[0]

_MISSING = object()


class Refusal(Exception):
    """A reason not to read. `main` prints it and exits EXIT_REFUSED."""


def _get(d: Any, path: Sequence[str]) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, Mapping) or p not in cur:
            return _MISSING
        cur = cur[p]
    return cur


def _variant(r: Mapping) -> str:
    return str((r.get("key") or [""] * 11)[KEY_VARIANT])


def _suite(r: Mapping) -> str:
    return str((r.get("key") or [""] * 11)[KEY_SUITE])


def _manifest(r: Mapping) -> Mapping | None:
    m = r.get(MANIFEST_FIELD)
    return m if isinstance(m, Mapping) else None


def _questioner_model(r: Mapping) -> str | None:
    m = _get(_manifest(r), (PINS_FIELD, QUESTIONER_ROLE, PIN_MODEL))
    return None if m is _MISSING or m is None else str(m)


def is_gateway_pin(model: str | None) -> bool:
    """A gateway (frontier-API) pin, routed `openai/*` by the proxy -- not a hosted_vllm route."""
    return str(model or "").startswith(GATEWAY_ROUTE_PREFIX)


def short_model(model: str) -> str:
    return str(model).rsplit("/", 1)[-1]


_DRIVER: Any = None


def extra_pin_driver() -> Any:
    """Lane D's extra-pin driver, loaded from its own file for its OWN constants (EXTRA_PIN,
    STORE_DIR, GATEWAY_ONLY, CAMPAIGN_SHA) and its job-index normaliser. Only the comparator-store
    reading loads it; its top level imports the standard library alone."""
    global _DRIVER
    if _DRIVER is None:
        if not DRIVER_PATH.is_file():
            raise Refusal(
                f"the extra-pin driver {DRIVER_PATH} is absent: the comparator store's layout, "
                "launch record and regime are defined there (RULES amendment 11)"
            )
        spec = importlib.util.spec_from_file_location("run_extra_pin", DRIVER_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _DRIVER = mod
    return _DRIVER


def comparator_store_dir(pin: str) -> str:
    """The directory the driver writes a pin's units to: its STORE_DIR (the pin with '/' as '__'),
    for the one pin it runs. The manifest's questioner pin must still be the pin itself."""
    drv = extra_pin_driver()
    if pin != drv.EXTRA_PIN:
        raise Refusal(
            f"--comparator-store is the extra-pin driver's arm, pin {drv.EXTRA_PIN}; "
            f"--comparator-model {pin} has no store directory there"
        )
    return drv.STORE_DIR


def comparator_job_dirs(root: Path, suites: Sequence[str]) -> list[Path]:
    """Every job the driver staged, `<root>/<suite>/_jobs/<label>/` holding a JOB.json."""
    out: list[Path] = []
    for suite in suites:
        jobs = root / suite / JOBS_DIR
        if jobs.is_dir():
            out += sorted(d for d in jobs.iterdir() if d.is_dir() and (d / JOB_RECORD).is_file())
    return out


def contrast_label(comparator: str) -> str:
    """The contrast a reading is labelled by: "trained vs prompted gpt-oss-120b"."""
    return f"trained vs prompted {short_model(comparator)}"


def _slot(r: Mapping) -> tuple:
    """(trace, k, run seed) from the key, which non-ok records carry when their fields do not."""
    k = r.get("key") or [None] * 11
    return (k[KEY_TRACE], k[KEY_PREFIX_K], k[KEY_SEED])


def _task_of_key(r: Mapping) -> str:
    return str((r.get("key") or [""] * 11)[KEY_TASK])


def _fnan(x: float) -> float | None:
    return None if isinstance(x, float) and math.isnan(x) else x


def _clean(o: Any) -> Any:
    """JSON without NaN: a NaN is written as null, which every reader reads as absent."""
    if isinstance(o, float):
        return _fnan(o)
    if isinstance(o, Mapping):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


# ============================================================================ statistics
def holm(ps: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order. A NaN (no informative task) enters as 1."""
    vals = [1.0 if (p is None or math.isnan(p)) else float(p) for p in ps]
    m = len(vals)
    order = sorted(range(m), key=lambda i: vals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * vals[i]))
        adj[i] = running
    return adj


def decided(intervals: Iterable[Mapping], deciding_n: int) -> bool:
    """DECIDED only if EVERY interval at the deciding resample count excludes zero, on one side.

    A bound within ZERO_TOL of 0 TOUCHES zero. Paired binary outcomes put bootstrap mass exactly
    at 0 and the percentile returns it with float noise (-1.84e-18 and 0.0 measured on the legacy
    retail population), so a strict `lo > 0` would let rounding decide a cell.
    """
    at = [iv for iv in intervals if iv["n_boot"] == deciding_n]
    if not at:
        return False
    above = all(iv["lo"] is not None and iv["lo"] > ZERO_TOL for iv in at)
    below = all(iv["hi"] is not None and iv["hi"] < -ZERO_TOL for iv in at)
    return above or below


def decided_side(intervals: Iterable[Mapping], deciding_n: int) -> str | None:
    """'negative' / 'positive' when every deciding interval excludes 0 on that side, else None."""
    at = [iv for iv in intervals if iv["n_boot"] == deciding_n]
    if not at:
        return None
    if all(iv["hi"] is not None and iv["hi"] < -ZERO_TOL for iv in at):
        return "negative"
    if all(iv["lo"] is not None and iv["lo"] > ZERO_TOL for iv in at):
        return "positive"
    return None


def success_ni(guard_tl: Mapping[str, Any]) -> dict[str, Any]:
    """The co-primary guard's non-inferiority at SUCCESS_NI_MARGIN: EVERY deciding lower bound, at
    every bootstrap seed, strictly above the margin."""
    n = guard_tl["deciding_n_boot"]
    lows = [iv["lo"] for iv in guard_tl["intervals"] if iv["n_boot"] == n]
    ok = bool(lows) and all(
        lo is not None and not math.isnan(lo) and lo > SUCCESS_NI_MARGIN for lo in lows
    )
    return {"margin": SUCCESS_NI_MARGIN, "n_boot": n, "lower_bounds": lows, "passes": ok}


def headline(primary_tl: Mapping[str, Any], guard_tl: Mapping[str, Any]) -> dict[str, Any]:
    """YES iff the follow-up delta is DECIDED negative AND success is non-inferior at the margin;
    otherwise NO, naming every condition that failed."""
    side = decided_side(primary_tl["intervals"], primary_tl["deciding_n_boot"])
    ni = success_ni(guard_tl)
    fu_neg = side == "negative"
    reasons = []
    if not fu_neg:
        reasons.append(
            "follow-up delta not DECIDED negative ("
            + ("decided positive" if side == "positive" else "undecided")
            + ")"
        )
    if not ni["passes"]:
        low = min((x for x in ni["lower_bounds"] if x is not None), default=float("nan"))
        reasons.append(
            f"success non-inferiority fails: a {ni['n_boot']}-resample lower bound is not above "
            f"{SUCCESS_NI_MARGIN:+.2f} (lowest {low:+.4f})"
        )
    return {
        "text": HEADLINE_YES
        if fu_neg and ni["passes"]
        else f"{HEADLINE_NO} -- " + "; ".join(reasons),
        "follow_up_decided_negative": fu_neg,
        "success_ni_passes": ni["passes"],
        "success_ni_margin": SUCCESS_NI_MARGIN,
        "success_ni": ni,
    }


def criteria_note(is_decided: bool, holm_p: float) -> str:
    """Name a disagreement between the two declared criteria; never resolve it here."""
    rejects = holm_p is not None and not math.isnan(holm_p) and holm_p < ALPHA
    if is_decided and not rejects:
        return f"DISAGREE: interval decided, Holm sign test not < {ALPHA:g}"
    if rejects and not is_decided:
        return f"DISAGREE: Holm sign test < {ALPHA:g}, interval undecided"
    return "agree"


def split_of(up: int, down: int, tied: int) -> str:
    """The informative-task split quoted beside every interval (amendment 1): `6-1 of 25, 18 tied`."""
    return f"{up}-{down} of {up + down + tied}, {tied} tied"


def task_stats(
    task_deltas: Mapping[str, float],
    resamples: Sequence[int],
    seeds: Sequence[int],
    printed: tuple[int, int] = PRINTED_INTERVAL,
) -> dict[str, Any]:
    """Task-level reading of one cell: counts, the sign test with its floor, and the bootstrap.

    The floor is the smallest p the INFORMATIVE task count can produce (ties are dropped by the
    sign test, so they cannot help it): a unanimous split's p equals it, so a p at its floor
    measures n and nothing else.

    One bootstrap unit per task, holding the task's mean paired delta: the units
    `paired_difference(..., clusters=None)` hands `cluster_bootstrap`, i.e. Table 1's call.
    """
    tasks = sorted(task_deltas)
    vals = [float(task_deltas[t]) for t in tasks]
    up = sum(v > 0 for v in vals)
    down = sum(v < 0 for v in vals)
    tied = len(vals) - up - down

    def boot(n_boot: int, seed: int) -> dict[str, Any]:
        _, lo, hi = cluster_bootstrap([[v] for v in vals], n_boot=n_boot, alpha=ALPHA, seed=seed)
        return {"n_boot": n_boot, "seed": seed, "lo": lo, "hi": hi}

    intervals = [boot(n, s) for n in resamples for s in seeds]
    deciding = max(resamples)
    return {
        "n_tasks": len(vals),
        "up": up,
        "down": down,
        "tied": tied,
        "split": split_of(up, down, tied),
        "mean_delta": statistics.fmean(vals) if vals else float("nan"),
        "printed_interval": boot(*printed),
        "sign_test_p": sign_test_p(down, up),
        "sign_test_floor": sign_test_p(up + down, 0) if up + down else float("nan"),
        "intervals": intervals,
        "deciding_n_boot": deciding,
        "decided": decided(intervals, deciding),
        "task_deltas": {t: task_deltas[t] for t in tasks},
    }


def _task_means(per_pair: Mapping[tuple, float], task_of: Mapping[tuple, str]) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for k, d in per_pair.items():
        groups[task_of[k]].append(d)
    return {t: statistics.fmean(v) for t, v in groups.items()}


# ============================================================================ output
class Emit:
    """stdout, with every line stamped when the population is a validation population."""

    def __init__(self, stamp: str | None) -> None:
        self.stamp = stamp

    def __call__(self, text: str = "") -> None:
        for line in (text or "").split("\n") or [""]:
            print(f"[{self.stamp}] {line}" if self.stamp else line)


def _fmt_p(p: float) -> str:
    return "  n/a " if p is None or math.isnan(p) else f"{p:.4f}"


def _fmt_iv(iv: Mapping) -> str:
    lo, hi = iv["lo"], iv["hi"]
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return "[  n/a  ]"
    return f"[{lo:+.4f}, {hi:+.4f}]"


def _print_task_stats(emit: Emit, s: Mapping, indent: str = "      ") -> None:
    pi = s["printed_interval"]
    emit(
        f"{indent}mean delta={s['mean_delta']:+.4f}  BCa 95% over tasks ({pi['n_boot']} "
        f"resamples, seed {pi['seed']}): {_fmt_iv(pi)}  split {s['split']}"
    )
    emit(
        f"{indent}  reported column (not a gate): sign p={_fmt_p(s['sign_test_p'])}  "
        f"floor({s['up'] + s['down']}-0)={_fmt_p(s['sign_test_floor'])}"
        + (
            "  [p AT ITS FLOOR: measures n only]"
            if s["up"] + s["down"]
            and not math.isnan(s["sign_test_p"])
            and abs(s["sign_test_p"] - s["sign_test_floor"]) < 1e-12
            else ""
        )
    )
    by_n: dict[int, list] = defaultdict(list)
    for iv in s["intervals"]:
        by_n[iv["n_boot"]].append(iv)
    for n_boot in sorted(by_n):
        row = "  ".join(f"seed {iv['seed']}: {_fmt_iv(iv)}" for iv in by_n[n_boot])
        emit(f"{indent}  verdict ladder, {n_boot:>6} resamples  {row}")
    verdict = "DECIDED" if s["decided"] else "UNDECIDED"
    emit(
        f"{indent}  {verdict} (every {s['deciding_n_boot']}-resample interval must exclude 0; "
        f"undecided is not a null)  split {s['split']}"
    )


# ============================================================================ reading
def load(
    args: argparse.Namespace, suites: Sequence[str], pins: Sequence[str]
) -> tuple[list[dict], dict]:
    """The records to read, each stamped with the pin sub-root it sat in when that is known.

    `--store <root>` reads the launcher's layout, `<root>/<suite>/<pin>/`, one sub-root per
    declared suite and questioner pin (0212fe0: s1 and s2 both run as `inquirer_trained` and a
    status carries no pin, so the sub-root is what keeps them apart). A declared sub-root that is
    missing refuses; anything else under a suite (`_jobs/`, an undeclared pin) is listed, not read.
    `--records` reads an extract, whose `_relpath` carries `<suite>/<pin>/<run_id>` when it was
    extracted from `<root>`.
    """
    if args.store:
        return load_store(Path(args.store), suites, pins)
    raw = Path(args.records).read_bytes()
    recs = list(json.loads(raw))
    for r in recs:
        parts = Path(str(r.get("_relpath") or "")).parts
        if len(parts) >= 2:
            r.setdefault(STORE_PIN, parts[-2])
        if len(parts) >= 3:
            r.setdefault(STORE_SUITE, parts[-3])
    src = {
        "kind": "records",
        "path": str(Path(args.records).resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return recs, src


def load_store(
    root: Path,
    suites: Sequence[str],
    pins: Sequence[str],
    dirs: Mapping[str, str] | None = None,
) -> tuple[list[dict], dict]:
    """`<root>/<suite>/<dir>/` for every declared suite and pin, where a pin's dir is the pin itself
    unless `dirs` names another (the extra-pin driver's STORE_DIR). Every run read is stamped with
    the PIN, so the manifest's questioner pin is checked against the pin, not the directory."""
    dirs = dict(dirs or {})
    recs: list[dict] = []
    subs = []
    for suite in suites:
        for pin in pins:
            d = dirs.get(pin, pin)
            sub = root / suite / d
            if not sub.is_dir():
                raise Refusal(
                    f"the pin sub-root {suite}/{d} is missing under {root}"
                    + (f" (the driver's store directory for {pin})" if pin in dirs else "")
                )
            got = extract_records.extract(sub)
            for r in got:
                r[STORE_SUITE], r[STORE_PIN] = suite, pin
            recs += got
            subs.append({"suite": suite, "pin": pin, "dir": d, "n_records": len(got)})
    heads = {Path(dirs.get(pin, pin)).parts[0] for pin in pins}
    not_read = sorted(
        f"{suite}/{d.name}"
        for suite in suites
        for d in (root / suite).iterdir()
        if d.is_dir() and d.name not in heads
    )
    src = {
        "kind": "store",
        "path": str(root.resolve()),
        "layout": STORE_LAYOUT,
        "sub_roots": subs,
        "not_read": not_read,
    }
    return recs, src


def load_launch(
    paths: Sequence[str] | None,
    *,
    records: Sequence[Mapping],
    suites: Sequence[str],
    pins: Sequence[str],
    legacy: bool,
    disabled: list[str],
) -> tuple[dict[tuple, dict[tuple, str]], list[dict]]:
    """The planned slots, from each suite's LAUNCH.json. Raises Refusal.

    Returns ({(suite, variant, pin): {(trace, k, seed): task_id}}, provenance). A planned unit the
    store never saw is invisible to every count the store can make, so without the plan the
    worst-case line could only impute what happened to be written.
    """
    docs: dict[str, dict] = {}
    prov = []
    for path in paths or []:
        doc, sha = _load_json(path, "launch record")
        suite = str((doc or {}).get(LAUNCH_SUITE))
        if suite not in suites:
            raise Refusal(f"launch record {path} is for {suite!r}, not a declared suite")
        if suite in docs:
            raise Refusal(f"two launch records for {suite}")
        units = _get(doc, LAUNCH_UNITS)
        if not isinstance(units, list) or not units:
            raise Refusal(f"launch record {path} holds no {'.'.join(LAUNCH_UNITS)}")
        docs[suite] = {"units": units}
        prov.append({"suite": suite, "path": path, "file_sha256": sha, "n_units": len(units)})
    missing = [s for s in suites if s not in docs]
    if missing and not legacy:
        raise Refusal(
            f"no launch record for declared suite(s) {missing} (--launch-record): a planned unit "
            "that never wrote a status would be invisible"
        )
    if missing:
        disabled.append(f"launch record for {missing}: planned-but-absent slots are not counted")

    planned: dict[tuple, dict[tuple, str]] = defaultdict(dict)
    plan_keys = set()
    for suite, d in docs.items():
        for u in d["units"]:
            if u.get("suite") != suite or u.get("pin") not in pins:
                raise Refusal(
                    f"launch record for {suite} plans a unit at suite {u.get('suite')!r}, pin "
                    f"{u.get('pin')!r}; declared pins: {list(pins)}"
                )
            slot = (u["trace_sha"], u["k"], u["seed"])
            planned[(suite, u["variant"], u["pin"])][slot] = str(u["task_id"])
            plan_keys.add((suite, u["pin"], u["arm_id"], u["variant"]) + slot)
    unplanned = [
        r
        for r in records
        if _suite(r) in docs
        and (_suite(r), _questioner_model(r), (r.get("key") or [None] * 3)[KEY_ARM], _variant(r))
        + _slot(r)
        not in plan_keys
    ]
    if unplanned:
        raise Refusal(
            f"{len(unplanned)} runs are not in the launch plan (e.g. {_e_g(unplanned)}): a run "
            "the launch did not plan belongs to another population"
        )
    return dict(planned), prov


def _parse_trained(specs: Sequence[str] | None) -> list[tuple[str, str, str]]:
    if not specs:
        return list(TRAINED_SEEDS)
    out = []
    for s in specs:
        label, _, rest = s.partition("=")
        model, _, prefix = rest.rpartition(":")
        if not (label and model and prefix):
            raise SystemExit(f"--trained {s!r}: expected LABEL=MODEL:SHA_PREFIX")
        out.append((label, model, prefix))
    return out


def _e_g(runs: Sequence[Mapping], n: int = 3) -> str:
    return ", ".join(str(r.get("run_id") or "<no run_id>") for r in runs[:n])


def _load_json(path: str, what: str) -> tuple[Any, str]:
    p = Path(path)
    if not p.is_file():
        raise Refusal(f"{what} {path} does not exist")
    raw = p.read_bytes()
    try:
        return json.loads(raw), hashlib.sha256(raw).hexdigest()
    except json.JSONDecodeError as exc:
        raise Refusal(f"{what} {path} is not JSON: {exc}") from exc


def _check_gate_charge(ok_runs: Sequence[Mapping]) -> dict[int, float]:
    """The budget identities of a gated unit (1ae52cb); raises Refusal. Returns {id(run): r}.

    (a) The ledger is not asks + gate alone: spent.retrieval_calls = (n_asks - rejected_user_asks)
        + query-expansion sub-retrievals + n_gate_charged, and the expansions have no field. So
        the residual r = spent - n_gate_charged - (n_asks - rejected_user_asks) is refused only
        when NEGATIVE; r > 0 is expansion spend and is printed per arm.
    (b) EXACT: n_gate_charged == the reader's own recount of ok, non-GENERIC env calls from index
        n_prefix_env_calls on, typed by conf/tau2/tool_types.json -- never by the run's own
        budget_exempt_tools, which must itself equal the map's GENERIC set -- and the runner's
        transcript census must agree with the gate (gate_census_agrees). The recount is the
        instrument that can disagree with the gate; the gate counting its own charges cannot.
    prefix + live may be fewer than n_env_calls: an unanswered call is in neither. Not refused.
    """
    need = (
        N_PREFIX_ENV_CALLS_FIELD,
        N_GATE_CHARGED_FIELD,
        TOOL_CALL_CAP_FIELD,
        GATE_ASKS_FIELD,
        GATE_CENSUS_AGREES_FIELD,
        BUDGET_EXEMPT_FIELD,
        ROLLOUT1_STOP_FIELD,
        ROLLOUT1_ASKS_FIELD,
        FIRST_LIVE_EXECUTED_FIELD,
        HANDOFF_FIELD,  # part of the refuse_and_tell_split status (lane A 4e09a07)
    )
    absent = [r for r in ok_runs if any(r.get(f) is None for f in need)]
    if absent:
        gone = sorted({f for r in absent for f in need if r.get(f) is None})
        raise Refusal(
            f"the gate fields {gone} are absent on {len(absent)} ok runs (e.g. {_e_g(absent)}); "
            "the live spend cannot be checked"
        )
    tool_types = fork_paired.load_tool_types()

    def generic(r: Mapping) -> list[str]:
        types = tool_types.get(fork_paired.domain_of(r), {})
        return sorted(n for n, t in types.items() if t in UNCHARGED_TOOL_TYPES)

    off_exempt = [r for r in ok_runs if sorted(r[BUDGET_EXEMPT_FIELD]) != generic(r)]
    if off_exempt:
        r = off_exempt[0]
        raise Refusal(
            f"{BUDGET_EXEMPT_FIELD} is not the map's GENERIC set on {len(off_exempt)} ok runs "
            f"(e.g. {r.get('run_id')}: {sorted(r[BUDGET_EXEMPT_FIELD])} vs {generic(r)})"
        )
    no_before = [r for r in ok_runs if SPENT_BEFORE_FIRST_CALL_FIELD not in r]
    if no_before:
        raise Refusal(
            f"{SPENT_BEFORE_FIRST_CALL_FIELD} is absent on {len(no_before)} ok runs (e.g. "
            f"{_e_g(no_before)}); on a gated run it is null only when no live call executed"
        )
    for r in ok_runs:
        first, before = r[FIRST_LIVE_EXECUTED_FIELD], r[SPENT_BEFORE_FIRST_CALL_FIELD]
        if (first is True) != (before is not None) or first not in (True, False):
            raise Refusal(
                f"{r.get('run_id')}: {FIRST_LIVE_EXECUTED_FIELD} is {first!r} but "
                f"{SPENT_BEFORE_FIRST_CALL_FIELD} is {before!r}: a spend exists iff a live call ran"
            )
        if before is not None and float(before) > float(_get(r, SPENT_RETRIEVAL_PATH)):
            raise Refusal(
                f"{r.get('run_id')}: {SPENT_BEFORE_FIRST_CALL_FIELD}={before} exceeds "
                f"{'.'.join(SPENT_RETRIEVAL_PATH)}={_get(r, SPENT_RETRIEVAL_PATH)}"
            )
    disagree = [r for r in ok_runs if r[GATE_CENSUS_AGREES_FIELD] is not True]
    if disagree:
        raise Refusal(
            f"{GATE_CENSUS_AGREES_FIELD} is not true on {len(disagree)} ok runs (e.g. "
            f"{_e_g(disagree)}): the transcript census and the gate disagree"
        )
    # ---- the tool budget (amendment 7): one declared cap, and the gate within it
    tool_caps = Counter(r[TOOL_CALL_CAP_FIELD] for r in ok_runs)
    if len(tool_caps) > 1:
        raise Refusal(
            f"{len(tool_caps)} {TOOL_CALL_CAP_FIELD} values among the ok runs: {dict(tool_caps)}"
        )
    over_tool = [r for r in ok_runs if r[N_GATE_CHARGED_FIELD] > r[TOOL_CALL_CAP_FIELD]]
    if over_tool:
        r = over_tool[0]
        raise Refusal(
            f"{N_GATE_CHARGED_FIELD} exceeds {TOOL_CALL_CAP_FIELD} on {len(over_tool)} ok runs (e.g. "
            f"{r.get('run_id')}: {r[N_GATE_CHARGED_FIELD]} > {r[TOOL_CALL_CAP_FIELD]})"
        )
    # (a) THE ASKS LEDGER ALONE. The gate no longer charges it, so the residual has no gate term:
    # r = spent - (n_asks - rejected_user_asks); negative refuses, positive is expansion spend.
    residual: dict[int, float] = {}
    for r in ok_runs:
        rejected = _get(r, REJECTED_ASKS_PATH)
        rejected = 0.0 if rejected in (_MISSING, None) else float(rejected)
        residual[id(r)] = float(_get(r, SPENT_RETRIEVAL_PATH)) - (r[GATE_ASKS_FIELD] - rejected)
    negative = [r for r in ok_runs if residual[id(r)] < 0]
    if negative:
        r = negative[0]
        raise Refusal(
            f"(a) the spend residual {'.'.join(SPENT_RETRIEVAL_PATH)} - ({GATE_ASKS_FIELD} - "
            f"rejected_user_asks) is negative on {len(negative)} ok runs (e.g. {r.get('run_id')}: "
            f"r = {residual[id(r)]:g}): the asks ledger holds less than the asks charged"
        )
    # A LEAK: the gate refuses only when the TOOL budget is spent, so a refused call on a unit
    # whose gate charged fewer than tool_call_cap means another budget reached the tool calls.
    leaked = [
        r
        for r in ok_runs
        if r.get(N_REFUSED_BUDGET_FIELD) and r[N_GATE_CHARGED_FIELD] < r[TOOL_CALL_CAP_FIELD]
    ]
    if leaked:
        r = leaked[0]
        raise Refusal(
            f"the split budget leaked on {len(leaked)} ok runs (e.g. {r.get('run_id')}: "
            f"{r.get(N_REFUSED_BUDGET_FIELD)} tool call(s) refused with {N_GATE_CHARGED_FIELD}="
            f"{r[N_GATE_CHARGED_FIELD]} < {TOOL_CALL_CAP_FIELD}={r[TOOL_CALL_CAP_FIELD]})"
        )
    # NO STARVATION REFUSAL (the coordinator's correction after 9e2c761). Under separate budgets
    # the asks reaching their OWN cap is legitimate, with or without a tool call after it; it is
    # reported as a descriptive stopping measure. The leak check above is the refusal.
    no_seq = [r for r in ok_runs if not isinstance(r.get(ENV_CALL_SEQ), list)]
    if no_seq:
        raise Refusal(
            f"{ENV_CALL_SEQ} is absent on {len(no_seq)} ok runs (e.g. {_e_g(no_seq)}): the gate's "
            "charge cannot be recounted. Re-extract with extract_records.py at this commit."
        )
    off_b = []
    for r in ok_runs:
        seq, prefix = r[ENV_CALL_SEQ], int(r[N_PREFIX_ENV_CALLS_FIELD])
        if prefix > len(seq):
            raise Refusal(
                f"{r.get('run_id')}: {N_PREFIX_ENV_CALLS_FIELD}={prefix} exceeds its "
                f"{len(seq)} recorded env calls"
            )
        types = tool_types.get(fork_paired.domain_of(r), {})
        live = seq[prefix:]
        if any(ok is None for _, ok in live):
            raise Refusal(f"{r.get('run_id')}: a live env call carries no ok flag")
        unmapped = sorted({name for name, _ in live if name not in types})
        if unmapped:
            raise Refusal(
                f"tools with no declared type, which would otherwise count as zero: {unmapped}"
            )
        recount = sum(
            1 for name, ok in live if ok is True and types[name] not in UNCHARGED_TOOL_TYPES
        )
        if recount != r[N_GATE_CHARGED_FIELD]:
            off_b.append((r, recount))
    if off_b:
        r, n = off_b[0]
        raise Refusal(
            f"(b) {N_GATE_CHARGED_FIELD} disagrees with the reader's recount of ok, non-GENERIC "
            f"live env calls on {len(off_b)} ok runs (e.g. {r.get('run_id')}: gate "
            f"{r[N_GATE_CHARGED_FIELD]}, recount {n})"
        )
    return residual


def expand_records(values: Sequence[str] | None, name_glob: str, what: str) -> list[str]:
    """Record files from repeatable arguments, each a FILE, a DIRECTORY (searched recursively,
    following symlinks, for `name_glob`) or a GLOB. HPC shards one campaign across LSF jobs, each
    writing its own record under `<root>/<suite>/_jobs/<job>/`. Nothing found refuses."""
    out: list[str] = []
    for v in values or []:
        if any(ch in v for ch in "*?["):
            out += sorted(globmod.glob(v))
        elif Path(v).is_dir():
            out += sorted(
                str(Path(d) / f)
                for d, _dirs, files in os.walk(v, followlinks=True)
                for f in files
                if fnmatch.fnmatch(f, name_glob)
            )
        else:
            out.append(v)
    if values and not out:
        raise Refusal(f"no {what} found at {list(values)}")
    return list(dict.fromkeys(out))


def harness_base_url_sha(url: str) -> str:
    """`ModelPin.base_url_sha` for `url`, computed by the harness's own route: a MeteredClient
    pinned to that base url, then `.pin(role).base_url_sha` -- not a re-implementation of the hash.
    `env={}` so no LITELLM_BASE_URL in this shell can stand in for the url being tested."""
    from pinq.budget import BudgetLedger
    from pinq_adapters.llm.litellm_client import MeteredClient
    from pinq_adapters.llm.pricing import PriceTable

    client = MeteredClient(
        BudgetLedger(0),
        price_table=PriceTable(version="reader", models={}, fallback={}, strict=False),
        models={QUESTIONER_ROLE: "reader-probe"},
        base_url=url,
        env={},
    )
    return client.pin(QUESTIONER_ROLE).base_url_sha


def check_weights(
    args: argparse.Namespace,
    trained: Sequence[tuple[str, str, str]],
    ok_design: Sequence[Mapping],
    t_arm: str,
    emit: Emit,
) -> dict[str, Any]:
    """Each trained seed's weights: the manifest's adapter_sha when it records one, else EVERY
    serve record, which must list the model, agree on its sha256 and match the expected prefix."""
    serves: list[tuple[str, str, Mapping]] = []
    weights: dict[str, Any] = {}
    for label, model, prefix in trained:
        runs = [
            r
            for r in ok_design
            if (r.get("key") or [None] * 3)[KEY_ARM] == t_arm and _questioner_model(r) == model
        ]
        shas = Counter(
            str(_get(_manifest(r), (PINS_FIELD, QUESTIONER_ROLE, PIN_ADAPTER_SHA)) or "")
            for r in runs
        )
        recorded = {x for x in shas if x}
        if recorded:
            if any(not x.startswith(prefix) for x in recorded) or "" in shas:
                raise Refusal(
                    f"{label} ({model}) runs record adapter_sha {sorted(shas)} in their "
                    f"manifests; expected every one to start {prefix}"
                )
            weights[label] = {
                "model": model,
                "source": "manifest adapter_sha",
                "sha256": sorted(recorded),
            }
            continue
        if not runs:
            continue
        if not serves:
            paths = expand_records(args.serve_record, SERVE_GLOB, "serve record")
            if not paths:
                raise Refusal(
                    f"{label} ({model}) runs record no adapter_sha in their manifests and no "
                    "serve record was given (--serve-record): their weights are unverified"
                )
            for path in paths:
                doc, sha = _load_json(path, "serve record")
                if not isinstance(doc, Mapping) or doc.get("schema") != SERVE_SCHEMA:
                    raise Refusal(
                        f"serve record {path} is not schema {SERVE_SCHEMA!r}; its field names "
                        "cannot be assumed"
                    )
                serves.append((path, sha, doc))
        got = {}
        for path, _, doc in serves:
            x = str(((doc.get(SERVE_ADAPTERS) or {}).get(model) or {}).get(SERVE_SHA) or "")
            if not x:
                raise Refusal(
                    f"trained model {model} ({label}) has no {SERVE_ADAPTERS}.{SERVE_SHA} entry "
                    f"in the serve record {path}"
                )
            got[path] = x
        if len(set(got.values())) > 1:
            raise Refusal(
                f"the serve records disagree on the sha256 of {model} ({label}): "
                + ", ".join(f"{Path(k).name}: {v[:16]}" for k, v in got.items())
            )
        (sha,) = set(got.values())
        if not sha.startswith(prefix):
            raise Refusal(
                f"the serve records map {model} ({label}) to sha256 {sha[:16]}..., expected "
                f"prefix {prefix}"
            )
        weights[label] = {"model": model, "source": "serve records", "sha256": sha}
    for label, w in weights.items():
        emit(f"  weights {label}: {w['model']} -> {str(w['sha256'])[:24]} ({w['source']})")
    return {
        "weights": weights,
        "serve_records": [
            {
                "path": path,
                "file_sha256": sha,
                **{f: doc.get(f) for f in SERVE_FIELDS_KEPT},
                "adapters": {
                    m: (a or {}).get(SERVE_SHA) for m, a in (doc.get(SERVE_ADAPTERS) or {}).items()
                },
            }
            for path, sha, doc in serves
        ],
    }


def _canon(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _params(e: Mapping) -> Mapping:
    lp = e.get("litellm_params")
    return lp if isinstance(lp, Mapping) else {}


def _route(e: Mapping) -> str:
    return str(_params(e).get("model") or "")


def _local_key(e: Mapping) -> str:
    """Check (b)'s comparison form: model_name plus the four litellm_params fields."""
    lp = _params(e)
    return _canon({"model_name": e.get("model_name"), **{k: lp.get(k) for k in LOCAL_ENTRY_FIELDS}})


def _without_api_key(e: Mapping) -> str:
    return _canon({**e, "litellm_params": {k: v for k, v in _params(e).items() if k != "api_key"}})


def _name(e: Mapping) -> str:
    return f"{e.get('model_name')} ({_route(e)})"


def config_agreement(docs: Sequence[tuple[str, str, Mapping]], emit: Emit) -> dict[str, Any]:
    """RULES amendment 10. Several config_sha256 values are one population only if every probe
    record's archived config (a) hashes to that record's config_sha256, (b) carries hosted_vllm
    entries byte-identical to every other config's, (c) differs from the others only in gateway
    entries' api_key, each an os.environ/<NAME>, and (d) agrees on every section but model_list.
    Prints the distinct shas, the jobs on each and every check's verdict; refuses otherwise. An api
    key's value is never printed."""
    jobs = Counter(str(d[PROBE_CONFIG_SHA]) for _, _, d in docs)
    emit(
        f"  probe records: {len(docs)} on {len(jobs)} config_sha256 values -- RULES amendment 10 "
        "requires their archived configs to agree:"
    )
    for sha, n in sorted(jobs.items(), key=lambda kv: (-kv[1], kv[0])):
        emit(f"    config_sha256 {sha[:16]}: {n} job{'' if n == 1 else 's'}")

    # (a) every archive present, named by its record's config_path, hashing to its record's sha
    archives: list[dict[str, Any]] = []
    configs: dict[str, Mapping] = {}
    bad_a: list[str] = []
    for path, _, d in docs:
        want = str(d[PROBE_CONFIG_SHA])
        named = str(d.get(PROBE_CONFIG_PATH) or "").replace("\\", "/").rsplit("/", 1)[-1]
        arch = Path(path).parent / named if named else None
        row: dict[str, Any] = {
            "probe_record": path,
            "archive": None if arch is None else str(arch),
            "config_sha256": want,
            "archive_sha256": None,
        }
        archives.append(row)
        if arch is None:
            bad_a.append(f"{path} names no {PROBE_CONFIG_PATH}")
            continue
        if not arch.is_file():
            bad_a.append(f"no archived config at {arch}")
            continue
        text = arch.read_text()
        # thinking_probe.py's own definition: sha256 of read_text().encode()
        row["archive_sha256"] = got = hashlib.sha256(text.encode()).hexdigest()
        if got != want:
            bad_a.append(f"{arch} hashes to {got[:12]}, its probe record says {want[:12]}")
            continue
        try:
            cfg = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            bad_a.append(f"{arch} is not YAML ({type(exc).__name__})")
            continue
        entries = cfg.get("model_list") if isinstance(cfg, Mapping) else None
        if not isinstance(entries, list) or not all(isinstance(e, Mapping) for e in entries):
            bad_a.append(f"{arch} has no model_list of entries")
            continue
        configs.setdefault(want, cfg)
    checks: dict[str, dict[str, Any]] = {
        "a": {
            "verdict": "FAIL" if bad_a else "PASS",
            "detail": "; ".join(bad_a[:3])
            + (f" (+{len(bad_a) - 3} more)" if len(bad_a) > 3 else "")
            if bad_a
            else f"{len(archives)} of {len(archives)} archives",
        }
    }
    unverified = sorted(sha[:12] for sha in jobs if sha not in configs)
    if unverified:
        for k in "bcd":
            checks[k] = {
                "verdict": "NOT EVALUATED",
                "detail": f"config_sha256 {unverified} has no verified archive",
            }
    else:
        order = sorted(configs, key=lambda sha: (-jobs[sha], sha))
        lists = {sha: configs[sha]["model_list"] for sha in order}

        # (b) hosted_vllm entries, compared as canonical JSON of the five fields, as multisets
        local = {
            sha: Counter(_local_key(e) for e in es if _route(e).startswith(LOCAL_ROUTE_PREFIX))
            for sha, es in lists.items()
        }
        ref = local[order[0]]
        diff_b = sorted(
            f"{sha[:12]} vs {order[0][:12]}: "
            + ", ".join(sorted({json.loads(k)["model_name"] for k in (ref - c) + (c - ref)}))
            for sha, c in local.items()
            if c != ref
        )
        n_local = sum(ref.values())
        checks["b"] = {
            "verdict": "PASS" if n_local and not diff_b else "FAIL",
            "detail": "; ".join(diff_b)
            if diff_b
            else (f"{n_local} entries" if n_local else "no hosted_vllm entry in any config"),
            "n_entries": n_local,
        }

        # (c) what differs is a gateway entry, only in api_key, and that key is an env name
        full = {sha: Counter(_canon(e) for e in es) for sha, es in lists.items()}
        common = full[order[0]]
        for c in full.values():
            common = common & c
        stripped = {sha: {_without_api_key(e) for e in es} for sha, es in lists.items()}
        bad_c: list[str] = []
        n_diff = 0
        for sha, es in lists.items():
            left = Counter(common)
            for e in es:
                key = _canon(e)
                if left[key]:
                    left[key] -= 1
                    continue
                n_diff += 1
                where = f"{_name(e)} in {sha[:12]}"
                if not _route(e).startswith(GATEWAY_ROUTE_PREFIX):
                    bad_c.append(f"{where} differs and is not a gateway entry")
                elif not ENV_KEY.fullmatch(str(_params(e).get("api_key") or "")):
                    bad_c.append(f"{where}: api_key is not of the form os.environ/<NAME>")
                elif any(_without_api_key(e) not in st for st in stripped.values()):
                    bad_c.append(f"{where} differs from the other configs beyond api_key")
        checks["c"] = {
            "verdict": "FAIL" if bad_c else "PASS",
            "detail": "; ".join(sorted(set(bad_c))[:3])
            + (f" (+{len(set(bad_c)) - 3} more)" if len(set(bad_c)) > 3 else "")
            if bad_c
            else f"{n_diff} entries differ",
            "n_entries_differing": n_diff,
        }

        # (d) every section but model_list, litellm_settings and general_settings included
        sections = sorted(
            {k for cfg in configs.values() for k in cfg} - {"model_list"}
            | {"litellm_settings", "general_settings"}
        )
        diff_d = [k for k in sections if len({_canon(configs[sha].get(k)) for sha in order}) != 1]
        checks["d"] = {
            "verdict": "FAIL" if diff_d else "PASS",
            "detail": f"differ: {diff_d}" if diff_d else f"{len(sections)} sections identical",
            "sections": sections,
        }
    for k in "abcd":
        emit(f"    ({k}) {CONFIG_CHECKS[k]}: {checks[k]['verdict']} -- {checks[k]['detail']}")
    failed = [k for k in "abcd" if checks[k]["verdict"] != "PASS"]
    if failed:
        raise Refusal(
            f"the probe records carry {len(jobs)} config_sha256 values and their archived configs "
            "do not agree as RULES amendment 10 requires: "
            + "; ".join(f"({k}) {checks[k]['verdict']}" for k in failed)
        )
    return {
        "rule": "RULES amendment 10",
        "jobs_per_config_sha256": dict(jobs),
        "archives": archives,
        "checks": checks,
    }


def check_probes(
    args: argparse.Namespace,
    trained: Sequence[tuple[str, str, str]],
    records: Sequence[Mapping],
    emit: Emit,
    reference: str | None = None,
) -> dict[str, Any]:
    """The thinking-probe records: ALL PASS, ALL on one proxy_url, their routes' union covering the
    comparator and the trained models, no route with thinking, and that proxy_url -- hashed by the
    harness -- equal to every run's questioner base_url_sha. One config_sha256, or several whose
    archived configs agree as RULES amendment 10 requires (`config_agreement`).
    chat_template_kwargs sits outside run identity, so these records are the only evidence that
    thinking was off."""
    if not args.probe_record:
        raise Refusal(
            "no thinking-probe record (--probe-record): the proxy's chat_template_kwargs sits "
            "outside run identity, so nothing else shows thinking was off"
        )
    paths = expand_records(args.probe_record, PROBE_GLOB, "probe record")
    docs = []
    for path in paths:
        doc, sha = _load_json(path, "probe record")
        verdict = doc.get(PROBE_VERDICT) if isinstance(doc, Mapping) else None
        if verdict != PROBE_PASS:
            raise Refusal(f"probe record {path} verdict is {verdict!r}, not {PROBE_PASS!r}")
        routes = doc.get(PROBE_ROUTES) or []
        if not isinstance(routes, list) or not all(isinstance(rt, Mapping) for rt in routes):
            raise Refusal(f"probe record {path} {PROBE_ROUTES!r} is malformed: expected objects")
        thinking = [
            str(rt.get(PROBE_ROUTE_MODEL))
            for rt in routes
            if rt.get(PROBE_ROUTE_THINKING) is not False
        ]
        if thinking:
            raise Refusal(
                f"probe record {path} says {PROBE_PASS} but {PROBE_ROUTE_THINKING} is not false "
                f"on routes {thinking}"
            )
        docs.append((path, sha, doc))
    vals = {str(d.get(PROBE_PROXY_URL) or "") for _, _, d in docs}
    if len(vals) != 1 or "" in vals:
        raise Refusal(
            f"the probe records do not share one {PROBE_PROXY_URL} ({sorted(vals)}): one campaign "
            "is one proxy"
        )
    no_sha = [path for path, _, d in docs if not d.get(PROBE_CONFIG_SHA)]
    if no_sha:
        raise Refusal(f"probe records {no_sha} carry no {PROBE_CONFIG_SHA}")
    n_shas = len({str(d[PROBE_CONFIG_SHA]) for _, _, d in docs})
    agreement = config_agreement(docs, emit) if n_shas > 1 else None
    # RULES amendment 11: coverage is required of the LOCAL (hosted_vllm) pins only; a gateway
    # pin is not thinking-probed and has a questioner probe record instead
    pins = {args.comparator_model, *(m for _, m, _ in trained), *([reference] if reference else [])}
    required = {m for m in pins if not is_gateway_pin(m)}
    gateway = sorted(pins - required)
    covered = {str(rt.get(PROBE_ROUTE_MODEL)) for _, _, d in docs for rt in d[PROBE_ROUTES]}
    uncovered = sorted(required - covered)
    if uncovered:
        raise Refusal(f"the probe records do not cover {uncovered}, which the runs require")
    proxy = str(docs[0][2][PROBE_PROXY_URL])
    proxy_sha = harness_base_url_sha(proxy)
    q = [
        r
        for r in records
        if _get(_manifest(r), (PINS_FIELD, QUESTIONER_ROLE, PIN_BASE_URL_SHA)) != proxy_sha
    ]
    if q:
        other = _get(_manifest(q[0]), (PINS_FIELD, QUESTIONER_ROLE, PIN_BASE_URL_SHA))
        raise Refusal(
            f"the probe's proxy_url hashes to {proxy_sha[:12]} (MeteredClient.pin), not the "
            f"questioner base_url_sha {str(other)[:12]} of {len(q)} runs: the probe proved a proxy "
            "the units did not use"
        )
    configs = (
        f"one config_sha256 {str(docs[0][2][PROBE_CONFIG_SHA])[:12]}"
        if agreement is None
        else f"{n_shas} config_sha256 values agreeing under RULES amendment 10"
    )
    emit(
        f"  probe records: {len(docs)} PASS, {configs}, one proxy_url, covering {sorted(required)}; "
        f"its harness hash equals the questioner base_url_sha of all {len(records)} runs"
    )
    if gateway:
        emit(
            f"  gateway pin(s) not thinking-probed: {gateway} -- the thinking probe probes "
            "hosted_vllm routes; each has a questioner probe record instead (amendment 11)"
        )
    return {
        "config_agreement": agreement,
        "gateway_pins_not_thinking_probed": gateway,
        "probe_records": [
            {
                "path": path,
                "file_sha256": sha,
                "verdict": d.get(PROBE_VERDICT),
                "proxy_url": d.get(PROBE_PROXY_URL),
                "config_sha256": d.get(PROBE_CONFIG_SHA),
                "routes": [
                    {
                        "model": rt.get(PROBE_ROUTE_MODEL),
                        "chat_template_kwargs": rt.get(PROBE_ROUTE_KWARGS),
                        "completion_tokens": rt.get("completion_tokens"),
                        "thinking_detected": rt.get(PROBE_ROUTE_THINKING),
                    }
                    for rt in d[PROBE_ROUTES]
                ],
            }
            for path, sha, d in docs
        ],
        "proxy_url_hash_equals_every_questioner_base_url_sha": True,
    }


def check_questioner_probe(
    pins: Sequence[str],
    records: Sequence[Mapping],
    emit: Emit,
    *,
    paths: Sequence[str],
    job_dirs: Sequence[Path],
    code_version: str,
    campaign_sha: str,
) -> dict[str, Any]:
    """RULES amendment 11: a gateway questioner pin is not thinking-probed; the driver's
    questioner-probe record stands in, ONE PER JOB -- every job directory the driver staged must
    hold its own, and `paths` may add more. Each must be a PASS for that pin, at a positive
    max_tokens with its own completion below the cap and non-empty content, on the campaign's
    harness sha, through a proxy_url hashing to that pin's questioner base_url_sha. Raises Refusal.
    Returns each pin's max_tokens, which the token-cap share is measured against."""
    if not pins:
        return {}
    lacking = [d for d in job_dirs if not (d / JOB_QPROBE).is_file()]
    if lacking:
        d = lacking[0]
        raise Refusal(
            f"job {d.parent.parent.name}/{JOBS_DIR}/{d.name} holds no {JOB_QPROBE} "
            f"({len(lacking)} job(s)): every job of a gateway questioner runs under its own"
        )
    found = [*paths, *(str(d / JOB_QPROBE) for d in job_dirs)]
    found = list({str(Path(x).resolve()): x for x in found}.values())
    if not found:
        raise Refusal(
            f"no questioner probe record (--questioner-probe-record, or a job of the comparator "
            f"store) for the gateway pin(s) {list(pins)}: the thinking probe does not probe a "
            "gateway route, and a reasoning questioner at its token cap returns empty content "
            "(amendment 11)"
        )
    by_pin: dict[str, list[dict]] = defaultdict(list)
    # THE WRAPPER'S COPY AND THE JOB'S COPY ARE ONE RECORD: the HPC wrapper writes
    # $ARM/_wrapper/<LSB_JOBID>/questioner_probe.<suite>.k<k>.json and the driver's _stage_job
    # copies the same bytes into the job directory. Counted by content, never by path.
    seen: dict[str, str] = {}
    copies = 0
    for path in found:
        doc, sha = _load_json(path, "questioner probe record")
        if sha in seen:
            copies += 1
            continue
        seen[sha] = path
        if not isinstance(doc, Mapping):
            raise Refusal(f"questioner probe record {path} is not a JSON object")
        verdict = doc.get(PROBE_VERDICT)
        if verdict != PROBE_PASS:
            raise Refusal(
                f"questioner probe record {path} verdict is {verdict!r}, not {PROBE_PASS!r}"
            )
        model = doc.get(QPROBE_MODEL)
        if model not in pins:
            raise Refusal(
                f"questioner probe record {path} is for {model!r}, not a gateway pin the runs "
                f"use ({list(pins)})"
            )
        mt = doc.get(QPROBE_MAX_TOKENS)
        if not isinstance(mt, int) or isinstance(mt, bool) or mt <= 0:
            raise Refusal(
                f"questioner probe record {path} carries no positive integer {QPROBE_MAX_TOKENS} "
                f"({mt!r}): the share of calls at the token cap has no cap to be measured against"
            )
        ct = doc.get(QPROBE_COMPLETION)
        if not isinstance(ct, int) or isinstance(ct, bool):
            raise Refusal(f"questioner probe record {path} carries no {QPROBE_COMPLETION}")
        if ct >= TOKEN_CAP_FRACTION * mt:
            raise Refusal(
                f"questioner probe record {path} says {PROBE_PASS} but its {QPROBE_COMPLETION} "
                f"{ct} is at the cap ({float(TOKEN_CAP_FRACTION):g} x {mt})"
            )
        chars = doc.get(QPROBE_CONTENT_CHARS)
        if chars is not None and not chars > 0:
            raise Refusal(f"questioner probe record {path} says {PROBE_PASS} on empty content")
        harness = doc.get(QPROBE_HARNESS)
        if harness != campaign_sha or harness != code_version:
            raise Refusal(
                f"questioner probe record {path}: {QPROBE_HARNESS} {harness!r} is not the "
                f"campaign sha {campaign_sha!r} and the runs' code_version {code_version!r}"
            )
        proxy = doc.get(PROBE_PROXY_URL)
        if not proxy:
            raise Refusal(f"questioner probe record {path} carries no {PROBE_PROXY_URL}")
        want = harness_base_url_sha(str(proxy))
        off = [
            r
            for r in records
            if _questioner_model(r) == model
            and _get(_manifest(r), (PINS_FIELD, QUESTIONER_ROLE, PIN_BASE_URL_SHA)) != want
        ]
        if off:
            raise Refusal(
                f"questioner probe record {path}: its {PROBE_PROXY_URL} hashes to {want[:12]}, not "
                f"the questioner base_url_sha of {len(off)} {model} runs"
            )
        by_pin[str(model)].append(
            {
                "path": path,
                "file_sha256": sha,
                "max_tokens": mt,
                "completion_tokens": ct,
                "reasoning_tokens": doc.get(QPROBE_REASONING),
                "harness_sha": harness,
                "proxy_url": proxy,
                "driver_sha256": doc.get("driver_sha256"),
                "suite": (doc.get("unit") or {}).get("suite"),
                **{f: doc.get(f) for f in LSB_FIELDS},
            }
        )
    missing = [pin for pin in pins if pin not in by_pin]
    if missing:
        raise Refusal(f"no questioner probe record for {missing} among {len(found)} record(s)")
    max_tokens: dict[str, int] = {}
    for pin, docs in sorted(by_pin.items()):
        mts = {d["max_tokens"] for d in docs}
        if len(mts) != 1:
            raise Refusal(
                f"the questioner probe records for {pin} disagree on {QPROBE_MAX_TOKENS} "
                f"({sorted(mts)})"
            )
        max_tokens[pin] = mts.pop()
        jobs = ", ".join(f"{d['suite']} {d['LSB_JOBID']}[{d['LSB_JOBINDEX']}]" for d in docs)
        emit(
            f"  questioner probe records for {pin}: {len(docs)} PASS (jobs {jobs}) at "
            f"{QPROBE_MAX_TOKENS} {max_tokens[pin]}, {QPROBE_HARNESS} {campaign_sha[:12]}; "
            f"{QPROBE_COMPLETION} {[d['completion_tokens'] for d in docs]}, {QPROBE_REASONING} "
            f"{[d['reasoning_tokens'] for d in docs]}; proxy_url hashes to the questioner "
            f"base_url_sha of its runs ({copies} byte-identical copies read once)"
        )
    return {
        "questioner_probe": {
            "records": dict(by_pin),
            "max_tokens": max_tokens,
            "n_byte_identical_copies": copies,
        }
    }


def load_arm_launch(
    paths: Sequence[str] | None,
    *,
    records: Sequence[Mapping],
    suites: Sequence[str],
    pin: str,
    c_arm: str,
    code_version: str,
    campaign_sha: str,
    legacy: bool,
    disabled: list[str],
    emit: Emit,
) -> tuple[dict[tuple, dict[tuple, str]], list[dict]]:
    """The comparator store's plan, from the driver's own ARM_LAUNCH.json (one per suite root; a
    file, a directory or a glob): the pin, arm, store directory and regime the driver declares, its
    harness at the campaign sha and the runs' code_version, one driver_sha256 across suites, and
    every comparator run inside its plan. Raises Refusal. Returns (planned slots, provenance)."""
    drv = extra_pin_driver()
    found = expand_records(paths, ARM_LAUNCH_NAME, "comparator launch record") if paths else []
    docs: dict[str, tuple[str, str, Mapping]] = {}
    want = {"arm_id": c_arm, "store_dir": comparator_store_dir(pin), "regime": drv.GATEWAY_ONLY}
    for path in found:
        doc, sha = _load_json(path, "comparator launch record")
        if not isinstance(doc, Mapping):
            raise Refusal(f"comparator launch record {path} is not a JSON object")
        suite = str(doc.get("suite"))
        if suite not in suites:
            raise Refusal(f"comparator launch record {path} is for {suite!r}, not a declared suite")
        if suite in docs:
            raise Refusal(f"two comparator launch records for {suite}: {docs[suite][0]}, {path}")
        if doc.get("pin") != pin:
            raise Refusal(
                f"comparator launch record {path} is for pin {doc.get('pin')!r}, not {pin!r}"
            )
        for f, w in want.items():
            if doc.get(f) != w:
                raise Refusal(f"comparator launch record {path}: {f} {doc.get(f)!r}, not {w!r}")
        h = doc.get("harness_sha")
        if h != campaign_sha:
            raise Refusal(
                f"comparator launch record {path}: harness_sha {h!r} is not the campaign sha "
                f"{campaign_sha!r} (RULES amendment 11: the exact staged harness)"
            )
        if h != code_version:
            raise Refusal(
                f"comparator launch record {path}: harness_sha {h!r} is not the runs' "
                f"code_version {code_version!r}"
            )
        plan = doc.get("plan")
        if not isinstance(plan, list) or not plan or not all(isinstance(u, Mapping) for u in plan):
            raise Refusal(f"comparator launch record {path} holds no plan of units")
        off = [
            u
            for u in plan
            if (u.get("suite"), u.get("pin"), u.get("arm_id")) != (suite, pin, c_arm)
        ]
        if off:
            raise Refusal(
                f"comparator launch record {path} plans {len(off)} unit(s) outside "
                f"{suite}/{pin}/{c_arm}, e.g. {dict(off[0])}"
            )
        docs[suite] = (path, sha, doc)
    missing = [s for s in suites if s not in docs]
    if missing and not legacy:
        raise Refusal(
            f"no comparator launch record for declared suite(s) {missing} "
            f"(--comparator-launch-record: the driver's {ARM_LAUNCH_NAME} per suite root): a "
            "planned unit that never wrote a status would be invisible"
        )
    if missing:
        disabled.append(f"comparator launch record for {missing}: planned-but-absent slots")
    drivers = {str(d.get("driver_sha256")) for _, _, d in docs.values()}
    if len(drivers) > 1:
        raise Refusal(
            f"the comparator launch records name {len(drivers)} driver_sha256 values "
            f"({sorted(x[:16] for x in drivers)}): one arm is one driver"
        )
    mine = drv.driver_sha256()
    planned: dict[tuple, dict[tuple, str]] = defaultdict(dict)
    plan_keys = set()
    prov = []
    for suite in suites:
        if suite not in docs:
            continue
        path, sha, doc = docs[suite]
        for u in doc["plan"]:
            slot = (u["trace_sha"], u["k"], u["seed"])
            planned[(suite, u["variant"], pin)][slot] = str(u["task_id"])
            plan_keys.add((suite, u["variant"]) + slot)
        ds = str(doc.get("driver_sha256"))
        prov.append(
            {
                "suite": suite,
                "path": path,
                "file_sha256": sha,
                "driver_sha256": ds,
                "reader_driver_sha256": mine,
                "harness_sha": doc["harness_sha"],
                "regime": doc["regime"],
                "pin": pin,
                "arm_id": c_arm,
                "store_dir": doc["store_dir"],
                "proxy_url": doc.get("proxy_url"),
                "pilot": doc.get("pilot"),
                "rerun_py_sha256": doc.get("rerun_py_sha256"),
                "n_units": len(doc["plan"]),
            }
        )
        emit(
            f"  comparator launch record {suite} ({ARM_LAUNCH_NAME}): driver_sha256 {ds[:16]} "
            + (
                "(= the reader's copy of the driver)"
                if ds == mine
                else f"(reader's copy {mine[:16]})"
            )
            + f", harness_sha {doc['harness_sha'][:12]}, regime {doc['regime']!r}, "
            f"{len(doc['plan'])} planned units, pilot {doc.get('pilot')}, proxy_url "
            f"{doc.get('proxy_url')}"
        )
    unplanned = [
        r
        for r in records
        if _suite(r) in docs and (_suite(r), _variant(r)) + _slot(r) not in plan_keys
    ]
    if unplanned:
        raise Refusal(
            f"{len(unplanned)} comparator runs are not in the arm's launch plan (e.g. "
            f"{_e_g(unplanned)}): a run the launch did not plan belongs to another population"
        )
    return dict(planned), prov


def check_comparator_jobs(
    job_dirs: Sequence[Path],
    *,
    launch_by_suite: Mapping[str, Mapping[str, Any]],
    pin: str,
    campaign_sha: str,
    emit: Emit,
) -> dict[str, Any]:
    """Every job the driver staged ran as its suite's ARM_LAUNCH.json says -- JOB.json's regime,
    driver_sha256, harness_sha and pin -- under a questioner probe written by that same LSF job
    (LSB_JOBID/LSB_JOBINDEX, normalised by the driver's own `_idx`); every launched suite has a
    job. Raises Refusal."""
    drv = extra_pin_driver()
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for d in job_dirs:
        suite = d.parent.parent.name
        where = f"job {suite}/{JOBS_DIR}/{d.name}"
        job, _ = _load_json(str(d / JOB_RECORD), "job record")
        q, _ = _load_json(str(d / JOB_QPROBE), "questioner probe record")
        if not isinstance(job, Mapping) or not isinstance(q, Mapping):
            raise Refusal(f"{where}: {JOB_RECORD} or {JOB_QPROBE} is not a JSON object")
        launch = launch_by_suite.get(suite)
        if launch is None:
            raise Refusal(f"{where}: no comparator launch record for {suite}")
        want = {
            "regime": drv.GATEWAY_ONLY,
            "driver_sha256": launch["driver_sha256"],
            "harness_sha": campaign_sha,
            "pin": pin,
        }
        for f, w in want.items():
            if job.get(f) != w:
                raise Refusal(
                    f"{where}: {JOB_RECORD} {f} {job.get(f)!r}, not {w!r} (its {ARM_LAUNCH_NAME})"
                )
        jid = (str(job.get(LSB_FIELDS[0]) or ""), drv._idx(job.get(LSB_FIELDS[1])))
        qid = (str(q.get(LSB_FIELDS[0]) or ""), drv._idx(q.get(LSB_FIELDS[1])))
        if jid != qid:
            raise Refusal(
                f"{where}: {JOB_RECORD} {'/'.join(LSB_FIELDS)} {jid} is not its questioner probe "
                f"record's {qid}: a record from another job"
            )
        by_suite[suite].append(
            {
                "label": d.name,
                "LSB_JOBID": jid[0],
                "LSB_JOBINDEX": jid[1],
                "shard": job.get("shard"),
                "n_units": job.get("n_units"),
            }
        )
    empty = [s for s in launch_by_suite if not by_suite.get(s)]
    if empty:
        raise Refusal(
            f"no job record ({JOB_RECORD} under {JOBS_DIR}/) for comparator suite(s) {empty}: a "
            "job's own questioner probe is what shows its questioner was not truncated"
        )
    for suite, js in by_suite.items():
        emit(
            f"  comparator jobs {suite}: {len(js)} ({', '.join(j['label'] for j in js)}), each "
            f"under its own PASS questioner probe; {JOB_RECORD} regime, driver_sha256 and "
            f"harness_sha agree with {ARM_LAUNCH_NAME}"
        )
    return {"comparator_jobs": dict(by_suite)}


def token_cap_block(
    runs: Sequence[Mapping], *, pin: str, max_tokens: int, suites: Sequence[str]
) -> dict[str, Any]:
    """RULES amendment 11: the share of the arm's questioner calls AT THE TOKEN CAP, from calls.jsonl
    (role inquirer, model == the pin, completed calls only). Raises Refusal where the share could
    only read absence: a run with no calls.jsonl, a questioner call off the pin, or no call."""
    no_calls = [r for r in runs if r.get(INQUIRER_CALLS_FIELD) is None]
    if no_calls:
        raise Refusal(
            f"{len(no_calls)} ok {pin} runs have no calls.jsonl (e.g. {_e_g(no_calls)}): the share "
            "of questioner calls at the token cap cannot be read"
        )
    off = [r for r in runs if any(c[0] != pin for c in r[INQUIRER_CALLS_FIELD])]
    if off:
        models = sorted({str(c[0]) for r in off for c in r[INQUIRER_CALLS_FIELD] if c[0] != pin})
        raise Refusal(
            f"questioner calls on {len(off)} {pin} runs name a model other than the pin ({models}; "
            f"e.g. {_e_g(off)}): filtering them out would hide a misroute"
        )
    cap = TOKEN_CAP_FRACTION * max_tokens

    def at_cap(c: Sequence) -> bool:
        return c[2] == 200 and (c[1] or 0) >= cap

    def tally(rs: Sequence[Mapping]) -> dict[str, Any]:
        calls = [c for r in rs for c in r[INQUIRER_CALLS_FIELD]]
        done = [c for c in calls if c[2] == 200]
        n_at = sum(1 for c in done if at_cap(c))
        return {
            "n_units": len(rs),
            "n_units_with_any_at_cap": sum(
                1 for r in rs if any(at_cap(c) for c in r[INQUIRER_CALLS_FIELD])
            ),
            "n_calls": len(done),
            "n_at_cap": n_at,
            "n_not_200_excluded": len(calls) - len(done),
            "share": n_at / len(done) if done else None,
        }

    total = tally(runs)
    if not total["n_calls"]:
        raise Refusal(
            f"no completed questioner call of {pin} is recorded in {len(runs)} ok runs: a zero "
            "here is absence, not a share"
        )
    return {
        "pin": pin,
        "max_tokens": max_tokens,
        "fraction": float(TOKEN_CAP_FRACTION),
        "threshold_tokens": float(cap),
        "limitation_above": TOKEN_CAP_LIMITATION,
        **total,
        "limitation": total["share"] > TOKEN_CAP_LIMITATION,
        "per_suite": {suite: tally([r for r in runs if _suite(r) == suite]) for suite in suites},
    }


def _print_token_cap(emit: Emit, tc: Mapping[str, Any]) -> None:
    emit("")
    emit(
        f"  QUESTIONER TOKEN CAP (amendment 11), prompted {short_model(tc['pin'])}: questioner calls "
        f"(role inquirer, model {tc['pin']}) with completion_tokens >= {tc['fraction']:g} x "
        f"max_tokens {tc['max_tokens']} = {tc['threshold_tokens']:g}; max_tokens from the questioner "
        "probe record -- descriptive"
    )
    for suite, t in tc["per_suite"].items():
        share = "n/a" if t["share"] is None else f"{t['share']:.2%}"
        emit(
            f"    {suite}: {t['n_at_cap']} of {t['n_calls']} calls ({share}) at the cap, on "
            f"{t['n_units_with_any_at_cap']} of {t['n_units']} units; {t['n_not_200_excluded']} "
            "non-200 call(s) excluded"
        )
    lim = f"{tc['limitation_above']:.0%}"
    emit(
        f"    population: {tc['n_at_cap']} of {tc['n_calls']} calls ({tc['share']:.2%}) -- "
        + (f"{TOKEN_CAP_LIMITATION_TEXT}: above {lim}" if tc["limitation"] else f"not above {lim}")
    )


def _reference_pairs(
    in_design: Sequence[Mapping],
    suite: str,
    variant: str,
    reference: str,
    comparator: str,
    c_arm: str,
) -> tuple[dict[tuple, dict[str, Mapping]], int, int, dict[tuple, str]]:
    """The prompted reference against the prompted comparator at one (suite, variant). Both arms
    share an arm_id, so copies are relabelled for `_pair_forks`. Returns (pairs keyed by arm
    REFERENCE_ARM / "comparator", unpaired, non-fork, task of each pair)."""
    here = [
        r
        for r in in_design
        if _suite(r) == suite
        and _variant(r) == variant
        and r.get("status") == "ok"
        and r.get("arm_id") == c_arm
    ]
    ref = [{**r, "arm_id": REFERENCE_ARM} for r in here if _questioner_model(r) == reference]
    cmp_ = [{**r, "arm_id": "comparator"} for r in here if _questioner_model(r) == comparator]
    try:
        pairs, n_unpaired, n_not_forks = _pair_forks(
            ref + cmp_, treatment=REFERENCE_ARM, control="comparator"
        )
    except ValueError as exc:
        raise Refusal(f"{suite}/{variant}/{REFERENCE_ARM}: {exc}") from exc
    mixed = [
        k for k, v in pairs.items() if v[REFERENCE_ARM]["task_id"] != v["comparator"]["task_id"]
    ]
    if mixed:
        raise Refusal(
            f"{suite}/{variant}/{REFERENCE_ARM}: {len(mixed)} pairs join two task_ids, e.g. "
            f"{mixed[0]}"
        )
    return (
        pairs,
        n_unpaired,
        n_not_forks,
        {k: str(v["comparator"]["task_id"]) for k, v in pairs.items()},
    )


def reference_contrast(
    in_design: Sequence[Mapping],
    *,
    suites: Sequence[str],
    variants: Sequence[str],
    reference: str,
    comparator: str,
    c_arm: str,
    printed: tuple[int, int],
) -> dict[str, Any]:
    """RULES amendment 11 point 2: the prompted 8B base against the prompted comparator, per suite
    and variant -- DESCRIPTIVE, NO CLAIM: task-level mean, printed interval and split only, in no
    Holm family. Both arms share an arm_id, so copies are relabelled for `_pair_forks`."""
    cells = []
    for suite in suites:
        for variant in variants:
            pairs, n_unpaired, n_not_forks, task_of = _reference_pairs(
                in_design, suite, variant, reference, comparator, c_arm
            )
            endpoints: dict[str, Any] = {}
            for name, fn in (
                (PRIMARY_ENDPOINT, lambda r: float(follow_ups(r))),
                (GUARD_ENDPOINT, lambda r: float(fork_paired.succeeded(r))),
            ):
                by_task: dict[str, list[tuple[float, float]]] = defaultdict(list)
                for k, v in pairs.items():
                    by_task[task_of[k]].append((fn(v[REFERENCE_ARM]), fn(v["comparator"])))
                if not by_task:
                    endpoints[name] = None
                    continue
                e = diff_stats(
                    {t: statistics.fmean(a - b for a, b in xs) for t, xs in by_task.items()},
                    printed,
                )
                e["reference_level"] = statistics.fmean(
                    statistics.fmean(a for a, _ in xs) for xs in by_task.values()
                )
                e["comparator_level"] = statistics.fmean(
                    statistics.fmean(b for _, b in xs) for xs in by_task.values()
                )
                endpoints[name] = e
            cells.append(
                {
                    "suite": suite,
                    "variant": variant,
                    "n_pairs": len(pairs),
                    "n_unpaired": n_unpaired,
                    "n_not_forks": n_not_forks,
                    "endpoints": endpoints,
                }
            )
    return {
        "label": REFERENCE_LABEL,
        "reference_model": reference,
        "comparator_model": comparator,
        "delta": f"{reference} minus {comparator}",
        "unit": "task",
        "cells": cells,
    }


def exploratory_ask_quality(
    aq_cells: Sequence[Mapping[str, Any]],
    in_design: Sequence[Mapping],
    *,
    suites: Sequence[str],
    variants: Sequence[str],
    rule: str,
    printed: tuple[int, int],
    contrast: str,
    reference: str | None,
    comparator: str,
    c_arm: str,
) -> dict[str, Any]:
    """RULES amendment 8's block: the per-seed cells, the seeds pooled within task by the primary's
    rule, and -- where a reference is read -- the prompted reference against the comparator."""
    pooled = []
    for suite in suites:
        for variant in variants:
            cs = [c for c in aq_cells if c["suite"] == suite and c["variant"] == variant]
            measures: dict[str, Any] = {}
            for name in AQ_MEASURES:
                e = _pooled_secondary(
                    cs, name, rule, printed, get=lambda c, name=name: c["measures"].get(name)
                )
                if e is not None:
                    es = [c["measures"][name] for c in cs]
                    if "mean_delta" in e:
                        e["trained_level"], e["control_level"] = _pooled_levels(es, rule)
                    e["undefined_units"] = {
                        c["seed_label"]: c["measures"][name]["undefined_units"] for c in cs
                    }
                measures[name] = e
            pooled.append(
                {
                    "suite": suite,
                    "variant": variant,
                    "seed_labels": [c["seed_label"] for c in cs],
                    "label": EXPLORATORY_LABEL,
                    "measures": measures,
                }
            )
    ref_cells = None
    if reference is not None:
        ref_cells = []
        for suite in suites:
            for variant in variants:
                pairs, _unpaired, _nf, task_of = _reference_pairs(
                    in_design, suite, variant, reference, comparator, c_arm
                )
                ref_cells.append(
                    {
                        "suite": suite,
                        "variant": variant,
                        "label": EXPLORATORY_LABEL,
                        "models": [reference, comparator],
                        "delta": f"{reference} minus {comparator}",
                        "n_pairs": len(pairs),
                        "measures": ask_quality_measures(
                            pairs,
                            task_of,
                            REFERENCE_ARM,
                            "comparator",
                            printed,
                            arms=("reference", "comparator"),
                        ),
                    }
                )
    return {
        "label": EXPLORATORY_LABEL,
        "declared": "RULES amendment 8, exploratory descriptive secondaries",
        "definitions": dict(AQ_DEFINITIONS),
        "omitted": AQ_OMITTED,
        "paraphrase_guard": f"{is_paraphrase.__module__}.{is_paraphrase.__qualname__}",
        "contrast": contrast,
        "direction": "trained minus prompted comparator",
        "pooling": rule,
        "cells": list(aq_cells),
        "pooled": pooled,
        "reference": ref_cells,
    }


def _print_ask_quality(emit: Emit, aq: Mapping[str, Any]) -> None:
    """Amendment 8's exploratory block. EVERY line carries the label."""
    lab = aq["label"]
    emit("")
    emit(
        f"  {lab} ===== ask quality ({aq['declared']}): per arm and the paired task-level "
        f"difference ({aq['direction']}) on the primary's own pairs; printed interval only; in no "
        "Holm family ====="
    )
    for name, d in aq["definitions"].items():
        emit(f"  {lab} [{name}] = {d}")
    emit(f"  {lab} omitted: {aq['omitted']}")

    def one(where: str, name: str, e: Mapping[str, Any] | None, arms: tuple, shown: tuple) -> str:
        head = f"  {lab} | {where} [{name}]"
        if e is None:
            return head + " NOT RECORDED -- not zero"
        und = e.get("undefined_units") or {}
        dropped = "; units with no ask (dropped, counted): " + ", ".join(
            f"{k} {v}" for k, v in und.items()
        )
        form = (
            pooled_secondary_form(e)
            if isinstance(e.get("n_pairs"), Mapping)
            else f"defined on {e['n_defined_pairs']} of {e['n_pairs']} pairs"
        )
        if "mean_delta" not in e:
            return head + f" {form}: defined on no pair" + dropped
        iv = e["printed_interval"]
        levels = "".join(
            f"{shown[i]} {e[f'{arms[i]}_level']:.4f}  " for i in (0, 1) if f"{arms[i]}_level" in e
        )
        return (
            head + f" {levels}paired delta={e['mean_delta']:+.4f} {_fmt_iv(iv)} ({iv['n_boot']}, "
            f"seed {iv['seed']})  split {e['split']}; {form}" + dropped
        )

    for c in aq["cells"]:
        where = f"{c['suite']} / {c['variant']} / {c['seed_label']} ({c['contrast']})"
        for name, e in c["measures"].items():
            emit(one(where, name, e, ("trained", "control"), ("trained", "comparator")))
    for c in aq["pooled"]:
        where = f"{c['suite']} / {c['variant']} / seeds-pooled {'+'.join(c['seed_labels'])}"
        for name, e in c["measures"].items():
            emit(one(where, name, e, ("trained", "control"), ("trained", "comparator")))
    for c in aq["reference"] or []:
        ref, cmp_ = (short_model(x) for x in c["models"])
        where = f"{c['suite']} / {c['variant']} ({ref} minus {cmp_}, descriptive, no claim)"
        for name, e in c["measures"].items():
            emit(one(where, name, e, ("reference", "comparator"), (ref, cmp_)))


def _print_reference(emit: Emit, rc: Mapping[str, Any]) -> None:
    ref, cmp_ = short_model(rc["reference_model"]), short_model(rc["comparator_model"])
    emit("")
    emit(
        f"  ===== prompted {ref} vs prompted {cmp_} -- {REFERENCE_LABEL} (RULES amendment 11 point "
        f"2; delta = {ref} minus {cmp_}; printed interval only; in no Holm family) ====="
    )
    for c in rc["cells"]:
        emit(
            f"    --- {c['suite']} / {c['variant']}: {c['n_pairs']} pairs ({c['n_unpaired']} "
            "unpaired) ---"
        )
        if not c["n_pairs"]:
            emit("      no pair: nothing to describe")
            continue
        for name, e in c["endpoints"].items():
            iv = e["printed_interval"]
            emit(
                f"      [{name}] {ref} {e['reference_level']:.4f}  {cmp_} "
                f"{e['comparator_level']:.4f}  delta={e['mean_delta']:+.4f} {_fmt_iv(iv)} "
                f"({iv['n_boot']}, seed {iv['seed']})  split {e['split']}"
            )


def _arm_label(
    r: Mapping,
    comparator: str,
    trained: Sequence[tuple[str, str, str]],
    reference: str | None = None,
) -> str:
    """'comparator', 'reference', a trained seed's label, or the run's arm_id when its pin says
    none of them."""
    model = _questioner_model(r)
    arm = (r.get("key") or [None] * 3)[KEY_ARM]
    if arm == CONTROL_ARM and reference is not None and model == reference:
        return REFERENCE_ARM
    if arm == CONTROL_ARM and model in (comparator, None):
        return "comparator"
    for label, m, _ in trained:
        if model == m:
            return label
    if model is None and arm == TREATMENT_ARM and len(trained) == 1:
        return trained[0][0]
    return str(arm)


def stopping_block(
    records: Sequence[Mapping],
    *,
    suites: Sequence[str],
    variants: Sequence[str],
    comparator: str,
    trained: Sequence[tuple[str, str, str]],
    cap: int,
    reference: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Does the questioner's asking leave the agent any budget? Per suite, variant and arm.

    At 132e3e8 the prompted base (thinking on) never stopped: its asks spent all 16 before the
    first tool call on 55/55 airline and 97/98 retail runs. If that persists, the primary is a
    STOPPING contrast by construction, and this block is what shows it. From the enforcement lane's
    fields: asks capped before the first tool call (exact; see _asks_capped), rollout 1's stop and
    asks, the leak check, zero executed non-GENERIC
    calls, and the spend before the first executed tool call (quoted on every unit, censored at the
    total spend where none executed; acting units only is descriptive). The final rollout's stop_reason and
    the asks over all rollouts are printed too, labelled.
    """
    labels = (
        ["comparator"] + ([REFERENCE_ARM] if reference else []) + [label for label, _, _ in trained]
    )
    rows, counts = [], []
    calls_fn = _unit_value(A4_CALLS_BEFORE_FIRST, cap)
    for suite in suites:
        for variant in variants:
            here = [r for r in records if _suite(r) == suite and _variant(r) == variant]
            for arm in labels:
                runs = [r for r in here if _arm_label(r, comparator, trained, reference) == arm]
                ok = [r for r in runs if r.get("status") == "ok"]
                counts.append(
                    {
                        "suite": suite,
                        "variant": variant,
                        "arm": arm,
                        "statuses": dict(
                            sorted(Counter(str(r.get("status")) for r in runs).items())
                        ),
                        "errors": dict(
                            Counter(
                                str(r.get("error") or "")[:120]
                                for r in runs
                                if r.get("status") != "ok"
                            )
                        ),
                    }
                )

                def frac(pred: Callable[[Mapping], bool | None]) -> float | None:
                    vals = [pred(r) for r in ok]
                    if not ok or any(v is None for v in vals):
                        return None
                    return sum(bool(v) for v in vals) / len(vals)

                def field_eq(field: str, want: Any) -> Callable[[Mapping], bool | None]:
                    return lambda r: None if r.get(field) is None else r.get(field) == want

                def median_of(field: str) -> float | None:
                    xs = [r.get(field) for r in ok]
                    return statistics.median(xs) if xs and None not in xs else None

                capped = [_asks_capped(r, cap) for r in ok]

                def leaked(r: Mapping) -> bool | None:
                    n, gate, tcap = (
                        r.get(N_REFUSED_BUDGET_FIELD),
                        r.get(N_GATE_CHARGED_FIELD),
                        r.get(TOOL_CALL_CAP_FIELD),
                    )
                    if n is None or gate is None or tcap is None:
                        return None
                    return bool(n) and gate < tcap

                leaks = [leaked(r) for r in ok]
                calls = [(r, calls_fn(r)) for r in ok]
                calls_recorded = bool(ok) and all(u[1] != NOT_RECORDED for _, u in calls)
                rows.append(
                    {
                        "suite": suite,
                        "variant": variant,
                        "arm": arm,
                        "n_ok": len(ok),
                        "n_not_ok": len(runs) - len(ok),
                        "frac_asks_capped": frac(lambda r: _asks_capped(r, cap)[0]),
                        "n_asks_capped": sum(1 for v, _ in capped if v),
                        "n_asks_capped_defined": sum(1 for v, _ in capped if v is not None),
                        "asks_capped_basis": sorted({b for v, b in capped if v is not None}),
                        "tool_calls_blocked_by_asks": TOOL_CALLS_BLOCKED_BY_ASKS,
                        "leak_refused_below_tool_cap": (
                            sum(1 for x in leaks if x) if ok and None not in leaks else None
                        ),
                        "frac_rollout1_policy_stop": frac(
                            field_eq(ROLLOUT1_STOP_FIELD, POLICY_STOP)
                        ),
                        "median_rollout1_asks": median_of(ROLLOUT1_ASKS_FIELD),
                        "frac_zero_executed_nongeneric": frac(field_eq(N_LIVE_NONGENERIC_FIELD, 0)),
                        "calls_before_first_executed_tool_call_quartiles": (
                            quartiles([u[0] for _, u in calls if u[1] == DEFINED])
                            if calls_recorded
                            else "NOT RECORDED"
                        ),
                        "n_no_live_call": (
                            sum(1 for r, _ in calls if r.get(FIRST_LIVE_EXECUTED_FIELD) is False)
                            if calls_recorded
                            else None
                        ),
                        "calls_before_first_quartiles_acting_only": (
                            quartiles(
                                [x for x in (_acted_before(r) for r, _ in calls) if x is not None]
                            )
                            if calls_recorded
                            else "NOT RECORDED"
                        ),
                        "frac_final_rollout_policy_stop": frac(
                            field_eq(STOP_REASON_FIELD, POLICY_STOP)
                        ),
                        "median_asks_per_unit": median_of(GATE_ASKS_FIELD),
                    }
                )
    return rows, counts


def _print_stopping(emit: Emit, rows: Sequence[Mapping], counts: Sequence[Mapping]) -> None:
    def f(x: float | None) -> str:
        return "NOT RECORDED" if x is None else f"{x:.3f}"

    emit("")
    emit("  ===== STOPPING -- the asks against their own cap (separate budgets) =====")
    emit(
        "    asks reached the ask cap before the first tool call: DESCRIPTIVE, not a refusal "
        "(separate budgets); calls before the first executed tool call are quoted on every unit, "
        "censored at the total spend where none executed. Final-rollout stop and all-rollout "
        "asks are labelled extras."
    )
    for r in rows:
        emit(
            f"    {r['suite']} / {r['variant']} / {r['arm']}: ok={r['n_ok']} not-ok={r['n_not_ok']}"
            + f"  asks reached the ask cap before the first tool call: "
            f"{f(r['frac_asks_capped'])} ({r['n_asks_capped']}/{r['n_asks_capped_defined']}, "
            f"basis {r['asks_capped_basis'] or 'NOT RECORDED'})"
            f"  tool calls blocked by asks: {r['tool_calls_blocked_by_asks']}; leak check: "
            f"refused calls below tool cap = "
            f"{'NOT RECORDED' if r['leak_refused_below_tool_cap'] is None else r['leak_refused_below_tool_cap']}"
            + f"  rollout-1 policy_stop: {f(r['frac_rollout1_policy_stop'])}"
            f"  median rollout-1 asks: {r['median_rollout1_asks']}"
            f"  zero executed non-GENERIC: {f(r['frac_zero_executed_nongeneric'])}"
            f"  calls before the first executed tool call [Q1, median, Q3], censored at total "
            f"spend: {r['calls_before_first_executed_tool_call_quartiles']}"
            f" ({r['n_no_live_call']} with no live call; acting units only, descriptive: "
            f"{r['calls_before_first_quartiles_acting_only']})"
            f"  final-rollout policy_stop (labelled): {f(r['frac_final_rollout_policy_stop'])}"
            f"  median asks/unit, all rollouts (labelled): {r['median_asks_per_unit']}"
        )
    emit("    unit status and errors per arm:")
    for c in counts:
        emit(
            f"      {c['suite']} / {c['variant']} / {c['arm']}: {c['statuses']}"
            + (f"  errors {c['errors']}" if c["errors"] else "")
        )


def _load_gold(
    args: argparse.Namespace, *, design: Sequence[Mapping], legacy: bool, disabled: list[str]
) -> "gold_objects.GoldObjects | None":
    """The gold objects, loaded at READING time from the tau2 data the harness reads. A gated
    reading refuses without them, without a task any run names, or without per-ask uids."""
    ok_design = [r for r in design if r.get("status") == "ok"]
    try:
        gold = gold_objects.GoldObjects.from_arg_or_env(args.tau2_data_dir)
    except (OSError, ValueError, KeyError) as exc:
        raise Refusal(
            f"the tau2 data ({args.tau2_data_dir or 'TAU2_DATA_DIR'}) cannot be read: {exc}"
        )
    if gold is None:
        if not legacy:
            raise Refusal(
                "no tau2 data (--tau2-data-dir or TAU2_DATA_DIR): the gold-object secondaries need "
                "the tasks.json and db.json the harness reads"
            )
        disabled.append("gold-object secondaries: no tau2 data given")
        return None
    unknown = sorted(
        {
            (_suite(r), str(r.get("task_id")))
            for r in ok_design
            if not gold.has_task(_suite(r), str(r.get("task_id")))
        }
    )
    no_uids = [r for r in ok_design if not isinstance(r.get(ASK_UIDS_FIELD), list)]
    if legacy:
        if unknown or no_uids:
            disabled.append("gold-object secondaries: tasks or per-ask uids not in these records")
            return None
        return gold
    if unknown:
        raise Refusal(f"tasks absent from the tau2 tasks.json at {gold.root}: {unknown[:5]}")
    if no_uids:
        raise Refusal(
            f"{ASK_UIDS_FIELD} is absent on {len(no_uids)} ok runs (e.g. {_e_g(no_uids)}): the "
            "per-ask retrieved uids are in turns.jsonl -- re-extract with extract_records.py"
        )
    return gold


def check_provenance(
    records: Sequence[dict],
    *,
    args: argparse.Namespace,
    trained: Sequence[tuple[str, str, str]],
    design: Sequence[dict],
    legacy: bool,
    emit: Emit,
    disabled: list[str],
    reference: str | None = None,
) -> dict[str, Any]:
    """Every refusal that does not need the pairing. Raises Refusal; returns provenance.

    ORDER IS ONLY FOR THE MESSAGE: the first defect found is the one named. `design` is the
    records inside the declared suites and variants, of the two declared arms.
    """
    t_arm, c_arm = args.treatment, args.control
    prov: dict[str, Any] = {}

    # ---- the store
    if not records:
        raise Refusal(
            "empty store: no run records were read -- an empty store reads like a clean one"
        )
    if not any(r.get("status") == "ok" for r in records):
        raise Refusal(f"none of {len(records)} records has status ok -- nothing to read")

    # ---- manifests
    no_manifest = [r for r in records if _manifest(r) is None]
    if no_manifest and not legacy:
        raise Refusal(
            f"{len(no_manifest)} of {len(records)} records carry no manifest (e.g. "
            f"{_e_g(no_manifest)}): the model pins, base_url_sha, adapter_sha and budget_cap are "
            "on the manifest. Re-extract with extract_records.py at this commit, or pass --store."
        )
    if no_manifest:
        disabled.append(
            f"manifest-dependent checks on {len(no_manifest)} records with no manifest: "
            "base_url_sha per role, questioner model pins (comparator and trained)"
        )

    # ---- a run sits in the sub-root of the pin its manifest names
    misfiled = [
        r
        for r in records
        if (r.get(STORE_PIN) is not None and _questioner_model(r) != r[STORE_PIN])
        or (r.get(STORE_SUITE) is not None and _suite(r) != r[STORE_SUITE])
    ]
    if misfiled:
        r = misfiled[0]
        raise Refusal(
            f"{len(misfiled)} runs sit under a pin sub-root their manifest does not name (e.g. "
            f"{r.get('run_id')}: sub-root {r.get(STORE_SUITE)}/{r.get(STORE_PIN)}, manifest "
            f"{_suite(r)}/{_questioner_model(r)})"
        )
    layout = "pin sub-roots" if any(r.get(STORE_PIN) for r in records) else "flat"
    prov["layout"] = layout
    emit(f"  layout: {layout} (runs attributed by their manifest's questioner pin)")

    # ---- code_version: one, by two routes
    split = [
        r
        for r in records
        if _manifest(r) is not None
        and _manifest(r).get(CODE_VERSION_FIELD) != r.get(CODE_VERSION_FIELD)
    ]
    if split:
        r = split[0]
        raise Refusal(
            f"status and manifest code_version disagree on {len(split)} runs, e.g. "
            f"{r.get('run_id')} ({r.get(CODE_VERSION_FIELD)} vs "
            f"{_manifest(r).get(CODE_VERSION_FIELD)})"
        )
    cvs = Counter(str(r.get(CODE_VERSION_FIELD) or "<absent>") for r in records)
    if len(cvs) != 1 or "<absent>" in cvs:
        raise Refusal(
            f"{len(cvs)} code_version values among the runs read ("
            + ", ".join(f"{k[:8]}: {v}" for k, v in sorted(cvs.items()))
            + "); one population is one code_version"
        )
    prov["code_version"] = next(iter(cvs))

    # ---- dirty: False by every route that carries it; absent is not clean
    not_clean = [
        r
        for r in records
        if r.get(DIRTY_FIELD) is not False
        or (_manifest(r) is not None and _manifest(r).get(DIRTY_FIELD) is not False)
    ]
    if not_clean:
        raise Refusal(
            f"{len(not_clean)} runs are dirty or carry no dirty flag (e.g. {_e_g(not_clean)}); "
            "a dirty tree is not a code_version"
        )

    # ---- one base_url_sha per role
    by_role: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        for role, pin in ((_manifest(r) or {}).get(PINS_FIELD) or {}).items():
            by_role[role][str((pin or {}).get(PIN_BASE_URL_SHA))] += 1
    multi = {role: dict(c) for role, c in by_role.items() if len(c) > 1}
    if multi:
        raise Refusal(
            "more than one base_url_sha for a role: "
            + "; ".join(
                f"{role}: " + ", ".join(f"{s[:12]} x{n}" for s, n in c.items())
                for role, c in sorted(multi.items())
            )
        )
    prov["base_url_sha_by_role"] = {role: next(iter(c)) for role, c in sorted(by_role.items())}
    for role, sha in prov["base_url_sha_by_role"].items():
        emit(f"  base_url_sha {role}: {sha[:12]} (one value)")

    # ---- retrieval k identical across arms (amendment 7)
    ks = Counter(
        str((_manifest(r) or {}).get(MANIFEST_K_FIELD)) for r in records if _manifest(r) is not None
    )
    if ks and set(ks) != {str(RETRIEVAL_K)}:
        raise Refusal(
            f"retrieval k is not {RETRIEVAL_K} on every manifest ({dict(ks)}): the arms must "
            "retrieve alike"
        )

    # ---- the key and the fields are two routes to one slot
    bad_key = []
    for r in records:
        if r.get("status") != "ok":
            continue
        k = r.get("key") or []
        if len(k) <= KEY_VARIANT or (
            k[KEY_TRACE] != r.get("foreign_trace_sha")
            or k[KEY_PREFIX_K] != r.get("foreign_prefix_k")
            or k[KEY_SEED] != r.get("seed")
            or str(k[KEY_TASK]) != str(r.get("task_id"))
            or k[KEY_ARM] != r.get("arm_id")
            or ("suite_id" in r and k[KEY_SUITE] != r.get("suite_id"))
        ):
            bad_key.append(r)
    if bad_key:
        raise Refusal(
            f"status key disagrees with its own fields on {len(bad_key)} ok runs "
            f"(e.g. {_e_g(bad_key)}); the slot a run occupies is ambiguous"
        )

    # ---- questioner pins: comparator is the base; trained is a declared seed
    declared = {m for _, m, _ in trained}
    t_models = Counter(
        _questioner_model(r) for r in records if (r.get("key") or [None] * 3)[KEY_ARM] == t_arm
    )
    undeclared = {m: n for m, n in t_models.items() if m is not None and m not in declared}
    if undeclared:
        raise Refusal(
            f"trained-arm runs pin undeclared questioner models {undeclared}; declared: "
            f"{sorted(declared)}"
        )
    if None in t_models and not (legacy and len(trained) == 1):
        raise Refusal(
            f"{t_models[None]} trained-arm runs have no questioner pin and cannot be attributed "
            "to a declared seed"
        )
    if None in t_models:
        emit(
            f"  trained runs attributed to {trained[0][0]} by arm_id alone "
            f"({t_models[None]} runs carry no manifest pin)"
        )
    c_models = Counter(
        _questioner_model(r) for r in records if (r.get("key") or [None] * 3)[KEY_ARM] == c_arm
    )
    allowed = {args.comparator_model, *([reference] if reference else [])}
    off = {m: n for m, n in c_models.items() if m is not None and m not in allowed}
    if off:
        raise Refusal(
            f"comparator-arm runs pin {off}, not the declared comparator "
            f"{args.comparator_model!r}"
            + (f" or the reference {reference!r}" if reference else "")
            + "; selecting the comparator by arm_id pools model pins"
        )
    required = {args.comparator_model} | declared | ({reference} if reference else set())
    outside = Counter(
        _questioner_model(r)
        for r in records
        if _questioner_model(r) is not None and _questioner_model(r) not in required
    )
    if outside:
        raise Refusal(
            f"runs pin questioner models outside the required set {sorted(required)}: "
            f"{dict(outside)}"
        )

    ok_design = [r for r in design if r.get("status") == "ok"]

    # ---- the budget: refuse-and-tell, no post-hoc overrun, nothing above the cap
    if legacy:
        disabled += [
            f"{BUDGET_ENFORCEMENT_FIELD} == {BUDGET_ENFORCEMENT_REQUIRED!r}",
            ".".join(POST_HOC_OVERRUN_PATH) + " not set",
            f"{'.'.join(SPENT_RETRIEVAL_PATH)} <= cap and manifest {MANIFEST_CAP_FIELD} == cap",
            f"status {STATUS_CAP_FIELD} == cap; len({REFUSED_CALLS_FIELD}) == "
            f"{N_REFUSED_BUDGET_FIELD}",
            f"gate identities: (a) spend residual >= 0; (b) {N_GATE_CHARGED_FIELD} == the "
            "reader's recount of live non-GENERIC calls; gate_census_agrees; budget_exempt_tools",
            f"rollout-1 and first-live-call fields ({ROLLOUT1_STOP_FIELD}, {ROLLOUT1_ASKS_FIELD}, "
            f"{FIRST_LIVE_EXECUTED_FIELD}, {SPENT_BEFORE_FIRST_CALL_FIELD}): NOT RECORDED if absent",
        ]
    else:
        wrong_mode: Counter = Counter()
        wrong_runs = []
        for r in ok_design:
            vs = r.get(BUDGET_ENFORCEMENT_FIELD, _MISSING)
            vm = (_manifest(r) or {}).get(BUDGET_ENFORCEMENT_FIELD, _MISSING)
            if vs is not _MISSING and vm is not _MISSING and vs != vm:
                raise Refusal(
                    f"{BUDGET_ENFORCEMENT_FIELD} is {vs!r} on the status and {vm!r} on the "
                    f"manifest of {r.get('run_id')}"
                )
            v = vs if vs is not _MISSING else vm
            if v != BUDGET_ENFORCEMENT_REQUIRED:
                wrong_mode["<absent>" if v is _MISSING else str(v)] += 1
                wrong_runs.append(r)
        if wrong_runs:
            raise Refusal(
                f"{len(wrong_runs)} ok runs lack {BUDGET_ENFORCEMENT_FIELD} == "
                f"{BUDGET_ENFORCEMENT_REQUIRED!r} ({dict(wrong_mode)}; e.g. {_e_g(wrong_runs)})"
            )
        # REVIVED ON A GATED UNIT (1ae52cb): it fires when (spent - n_gate_charged) +
        # n_live_env_calls_nongeneric_ok > cap, i.e. the transcript shows a call the cap forbade.
        over = [r for r in ok_design if _get(r, POST_HOC_OVERRUN_PATH) not in (_MISSING, None, 0)]
        if over:
            raise Refusal(
                f"{'.'.join(POST_HOC_OVERRUN_PATH)} is set on {len(over)} ok runs "
                f"(e.g. {_e_g(over)}): the cap was charged after the dialogue, not enforced"
            )
        # RULES amendment 12: the ledger creates a currency on its first debit, so a run that
        # asked nothing has no retrieval_calls key. That absence is a true zero ONLY when the
        # ledger was written and no ask, turn or question is recorded (pi_run.compact reads it
        # as 0 too); it is filled here, before every budget identity, and named. Any other
        # absence falls through to the refusal below.
        asked_nothing = [
            r
            for r in ok_design
            if _get(r, SPENT_RETRIEVAL_PATH) is _MISSING
            and isinstance(r.get("spent"), Mapping)
            and r["spent"]
            and r.get(GATE_ASKS_FIELD) == 0
            and r.get(ROLLOUT1_ASKS_FIELD) == 0
            and r.get("n_turns") == 0
            and r.get(SPENT_BEFORE_FIRST_CALL_FIELD) in (None, 0, 0.0)
            and not r.get(ASK_UIDS_FIELD)
            and not r.get(ASK_QUESTIONS_FIELD)
        ]
        for r in asked_nothing:
            r["spent"][SPENT_RETRIEVAL_PATH[-1]] = 0.0
        if asked_nothing:
            emit(
                f"  amendment 12: {'.'.join(SPENT_RETRIEVAL_PATH)} absent and read as 0 on "
                f"{len(asked_nothing)} ok run{'s' if len(asked_nothing) != 1 else ''} that asked "
                f"nothing (n_asks, rollout1_n_asks and n_turns 0; no question recorded): "
                f"{', '.join(str(r.get('run_id')) for r in asked_nothing)}"
            )
        prov["asked_nothing_read_as_zero"] = [r.get("run_id") for r in asked_nothing]
        no_spend = [r for r in ok_design if _get(r, SPENT_RETRIEVAL_PATH) in (_MISSING, None)]
        if no_spend:
            raise Refusal(
                f"{'.'.join(SPENT_RETRIEVAL_PATH)} is absent on {len(no_spend)} ok runs "
                f"(e.g. {_e_g(no_spend)}); spend cannot be checked against the cap"
            )
        above = [r for r in ok_design if float(_get(r, SPENT_RETRIEVAL_PATH)) > args.cap]
        if above:
            worst = max(float(_get(r, SPENT_RETRIEVAL_PATH)) for r in above)
            raise Refusal(
                f"{len(above)} ok runs spent retrieval calls above the cap {args.cap} "
                f"(max {worst:g}; e.g. {_e_g(above)})"
            )
        cap_off = [r for r in ok_design if (_manifest(r) or {}).get(MANIFEST_CAP_FIELD) != args.cap]
        if cap_off:
            caps = Counter(str((_manifest(r) or {}).get(MANIFEST_CAP_FIELD)) for r in cap_off)
            raise Refusal(
                f"manifest {MANIFEST_CAP_FIELD} is not the declared cap {args.cap} on "
                f"{len(cap_off)} ok runs ({dict(caps)})"
            )
        status_cap_off = [r for r in ok_design if r.get(STATUS_CAP_FIELD) != args.cap]
        if status_cap_off:
            caps = Counter(str(r.get(STATUS_CAP_FIELD)) for r in status_cap_off)
            raise Refusal(
                f"status {STATUS_CAP_FIELD} is not the declared cap {args.cap} on "
                f"{len(status_cap_off)} ok runs ({dict(caps)})"
            )
        bad_list = [
            r
            for r in ok_design
            if not isinstance(r.get(REFUSED_CALLS_FIELD), list)
            or len(r[REFUSED_CALLS_FIELD]) != r.get(N_REFUSED_BUDGET_FIELD)
        ]
        if bad_list:
            raise Refusal(
                f"{REFUSED_CALLS_FIELD} is absent or its length is not {N_REFUSED_BUDGET_FIELD} "
                f"on {len(bad_list)} ok runs (e.g. {_e_g(bad_list)})"
            )
        residual = _check_gate_charge(ok_design)
        emit(
            f"  budget: {len(ok_design)} ok runs carry {BUDGET_ENFORCEMENT_FIELD}="
            f"{BUDGET_ENFORCEMENT_REQUIRED!r}, no post-hoc overrun, spend <= cap {args.cap}; "
            f"the gate charge == the reader's recount and {GATE_CENSUS_AGREES_FIELD} on all"
        )
        by_arm: dict[str, Counter] = defaultdict(Counter)
        for r in ok_design:
            by_arm[_arm_label(r, args.comparator_model, trained, reference)][
                f"{residual[id(r)]:g}"
            ] += 1
        prov["residual_by_arm"] = {a: dict(sorted(c.items())) for a, c in sorted(by_arm.items())}
        for a, c in prov["residual_by_arm"].items():
            emit(
                f"  spend residual r (query-expansion sub-retrievals; r > 0 is expansion spend) "
                f"{a}: {c}"
            )

    # ---- the trained weights: manifest adapter_sha, else every serve record, all agreeing
    if legacy:
        disabled.append("trained weights: adapter_sha / serve-record sha256 prefix per seed")
    else:
        prov.update(check_weights(args, trained, ok_design, t_arm, emit))

    # ---- thinking off: every probe record PASS, one config (or amendment 10's agreeing archives),
    # one proxy, the required set covered
    if legacy:
        disabled.append(
            "thinking-probe records (PASS, one config_sha256 or archived configs agreeing under "
            "amendment 10, one proxy_url, covering the "
            "comparator and the trained models, proxy hash == every questioner base_url_sha)"
        )
    else:
        prov.update(check_probes(args, trained, records, emit, reference))

    # ---- the endpoints are present, and the endpoint routes agree
    no_primary = [r for r in ok_design if any(r.get(f) is None for f in PRIMARY_FIELDS)]
    if no_primary:
        gone = sorted({f for r in no_primary for f in PRIMARY_FIELDS if r.get(f) is None})
        raise Refusal(
            f"the primary's fields {gone} are absent on {len(no_primary)} ok runs (e.g. "
            f"{_e_g(no_primary)}); follow_ups() would read the absence as 0"
        )
    no_reward = [r for r in ok_design if _get(r, REWARD_PATH) in (_MISSING, None)]
    if no_reward:
        raise Refusal(
            f"{len(no_reward)} ok runs carry no {'.'.join(REWARD_PATH)} (e.g. {_e_g(no_reward)}); "
            "a missing reward is absence, not failure"
        )
    checked = [r for r in ok_design if r.get(ENV_CALLS_FROM_OUTCOME) is not None]
    bad = [r for r in checked if r.get("n_env_calls") != r[ENV_CALLS_FROM_OUTCOME]]
    if bad:
        raise Refusal(
            f"n_env_calls disagrees with len(outcome.env_calls) on {len(bad)} of {len(checked)} "
            f"ok runs, e.g. {bad[0].get('run_id')}"
        )
    for field in (ENV_TOOL_COUNTS, ENV_REQUESTOR_COUNTS):
        have = [r for r in ok_design if isinstance(r.get(field), Mapping)]
        bad = [r for r in have if sum(r[field].values()) != r.get("n_env_calls")]
        if bad:
            raise Refusal(
                f"{field} does not sum to n_env_calls on {len(bad)} ok runs, e.g. "
                f"{bad[0].get('run_id')}"
            )
    tool_types = fork_paired.load_tool_types()
    unmapped = sorted(
        {
            (fork_paired.domain_of(r), name)
            for r in ok_design
            for name in (r.get(ENV_TOOL_COUNTS) or {})
            if name not in tool_types.get(fork_paired.domain_of(r), {})
        }
    )
    if unmapped:
        raise Refusal(
            f"tools with no declared type, which would otherwise count as zero: {unmapped}"
        )
    emit(
        f"  endpoint routes: n_env_calls == len(outcome.env_calls) on {len(checked)} of "
        f"{len(ok_design)} ok runs in the design"
    )
    return prov


def _secondary(
    pairs: Mapping[tuple, Mapping],
    task_of: Mapping[tuple, str],
    t: str,
    c: str,
    tool_types: Mapping,
    cap: int = CAP,
    printed: tuple[int, int] = PRINTED_INTERVAL,
    gold: "gold_objects.GoldObjects | None" = None,
    suite: str = "",
) -> dict[str, Any]:
    """Every secondary on the primary's OWN pairs -- no unit is dropped or selected here, and none
    is conditioned on a post-treatment value. Per run, never a per-turn average; both arms' task-
    level levels, the paired task-level difference and the printed interval (amendment 4)."""
    runs = [v[a] for v in pairs.values() for a in (t, c)]
    if not runs:  # a cell with no pairs is refused by the caller; nothing to contrast here
        return {}

    def present(*names: str) -> bool:
        # `follow_ups` and `endpoint_asks` read an absent field as 0; absence must read as absence.
        return all(r.get(n) is not None for r in runs for n in names)

    endpoints: list[tuple[str, Callable[[Mapping], float] | None]] = [
        (
            "follow-up user turns",
            follow_ups if present("n_user_turns", "n_prefix_user_turns") else None,
        ),
        ("asks (n_asks)", fork_paired.endpoint_asks if present("n_asks") else None),
        (
            "env tool calls (n_env_calls)",
            fork_paired.endpoint_calls if present("n_env_calls") else None,
        ),
    ]
    if all(r.get(ENV_TOOL_COUNTS) is not None for r in runs):
        endpoints += [
            ("READ calls", fork_paired.typed_calls(tool_types, "READ")),
            ("WRITE calls", fork_paired.typed_calls(tool_types, "WRITE")),
        ]
    else:
        endpoints += [("READ calls", None), ("WRITE calls", None)]
    if all(isinstance(r.get(ENV_REQUESTOR_COUNTS), Mapping) for r in runs):
        roles = sorted({role for r in runs for role in r[ENV_REQUESTOR_COUNTS]})
        for role in roles:
            endpoints.append(
                (
                    f"env tool calls by requestor={role}",
                    lambda r, role=role: float(r[ENV_REQUESTOR_COUNTS].get(role, 0)),
                )
            )
    else:
        endpoints.append(("env tool calls by requestor role", None))

    def always(fn: Callable[[Mapping], float]) -> Callable[[Mapping], tuple[float, str]]:
        return lambda r: (float(fn(r)), DEFINED)

    tagged: list[tuple[str, Callable[[Mapping], tuple[float | None, str]] | None]] = [
        (name, None if fn is None else always(fn)) for name, fn in endpoints
    ]
    tagged += [
        (name, _unit_value(name, cap))
        for name in (A4_STARVED, A4_ROLLOUT1_STOP, A4_ZERO_NONGENERIC, A4_CALLS_BEFORE_FIRST)
    ]
    tagged += [(name, _a7_value(name, gold, suite)) for name in (A7_HIT, A7_RECALL, A7_ASKS)]

    def refused_any(r: Mapping) -> tuple[float | None, str]:
        v = r.get(N_REFUSED_BUDGET_FIELD)
        return (None, NOT_RECORDED) if v is None else (float(v > 0), DEFINED)

    tagged.append((A9_REFUSED_UNITS, refused_any))

    out: dict[str, Any] = {}
    for name, fn in tagged:
        if fn is None:
            out[name] = None
            continue
        units = {k: (fn(v[t]), fn(v[c])) for k, v in pairs.items()}
        if any(u[1] == NOT_RECORDED for pair in units.values() for u in pair):
            out[name] = None
            continue
        entry = _tagged_entry(units, task_of, printed)
        if name == A4_CALLS_BEFORE_FIRST:
            entry["defined_label"] = CALLS_QUOTED_LABEL
            for arm, i, a_ in (("trained", 0, t), ("control", 1, c)):
                rs = [v[a_] for v in pairs.values()]
                entry[f"{arm}_quartiles"] = quartiles(
                    [units[k][i][0] for k in pairs if units[k][i][1] == DEFINED]
                )
                entry[f"{arm}_n_censored"] = sum(
                    1 for r in rs if r.get(FIRST_LIVE_EXECUTED_FIELD) is False
                )
                entry[f"{arm}_quartiles_acting_only"] = quartiles(
                    [x for x in (_acted_before(r) for r in rs) if x is not None]
                )
            # DESCRIPTIVE ONLY: conditioning on both arms having acted selects on a post-treatment
            # variable, so it carries no interval and is never the quoted difference.
            both = {
                k: (_acted_before(v[t]), _acted_before(v[c]))
                for k, v in pairs.items()
                if _acted_before(v[t]) is not None and _acted_before(v[c]) is not None
            }
            by_task: dict[str, list[float]] = defaultdict(list)
            for k in sorted(both, key=repr):
                by_task[task_of[k]].append(both[k][0] - both[k][1])
            tm = {task: statistics.fmean(d) for task, d in sorted(by_task.items())}
            entry["descriptive_conditional_on_both_acting"] = {
                "label": CALLS_DESCRIPTIVE_LABEL,
                "n_pairs": len(both),
                "task_deltas": tm,
                "mean_delta": statistics.fmean(tm.values()) if tm else None,
            }
        if name == A9_REFUSED_UNITS:
            entry["trained_units_refused"] = sum(1 for a, _ in units.values() if a[0])
            entry["control_units_refused"] = sum(1 for _, b in units.values() if b[0])
        if name == A7_HIT:
            # WHY EACH PAIR WAS DROPPED, never imputed: an arm that asked nothing has no rate
            no_gold = [k for k, v in pairs.items() if not gold.uids(suite, task_of[k])]
            entry["dropped_units"] = {
                "trained_asked_nothing": sum(
                    1 for k, v in pairs.items() if k not in no_gold and not v[t][ASK_UIDS_FIELD]
                ),
                "comparator_asked_nothing": sum(
                    1 for k, v in pairs.items() if k not in no_gold and not v[c][ASK_UIDS_FIELD]
                ),
            }
            entry["dropped_pairs_no_gold_object"] = len(no_gold)
            for arm, a_ in (("trained", t), ("control", c)):
                hits = asked = outside = 0
                for k, v in pairs.items():
                    g = gold.uids(suite, task_of[k])
                    asks = v[a_][ASK_UIDS_FIELD]
                    if not g:
                        outside += len(asks)  # no gold object to hit: outside the share, counted
                        continue
                    asked += len(asks)
                    hits += sum(1 for x in asks if set(x) & g)
                entry[f"{arm}_share_of_asks_hit"] = hits / asked if asked else None
                entry[f"{arm}_asks_in_tasks_without_gold"] = outside
        out[name] = entry
    return out


def _tagged_entry(
    units: Mapping[tuple, tuple[tuple[float | None, str], tuple[float | None, str]]],
    task_of: Mapping[tuple, str],
    printed: tuple[int, int],
) -> dict[str, Any]:
    """One tagged endpoint over its pairs, from each pair's ((value, status), (value, status)):
    the pair-level values within each task, the task-level difference, both arms' task-level levels
    and the printed interval. A PAIR IS DEFINED ONLY WHEN BOTH UNITS ARE; an undefined unit is never
    imputed here."""
    vals = {k: (a[0], b[0]) for k, (a, b) in units.items() if a[1] == DEFINED and b[1] == DEFINED}
    task_pairs: dict[str, list[list[float]]] = defaultdict(list)
    for k in sorted(vals, key=repr):
        task_pairs[task_of[k]].append(list(vals[k]))
    deltas = {task: [a - b for a, b in xs] for task, xs in task_pairs.items()}
    entry: dict[str, Any] = {
        "n_pairs": len(units),
        "n_defined_pairs": len(vals),
        "defined_on_all_units": len(vals) == len(units),
        "task_pair_deltas": dict(deltas),
        "task_pair_values": dict(task_pairs),
    }
    if vals:
        entry.update(diff_stats({t_: statistics.fmean(d) for t_, d in deltas.items()}, printed))
        entry.update(
            trained_level=statistics.fmean(
                statistics.fmean(a for a, _ in xs) for xs in task_pairs.values()
            ),
            control_level=statistics.fmean(
                statistics.fmean(b for _, b in xs) for xs in task_pairs.values()
            ),
            pair_mean_delta_NOT_QUOTABLE=statistics.fmean(a - b for a, b in vals.values()),
        )
    return entry


def _normalised_ask(q: str) -> str:
    """Amendment 8's exact comparison: lowercased, whitespace runs collapsed."""
    return " ".join(str(q).split()).lower()


def _ask_quality_value(name: str) -> Callable[[Mapping], tuple[float | None, str]]:
    """One unit's (value, status) for an amendment-8 exploratory measure: UNDEFINED where the unit
    asked nothing, NOT_RECORDED where the extract carries no per-ask field -- never a zero."""

    def value(r: Mapping) -> tuple[float | None, str]:
        if name == AQ_RETRIEVE:
            uids = r.get(ASK_UIDS_FIELD)
            if uids is None:
                return None, NOT_RECORDED
            if not uids:
                return None, UNDEFINED
            return sum(1 for u in uids if u) / len(uids), DEFINED
        qs = r.get(ASK_QUESTIONS_FIELD)
        if qs is None:
            return None, NOT_RECORDED
        if not qs:
            return None, UNDEFINED
        if name == AQ_EXACT:
            norm = [_normalised_ask(q) for q in qs]
            n = sum(1 for i, q in enumerate(norm) if q in norm[:i])
        else:
            n = sum(1 for i in range(len(qs)) if any(is_paraphrase(qs[i], qs[j]) for j in range(i)))
        return n / len(qs), DEFINED

    return value


def ask_quality_measures(
    pairs: Mapping[tuple, Mapping],
    task_of: Mapping[tuple, str],
    t: str,
    c: str,
    printed: tuple[int, int],
    arms: tuple[str, str] = ("trained", "control"),
) -> dict[str, Any]:
    """RULES amendment 8's exploratory ask-quality measures on the given pairs (the primary's own,
    or the reference contrast's), each read by `_tagged_entry`, with each arm's no-ask units
    counted. A measure some unit does not record is None -- NOT RECORDED, not zero."""
    out: dict[str, Any] = {}
    for name in AQ_MEASURES:
        fn = _ask_quality_value(name)
        units = {k: (fn(v[t]), fn(v[c])) for k, v in pairs.items()}
        if any(u[1] == NOT_RECORDED for pair in units.values() for u in pair):
            out[name] = None
            continue
        e = _tagged_entry(units, task_of, printed)
        e["undefined_units"] = {
            arms[0]: sum(1 for a, _ in units.values() if a[1] == UNDEFINED),
            arms[1]: sum(1 for _, b in units.values() if b[1] == UNDEFINED),
        }
        for old, new in (
            ("trained_level", f"{arms[0]}_level"),
            ("control_level", f"{arms[1]}_level"),
        ):
            if old in e and old != new:
                e[new] = e.pop(old)
        out[name] = e
    return out


def _asks_capped(r: Mapping, cap: int) -> tuple[bool | None, str]:
    """(asks capped, basis). EXACT from the first-live-call fields: no live call executed and the
    spend reached the cap, or the spend before the first live call already had. FALLBACK, labelled,
    where those fields are absent: questioner-side spend (spent - n_gate_charged) reached the cap
    with nothing charged. (None, NOT_RECORDED) where neither can be read."""
    spent = _get(r, SPENT_RETRIEVAL_PATH)
    if spent in (_MISSING, None):
        return None, NOT_RECORDED
    run_cap = r.get(STATUS_CAP_FIELD) or cap
    first = r.get(FIRST_LIVE_EXECUTED_FIELD)
    if first is not None and SPENT_BEFORE_FIRST_CALL_FIELD in r:
        before = r[SPENT_BEFORE_FIRST_CALL_FIELD]
        hit = (first is False and float(spent) >= run_cap) or (
            before is not None and float(before) >= run_cap
        )
        return hit, STARVED_EXACT
    gate = r.get(N_GATE_CHARGED_FIELD)
    if gate is None:
        return None, NOT_RECORDED
    return float(spent) - gate >= run_cap and gate == 0, STARVED_FALLBACK


def _unit_value(name: str, cap: int) -> Callable[[Mapping], tuple[float | None, str]]:
    """Amendment 4's secondaries as (value, DEFINED | UNDEFINED | NOT_RECORDED) per unit."""

    def value(r: Mapping) -> tuple[float | None, str]:
        if name == A4_STARVED:
            v, _ = _asks_capped(r, cap)
            return (None, NOT_RECORDED) if v is None else (float(v), DEFINED)
        if name == A4_ROLLOUT1_STOP:
            v = r.get(ROLLOUT1_STOP_FIELD)
            return (None, NOT_RECORDED) if v is None else (float(v == POLICY_STOP), DEFINED)
        if name == A4_ZERO_NONGENERIC:
            v = r.get(N_LIVE_NONGENERIC_FIELD)
            return (None, NOT_RECORDED) if v is None else (float(v == 0), DEFINED)
        if name == A4_CALLS_BEFORE_FIRST:
            first = r.get(FIRST_LIVE_EXECUTED_FIELD)
            if first is None or SPENT_BEFORE_FIRST_CALL_FIELD not in r:
                return None, NOT_RECORDED
            before = r[SPENT_BEFORE_FIRST_CALL_FIELD]
            if before is not None:
                return float(before), DEFINED
            spent = _get(r, SPENT_RETRIEVAL_PATH)
            if first is False and spent not in (_MISSING, None):
                return float(spent), DEFINED  # censored at the total spend
            return None, NOT_RECORDED
        raise ValueError(name)

    return value


def _a7_value(
    name: str, gold: "gold_objects.GoldObjects | None", suite: str
) -> Callable[[Mapping], tuple[float | None, str]]:
    """Amendment 7's secondaries per unit. UNDEFINED where the arm asked nothing (hit rate) or the
    task has no pre-dialogue gold object (hit rate and recall) -- never 0 or 1 by imputation."""

    def value(r: Mapping) -> tuple[float | None, str]:
        asks = r.get(ASK_UIDS_FIELD)
        if gold is None or not isinstance(asks, list):
            return None, NOT_RECORDED
        if name == A7_ASKS:
            return float(len(asks)), DEFINED
        g = gold.uids(suite, str(r.get("task_id")))
        if not g:
            return None, UNDEFINED
        if name == A7_HIT:
            if not asks:
                return None, UNDEFINED
            return sum(1 for a in asks if set(a) & g) / len(asks), DEFINED
        if name == A7_RECALL:
            touched = set().union(*(set(a) for a in asks)) if asks else set()
            return len(touched & g) / len(g), DEFINED
        raise ValueError(name)

    return value


def _acted_before(r: Mapping) -> float | None:
    """spent_before_first_live_call on a unit that ACTED (a live call executed); else None."""
    if r.get(FIRST_LIVE_EXECUTED_FIELD) is not True:
        return None
    v = r.get(SPENT_BEFORE_FIRST_CALL_FIELD)
    return None if v is None else float(v)


def quartiles(xs: Sequence[float]) -> list[float] | None:
    """[Q1, median, Q3], inclusive (linear) method; a single value is its own quartiles."""
    xs = sorted(float(x) for x in xs)
    if not xs:
        return None
    if len(xs) == 1:
        return [xs[0]] * 3
    return [float(q) for q in statistics.quantiles(xs, n=4, method="inclusive")]


def combine_within_task(per_seed: Sequence[Mapping[str, Sequence[float]]], rule: str) -> dict:
    """Training seeds combined within each task: `seed_mean` averages each seed's task mean,
    `pair_mean` averages every seed's pairs (table1_by_seed.pooled_symmetric's arithmetic)."""
    by_task: dict[str, list[list[float]]] = defaultdict(list)
    for cell in per_seed:
        for task, xs in cell.items():
            by_task[task].append(list(xs))
    if rule == "seed_mean":
        return {t: statistics.fmean(statistics.fmean(x) for x in v) for t, v in by_task.items()}
    return {t: statistics.fmean(y for x in v for y in x) for t, v in by_task.items()}


def diff_stats(task_deltas: Mapping[str, float], printed: tuple[int, int]) -> dict[str, Any]:
    """A secondary's task-level difference: counts, split, mean, and ONLY the printed interval
    (amendment 4: the 1k/10k/50k ladder is the primary's)."""
    vals = [float(task_deltas[t]) for t in sorted(task_deltas)]
    up, down = sum(x > 0 for x in vals), sum(x < 0 for x in vals)
    if vals:
        _, lo, hi = cluster_bootstrap(
            [[v] for v in vals], n_boot=printed[0], alpha=ALPHA, seed=printed[1]
        )
    else:
        lo = hi = float("nan")
    return {
        "n_tasks": len(vals),
        "up": up,
        "down": down,
        "tied": len(vals) - up - down,
        "split": split_of(up, down, len(vals) - up - down),
        "mean_delta": statistics.fmean(vals) if vals else float("nan"),
        "sign_test_p": sign_test_p(down, up),
        "sign_test_floor": sign_test_p(up + down, 0) if up + down else float("nan"),
        "printed_interval": {"n_boot": printed[0], "seed": printed[1], "lo": lo, "hi": hi},
        "task_deltas": {t: task_deltas[t] for t in sorted(task_deltas)},
    }


def _refused_budget(runs: Sequence[Mapping]) -> dict[str, Any] | None:
    vals = [r.get(N_REFUSED_BUDGET_FIELD) for r in runs]
    if not runs or any(v is None for v in vals):
        return None
    return {
        "n_runs": len(vals),
        "total": float(sum(vals)),
        "runs_with_any": sum(1 for v in vals if v),
        "per_run_mean": statistics.fmean(float(v) for v in vals),
    }


def read(args: argparse.Namespace, emit: Emit) -> dict[str, Any]:
    legacy = bool(args.validation_legacy_population)
    trained = _parse_trained(args.trained)
    suites = tuple(args.suites.split(",")) if args.suites else SUITES
    variants = tuple(args.variants.split(",")) if args.variants else VARIANTS
    resamples = tuple(int(x) for x in args.boot_resamples.split(","))
    boot_seeds = tuple(int(x) for x in args.boot_seeds.split(","))
    pn, _, ps = args.printed_interval.partition(":")
    printed = (int(pn), int(ps))
    t_arm, c_arm = args.treatment, args.control

    # RULES amendment 11: with --comparator-store the comparator pin is read from ITS OWN ROOT and
    # the main source holds the trained seeds and the prompted reference (the 8B base)
    comparator = args.comparator_model
    cstore = args.comparator_store
    reference = args.reference_model if cstore else None
    contrast_text = contrast_label(comparator)
    main_pins = [reference if cstore else comparator] + [m for _, m, _ in trained]
    if cstore and comparator in main_pins:
        raise Refusal(
            f"--comparator-model {comparator} is also a main-source pin {main_pins}: the "
            "comparator store would pool it"
        )
    main_records, src = load(args, suites, main_pins)
    c_records: list[dict] = []
    if cstore:
        c_records, src["comparator_store"] = load_store(
            Path(cstore), suites, [comparator], dirs={comparator: comparator_store_dir(comparator)}
        )
    records = main_records + c_records
    disabled: list[str] = []

    ok = [r for r in records if r.get("status") == "ok"]
    in_design = [r for r in records if _suite(r) in suites and _variant(r) in variants]
    arms = (t_arm, c_arm)
    design = [r for r in in_design if (r.get("key") or [None] * 3)[KEY_ARM] in arms]
    emit(
        f"===== tau2 transfer rerun reader: {len(records)} records read, {len(ok)} status ok ====="
    )
    emit(f"  rules: {RULES_DOC}   source: {src['kind']} {src['path']}")
    emit(
        f"  CONTRAST: {contrast_text} -- comparator pin {comparator}, arm {c_arm}"
        + (
            f", from its own root {src['comparator_store']['path']} ({len(c_records)} records); "
            f"the prompted reference {reference} from the main source"
            if cstore
            else ""
        )
    )
    if cstore:
        stray = [r for r in main_records if _questioner_model(r) == comparator]
        if stray:
            raise Refusal(
                f"{len(stray)} runs of the comparator pin {comparator} are in the main source (e.g. "
                f"{_e_g(stray)}); with --comparator-store it is read from its own root only"
            )
    outside = len(records) - len(design)
    if outside:
        emit(
            f"  {outside} records are outside the declared suites {list(suites)}, variants "
            f"{list(variants)} or arms {list(arms)}, and are not read"
        )

    if not args.pilot and (resamples != BOOT_RESAMPLES or boot_seeds != BOOT_SEEDS):
        emit(
            f"  resamples {list(resamples)} x seeds {list(boot_seeds)} are NOT the declared "
            f"{list(BOOT_RESAMPLES)} x {list(BOOT_SEEDS)}: no interval below may be quoted"
        )
    cvs = Counter(str(r.get(CODE_VERSION_FIELD) or "<absent>") for r in records)
    emit("  code_version: " + ", ".join(f"{k[:8]}: {v}" for k, v in sorted(cvs.items())))

    prov = check_provenance(
        records,
        args=args,
        trained=trained,
        design=design,
        legacy=legacy,
        emit=emit,
        disabled=disabled,
        reference=reference,
    )
    # ---- RULES amendment 11: a gateway questioner's own preflight, one per job of its store
    job_dirs = comparator_job_dirs(Path(cstore), suites) if cstore else []
    gateway = prov.get("gateway_pins_not_thinking_probed") or []
    if gateway and not legacy:
        prov.update(
            check_questioner_probe(
                gateway,
                records,
                emit,
                paths=expand_records(
                    args.questioner_probe_record, QPROBE_GLOB, "questioner probe record"
                )
                if args.questioner_probe_record
                else [],
                job_dirs=job_dirs,
                code_version=prov["code_version"],
                campaign_sha=extra_pin_driver().CAMPAIGN_SHA,
            )
        )
    planned, launch_prov = load_launch(
        args.launch_record,
        records=main_records,
        suites=suites,
        pins=main_pins,
        legacy=legacy,
        disabled=disabled,
    )
    prov["launch_records"] = launch_prov
    if cstore:
        drv = extra_pin_driver()
        c_planned, prov["comparator_launch"] = load_arm_launch(
            args.comparator_launch_record,
            records=c_records,
            suites=suites,
            pin=comparator,
            c_arm=c_arm,
            code_version=prov["code_version"],
            campaign_sha=drv.CAMPAIGN_SHA,
            legacy=legacy,
            disabled=disabled,
            emit=emit,
        )
        planned.update(c_planned)
        if not legacy:
            prov.update(
                check_comparator_jobs(
                    job_dirs,
                    launch_by_suite={x["suite"]: x for x in prov["comparator_launch"]},
                    pin=comparator,
                    campaign_sha=drv.CAMPAIGN_SHA,
                    emit=emit,
                )
            )
    gold = _load_gold(args, design=design, legacy=legacy, disabled=disabled)
    if gold is not None:
        prov["tau2_data"] = {"root": str(gold.root), "files": gold.files}
    for d in disabled:
        emit(f"  DISABLED by --validation-legacy-population: {d}")
    stopping, status_counts = stopping_block(
        design,
        suites=suites,
        variants=variants,
        comparator=args.comparator_model,
        trained=trained,
        cap=args.cap,
        reference=reference,
    )
    _print_stopping(emit, stopping, status_counts)
    if args.pilot:
        run_ids = sorted(str(r.get("run_id") or "") for r in records)
        return {
            "stamp": emit.stamp,
            "pilot": True,
            "reader": "scripts/tau2_concordance/transfer_rerun.py",
            "rules_doc": RULES_DOC,
            "disabled_checks": disabled,
            "source": src,
            "provenance": {
                **prov,
                "n_records": len(records),
                "n_ok": len(ok),
                "run_ids": run_ids,
                "run_ids_sha256": hashlib.sha256("\n".join(run_ids).encode()).hexdigest(),
            },
            "stopping": stopping,
            "status_counts": status_counts,
        }
    # ---- RULES amendment 11: where a gateway questioner's calls hit their token cap
    token_cap = None
    if is_gateway_pin(comparator) and not legacy:
        token_cap = token_cap_block(
            [
                r
                for r in design
                if r.get("status") == "ok"
                and r.get("arm_id") == c_arm
                and _questioner_model(r) == comparator
            ],
            pin=comparator,
            max_tokens=prov["questioner_probe"]["max_tokens"][comparator],
            suites=suites,
        )
        _print_token_cap(emit, token_cap)
    emit("")
    emit(f"  interval: {INTERVAL_STATEMENT}")
    emit(
        f"  DECIDED = every {max(resamples)}-resample interval at seeds {list(boot_seeds)} excludes "
        "0 (the gate); the sign test, its floor and Holm are a reported column, not a gate"
    )

    # ---- attribution of trained runs to a declared seed
    by_model = {model: label for label, model, _ in trained}
    label_of: dict[int, str | None] = {}
    for r in records:
        if r.get("arm_id") != t_arm and (r.get("key") or [None] * 3)[KEY_ARM] != t_arm:
            continue
        m = _questioner_model(r)
        if m is not None:
            label_of[id(r)] = by_model.get(m)
        elif legacy and len(trained) == 1:
            label_of[id(r)] = trained[0][0]
        else:
            label_of[id(r)] = None

    tool_types = fork_paired.load_tool_types()

    # ---- cells
    cells: list[dict[str, Any]] = []
    # amendment 8's exploratory block reads the SAME pairs and is kept APART from the cells: the
    # primary's reading cannot move with it
    aq_cells: list[dict[str, Any]] = []
    for suite in suites:
        for variant in variants:
            recs_sv = [r for r in in_design if _suite(r) == suite and _variant(r) == variant]
            # the comparator by arm AND pin: the prompted reference shares its arm_id
            c_all = [
                r
                for r in recs_sv
                if (r.get("key") or [None] * 3)[KEY_ARM] == c_arm
                and _questioner_model(r) in (comparator, None)
            ]
            c_ok = [r for r in c_all if r.get("status") == "ok" and r.get("arm_id") == c_arm]
            for label, model, prefix in trained:
                t_all = [
                    r
                    for r in recs_sv
                    if (r.get("key") or [None] * 3)[KEY_ARM] == t_arm
                    and label_of.get(id(r)) == label
                ]
                t_ok = [r for r in t_all if r.get("status") == "ok"]
                try:
                    pairs, n_unpaired, n_not_forks = _pair_forks(
                        c_ok + t_ok, treatment=t_arm, control=c_arm
                    )
                except ValueError as exc:  # a second run of one arm at one fork key
                    raise Refusal(f"{suite}/{variant}/{label}: {exc}") from exc
                mixed = [k for k, v in pairs.items() if v[t_arm]["task_id"] != v[c_arm]["task_id"]]
                if mixed:
                    raise Refusal(
                        f"{suite}/{variant}/{label}: {len(mixed)} pairs join two task_ids, e.g. "
                        f"{mixed[0]}"
                    )
                cells.append(
                    _read_cell(
                        suite=suite,
                        variant=variant,
                        label=label,
                        model=model,
                        recs_sv=recs_sv,
                        t_all=t_all,
                        c_all=c_all,
                        c_ok=c_ok,
                        t_ok=t_ok,
                        pairs=pairs,
                        n_unpaired=n_unpaired,
                        n_not_forks=n_not_forks,
                        t_arm=t_arm,
                        c_arm=c_arm,
                        tool_types=tool_types,
                        resamples=resamples,
                        boot_seeds=boot_seeds,
                        printed=printed,
                        cap=args.cap,
                        planned_c=planned.get((suite, variant, args.comparator_model), {}),
                        planned_t=planned.get((suite, variant, model), {}),
                        stamp=emit.stamp,
                        gold=gold,
                    )
                )
                cells[-1].update(contrast=contrast_text, comparator_model=comparator)
                if args.exploratory_ask_quality:
                    aq_cells.append(
                        {
                            "suite": suite,
                            "variant": variant,
                            "seed_label": label,
                            "contrast": contrast_text,
                            "label": EXPLORATORY_LABEL,
                            "measures": ask_quality_measures(
                                pairs,
                                {k: str(v[c_arm].get("task_id")) for k, v in pairs.items()},
                                t_arm,
                                c_arm,
                                printed,
                            ),
                        }
                    )

    # A CELL OF ZEROS IS NOT A CELL. With no pairs every count prints as 0 and reads as a finding.
    empty = [f"{c['suite']}/{c['variant']}/{c['seed_label']}" for c in cells if not c["n_pairs"]]
    if empty:
        raise Refusal(
            f"zero pairs in {len(empty)} declared cell(s): {', '.join(empty)} -- a declared cell "
            "with nothing in it would print as a result"
        )

    # ---- seeds-pooled lines: seeds combined within a task, then the bootstrap over tasks
    pooled = [
        _pooled_line(
            [c for c in cells if c["suite"] == suite and c["variant"] == variant],
            suite=suite,
            variant=variant,
            rule=args.seed_pooling,
            resamples=resamples,
            boot_seeds=boot_seeds,
            printed=printed,
            stamp=emit.stamp,
        )
        for suite in suites
        for variant in variants
    ]

    # ---- Holm over the declared primary family
    if args.holm_family == "suite_variant_seed":
        families = {"all": list(range(len(cells)))}
    else:
        families = {v: [i for i, c in enumerate(cells) if c["variant"] == v] for v in variants}
    holm_out: dict[str, Any] = {
        "contrast": contrast_text,
        "family_rule": args.holm_family,
        "family": [],
        "families": {},
    }
    for fam, idxs in families.items():
        members = []
        for block in ("primary", "guard"):  # the same family, a Holm column for each endpoint
            adj = holm([cells[i][block]["task_level"]["sign_test_p"] for i in idxs])
            for i, a in zip(idxs, adj):
                tl = cells[i][block]["task_level"]
                tl["holm_p"] = a
                tl["criteria"] = criteria_note(tl["decided"], a)
        for i in idxs:
            members.append(f"{cells[i]['suite']}/{cells[i]['variant']}/{cells[i]['seed_label']}")
        holm_out["families"][fam] = members
        holm_out["family"] += members

    # ---- print
    for c in cells:
        _print_cell(emit, c, wording_note(stopping, c["suite"], c["variant"], [c["seed_label"]]))
    emit("")
    emit(
        f"  ===== SEEDS-POOLED (seed pooling {args.seed_pooling}: training seeds combined WITHIN "
        "the task, then the bootstrap over tasks; not in the Holm family) ====="
    )
    for p in pooled:
        emit(
            f"    --- {p['suite']} / {p['variant']} / seeds-pooled {'+'.join(p['seed_labels'])}: "
            f"{p['n_tasks_in_every_seed']} tasks in every seed, "
            f"{p['n_tasks_in_fewer_seeds']} in fewer, "
            f"{p['n_tasks_unbalanced_across_seeds']} unbalanced across seeds ---"
        )
        note = wording_note(stopping, p["suite"], p["variant"], p["seed_labels"])
        if note:
            emit(f"      {note}")
        emit(f"      PRIMARY {PRIMARY_ENDPOINT}:")
        _print_task_stats(emit, p["task_level"])
        for which, block in (("PRIMARY", p), ("CO-PRIMARY GUARD", p["guard"])):
            alt = block["alternative"]
            if which == "CO-PRIMARY GUARD":
                emit(f"      CO-PRIMARY GUARD {GUARD_ENDPOINT}:")
                _print_task_stats(emit, block["task_level"])
                _print_ni(emit, p["headline"]["success_ni"])
                _print_tool_cap(emit, block["refused_tool_call_units"], pooled=True)
            if p["n_tasks_unbalanced_across_seeds"]:
                emit(
                    f"      {which}: the two pooling rules DIFFER here: under "
                    f"{alt['seed_pooling']} the mean delta is {alt['mean_delta']:+.4f} "
                    f"(split {alt['split']})"
                )
        emit(f"      HEADLINE (seeds-pooled): {p['headline']['text']}")
        emit("      SECONDARY, seeds-pooled within task (printed interval only):")
        for name, e in p["secondary"].items():
            if e is None:
                emit(f"        [{name}] NOT RECORDED in these records -- not zero")
                continue
            form = pooled_secondary_form(e)
            if "mean_delta" not in e:
                emit(f"        [{name}] {form}: defined on no pair")
                continue
            emit(
                f"        [{name}] {form}: paired delta={e['mean_delta']:+.4f} "
                f"{_fmt_iv(e['printed_interval'])}  split {e['split']}"
            )
    aq_block = None
    if args.exploratory_ask_quality:
        aq_block = exploratory_ask_quality(
            aq_cells,
            in_design,
            suites=suites,
            variants=variants,
            rule=args.seed_pooling,
            printed=printed,
            contrast=contrast_text,
            reference=reference,
            comparator=comparator,
            c_arm=c_arm,
        )
        _print_ask_quality(emit, aq_block)
    ref_contrast = None
    if reference is not None:
        ref_contrast = reference_contrast(
            in_design,
            suites=suites,
            variants=variants,
            reference=reference,
            comparator=comparator,
            c_arm=c_arm,
            printed=printed,
        )
        _print_reference(emit, ref_contrast)
    emit("")
    for fam, members in holm_out["families"].items():
        emit(
            f"  Holm family ({len(members)} cells, rule {args.holm_family}"
            + ("" if fam == "all" else f", variant {fam}")
            + f"; contrast {contrast_text}): {', '.join(members)}"
        )
    for c in cells:
        for which, block in (("primary", "primary"), ("guard", "guard")):
            tl = c[block]["task_level"]
            emit(
                f"    {c['suite']}/{c['variant']}/{c['seed_label']} {which}: "
                f"sign p={_fmt_p(tl['sign_test_p'])}  Holm-adjusted={_fmt_p(tl['holm_p'])}  "
                f"interval {'DECIDED' if tl['decided'] else 'UNDECIDED'}  "
                f"criteria: {tl['criteria']}"
            )

    run_ids = sorted(str(r.get("run_id") or "") for r in records)
    return {
        "stamp": emit.stamp,
        "reader": "scripts/tau2_concordance/transfer_rerun.py",
        "rules_doc": RULES_DOC,
        "rules": {
            "treatment_arm": t_arm,
            "control_arm": c_arm,
            "comparator_model": args.comparator_model,
            "contrast": contrast_text,
            "comparator_store": cstore,
            "extra_pin_driver": {
                "path": str(DRIVER_PATH),
                "sha256": extra_pin_driver().driver_sha256(),
            }
            if cstore
            else None,
            "reference_model": reference,
            "token_cap_rule": f"questioner calls with completion_tokens >= "
            f"{float(TOKEN_CAP_FRACTION):g} x max_tokens (questioner probe record); a share above "
            f"{TOKEN_CAP_LIMITATION:g} is {TOKEN_CAP_LIMITATION_TEXT}",
            "trained": [{"label": a, "model": b, "sha_prefix": p} for a, b, p in trained],
            "suites": list(suites),
            "variants": list(variants),
            "cap": args.cap,
            "reward_field": ".".join(REWARD_PATH),
            "primary": f"{PRIMARY_ENDPOINT} (fork_report.follow_ups: n_user_turns - "
            "n_prefix_user_turns), trained minus comparator; a negative delta is fewer turns",
            "co_primary_guard": f"{GUARD_ENDPOINT} (fork_paired.succeeded: tau_reward > 0), "
            f"non-inferior at {SUCCESS_NI_MARGIN:+.2f}",
            "success_ni_margin": SUCCESS_NI_MARGIN,
            "headline_rule": f"'{HEADLINE_YES}' iff the follow-up delta is DECIDED negative and "
            "every deciding success lower bound is above the margin",
            "retrieval_k": RETRIEVAL_K,
            "gold_object_definition": A7_DEFINITION,
            "pairing": "pi_eval.fork_report._pair_forks per (suite, variant, trained seed)",
            "unit": "task",
            "boot_resamples": list(resamples),
            "boot_seeds": list(boot_seeds),
            "printed_interval": {"n_boot": printed[0], "seed": printed[1]},
            "seed_pooling": args.seed_pooling,
            "interval_statement": INTERVAL_STATEMENT,
            "alpha": ALPHA,
            "holm_family": args.holm_family,
            "budget_enforcement_required": BUDGET_ENFORCEMENT_REQUIRED,
        },
        "disabled_checks": disabled,
        "source": src,
        "provenance": {
            **prov,
            "n_records": len(records),
            "n_ok": len(ok),
            "code_version_counts": dict(cvs),
            "run_ids": run_ids,
            "run_ids_sha256": hashlib.sha256("\n".join(run_ids).encode()).hexdigest(),
        },
        "cells": cells,
        "pooled": pooled,
        "stopping": stopping,
        "status_counts": status_counts,
        "holm": holm_out,
        "questioner_token_cap": token_cap,
        "reference_contrast": ref_contrast,
        "exploratory_ask_quality": aq_block,
    }


def _pooled_secondary(
    cs: Sequence[Mapping[str, Any]],
    name: str,
    rule: str,
    printed: tuple[int, int],
    get: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any] | None:
    """One secondary pooled across seeds by the primary's rule, over DEFINED pairs: under pair_mean
    each seed weighs by its defined pairs. The form and the per-seed counts travel with it. `get`
    reads the per-cell entry from somewhere other than `cell["secondary"][name]`."""
    es = [(get or (lambda c: c["secondary"].get(name)))(c) for c in cs]
    if any(e is None for e in es):
        return None
    out: dict[str, Any] = {
        "pooling": rule,
        "n_pairs": {c["seed_label"]: e["n_pairs"] for c, e in zip(cs, es)},
        "n_defined_pairs": {c["seed_label"]: e["n_defined_pairs"] for c, e in zip(cs, es)},
        "defined_on_all_units": all(e["defined_on_all_units"] for e in es),
    }
    labels = {e.get("defined_label") for e in es}
    if out["defined_on_all_units"] and len(labels) == 1 and None not in labels:
        out["defined_label"] = labels.pop()
    if any(e["n_defined_pairs"] for e in es):
        out.update(
            diff_stats(combine_within_task([e["task_pair_deltas"] for e in es], rule), printed)
        )
    return out


def _pooled_levels(es: Sequence[Mapping[str, Any]], rule: str) -> tuple[float, float]:
    """Each arm's level pooled by THE SAME RULE as the delta, then averaged over tasks, so the
    first minus the second is the pooled mean_delta (on the same defined pairs)."""
    levels = []
    for i in (0, 1):
        per_task = combine_within_task(
            [{t_: [xy[i] for xy in xs] for t_, xs in e["task_pair_values"].items()} for e in es],
            rule,
        )
        levels.append(statistics.fmean(per_task.values()))
    return levels[0], levels[1]


def pooled_secondary_form(e: Mapping[str, Any]) -> str:
    """How a pooled secondary was formed: its pooling rule and either its definition label or its
    defined pairs per seed."""
    if e["defined_on_all_units"]:
        return f"{e['pooling']}, {e.get('defined_label') or 'defined on all units'}"
    return f"{e['pooling']} over defined pairs: " + ", ".join(
        f"{lab} {e['n_defined_pairs'][lab]} of {e['n_pairs'][lab]}" for lab in e["n_pairs"]
    )


def _pooled_endpoint(
    cs: Sequence[Mapping[str, Any]],
    block: str,
    rule: str,
    resamples: Sequence[int],
    boot_seeds: Sequence[int],
    printed: tuple[int, int],
) -> dict[str, Any]:
    """One endpoint's seeds-pooled reading: seeds combined within each task by `rule`, the other
    rule carried beside it."""
    by_task: dict[str, list[list[float]]] = defaultdict(list)
    for c in cs:
        for task, ds in c[block]["task_pair_deltas"].items():
            by_task[task].append(list(ds))
    seed_mean = {t: statistics.fmean(statistics.fmean(d) for d in v) for t, v in by_task.items()}
    pair_mean = {t: statistics.fmean(x for d in v for x in d) for t, v in by_task.items()}
    chosen, other = (seed_mean, pair_mean) if rule == "seed_mean" else (pair_mean, seed_mean)
    other_vals = list(other.values())
    o_up, o_down = sum(x > 0 for x in other_vals), sum(x < 0 for x in other_vals)
    return {
        "task_level": task_stats(chosen, resamples, boot_seeds, printed),
        "alternative": {
            "seed_pooling": SEED_POOLINGS[1] if rule == SEED_POOLINGS[0] else SEED_POOLINGS[0],
            "mean_delta": statistics.fmean(other_vals) if other_vals else float("nan"),
            "split": split_of(o_up, o_down, len(other_vals) - o_up - o_down),
            "task_deltas": dict(sorted(other.items())),
        },
        "by_task": by_task,
    }


def _pooled_line(
    cs: Sequence[Mapping[str, Any]],
    *,
    suite: str,
    variant: str,
    rule: str,
    resamples: Sequence[int],
    boot_seeds: Sequence[int],
    printed: tuple[int, int],
    stamp: str | None,
) -> dict[str, Any]:
    """The seeds-pooled line of one (suite, variant): training seeds combined within each task.

    `pair_mean` (default, Table 1's arithmetic) averages every seed's pairs in the task; `seed_mean`
    averages each seed's task delta. The declared one is read, the other is carried, for the
    primary and for the guard alike.
    """
    prim = _pooled_endpoint(cs, "primary", rule, resamples, boot_seeds, printed)
    guard = _pooled_endpoint(cs, "guard", rule, resamples, boot_seeds, printed)
    guard["refused_tool_call_units"] = tc = _pooled_secondary(
        cs, A9_REFUSED_UNITS, rule, printed, get=lambda c: c["guard"].get("refused_tool_call_units")
    )
    if tc is not None:
        es = [c["guard"]["refused_tool_call_units"] for c in cs]
        for k in ("trained_units_refused", "control_units_refused"):
            tc[k] = {c["seed_label"]: e[k] for c, e in zip(cs, es)}
        if "mean_delta" in tc:
            tc["trained_level"], tc["control_level"] = _pooled_levels(es, rule)
    by_task = prim.pop("by_task")
    guard.pop("by_task")
    unbalanced = sum(
        1 for v in by_task.values() if len(v) < len(cs) or len({len(d) for d in v}) > 1
    )
    return {
        "suite": suite,
        "variant": variant,
        "seed_labels": [c["seed_label"] for c in cs],
        "seed_pooling": rule,
        "label": f"seeds-pooled ({rule}): training seeds combined within the task",
        "n_tasks_in_every_seed": sum(len(v) == len(cs) for v in by_task.values()),
        "n_tasks_in_fewer_seeds": sum(len(v) < len(cs) for v in by_task.values()),
        "n_tasks_unbalanced_across_seeds": unbalanced,
        "task_level": prim["task_level"],
        "alternative": prim["alternative"],
        "guard": guard,
        "headline": headline(prim["task_level"], guard["task_level"]),
        "secondary": {
            name: _pooled_secondary(cs, name, rule, printed)
            for name in (cs[0]["secondary"] if cs else {})
        },
        "stamp": stamp,
    }


def _endpoint_block(
    name: str,
    fn: Callable[[Mapping], float],
    *,
    pairs: Mapping[tuple, Mapping],
    task_of: Mapping[tuple, str],
    t_arm: str,
    c_arm: str,
    resamples: Sequence[int],
    boot_seeds: Sequence[int],
    printed: tuple[int, int],
    slot_task: Mapping[tuple, str],
    t_by: Mapping[tuple, Mapping],
    c_by: Mapping[tuple, Mapping],
    fills: Mapping[str, tuple[float, float]],
    binary: bool,
) -> dict[str, Any]:
    """One endpoint read by the declared machinery: pair deltas, the task unit, the ladder, and the
    worst-case imputation of every incomplete slot (`fills`: trained value, comparator value)."""
    deltas = {k: fn(v[t_arm]) - fn(v[c_arm]) for k, v in pairs.items()}
    task_pair_deltas: dict[str, list[float]] = defaultdict(list)
    for k in sorted(deltas, key=repr):
        task_pair_deltas[task_of[k]].append(deltas[k])
    pair_level: dict[str, Any] = {
        "n_pairs": len(pairs),
        "mean_delta": statistics.fmean(deltas.values()) if deltas else float("nan"),
    }
    if binary:
        vals = [(fn(v[t_arm]), fn(v[c_arm])) for v in pairs.values()]
        pair_level.update(
            both=sum(1 for a, b in vals if a and b),
            trained_only=sum(1 for a, b in vals if a and not b),
            control_only=sum(1 for a, b in vals if b and not a),
            neither=sum(1 for a, b in vals if not a and not b),
        )
    sensitivity: dict[str, Any] = {}
    incomplete = [x for x in slot_task if not (x in t_by and x in c_by)]
    if incomplete:
        for label, (t_fill, c_fill) in fills.items():
            imp = {
                x: (fn(t_by[x]) if x in t_by else t_fill) - (fn(c_by[x]) if x in c_by else c_fill)
                for x in slot_task
            }
            sensitivity[label] = {
                "n_imputed_slots": len(incomplete),
                "imputation_range": [min(t_fill, c_fill), max(t_fill, c_fill)],
                "task_level": task_stats(
                    _task_means(imp, slot_task), (max(resamples),), boot_seeds, printed
                ),
            }
    return {
        "endpoint": name,
        "task_level": task_stats(_task_means(deltas, task_of), resamples, boot_seeds, printed),
        "task_pair_deltas": dict(task_pair_deltas),
        "pair_level_NOT_QUOTABLE": pair_level,
        "sensitivity": sensitivity,
    }


def _handoffs(t_ok: Sequence[Mapping], c_ok: Sequence[Mapping]) -> dict[str, Any] | None:
    if any(r.get(HANDOFF_FIELD) is None for r in [*t_ok, *c_ok]):
        return None
    return {
        "trained": {"total": sum(int(r[HANDOFF_FIELD]) for r in t_ok), "n_units": len(t_ok)},
        "control": {"total": sum(int(r[HANDOFF_FIELD]) for r in c_ok), "n_units": len(c_ok)},
    }


def _read_cell(
    *,
    suite: str,
    variant: str,
    label: str,
    model: str,
    recs_sv: Sequence[Mapping],
    t_all: Sequence[Mapping],
    c_all: Sequence[Mapping],
    c_ok: Sequence[Mapping],
    t_ok: Sequence[Mapping],
    pairs: Mapping[tuple, Mapping],
    n_unpaired: int,
    n_not_forks: int,
    t_arm: str,
    c_arm: str,
    tool_types: Mapping,
    resamples: Sequence[int],
    boot_seeds: Sequence[int],
    printed: tuple[int, int],
    cap: int,
    planned_c: Mapping[tuple, str],
    planned_t: Mapping[tuple, str],
    stamp: str | None,
    gold: "gold_objects.GoldObjects | None" = None,
) -> dict[str, Any]:
    task_of = {k: str(v[c_arm].get("task_id")) for k, v in pairs.items()}

    # ---- census: every slot any run of either arm occupies, whatever its status (the comparator
    # by arm and pin, as the caller selected it)
    slot_task: dict[tuple, str] = {}
    for r in list(c_all) + [r for r in recs_sv if (r.get("key") or [None] * 3)[KEY_ARM] == t_arm]:
        if (r.get("key") or [None] * 8)[KEY_TRACE]:
            slot_task[_slot(r)] = _task_of_key(r)
    # PLANNED SLOTS the store never saw join the census: a unit that wrote no status at all is
    # missing, not absent from the design.
    seen = set(slot_task)
    planned_slots: dict[tuple, str] = {}
    for plan in (planned_c, planned_t):
        for s, task in plan.items():
            if planned_slots.get(s, task) != task or slot_task.get(s, task) != task:
                raise Refusal(f"{suite}/{variant}/{label}: slot {s} is planned under two task_ids")
            planned_slots[s] = task
    for s, task in planned_slots.items():
        slot_task.setdefault(s, task)
    t_by = {_slot(r): r for r in t_ok}
    c_by = {_slot(r): r for r in c_ok}
    census = {"n_slots": len(slot_task), "complete": 0}
    census.update(
        n_planned_slots=len(planned_slots),
        n_absent_from_store=len(set(planned_slots) - seen),
    )
    census.update(trained_missing=0, control_missing=0, both_missing=0)
    for s in slot_task:
        ht, hc = s in t_by, s in c_by
        if ht and hc:
            census["complete"] += 1
        elif hc:
            census["trained_missing"] += 1
        elif ht:
            census["control_missing"] += 1
        else:
            census["both_missing"] += 1
    non_ok = {
        "trained": dict(Counter(str(r.get("status")) for r in t_all if r.get("status") != "ok")),
        "control": dict(Counter(str(r.get("status")) for r in c_all if r.get("status") != "ok")),
    }

    # ---- the PRIMARY (follow-ups) and the CO-PRIMARY GUARD (success), one machinery. Worst cases:
    # success at its bounds (0, 1); follow-ups at the range the cell observed, since fewer
    # follow-ups is the direction a trained arm would want.
    def success(r: Mapping) -> float:
        return float(fork_paired.succeeded(r))

    fu_seen = [float(follow_ups(r)) for r in [*t_ok, *c_ok]] or [0.0]
    lo, hi = min(fu_seen), max(fu_seen)
    common = dict(
        pairs=pairs,
        task_of=task_of,
        t_arm=t_arm,
        c_arm=c_arm,
        resamples=resamples,
        boot_seeds=boot_seeds,
        printed=printed,
        slot_task=slot_task,
        t_by=t_by,
        c_by=c_by,
    )
    primary = _endpoint_block(
        PRIMARY_ENDPOINT,
        lambda r: float(follow_ups(r)),
        fills={"worst_for_trained": (hi, lo), "worst_for_comparator": (lo, hi)},
        binary=False,
        **common,
    )
    guard = _endpoint_block(
        GUARD_ENDPOINT,
        success,
        fills={"worst_for_trained": (0.0, 1.0), "worst_for_comparator": (1.0, 0.0)},
        binary=True,
        **common,
    )

    secondary = _secondary(
        pairs, task_of, t_arm, c_arm, tool_types, cap, printed, gold=gold, suite=suite
    )
    # amendment 9 lives BESIDE THE GUARD, not among the secondaries
    guard["refused_tool_call_units"] = secondary.pop(A9_REFUSED_UNITS, None)
    rb_t = _refused_budget(t_ok)
    rb_c = _refused_budget(c_ok)
    return {
        "suite": suite,
        "variant": variant,
        "seed_label": label,
        "trained_model": model,
        "stamp": stamp,
        "n_runs_trained_ok": len(t_ok),
        "n_runs_control_ok": len(c_ok),
        "n_pairs": len(pairs),
        "n_unpaired": n_unpaired,
        "n_not_forks": n_not_forks,
        "census": census,
        "non_ok": non_ok,
        "primary": primary,
        "guard": guard,
        "headline": headline(primary["task_level"], guard["task_level"]),
        "handoffs": _handoffs(t_ok, c_ok),
        "secondary": secondary,
        "n_refused_budget": {"trained": rb_t, "control": rb_c},
        "run_ids": sorted(str(v[a].get("run_id")) for v in pairs.values() for a in (t_arm, c_arm)),
    }


def wording_note(
    stopping: Sequence[Mapping], suite: str, variant: str, labels: Sequence[str]
) -> str | None:
    """The asks-capped counts for this cell, printed beside its primary line at EVERY share.
    Descriptive: amendment 4's shared-budget wording rule is superseded by amendment 7's separate
    budgets, and the aid says so rather than inviting shared-budget wording for this run."""
    rows = {r["arm"]: r for r in stopping if r["suite"] == suite and r["variant"] == variant}
    if "comparator" not in rows or not all(label in rows for label in labels):
        return None
    cx, cn = rows["comparator"]["n_asks_capped"], rows["comparator"]["n_asks_capped_defined"]
    qx = sum(rows[label]["n_asks_capped"] for label in labels)
    qn = sum(rows[label]["n_asks_capped_defined"] for label in labels)

    def count(x: int, n: int) -> str:
        return f"{x}/{n}" if n else "NOT RECORDED"

    return (
        f"{A4_STARVED}: comparator {count(cx, cn)}, questioner {count(qx, qn)} {WORDING_AID_TAIL}"
    )


def _print_sensitivity(emit: Emit, sens: Mapping[str, Any]) -> None:
    if not sens:
        emit(
            "        sensitivity: no missing or non-ok slot in this cell; imputation changes nothing"
        )
    for name, s in sens.items():
        tl = s["task_level"]
        ivs = "  ".join(f"seed {iv['seed']}: {_fmt_iv(iv)}" for iv in tl["intervals"])
        emit(
            f"        WORST CASE ({name.replace('_', ' ')}, {s['n_imputed_slots']} slot(s) imputed "
            f"within {s['imputation_range']}): split {tl['split']}  "
            f"mean delta={tl['mean_delta']:+.4f}  sign p={_fmt_p(tl['sign_test_p'])}  "
            f"{tl['deciding_n_boot']} resamples {ivs}  "
            f"{'DECIDED' if tl['decided'] else 'UNDECIDED'}"
        )


def _print_tool_cap(emit: Emit, e: Mapping[str, Any] | None, pooled: bool = False) -> None:
    """Amendment 9, beside the success guard: where the tool cap bound, per arm and paired."""
    if e is None:
        emit(f"        [{A9_REFUSED_UNITS}] NOT RECORDED -- not zero")
        return
    if "mean_delta" not in e:
        emit(f"        [{A9_REFUSED_UNITS}] defined on no pair")
        return
    iv = e["printed_interval"]
    if pooled:

        def per_seed(k: str) -> str:
            return ", ".join(f"{s} {e[k][s]}/{e['n_pairs'][s]}" for s in e["n_pairs"]) + " units"

        head = (
            f"seeds-pooled {pooled_secondary_form(e)}: trained {e['trained_level']:.4f} "
            f"({per_seed('trained_units_refused')}), comparator {e['control_level']:.4f} "
            f"({per_seed('control_units_refused')})"
        )
    else:
        head = (
            f"trained {e['trained_units_refused']}/{e['n_pairs']} units, comparator "
            f"{e['control_units_refused']}/{e['n_pairs']} units"
        )
    emit(
        f"        [{A9_REFUSED_UNITS}] {head}; paired delta={e['mean_delta']:+.4f} {_fmt_iv(iv)} "
        f"({iv['n_boot']}, seed {iv['seed']})  split {e['split']} -- descriptive: where the tool "
        "cap bound"
    )


def _print_ni(emit: Emit, ni: Mapping[str, Any]) -> None:
    lows = ", ".join(_fmt_p(x) if x is None else f"{x:+.4f}" for x in ni["lower_bounds"])
    emit(
        f"        success NI guard at {ni['margin']:+.2f}: {ni['n_boot']}-resample lower bounds "
        f"[{lows}] -> {'PASS' if ni['passes'] else 'FAIL'}"
    )


def _print_cell(emit: Emit, c: Mapping, note: str | None = None) -> None:
    emit("")
    emit(
        f"  --- {c['suite']} / {c['variant']} / {c['seed_label']} ({c['trained_model']}) "
        f"vs prompted {short_model(c['comparator_model'])}: {c['n_pairs']} pairs "
        f"({c['n_unpaired']} unpaired, "
        f"{c['n_not_forks']} non-fork) ---"
    )
    cen = c["census"]
    emit(
        f"      census: {cen['n_slots']} slots seen; {cen['complete']} complete; "
        f"trained missing at {cen['trained_missing']} slot(s); "
        f"comparator missing at {cen['control_missing']} slot(s); "
        f"both missing at {cen['both_missing']} slot(s); planned {cen['n_planned_slots']}, "
        f"of which {cen['n_absent_from_store']} never reached the store"
    )
    emit(
        f"      non-ok units: trained {c['non_ok']['trained'] or 'none'}; "
        f"comparator {c['non_ok']['control'] or 'none'}"
    )
    ho = c["handoffs"]
    emit(
        f"      hand-offs ({HANDOFF_FIELD}): "
        + (
            "NOT RECORDED -- not zero"
            if ho is None
            else f"trained {ho['trained']['total']} over {ho['trained']['n_units']} units, "
            f"comparator {ho['control']['total']} over {ho['control']['n_units']} units"
        )
    )
    p = c["primary"]
    pl = p["pair_level_NOT_QUOTABLE"]
    emit(
        f"      PRIMARY {PRIMARY_ENDPOINT} -- pair level, NOT QUOTABLE: {pl['n_pairs']} pairs, "
        f"pair mean delta={pl['mean_delta']:+.4f}"
    )
    emit(
        f"      PRIMARY {PRIMARY_ENDPOINT} -- task level (the unit), trained minus comparator "
        "(a negative delta is fewer follow-up turns):" + (f"  {note}" if note else "")
    )
    _print_task_stats(emit, p["task_level"])
    _print_sensitivity(emit, p["sensitivity"])
    g = c["guard"]
    gl = g["pair_level_NOT_QUOTABLE"]
    emit(
        f"      CO-PRIMARY GUARD {GUARD_ENDPOINT} -- pair level, NOT QUOTABLE: both={gl['both']} "
        f"trained_only={gl['trained_only']} control_only={gl['control_only']} "
        f"neither={gl['neither']}  pair mean delta={gl['mean_delta']:+.4f}"
    )
    emit(f"      CO-PRIMARY GUARD {GUARD_ENDPOINT} -- task level, trained minus comparator:")
    _print_task_stats(emit, g["task_level"])
    _print_ni(emit, c["headline"]["success_ni"])
    _print_tool_cap(emit, g["refused_tool_call_units"])
    _print_sensitivity(emit, g["sensitivity"])
    emit(f"      HEADLINE: {c['headline']['text']}")
    emit("      SECONDARY (task level, per run, never per-turn; not in the Holm family):")
    for name, s in c["secondary"].items():
        if s is None:
            emit(f"        [{name}] NOT RECORDED in these records -- not zero")
            continue
        defined = (
            s.get("defined_label") or "defined on all units"
            if s["defined_on_all_units"]
            else f"defined on {s['n_defined_pairs']} of {s['n_pairs']} pairs"
        )
        if not s["n_defined_pairs"]:
            if "dropped_units" in s:
                du = s["dropped_units"]
                emit(
                    f"        [{name}] ({defined}): trained asked nothing: "
                    f"{du['trained_asked_nothing']} unit(s) dropped; comparator asked nothing: "
                    f"{du['comparator_asked_nothing']} unit(s) dropped; no pre-dialogue gold "
                    f"object: {s['dropped_pairs_no_gold_object']} pair(s) dropped; asks in tasks "
                    f"without gold objects: trained {s['trained_asks_in_tasks_without_gold']}, "
                    f"comparator {s['control_asks_in_tasks_without_gold']}"
                )
                continue
            emit(
                f"        [{name}] ({defined})"
                + (
                    f"  quartiles, censored: trained {s['trained_quartiles']} "
                    f"({s['trained_n_censored']} censored) comparator {s['control_quartiles']} "
                    f"({s['control_n_censored']} censored)"
                    if "trained_quartiles" in s
                    else ""
                )
            )
            continue
        emit(
            f"        [{name}] ({defined}) trained {s['trained_level']:.4f}  comparator "
            f"{s['control_level']:.4f}  paired delta={s['mean_delta']:+.4f} "
            f"{_fmt_iv(s['printed_interval'])} ({s['printed_interval']['n_boot']}, seed "
            f"{s['printed_interval']['seed']})  split {s['split']}  "
            f"sign p={_fmt_p(s['sign_test_p'])} floor={_fmt_p(s['sign_test_floor'])}"
            + (
                f"  quartiles, censored: trained {s['trained_quartiles']} "
                f"({s['trained_n_censored']} censored) comparator {s['control_quartiles']} "
                f"({s['control_n_censored']} censored)"
                if "trained_quartiles" in s
                else ""
            )
        )
        if "dropped_units" in s:
            du = s["dropped_units"]
            emit(
                f"        [{name}] trained asked nothing: {du['trained_asked_nothing']} unit(s) "
                f"dropped; comparator asked nothing: {du['comparator_asked_nothing']} unit(s) "
                f"dropped; no pre-dialogue gold object: {s['dropped_pairs_no_gold_object']} "
                f"pair(s) dropped; share of asks hit: trained "
                f"{_fmt_p(s['trained_share_of_asks_hit'])} comparator "
                f"{_fmt_p(s['control_share_of_asks_hit'])} (asks in tasks without gold objects: "
                f"trained {s['trained_asks_in_tasks_without_gold']}, comparator "
                f"{s['control_asks_in_tasks_without_gold']}) -- read together, never alone, with "
                f"[{A7_RECALL}] and [{A7_ASKS}]"
            )
        d = s.get("descriptive_conditional_on_both_acting")
        if d is not None:
            emit(
                f"        [{name}] {d['label']}: {d['n_pairs']} pair(s)"
                + (
                    f", task-level mean delta {d['mean_delta']:+.4f} over {len(d['task_deltas'])} "
                    f"task(s); quartiles acting only: trained {s['trained_quartiles_acting_only']} "
                    f"comparator {s['control_quartiles_acting_only']}"
                    if d["n_pairs"]
                    else "; no pair where both arms acted"
                )
            )
    for arm in ("trained", "control"):
        rb = c["n_refused_budget"][arm]
        if rb is None:
            emit(f"        [{arm}] {N_REFUSED_BUDGET_FIELD} NOT RECORDED -- not zero")
        else:
            emit(
                f"        [{arm}] {N_REFUSED_BUDGET_FIELD}: total={rb['total']:.0f} over "
                f"{rb['n_runs']} runs, {rb['runs_with_any']} with any"
            )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--records", help="extract_records.py JSON array")
    src.add_argument("--store", help="a runs root, read through extract_records.extract")
    ap.add_argument("--treatment", default=TREATMENT_ARM)
    ap.add_argument("--control", default=CONTROL_ARM)
    ap.add_argument("--comparator-model", default=COMPARATOR_MODEL)
    ap.add_argument(
        "--comparator-store",
        help="RULES amendment 11: read the comparator pin from this root (<root>/<suite>/<pin>/), "
        "paired against the trained seeds from the main source; every provenance refusal reads "
        "both stores together",
    )
    ap.add_argument(
        "--comparator-launch-record",
        action="append",
        help="the extra-pin driver's ARM_LAUNCH.json, one per suite root (repeatable); a file, a "
        "directory (the arm root) or a glob",
    )
    ap.add_argument(
        "--questioner-probe-record",
        action="append",
        help="run_extra_pin.py questioner-probe records beyond those in the comparator store's job "
        "directories (read always, one per job), repeatable; a file, a directory or a glob",
    )
    ap.add_argument(
        "--reference-model",
        default=COMPARATOR_MODEL,
        help="with --comparator-store: the prompted pin in the main source, set against the "
        "comparator as a descriptive block with no claim",
    )
    ap.add_argument(
        "--trained",
        action="append",
        help="LABEL=MODEL:SHA_PREFIX, repeatable; default "
        + ", ".join(f"{a}={b}:{c}" for a, b, c in TRAINED_SEEDS),
    )
    ap.add_argument("--suites", help="comma list; default " + ",".join(SUITES))
    ap.add_argument("--variants", help="comma list; default " + ",".join(VARIANTS))
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument(
        "--serve-record",
        action="append",
        help="SERVED.<job>.json, repeatable; a file, a directory (searched for SERVED.*.json) or "
        "a glob. Every record must agree on each adapter's sha256.",
    )
    ap.add_argument(
        "--probe-record",
        action="append",
        help="thinking_probe.json, repeatable; a file, a directory or a glob. All must PASS on one "
        "proxy_url and one config_sha256 -- or several, if the configs archived next to the "
        "records agree as RULES amendment 10 requires.",
    )
    ap.add_argument("--boot-resamples", default=",".join(str(x) for x in BOOT_RESAMPLES))
    ap.add_argument("--boot-seeds", default=",".join(str(x) for x in BOOT_SEEDS))
    ap.add_argument("--holm-family", choices=HOLM_FAMILIES, default=HOLM_FAMILY)
    ap.add_argument(
        "--printed-interval",
        default=f"{PRINTED_INTERVAL[0]}:{PRINTED_INTERVAL[1]}",
        help="N_BOOT:SEED of the interval printed beside the split",
    )
    ap.add_argument("--seed-pooling", choices=SEED_POOLINGS, default=SEED_POOLING)
    ap.add_argument(
        "--tau2-data-dir",
        help="the tau2 data root the harness reads (TAU2_DATA_DIR by default): db.json and "
        "tasks.json per domain, for the gold-object secondaries -- read here, never by a rollout",
    )
    ap.add_argument(
        "--launch-record",
        action="append",
        help="the launcher's LAUNCH.json, one per suite (repeatable): its plan.units are the "
        "planned slots, so a planned unit absent from the store counts as missing",
    )
    ap.add_argument(
        "--validation-legacy-population",
        action="store_true",
        help="read a withdrawn population to validate the reader: disables ONLY the enforcement, "
        "weights and thinking-probe refusals, and stamps every line and the JSON",
    )
    ap.add_argument(
        "--pilot",
        action="store_true",
        help="print ONLY the stopping block and the per-arm status and error counts, stamped "
        "PILOT, after every provenance refusal; no pairing, no bootstrap",
    )
    ap.add_argument(
        "--exploratory-ask-quality",
        action="store_true",
        help="RULES amendment 8's EXPLORATORY ask-quality block (share of asks retrieving a "
        "record; exact and near-exact repeat rates), per arm and paired on the primary's own "
        "pairs, labelled on every line and in the JSON; the primary reading is unchanged",
    )
    ap.add_argument("--out")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stamps = [VALIDATION_STAMP] if args.validation_legacy_population else []
    stamps += [PILOT_STAMP] if args.pilot else []
    emit = Emit(" | ".join(stamps) or None)
    try:
        report = read(args, emit)
    except Refusal as exc:
        emit(f"  REFUSING: {exc}")
        return EXIT_REFUSED
    if args.out:
        Path(args.out).write_text(json.dumps(_clean(report), indent=2, default=str))
        emit(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
