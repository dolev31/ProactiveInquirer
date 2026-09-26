"""Two tau2-only Inquirer templates, layered, and the QA prompt neither of them may touch.

WHY A VARIANT AT ALL. On the fork campaigns 60-65% of dialogues on every domain ended at the
turn cap, and the retail transfer result is carried by the tail rather than by the typical
dialogue, so how the Inquirer decides to stop is the quantity under study on tau2 and nowhere
else. The QA suites' stopping paragraph is not under study, and it is rendered by every
confirmatory run in the paper.

WHY THERE ARE NOW TWO. Commit 3052546 ("teach the Inquirer that a question reaches records,
never the customer") edited `inquirer_prompted.txt` on branch `convlog-work` and was never
merged: it is not an ancestor of HEAD, so this line of history still renders the pre-3052546
bytes. MEASURED over every tau2 fork run on disk, holding the arm and the Inquirer model
fixed (`inquirer_prompted`, claude-sonnet-5), the answered-ask rate is

    template 3006f5bf (post-3052546)   retail 9271/10620 = 0.873   airline 3560/3920 = 0.908
    template 814658ea (this history)   retail  819/ 1847 = 0.443   airline  694/1430 = 0.485

-- the policy spends its turns asking the CUSTOMER for facts that are in no record, so the
retrieval channel returns nothing. That instruction is meaningless on the QA suites, which have
no customer, and porting it into the shared template would move every held-out QA number in the
paper. So it goes into a tau2-only template instead, and the layering is:

    inquirer_prompted            the QA prompt, 814658ea, UNTOUCHED
      + the 3052546 hunk      -> inquirer_prompted_tau2_base   ROUTING ONLY  (arms 1-4)
      + the stopping hunk     -> inquirer_prompted_tau2        ROUTING + STOP (arms 5-6)

so the 5/6-vs-1/4 contrast is still exactly the stopping paragraph and nothing else.

WHAT THIS FILE HOLDS, and why each half is necessary:

  * THE QA PROMPT'S BYTES ARE PINNED. `inquirer_prompted.txt` is inside `prompt_hashes`, which
    is inside `semantic_hash`: editing one word of it moves the run_id of every QA run ever
    made, which is correct only if it is deliberate. Pinned the way `retriever_select`'s sha is
    pinned in tests/test_prompt_templates_exist.py, and for the same reason.

  * EACH LAYER DIFFERS IN EXACTLY ITS OWN HUNK. A base variant that also drifted in the output
    contract, or a stop variant that also drifted in HOW TO CHOOSE, would confound one change
    with another and nothing downstream could separate them afterwards. The base is checked
    against the QA prompt twice over -- by reconstructing the 3052546 hunk from its own diff
    and by the sha 1,159 recorded retail and 536 airline fork runs pin -- and the stop variant
    is compared to the BASE block by block over the whole file.

  * SELECTING ONE IS OPT-IN AND LABELLED. An unset `UnitSpec.prompt_variant` must render what
    the recorded campaigns rendered, and a run under a variant must be separable from them
    afterwards -- which it is, through `prompt_variant_id` (in `pinq.ids.SEMANTIC_FIELDS`) and
    through the Inquirer's own `prompt_hashes`.

No LLM, no environment and no gold: `apply_prompt_variant` takes any object carrying a
`prompt_name`, which is what the policies expose.
"""

from __future__ import annotations

import json

import pytest
from scripts.run_tau2_forks import build_specs

from pi_run.stages import tau2_runner
from pi_run.worker import UnitSpec
from pinq import promptlib
from pinq_expt.policies.prompted import PromptedInquirer

QA = "inquirer_prompted"
BASE = tau2_runner.TAU2_BASE_TEMPLATE
VARIANT = tau2_runner.TAU2_STOP_TEMPLATE

# The bytes every QA run in the paper rendered. If this moves, say so and re-run them.
QA_SHA = "814658eaa2f09eb798ab6f1019ea9ba728b68a3c00c4aa722f300e87f49b69f7"

# The bytes of `src/pinq/prompts/inquirer_prompted.txt` AT COMMIT 3052546, read out of git with
# `git show 3052546:src/pinq/prompts/inquirer_prompted.txt | shasum -a 256`. This is the
# template sha 1,159 recorded tau2_retail and 536 tau2_airline fork runs carry in their
# `prompt_hashes`, so the tau2 base variant renders byte-for-byte what produced the
# published-era answered-ask rates quoted above -- not a re-typing of it.
BASE_SHA = "3006f5bfc514c7cb53549b3c89e54653db7ca96590b2f6a729cfd1705ae571cc"

STOP_HEADING = "WHEN TO STOP"
CHOOSE_HEADING = "HOW TO CHOOSE"
RESERVED_HEADING = "RESERVED"

# --- the 3052546 hunk, transcribed from `git show 3052546 -- src/pinq/prompts/inquirer_prompted.txt`

# ADDED: a paragraph at the end of HOW TO CHOOSE.
ROUTING_PARAGRAPH = """A question reaches records, never the customer. What only the customer knows -- their email,
which order or reservation they mean, what they prefer, whether they agree -- is in no record,
so asking for it returns nothing and spends the turn. Ask only when you can name the record
holding the answer: an order, reservation or user id in the evidence or the task. If only the
customer can supply the fact, stop; the answer being drafted is what asks them.
"""

# REMOVED: the RESERVED block, which exists to be inert padding and is what pays for the
# paragraph above -- which is why the variant stays length-matched with the prompt it replaces.
RESERVED_BLOCK = """RESERVED
This block is reserved. It appears in every condition so that the amount of text surrounding the
task is the same in each. It states no requirement and expresses no preference. It carries no
fact about the task, about the evidence or about the answer. It is not a hint, and nothing in it
should be treated as one. It describes neither what to ask nor when to stop. It names no entity
and refers to no document.
"""


def _apply_3052546(qa_text: str) -> str:
    """The commit's diff for this file, re-applied as a transformation.

    Written from the diff rather than from the resulting file, so that this and `BASE_SHA`
    are two INDEPENDENT statements about the same bytes: one says "the hunk is exactly this",
    the other says "the result is exactly what 1,695 recorded runs rendered".
    """
    out = qa_text.replace(f"\n{STOP_HEADING}\n", f"\n{ROUTING_PARAGRAPH}\n{STOP_HEADING}\n", 1)
    return out.replace(f"{RESERVED_BLOCK}\n", "", 1)


def _blocks(name: str) -> dict[str, str]:
    """The template split on its ALL-CAPS headings, so every section can be compared."""
    out: dict[str, str] = {}
    head = "(preamble)"
    for line in promptlib.load(name).splitlines():
        if line.strip() and line == line.upper() and line.strip() == line and len(line) < 60:
            head = line.strip()
            out.setdefault(head, "")
            continue
        out[head] = out.get(head, "") + line + "\n"
    return out


# --------------------------------------------------------------------- the QA prompt is frozen


def test_the_qa_inquirer_prompt_bytes_are_unchanged():
    """The variants are NEW FILES precisely so this sha does not move."""
    assert promptlib.sha(QA) == QA_SHA, (
        "inquirer_prompted.txt changed. Every QA run's prompt_hashes, semantic_hash and run_id "
        "move with it, so this has to be a deliberate change with its own re-run, not a "
        "side effect of editing a tau2-only template."
    )


def test_the_shipped_policy_still_renders_the_qa_template():
    """The class default is what every non-tau2 caller gets; a variant is set per instance."""
    assert PromptedInquirer.prompt_name == QA


# ------------------------------------------- the base variant is the 3052546 hunk and nothing else


def test_the_tau2_base_variant_is_byte_identical_to_the_template_the_fork_corpus_pinned():
    """Not "carries the same instruction" -- the SAME BYTES. A re-typing would render a
    different prompt_hashes entry than the recorded runs whose answered-ask rates are the
    comparison this campaign is measured against."""
    assert promptlib.sha(BASE) == BASE_SHA


def test_the_tau2_base_variant_applies_exactly_the_3052546_hunk_to_the_qa_prompt():
    assert promptlib.load(BASE) == _apply_3052546(promptlib.load(QA))


def test_the_base_variant_adds_the_routing_paragraph_and_removes_only_the_reserved_block():
    """Block by block over the whole file, so a drift anywhere else is visible here rather
    than confounded with the routing change three stages downstream."""
    qa, base = _blocks(QA), _blocks(BASE)
    assert set(qa) - set(base) == {RESERVED_HEADING}, (sorted(qa), sorted(base))
    assert set(base) - set(qa) == set(), (sorted(qa), sorted(base))
    differing = sorted(h for h in base if qa.get(h) != base[h])
    assert differing == [CHOOSE_HEADING], (
        f"the tau2 base variant differs from the QA prompt in {differing}; the 3052546 hunk "
        f"touches {CHOOSE_HEADING!r} and deletes {RESERVED_HEADING!r}, and nothing else"
    )
    assert ROUTING_PARAGRAPH in base[CHOOSE_HEADING]
    assert qa[CHOOSE_HEADING] in base[CHOOSE_HEADING], "the original paragraph must survive"
    assert base[STOP_HEADING] == qa[STOP_HEADING], "the base variant is ROUTING ONLY"


# ----------------------------------------- the stop variant is the base plus the stopping hunk


def test_the_stop_variant_differs_from_the_tau2_base_only_in_the_stopping_block():
    """THE CONTRAST THE CAMPAIGN MEASURES. Arms 5 and 6 render this file and arms 1-4 render
    the base; if they differed anywhere but here, the stopping effect would be confounded."""
    base, var = _blocks(BASE), _blocks(VARIANT)
    assert set(base) == set(var), (sorted(base), sorted(var))
    differing = sorted(h for h in base if base[h] != var[h])
    assert differing == [STOP_HEADING], (
        f"the tau2 stop variant differs from the tau2 base in {differing}; it may differ in "
        f"{STOP_HEADING!r} and in nothing else, or the stopping change is confounded with "
        "whatever else moved"
    )
    assert var[STOP_HEADING].strip(), "the variant's stopping block is empty"


def test_the_stop_variant_also_carries_the_routing_instruction():
    """Arms 5 and 6 are routing PLUS stopping. If the routing paragraph were absent here, the
    stop arms would be running the dead channel and the contrast would be routing, not
    stopping."""
    text = promptlib.load(VARIANT)
    assert ROUTING_PARAGRAPH in text
    assert RESERVED_BLOCK not in text


def test_the_stop_variant_is_the_base_with_its_stopping_paragraph_replaced():
    base, var = promptlib.load(BASE), promptlib.load(VARIANT)
    head_b, _, tail_b = base.partition(f"\n{STOP_HEADING}\n")
    head_v, _, tail_v = var.partition(f"\n{STOP_HEADING}\n")
    assert head_b == head_v, "everything before WHEN TO STOP must be identical"
    assert tail_b.partition("\n\n")[2] == tail_v.partition("\n\n")[2], (
        "everything after the stopping paragraph must be identical"
    )
    assert tail_b.partition("\n\n")[0] != tail_v.partition("\n\n")[0]


# ------------------------------------------------------ both variants render the same state


@pytest.mark.parametrize("name", [lambda: BASE, lambda: VARIANT])
def test_a_variant_renders_the_same_holes_and_announces_no_budget(name):
    """Same placeholders, so the same state reaches it; and a stopping paragraph is exactly
    where a cap would be most tempting to mention (see promptlib's rule 3)."""
    n = name()
    assert promptlib.placeholders(n) == promptlib.placeholders(QA)
    assert not (promptlib.placeholders(n) & promptlib.FORBIDDEN_PLACEHOLDERS)
    assert "budget" not in promptlib.load(n).lower()


@pytest.mark.parametrize("name", [lambda: BASE, lambda: VARIANT])
def test_a_variant_is_length_matched_with_the_prompt_it_replaces(name):
    """Arms are compared on what the scaffolding SAYS, not on how much of it there is. The
    3052546 hunk pays for its paragraph out of RESERVED for exactly this reason."""
    n = name()
    qa, var = promptlib.tokens(QA), promptlib.tokens(n)
    assert abs(var - qa) / qa <= 0.10, (qa, var)


# --------------------------------------------------------------- selecting one, and not selecting one


class _Inq:
    prompt_name = QA


def test_no_variant_leaves_the_prompt_alone_and_labels_the_run_v1(monkeypatch):
    monkeypatch.delenv(promptlib.OVERLAY_ENV, raising=False)
    inq = _Inq()
    assert tau2_runner.apply_prompt_variant(inq, "") == "v1"
    assert inq.prompt_name == QA


@pytest.mark.parametrize(
    "variant_of,template_of",
    [
        (lambda: tau2_runner.TAU2_BASE_VARIANT, lambda: BASE),
        (lambda: tau2_runner.TAU2_STOP_VARIANT, lambda: VARIANT),
    ],
)
def test_a_variant_repoints_the_policy_and_labels_the_run_with_a_digest(variant_of, template_of):
    variant, template = variant_of(), template_of()
    inq = _Inq()
    label = tau2_runner.apply_prompt_variant(inq, variant)
    assert inq.prompt_name == template
    assert label == f"{variant}-{promptlib.sha(template)[:8]}"
    # The digest is of that template's own bytes, so editing it cannot reuse this label.
    assert not label.endswith(promptlib.sha(QA)[:8])


def test_the_two_variants_do_not_share_a_label():
    a = tau2_runner.apply_prompt_variant(_Inq(), tau2_runner.TAU2_BASE_VARIANT)
    b = tau2_runner.apply_prompt_variant(_Inq(), tau2_runner.TAU2_STOP_VARIANT)
    assert a != b
    assert a.split("-")[1] != b.split("-")[1], "two templates, two digests"


@pytest.mark.parametrize(
    "variant_of,template_of",
    [
        (lambda: tau2_runner.TAU2_BASE_VARIANT, lambda: BASE),
        (lambda: tau2_runner.TAU2_STOP_VARIANT, lambda: VARIANT),
    ],
)
def test_a_real_policy_under_a_variant_records_that_variants_hash(variant_of, template_of):
    """`prompt_hashes` is a property over `prompt_name`, so run identity follows the swap.

    THE MIDDLE ASSERTION IS THE ONE THIS TEST IS NAMED FOR. Parametrising an earlier version of
    it dropped `after[template] == sha(template)` and kept only "the QA key is gone and the dict
    changed" -- which a swap to ANY other template satisfies, including a wrong one. The
    template is parametrised alongside the variant so the pair stays checkable.
    """
    variant, template = variant_of(), template_of()
    inq = PromptedInquirer(llm=object())
    before = dict(inq.prompt_hashes)
    tau2_runner.apply_prompt_variant(inq, variant)
    after = dict(inq.prompt_hashes)
    assert before[QA] == promptlib.sha(QA)
    assert after[template] == promptlib.sha(template), (
        f"the policy must record {template}'s own bytes; run identity is downstream of this"
    )
    assert QA not in after and before != after


def test_an_unknown_variant_refuses_rather_than_running_the_shipped_prompt():
    with pytest.raises(tau2_runner.Tau2DriverError) as e:
        tau2_runner.apply_prompt_variant(_Inq(), "tau2_stopp")
    assert "unknown prompt variant" in str(e.value)
    # both known names are offered, or the error teaches the wrong one
    assert tau2_runner.TAU2_BASE_VARIANT in str(e.value)
    assert tau2_runner.TAU2_STOP_VARIANT in str(e.value)


@pytest.mark.parametrize(
    "variant_of", [lambda: tau2_runner.TAU2_BASE_VARIANT, lambda: tau2_runner.TAU2_STOP_VARIANT]
)
def test_an_arm_that_renders_another_template_refuses(variant_of):
    """A silent no-op would produce a run whose manifest names a variant and whose policy read
    the paragraph that variant exists to replace."""

    class _Other:
        prompt_name = "inquirer_noevidence"

    with pytest.raises(tau2_runner.Tau2DriverError) as e:
        tau2_runner.apply_prompt_variant(_Other(), variant_of())
    assert "inquirer_noevidence" in str(e.value)


# --------------------------------------------------------------------- it cannot collide on disk


def test_units_differing_only_in_the_variant_do_not_share_a_reassembly_key():
    """`run_sweep` returns `[by_key[spec.key] for spec in specs]`: a shared key hands back two
    aliases of one status and hides whichever of the two errored."""
    base = dict(
        suite_id="tau2_retail",
        corpus_dir="",
        task_id="55",
        arm_id="inquirer_prompted",
        seed=0,
        runs_root="runs",
        cache_root="cache",
        foreign_trace_sha="ab" * 32,
        foreign_prefix_k=34,
    )
    plain = UnitSpec(**base)
    routing = UnitSpec(**base, prompt_variant=tau2_runner.TAU2_BASE_VARIANT)
    stopping = UnitSpec(**base, prompt_variant=tau2_runner.TAU2_STOP_VARIANT)
    assert len({plain.key, routing.key, stopping.key}) == 3
    assert plain.prompt_variant == ""


@pytest.mark.parametrize(
    "variant_of", [lambda: tau2_runner.TAU2_BASE_VARIANT, lambda: tau2_runner.TAU2_STOP_VARIANT]
)
def test_the_launcher_passes_the_variant_through_to_every_spec(variant_of):
    points = [{"task_id": "55", "trace_sha": "ab" * 32, "k": 34}]
    specs = build_specs(
        points,
        suite="tau2_retail",
        arms=["inquirer_prompted"],
        seeds=[0, 1],
        prompt_variant=variant_of(),
    )
    assert [s.prompt_variant for s in specs] == [variant_of()] * 2
    assert all(
        s.prompt_variant == ""
        for s in build_specs(points, suite="tau2_retail", arms=["inquirer_prompted"], seeds=[0])
    )


# ------------------------------------------------------- the driver's own two call sites


@pytest.mark.integration
def test_the_driver_stamps_the_label_and_refuses_an_arm_that_cannot_render_a_variant(tmp_path):
    """The unit tests above call `apply_prompt_variant` directly; this reaches it the way a
    campaign does, through `run_tau2_unit`, which is where the field was dropped before.

    LLM-free (`fake_chain`, a scripted customer), so it costs nothing and needs no key.
    """
    from tests.test_adapters_tau2 import _scripted_user_class

    from pi_run.stages.tau2_runner import run_tau2_unit
    from pinq_adapters.tau2._probe import available
    from pinq_adapters.tau2.suite import Tau2Suite

    ok, why = available()
    if not ok:
        pytest.skip(why)

    def _spec(**over):
        return UnitSpec(
            suite_id="tau2",
            corpus_dir=str(tmp_path),
            task_id=Tau2Suite().task_ids()[0],
            arm_id="fake_chain",
            seed=0,
            runs_root=str(tmp_path / "runs"),
            cache_root=str(tmp_path / "cache"),
            code_version="t",
            dirty=False,
            max_turns=2,
            k=3,
            **over,
        )

    plain = _spec()
    res = run_tau2_unit(plain, user=_scripted_user_class()(["Hello."]))
    man = json.loads((tmp_path / "runs" / res["run_id"] / "manifest.json").read_text())
    assert man["prompt_variant_id"] == "v1", (
        "an unset variant must stamp what every run on disk has"
    )

    # fake_chain's Inquirer renders no template at all, so asking for either variant there must
    # fail the UNIT rather than silently run the shipped prompt under a variant's label.
    for variant in (tau2_runner.TAU2_BASE_VARIANT, tau2_runner.TAU2_STOP_VARIANT):
        bad = run_tau2_unit(_spec(prompt_variant=variant), user=_scripted_user_class()(["Hello."]))
        assert bad["status"] != "ok"
        assert "prompt variant" in str(bad.get("error", ""))


@pytest.mark.integration
def test_the_manifest_carries_whatever_the_labeller_returned_not_a_hardcoded_v1(
    tmp_path, monkeypatch
):
    """THE OTHER HALF OF THE STAMP. The test above proves "v1" reaches `manifest.json`, which a
    hardcoded default would also satisfy; this proves the driver writes what
    `apply_prompt_variant` ACTUALLY RETURNED, which is the half that carries a variant's label
    on a paid unit. The labeller's own return value is pinned by the unit tests above, so the
    two together cover the claim without needing a provider.
    """
    from tests.test_adapters_tau2 import _scripted_user_class

    from pi_run.stages import tau2_runner as tr
    from pinq_adapters.tau2._probe import available
    from pinq_adapters.tau2.suite import Tau2Suite

    ok, why = available()
    if not ok:
        pytest.skip(why)

    sentinel = "tau2_base-deadbeef"
    monkeypatch.setattr(tr, "apply_prompt_variant", lambda inquirer, variant: sentinel)
    res = tr.run_tau2_unit(
        UnitSpec(
            suite_id="tau2",
            corpus_dir=str(tmp_path),
            task_id=Tau2Suite().task_ids()[0],
            arm_id="fake_chain",
            seed=0,
            runs_root=str(tmp_path / "runs"),
            cache_root=str(tmp_path / "cache"),
            code_version="t",
            dirty=False,
            max_turns=2,
            k=3,
        ),
        user=_scripted_user_class()(["Hello."]),
    )
    man = json.loads((tmp_path / "runs" / res["run_id"] / "manifest.json").read_text())
    assert man["prompt_variant_id"] == sentinel
