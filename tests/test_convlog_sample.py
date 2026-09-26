"""Decision points -> a blind B-family bundle and its key.

THE BUNDLE IS WHAT AN ANNOTATOR READS AND THE KEY IS EVERYTHING THAT WOULD GIVE THE ANSWER
AWAY. They are separate objects because the bundle is handed to a rater verbatim: an item that
announced which verdict the pipeline expects would measure agreement with a hint.

The foils are the validity check. A planted B1 item whose answer is sitting in the prior turns
must come back `answer_in_state`; a planted B2 item copied verbatim out of an earlier turn must
come back `stated_already`. A rater who misses those is not reading the state, and the A7
campaign is the reason this is not optional: one model caught 99 of 99 foils and was still
quarantined for answering the slot instead of the content.
"""

import pytest

from pi_eval.annotate import bundle_shape_errors
from pi_eval.build.convlog_parse import DecisionPoint
from pi_eval.build.convlog_sample import sample_b_bundle, split_into_items
from pinq_adapters.convlog.types import ConvState, ToolStep


def dp(
    i,
    *,
    kind="yield",
    action="ASK_USER",
    question="Which timeout should I use?",
    nxt="use 30 seconds, also update the changelog",
    prior=("set up the uploader",),
):
    state = ConvState(
        session_id=f"s{i % 5}",
        dp_index=i,
        cwd_norm="~/proj",
        git_branch_norm="main",
        after_compaction=False,
        task_statement=f"Task number {i}: add retries to the uploader.",
        prior_user_turns=tuple(prior),
        prior_assistant_text=("I can add a fixed retry.",),
        tool_trace=(ToolStep("Bash", "cat upload.py", True, 42, "def upload"),),
    )
    return DecisionPoint(
        state=state,
        kind=kind,
        observed_action=action,
        asked_question=question,
        final_assistant_text=question or "I rewrote the config loader and it builds.",
        next_user_turn=nxt,
        reaction="extra_item",
        repeats_earlier_turn=False,
    )


POOL = (
    [dp(i) for i in range(40)]
    + [dp(100 + i, kind="yield", action="REPORT", question="") for i in range(40)]
    + [
        dp(
            200 + i,
            kind="interrupt",
            action="ACTING",
            question="",
            nxt="[Request interrupted by user] no, leave the config alone",
        )
        for i in range(20)
    ]
)


# --------------------------------------------------------------------------- item splitting


def test_a_multi_request_turn_splits_into_its_requests():
    items = split_into_items("use 30 seconds. Also update the changelog, and bump the version.")
    assert len(items) >= 2
    assert any("changelog" in x for x in items)


def test_a_single_request_stays_one_item():
    assert len(split_into_items("use 30 seconds")) == 1


def test_splitting_is_capped_so_one_essay_cannot_dominate_a_batch():
    long_turn = ". ".join(f"do thing number {i}" for i in range(50))
    assert len(split_into_items(long_turn, max_items=6)) <= 6


def test_empty_and_whitespace_turns_yield_no_items():
    assert split_into_items("") == []
    assert split_into_items("   \n ") == []


# --------------------------------------------------------------------------- the bundle


@pytest.fixture(scope="module")
def built():
    return sample_b_bundle(POOL, n_b1=8, n_b2=8, n_b3=4, foil_rate=0.25, seed=7)


def test_the_bundle_holds_exactly_what_was_asked_for(built):
    bundle, _key = built
    kinds = [i["task_type"] for i in bundle["items"]]
    assert kinds.count("B1") == 8
    assert kinds.count("B2") == 8
    assert kinds.count("B3") == 4


def test_every_item_satisfies_the_annotator_tool_contract(built):
    bundle, _key = built
    assert bundle_shape_errors(bundle) == []


def test_no_item_carries_a_verdict_a_foil_mark_or_the_pipeline_s_own_classification(built):
    """The whole reason there are two files.

    Checked per ITEM and per key, not by grepping the whole bundle for words. The first version
    of this test banned the substring "foil" anywhere in `repr(bundle)`, which had two faults:
    it pushed the manifest into naming its own rate `plant_rate` to get past a string check, and
    on real transcripts it fires on the word "expected" appearing in somebody's prose. What
    actually must not leak is per-item: which verdict the pipeline expects, whether this item is
    planted, and how the pipeline classified the reply.
    """
    bundle, key = built
    banned = {
        "expected",
        "foil",
        "is_foil",
        "reaction",
        "observed_action",
        "verdict",
        "repeats_earlier_turn",
        "next_user_turn",
    }
    for item in bundle["items"]:
        for section in ("context", "payload", "provenance"):
            assert not (set(item.get(section) or {}) & banned), f"{section} leaks {banned}"
        assert not (set(item) & banned)
    # And the key really does hold them, or the split is protecting nothing.
    assert any(v.get("foil") for v in key["items"].values())
    assert all("observed_action" in v for v in key["items"].values())


def test_the_manifest_names_its_own_foil_rate_honestly(built):
    """A field renamed to get past a test is a field whose name no longer says what it is."""
    bundle, _key = built
    assert bundle["manifest"]["foil_rate"] == 0.25


def test_the_key_covers_every_item_and_holds_the_provenance(built):
    bundle, key = built
    ids = {i["item_id"] for i in bundle["items"]}
    assert set(key["items"]) == ids
    some = key["items"][next(iter(ids))]
    assert "session_id" in some and "dp_index" in some


def test_foils_are_planted_at_the_requested_rate_and_marked_only_in_the_key(built):
    bundle, key = built
    foils = [k for k, v in key["items"].items() if v.get("foil")]
    b12 = [i for i in bundle["items"] if i["task_type"] in ("B1", "B2")]
    assert 0 < len(foils) <= len(b12)
    assert abs(len(foils) / len(b12) - 0.25) < 0.2
    for fid in foils:
        assert key["items"][fid]["expected"] in ("answer_in_state", "stated_already")


def test_a_b1_foil_puts_its_own_answer_in_the_prior_turns(built):
    """The foil has to be answerable from the state, or it tests nothing."""
    bundle, key = built
    items = {i["item_id"]: i for i in bundle["items"]}
    for fid, k in key["items"].items():
        if k.get("foil") and k["expected"] == "answer_in_state":
            ctx = items[fid]["context"]
            assert any(w in ctx["prior_turns"] for w in ctx["asked_question"].split() if len(w) > 6)


def test_a_b3_item_shows_what_the_agent_actually_said_last(built):
    """It reads `final_assistant_text` off the record. Reconstructing it from
    `prior_assistant_text` gets the PREVIOUS statement on every REPORT yield, because a
    yield's own final text is deliberately excluded from its prior context."""
    bundle, _key = built
    for i in bundle["items"]:
        if i["task_type"] == "B3":
            assert i["context"]["final_assistant_text"].strip()


def test_b3_is_drawn_from_yields_and_interrupts_only(built):
    """B3 asks whether stopping was right. An explicit AskUserQuestion is not a stop."""
    bundle, key = built
    for i in bundle["items"]:
        if i["task_type"] == "B3":
            assert key["items"][i["item_id"]]["kind"] in ("yield", "interrupt")


def test_b1_is_drawn_only_from_decision_points_that_actually_asked(built):
    bundle, key = built
    for i in bundle["items"]:
        if i["task_type"] == "B1":
            assert key["items"][i["item_id"]]["observed_action"] == "ASK_USER"
            assert i["context"]["asked_question"].strip()


def test_the_sample_is_deterministic_under_a_seed():
    a, _ = sample_b_bundle(POOL, n_b1=6, n_b2=6, n_b3=3, foil_rate=0.2, seed=11)
    b, _ = sample_b_bundle(POOL, n_b1=6, n_b2=6, n_b3=3, foil_rate=0.2, seed=11)
    assert [i["item_id"] for i in a["items"]] == [i["item_id"] for i in b["items"]]
    c, _ = sample_b_bundle(POOL, n_b1=6, n_b2=6, n_b3=3, foil_rate=0.2, seed=12)
    assert [i["item_id"] for i in a["items"]] != [i["item_id"] for i in c["items"]]


def test_items_are_drawn_by_stride_from_a_shuffled_pool_not_by_position():
    """The A-campaign's batch-drift finding: annotators drift toward a batch's own base rate,
    so a batch must not be a contiguous slice of one conversation. With five sessions in the
    pool, a bundle of eight must touch more than one of them."""
    bundle, key = sample_b_bundle(POOL, n_b1=8, n_b2=0, n_b3=0, foil_rate=0.0, seed=3)
    sessions = {key["items"][i["item_id"]]["session_id"] for i in bundle["items"]}
    assert len(sessions) > 1


def test_asking_for_more_items_than_exist_returns_what_exists_rather_than_repeating():
    """A duplicated item_id spends two annotation slots on one question and produces two
    records claiming one id, which `bundle_shape_errors` refuses outright."""
    bundle, _ = sample_b_bundle(POOL[:5], n_b1=50, n_b2=50, n_b3=50, foil_rate=0.0, seed=1)
    ids = [i["item_id"] for i in bundle["items"]]
    assert len(ids) == len(set(ids))
    assert bundle_shape_errors(bundle) == []


def test_the_bundle_shows_the_same_state_text_the_policy_would_be_shown():
    """One definition of "how a conversation state becomes text", not two.

    The sampler had its own copy of the two renderers. They were identical when written and
    already diverging: the policy-facing one caps the rendered state (median real prompt was
    45,585 characters, max 276,143, at which size the model's own context window truncates from
    the front and silently) and the sampler's copy did not. An annotator judging an uncapped
    state and a policy acting on a capped one are not answering the same question.
    """
    from pinq_adapters.convlog.render import render_prior_turns, render_tool_trace

    d = dp(
        1,
        prior=tuple(f"turn {i} " + "z" * 800 for i in range(40)),
    )
    bundle, _ = sample_b_bundle([d, dp(2), dp(3)], n_b1=3, n_b2=0, n_b3=0, foil_rate=0.0, seed=5)
    item = next(i for i in bundle["items"] if "turn 39" in i["context"]["prior_turns"])
    assert item["context"]["prior_turns"] == render_prior_turns(d.state)
    assert item["context"]["tool_trace"] == render_tool_trace(d.state)
    assert "elided" in item["context"]["prior_turns"]


def test_b3_is_drawn_where_over_action_can_actually_happen():
    """The same mistake the A7 campaign found in its own fork frame, and its words for it:
    "`frontier-states --max-turn 2` concentrates forks at t0, so the pipeline has been sampling
    the states where its own target behaviour cannot occur."

    B3's positive verdict is `should_have_asked` -- the agent acted where it should have
    checked. The observable signature of that is the person interrupting or correcting. Drawing
    B3 from every yield put almost all of its items where the phenomenon cannot occur, and the
    pilot showed exactly that: both strong raters returned `should_have_asked` ZERO times in 39
    and 40 items, so `over_action_rate` was 0 by construction of the sample.

    So B3 prefers decision points whose observed reaction is an interrupt or a correction, and
    falls back to the rest only once those run out.
    """
    plain = [dp(i, nxt="thanks, looks good") for i in range(40)]
    for d in plain:
        object.__setattr__(d, "reaction", "other")
    hot = []
    for i in range(6):
        d = dp(
            500 + i,
            kind="interrupt",
            action="ACTING",
            question="",
            nxt="[Request interrupted by user] no, leave the config alone",
        )
        object.__setattr__(d, "reaction", "interrupt")
        hot.append(d)
    for i in range(6):
        d = dp(600 + i, nxt="no, that is wrong, revert it")
        object.__setattr__(d, "reaction", "correction")
        hot.append(d)

    bundle, key = sample_b_bundle(plain + hot, n_b1=0, n_b2=0, n_b3=10, foil_rate=0.0, seed=4)
    b3 = [i for i in bundle["items"] if i["task_type"] == "B3"]
    assert len(b3) == 10
    reactions = [key["items"][i["item_id"]]["reaction"] for i in b3]
    assert sum(r in ("interrupt", "correction") for r in reactions) == 10


def test_b3_falls_back_to_ordinary_yields_when_the_hot_ones_run_out():
    """Preference, not a filter. A corpus with three interrupts must still yield a B3 sample."""
    plain = [dp(i, nxt="thanks") for i in range(30)]
    for d in plain:
        object.__setattr__(d, "reaction", "other")
    hot = []
    for i in range(3):
        d = dp(
            700 + i,
            kind="interrupt",
            action="ACTING",
            question="",
            nxt="[Request interrupted by user] stop",
        )
        object.__setattr__(d, "reaction", "interrupt")
        hot.append(d)
    bundle, key = sample_b_bundle(plain + hot, n_b1=0, n_b2=0, n_b3=10, foil_rate=0.0, seed=4)
    b3 = [i for i in bundle["items"] if i["task_type"] == "B3"]
    assert len(b3) == 10
    reactions = [key["items"][i["item_id"]]["reaction"] for i in b3]
    assert sum(r in ("interrupt", "correction") for r in reactions) == 3


def test_the_key_carries_everything_the_exporter_needs_to_build_a_row():
    """The key is where the unblinding side lives, and a training row is the most unblinded
    thing there is: it needs the rendered prompt the policy would see and the question the
    agent actually asked. Re-deriving those at export time would mean a second renderer, and
    the pilot already showed what two copies of one renderer do."""
    bundle, key = sample_b_bundle(POOL, n_b1=4, n_b2=2, n_b3=2, foil_rate=0.0, seed=9)
    for item in bundle["items"]:
        e = key["items"][item["item_id"]]
        for field in (
            "state_text",
            "asked_question",
            "final_assistant_text",
            "repeats_earlier_turn",
            "session_id",
            "dp_index",
        ):
            assert field in e, f"{item['task_type']} key entry lacks {field}"
        assert e["state_text"].strip()


def test_a_b1_foils_rewritten_question_is_the_one_the_key_records():
    """A foil shows a question the agent never asked. Exporting the LOGGED question for it
    would train on a target that does not match the state the rater judged, so the key holds
    the shown text and the exporter refuses foils outright."""
    bundle, key = sample_b_bundle(POOL, n_b1=8, n_b2=0, n_b3=0, foil_rate=0.5, seed=9)
    items = {i["item_id"]: i for i in bundle["items"]}
    for iid, e in key["items"].items():
        if e.get("foil"):
            assert e["asked_question"] == items[iid]["context"]["asked_question"]


def test_a_foil_and_its_non_foil_twin_do_not_share_an_item_id():
    """`annotate.item_id` hashes (task_type, provenance, payload) and NOT context, while a B1
    foil rewrites the question in CONTEXT. So the same decision point produced the same id
    whether or not it was planted, and the two are different questions.

    Measured when it bit: carrying records between two bundles with different foil draws left
    114 to 157 items per rater holding two records with two different prompts, because
    `--resume` keys on (item_id, prompt_sha) and the id said "already done" while the prompt
    had changed underneath it. The fix is a digest of the shown context in provenance, present
    on every item so its presence cannot mark a foil.
    """
    pool = [dp(i) for i in range(12)]
    plain, key_plain = sample_b_bundle(pool, n_b1=12, n_b2=0, n_b3=0, foil_rate=0.0, seed=2)
    foiled, key_foiled = sample_b_bundle(pool, n_b1=12, n_b2=0, n_b3=0, foil_rate=1.0, seed=2)
    foils = {k for k, v in key_foiled["items"].items() if v.get("foil")}
    assert foils, "the fixture must actually plant one"
    assert not (foils & set(key_plain["items"])), "a planted item kept its unplanted id"


def test_the_shown_context_digest_is_on_every_item_not_only_on_foils():
    """A field present only on planted items marks them, which is what the key exists to hide."""
    bundle, _key = sample_b_bundle(POOL, n_b1=6, n_b2=4, n_b3=2, foil_rate=0.5, seed=2)
    assert all("shown_sha" in (i.get("provenance") or {}) for i in bundle["items"])


# ------------------------------------------- B4: propose a question, then validate it as B1


def test_b4_is_drawn_only_where_the_agent_did_not_ask():
    """A state where the agent already asked has no missing question to propose. Including one
    would let the proposer restate the question that is already there and score a hit."""
    from pi_eval.build.convlog_sample import sample_b4_bundle

    asked = [dp(i) for i in range(20)]  # ASK_USER
    silent = [dp(100 + i, action="REPORT", question="") for i in range(20)]
    bundle, key = sample_b4_bundle(asked + silent, n=15, seed=3)
    assert len(bundle["items"]) == 15
    for i in bundle["items"]:
        assert i["task_type"] == "B4"
        assert key["items"][i["item_id"]]["observed_action"] in ("REPORT", "ACTING")


def test_a_b4_item_shows_the_state_but_never_the_reply():
    from pi_eval.build.convlog_sample import sample_b4_bundle

    silent = [dp(100 + i, action="REPORT", question="", nxt="SECRET REPLY TEXT") for i in range(6)]
    bundle, _ = sample_b4_bundle(silent, n=4, seed=3)
    for i in bundle["items"]:
        assert "SECRET REPLY TEXT" not in repr(i)
        assert set(i["context"]) == {"task", "prior_turns", "tool_trace", "final_assistant_text"}


def test_a_proposal_becomes_a_b1_item_indistinguishable_from_a_logged_one():
    """The validators must not be able to tell a proposed question from one the agent really
    asked, or the acceptance rates stop being comparable. Provenance lives in the key."""
    from pi_eval.build.convlog_sample import proposals_to_b1_bundle, sample_b4_bundle

    silent = [dp(100 + i, action="REPORT", question="") for i in range(6)]
    b4, k4 = sample_b4_bundle(silent, n=4, seed=3)
    proposals = {i["item_id"]: "Which config file did you mean?" for i in b4["items"]}
    bundle, key = proposals_to_b1_bundle(b4, k4, proposals)
    assert [i["task_type"] for i in bundle["items"]] == ["B1"] * 4
    for i in bundle["items"]:
        assert i["context"]["asked_question"] == "Which config file did you mean?"
        assert "propos" not in repr(i).lower(), "the bundle must not mark the question's origin"
        assert key["items"][i["item_id"]]["question_source"] == "proposed"


def test_a_state_with_no_proposal_produces_no_validation_item():
    from pi_eval.build.convlog_sample import proposals_to_b1_bundle, sample_b4_bundle

    silent = [dp(100 + i, action="REPORT", question="") for i in range(6)]
    b4, k4 = sample_b4_bundle(silent, n=4, seed=3)
    only_one = {b4["items"][0]["item_id"]: "Which file?"}
    bundle, _key = proposals_to_b1_bundle(b4, k4, only_one)
    assert len(bundle["items"]) == 1
