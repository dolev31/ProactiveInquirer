"""Decision points -> a blind B-family annotation bundle, plus the key that unblinds it.

THE BUNDLE/KEY SPLIT, AGAIN, ONE LEVEL DOWN THE PIPELINE. `pi_eval.annotate`'s module
docstring explains why a bundle and a key are different files for the A instruments; the B
family needs the same discipline for the same reason. A B1 item is "was this question
necessary" and a B3 item is "was stopping here right" -- if the bundle also carried
`observed_action`, `reaction` or `repeats_earlier_turn`, an annotator would be reading the
pipeline's own classification of the answer, not judging the conversation. So those fields,
plus `session_id`/`dp_index` for anything that would let an annotator recover which real
session an item came from mid-campaign, live ONLY in the key this module returns second.

WHY THE FOILS EXIST. A rater who says "necessary" to every B1 question, or "new information"
to every B2 item, is indistinguishable from an attentive one until something in the bundle has
a knowably correct answer. A planted B1 foil asks about something the state already answers
verbatim (expected `answer_in_state`); a planted B2 foil is a request copied verbatim out of an
earlier turn (expected `stated_already`). Both must be GENUINELY answerable from what is shown,
not merely labeled as foils, or a rater who passes them proves nothing -- see the module
docstring on `tests/test_convlog_sample.py` and the A7 campaign it cites, where a rater caught
every foil and was still quarantined for reading the slot rather than the content.

WHY SPLITTING A TURN INTO ITS REQUESTS IS MECHANICAL, NOT AN LLM CALL. `split_into_items` feeds
B2, whose whole question is "did the agent already have what it needed for each of these
requests". If an LLM did the decomposition, a wrong B2 score could mean the policy missed a
request OR the decomposer invented/merged one -- two different failures wearing one number. A
regex over sentence boundaries and a fixed list of coordinating cues ("also", "and then", "in
addition", "plus") cannot invent a request that was not typed, so B2's error is always the
policy's.

WHY STRIDE, NOT A SLICE, OFF A SHUFFLED POOL. The A-campaign measured annotators drifting
toward a batch's own base rate -- a batch that happens to be one conversation in a row teaches
the rater that session's quirks instead of the general question. Shuffling breaks the pool's
original (typically per-session-contiguous) order; striding across the shuffled pool on top of
that spreads picks further apart than an accidental run of adjacent shuffled slots would, at no
cost since sampling here is already without replacement.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Sequence

from pi_eval.annotate import item_id, item_set_hash
from pi_eval.build.convlog_parse import DecisionPoint
from pinq_adapters.convlog.render import (
    render_conv_state,
    render_prior_turns,
    render_task,
    render_tool_trace,
)
from pinq_adapters.convlog.types import ConvState

_NONE = "(none)"

# --------------------------------------------------------------------------- item splitting

# Order doesn't matter between these two passes' OWN matches (each cue is a distinct phrase,
# none a substring of another with a word boundary between), only that sentence boundaries are
# tried first: a period ends a request the way a cue word joins two requests within one
# sentence, and running cue-splitting first would let "also" inside an already-separate
# sentence steal text that belongs to the sentence after it.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CUE_RE = re.compile(r"\b(?:also|and then|in addition|plus)\b", re.I)
_WORD_RE = re.compile(r"[A-Za-z']+")


def split_into_items(text: str, *, max_items: int = 6) -> list[str]:
    """One user turn -> the separate requests packed into it, mechanically (see module
    docstring on why this is never an LLM call). Capped at `max_items` so one run-on turn
    cannot dominate a sampled batch the way a single long conversation must not either."""
    text = text.strip()
    if not text:
        return []
    items: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        for piece in _CUE_RE.split(sentence):
            cleaned = piece.strip(" ,.")
            if cleaned:
                items.append(cleaned)
    return items[:max_items]


# --------------------------------------------------------------------------- local rendering

# `render_conv_state` in pinq_adapters.convlog.render does this same job for the POLICY's
# prompt, but its helpers are private and pi_eval may not reach across the package boundary to
# use them (pinq_adapters may import pi_eval, never the reverse -- see convlog_parse.py's
# module docstring). These reproduce the same two rules that renderer earns its keep on: never
# render `ToolStep.head` (un-vetted tool output, possibly the very thing a label is graded
# against), and say "(none)" for an empty section rather than rendering it blank, so a section
# that is genuinely empty cannot be mistaken for one a bug silently dropped.


def _base_context(state: ConvState) -> dict[str, str]:
    """The three fields every B-family item shows: the task as stated, what has been said
    since, and what the agent already did. One function, not three copy-pasted dicts, so B1,
    B2 and B3 can never drift into showing three different slices of the same state."""
    return {
        "task": render_task(state.task_statement),
        "prior_turns": render_prior_turns(state),
        "tool_trace": render_tool_trace(state),
    }


def _final_assistant_text(d: DecisionPoint) -> str:
    """Read off the record, never reconstructed.

    This used to fall back to `state.prior_assistant_text[-1]` when `asked_question` was empty.
    That is wrong on exactly the population B3 is about: a yield's own final text is excluded
    from its prior context (it is the action, not background to it), so on a REPORT yield the
    fallback silently returned the statement BEFORE the one being judged. `DecisionPoint` now
    carries the text itself.
    """
    return d.final_assistant_text


def _provenance(state: ConvState) -> dict[str, object]:
    """Enough to trace an item back to the exact decision point it came from, never enough to
    reveal what happened there. `item_id` hashes only (task_type, provenance, payload) -- NOT
    context -- and a B1 or B3 item's payload is always `{}`, so without `session_id`/`dp_index`
    here every B1 item would collide on one id. Carrying them is safe: they identify WHICH
    state this is, not what the pipeline believes about it, and neither is on the
    forbidden-giveaway list a foil's answer key sits behind (verdict, foil mark, reaction,
    observed_action, repeats_earlier_turn)."""
    return {"suite": "convlog", "session_id": state.session_id, "dp_index": state.dp_index}


def _shown_sha(context: dict) -> str:
    """A digest of what the annotator is actually SHOWN, folded into the id.

    `item_id` hashes (task_type, provenance, payload) and NOT context, while a B1 foil rewrites
    the question in context. So one decision point produced ONE id whether or not it was
    planted, and the two are different questions. Measured when it bit: carrying records between
    two bundles whose foil draws differed left 114 to 157 items per rater holding two records
    with two different prompts, because `--resume` keys on (item_id, prompt_sha) and the id said
    "already done" while the prompt had changed underneath it.

    ON EVERY ITEM, not only on planted ones. A field that appears only on foils marks them,
    which is exactly what the separate key file exists to prevent.
    """
    blob = json.dumps(context, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _key_entry(d: DecisionPoint) -> dict:
    """Everything the key carries for every B-family item, foil or not: the unblinding side
    that must never reach the bundle an annotator reads."""
    return {
        "session_id": d.state.session_id,
        "dp_index": d.state.dp_index,
        "kind": d.kind,
        "observed_action": d.observed_action,
        "reaction": d.reaction,
        "repeats_earlier_turn": d.repeats_earlier_turn,
        # THE EXPORT SIDE. A training row needs the prompt the policy would actually see and
        # the question the agent actually asked. Both are carried here rather than re-derived
        # downstream, because re-deriving means a second renderer and this file already
        # learned what two copies of one renderer do to a measurement.
        "state_text": render_conv_state(d.state),
        "asked_question": d.asked_question,
        "final_assistant_text": d.final_assistant_text,
    }


# --------------------------------------------------------------------------- foils


def _foil_word(state: ConvState) -> str | None:
    """A word already sitting in the state's prior turns, long enough (>6 chars) to be a
    specific answer rather than a filler word, and stripped to letters only so the substring
    check a B1 foil's validity depends on can never be broken by a trailing '.', ',' or '?' in
    the source text. None means this state has nothing to build a genuine foil from."""
    text = " ".join((*state.prior_user_turns, *state.prior_assistant_text))
    for word in _WORD_RE.findall(text):
        if len(word) > 6:
            return word
    return None


def _b1_foil_question(word: str) -> str:
    # `word` sits mid-sentence, not at the end, so nothing of ours (a trailing "?") ever
    # attaches to it and breaks the plain substring check the foil test runs against
    # `prior_turns`.
    return f"Didn't the user already mention {word} earlier in this conversation?"


def _plan_foils(
    b1_dps: Sequence[DecisionPoint],
    b2_dps: Sequence[DecisionPoint],
    foil_rate: float,
    rng: random.Random,
) -> tuple[set[int], set[int]]:
    """Which drawn B1 and B2 items (by their index into `b1_dps`/`b2_dps`) become foils.

    Eligibility is checked FIRST, the rate SECOND: a decision point with nothing to build a
    genuine foil from must not count toward foil_rate's denominator, or asking for foils this
    pool cannot supply would silently plant fewer than requested with no signal why (a real
    export would want to see that as a refusal count; this sampler simply never proposes an
    unanswerable foil in the first place -- see the module docstring on why a foil must be
    genuinely answerable or it tests nothing).
    """
    eligible: list[tuple[str, int]] = [
        ("b1", i) for i, d in enumerate(b1_dps) if _foil_word(d.state) is not None
    ]
    eligible += [("b2", i) for i, d in enumerate(b2_dps) if d.state.prior_user_turns]
    rng.shuffle(eligible)
    n_foils = min(len(eligible), round(foil_rate * (len(b1_dps) + len(b2_dps))))
    b1_foils: set[int] = set()
    b2_foils: set[int] = set()
    for kind, i in eligible[:n_foils]:
        (b1_foils if kind == "b1" else b2_foils).add(i)
    return b1_foils, b2_foils


# --------------------------------------------------------------------------- sampling


def _stride_sample(
    pool: Sequence[DecisionPoint], n: int, rng: random.Random
) -> list[DecisionPoint]:
    """`n` items from `pool`, drawn at a stride across a shuffled copy -- never a contiguous
    slice of the pool as it arrived (see module docstring on batch drift). Asking for more than
    `len(pool)` returns the whole (shuffled) pool rather than repeating anything: a repeated
    item_id would spend two annotation slots on one question, which `bundle_shape_errors`
    refuses outright."""
    if n <= 0 or not pool:
        return []
    shuffled = list(pool)
    rng.shuffle(shuffled)
    m = len(shuffled)
    if n >= m:
        return shuffled
    stride = max(1, m // n)
    used: set[int] = set()
    out: list[DecisionPoint] = []
    for i in range(n):
        idx = (i * stride) % m
        while idx in used:  # a stride can collide when m is not a clean multiple of n
            idx = (idx + 1) % m
        used.add(idx)
        out.append(shuffled[idx])
    return out


def sample_b4_bundle(dps: Sequence[DecisionPoint], *, n: int, seed: int) -> tuple[dict, dict]:
    """States where the agent did NOT ask, for a proposer to suggest the missing question.

    B3 asked whether a question was missing and three strong raters said yes, unanimously,
    ONCE in 848 units. A recognition task with a base rate that low leaves the dataset's ASK
    supervision resting entirely on questions the agent already thought to ask. B4 inverts it.

    DRAWN ONLY WHERE NOTHING WAS ASKED. A state where the agent already asked has no missing
    question, and including one would let the proposer restate the question that is already
    there and score a hit.

    THE REPLY IS NEVER SHOWN. B4's context is B3's exactly, which omits `next_turn`: a proposer
    that could read what the person said next would be copying rather than proposing, and every
    proposal would come back trivially necessary.
    """
    rng = random.Random(seed)
    pool = [
        d for d in dps if d.observed_action in ("REPORT", "ACTING") and _final_assistant_text(d)
    ]
    drawn = _stride_sample(pool, n, rng)
    items, key_items = [], {}
    for d in drawn:
        context = {**_base_context(d.state), "final_assistant_text": _final_assistant_text(d)}
        iid = item_id("B4", dict(_provenance(d.state), shown_sha=_shown_sha(context)), {})
        items.append(
            {
                "item_id": iid,
                "task_type": "B4",
                "provenance": dict(_provenance(d.state), shown_sha=_shown_sha(context)),
                "context": context,
                "payload": {},
            }
        )
        key_items[iid] = _key_entry(d)
    return (
        {
            "manifest": {
                "bundle_id": f"convlog-b4-{item_set_hash(items)[:12]}",
                "counts": {"B4": len(items)},
                "item_set_hash": item_set_hash(items),
                "seed": seed,
            },
            "items": items,
        },
        {"items": key_items},
    )


def proposals_to_b1_bundle(
    b4_bundle: dict, b4_key: dict, proposals: dict[str, str]
) -> tuple[dict, dict]:
    """Proposed questions -> a B1 bundle a validator cannot distinguish from a logged one.

    THE PROMPT IS B1's VERBATIM, and that is the design rather than an economy. A proposed
    question and a real one reach the validators in identical form, so the blinding is genuine
    AND the two acceptance rates are directly comparable -- "are invented questions accepted
    more often than real ones?" becomes a number instead of a worry. A much higher rate would
    say the proposer produces agreeable noise; a much lower one, that it invents.

    The origin therefore lives in the key, next to every other thing the bundle must not say.
    """
    items, key_items = [], {}
    src_key = dict(b4_key.get("items") or {})
    for it in b4_bundle.get("items") or ():
        q = str(proposals.get(it["item_id"]) or "").strip()
        if not q:
            continue
        context = {k: v for k, v in it["context"].items() if k != "final_assistant_text"}
        context["asked_question"] = q
        prov = dict(it["provenance"])
        prov["shown_sha"] = _shown_sha(context)
        iid = item_id("B1", prov, {})
        items.append(
            {
                "item_id": iid,
                "task_type": "B1",
                "provenance": prov,
                "context": context,
                "payload": {},
            }
        )
        entry = dict(src_key.get(it["item_id"]) or {})
        entry.update(
            {"question_source": "proposed", "asked_question": q, "b4_item_id": it["item_id"]}
        )
        key_items[iid] = entry
    return (
        {
            "manifest": {
                "bundle_id": f"convlog-b4v-{item_set_hash(items)[:12]}",
                "counts": {"B1": len(items)},
                "item_set_hash": item_set_hash(items),
            },
            "items": items,
        },
        {"items": key_items},
    )


def sample_b_bundle(
    dps: Sequence[DecisionPoint],
    *,
    n_b1: int,
    n_b2: int,
    n_b3: int,
    foil_rate: float,
    seed: int,
) -> tuple[dict, dict]:
    """Decision points -> `(bundle, key)`. See the module docstring for what each file may and
    may not carry, and why. One `random.Random(seed)` drives every draw in a fixed order (B1
    pool, then B2, then B3, then the foil plan), which is what makes the whole bundle
    reproducible from `seed` alone and different seeds diverge.
    """
    rng = random.Random(seed)

    # Each pool's eligibility rule is pinned by its own test in tests/test_convlog_sample.py;
    # see that file for the reasoning each one encodes.
    b1_pool = [d for d in dps if d.observed_action == "ASK_USER" and d.asked_question.strip()]
    b2_pool = [d for d in dps if split_into_items(d.next_user_turn)]
    b3_pool = [d for d in dps if d.kind in ("yield", "interrupt") and _final_assistant_text(d)]

    b1_dps = _stride_sample(b1_pool, n_b1, rng)
    b2_dps = _stride_sample(b2_pool, n_b2, rng)
    # B3 PREFERS THE STATES WHERE ITS TARGET BEHAVIOUR CAN OCCUR. Its positive verdict is
    # "the agent acted where it should have checked", and the observable signature of that is
    # the person interrupting or correcting. Drawing from every yield put almost all items
    # where the phenomenon cannot happen, and the first pilot showed it: both strong raters
    # returned `should_have_asked` ZERO times in 39 and 40 items, so `over_action_rate` was
    # zero by construction of the sample rather than as a finding about the corpus.
    #
    # This is the A7 campaign's own mistake, in its own words: "`frontier-states --max-turn 2`
    # concentrates forks at t0, so the pipeline has been sampling the states where its own
    # target behaviour cannot occur."
    #
    # A PREFERENCE, NOT A FILTER. The corpus holds 33 interrupts against 2,146 decision points,
    # so a hard filter would cap B3 at 33 forever and make the sample unrepresentative in the
    # opposite direction. The remainder is drawn from the rest and the key records which is
    # which, so the two strata stay separable after the fact.
    hot = [d for d in b3_pool if d.reaction in ("interrupt", "correction")]
    cold = [d for d in b3_pool if d.reaction not in ("interrupt", "correction")]
    b3_dps = _stride_sample(hot, n_b3, rng)
    if len(b3_dps) < n_b3:
        b3_dps += _stride_sample(cold, n_b3 - len(b3_dps), rng)
    b1_foils, b2_foils = _plan_foils(b1_dps, b2_dps, foil_rate, rng)

    items: list[dict] = []
    key_items: dict[str, dict] = {}

    for i, d in enumerate(b1_dps):
        context = {**_base_context(d.state), "asked_question": d.asked_question}
        entry = _key_entry(d)
        if i in b1_foils:
            # The rewrite happens BEFORE the id is minted, which is what keeps a planted
            # item distinct from its unplanted twin: `item_id` ignores context, so the id has
            # to carry a digest of what was shown (`_shown_sha`) or the two collide.
            context["asked_question"] = _b1_foil_question(_foil_word(d.state))
            # The key records the question the rater was SHOWN. Exporting the logged one would
            # train on a target that does not match the state the verdict was given for, so
            # the exporter refuses foils outright and this keeps the record honest either way.
            entry["asked_question"] = context["asked_question"]
            entry["foil"] = True
            entry["expected"] = "answer_in_state"
        payload: dict = {}
        iid = item_id("B1", dict(_provenance(d.state), shown_sha=_shown_sha(context)), payload)
        items.append(
            {
                "item_id": iid,
                "task_type": "B1",
                "context": context,
                "payload": payload,
                "provenance": dict(_provenance(d.state), shown_sha=_shown_sha(context)),
            }
        )
        key_items[iid] = entry

    for i, d in enumerate(b2_dps):
        sub_items = [
            {"item_key": str(k), "text": text}
            for k, text in enumerate(split_into_items(d.next_user_turn))
        ]
        entry = _key_entry(d)
        if i in b2_foils:
            # A continuing numeric key, not a labeled "foil" one: the item_key rides in the
            # PAYLOAD, which is bundle-visible, and the word "foil" anywhere in the bundle is
            # exactly what tests/test_convlog_sample.py's repr-scan refuses.
            sub_items.append(
                {"item_key": str(len(sub_items)), "text": d.state.prior_user_turns[-1]}
            )
            entry["foil"] = True
            entry["expected"] = "stated_already"
        payload = {"items": sub_items}
        context = {**_base_context(d.state), "next_turn": d.next_user_turn}
        iid = item_id("B2", dict(_provenance(d.state), shown_sha=_shown_sha(context)), payload)
        items.append(
            {
                "item_id": iid,
                "task_type": "B2",
                "context": context,
                "payload": payload,
                "provenance": dict(_provenance(d.state), shown_sha=_shown_sha(context)),
            }
        )
        key_items[iid] = entry

    for d in b3_dps:
        context = {**_base_context(d.state), "final_assistant_text": _final_assistant_text(d)}
        payload = {}
        iid = item_id("B3", dict(_provenance(d.state), shown_sha=_shown_sha(context)), payload)
        items.append(
            {
                "item_id": iid,
                "task_type": "B3",
                "context": context,
                "payload": payload,
                "provenance": dict(_provenance(d.state), shown_sha=_shown_sha(context)),
            }
        )
        key_items[iid] = _key_entry(d)

    ish = item_set_hash(items)
    manifest = {
        "bundle_id": f"convlog-b-{seed}-{ish[:12]}",
        "item_set_hash": ish,
        "counts": {"B1": len(b1_dps), "B2": len(b2_dps), "B3": len(b3_dps)},
        "seed": seed,
        # The rate items were PLANTED at, not the word "foil" itself -- that word is exactly
        # what the repr-scan in test_the_bundle_carries_no_expected_answer_anywhere refuses
        # anywhere in the bundle, manifest included.
        "foil_rate": foil_rate,
    }
    bundle = {"manifest": manifest, "items": items}
    key = {"items": key_items}
    return bundle, key
