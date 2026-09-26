"""The convlog Inquirer template.

Two properties, both of which a later edit could break silently.
"""

from pinq import promptlib

NAME = "inquirer_convlog"


def test_the_template_is_discoverable_and_renders():
    assert NAME in promptlib.names()
    out = promptlib.render(
        NAME,
        task="t",
        prior_turns="p",
        tool_trace="tt",
        workspace="~/x",
        user_channel=promptlib.load("fragment_user_channel"),
    )
    assert "{{" not in out
    assert "THE USER CHANNEL IS OPEN" in out


def test_its_placeholders_are_exactly_the_conv_state_it_is_rendered_from():
    """A placeholder this template gains without a renderer field raises at render time, which
    is right -- but a placeholder it LOSES fails silently, and a section quietly dropped from
    every state is an ablation nobody declared."""
    assert promptlib.placeholders(NAME) == frozenset(
        {"task", "prior_turns", "tool_trace", "workspace", "user_channel"}
    )


def test_it_is_deliberately_outside_every_parity_family():
    """Membership of a parity family is the claim 'these arms differ in what the prompt says,
    not in how long it is'. This template is rendered from a conversation state and is never
    compared against the benchmark Inquirer arms, so token-matching it to them would assert a
    comparison that is not made."""
    for family, members in promptlib.PARITY_FAMILIES.items():
        assert NAME not in members, f"{NAME} must not be in the {family!r} parity family"


def test_it_never_tells_the_policy_its_cap():
    """Same rule as State: a policy told how much is left confounds stopping with the cap it
    was shown.

    The first version of this test banned the SUBSTRING "you have", and failed on the sentence
    "Stop when the work can proceed on what you have" -- which announces no budget at all. A
    substring blacklist over English prose reports the wrong thing; the invariant that is
    actually checkable is the one rung 0 already enforces on every mutated prompt, so this
    reuses that list rather than inventing a second, worse one.
    """
    assert promptlib.placeholders(NAME) & promptlib.FORBIDDEN_PLACEHOLDERS == frozenset()
    text = promptlib.load(NAME).lower()
    # A cap can also arrive as prose rather than as a placeholder. These phrases each state a
    # QUANTITY of remaining allowance, which "what you have" does not.
    for phrase in (
        "turns left",
        "turns remaining",
        "questions left",
        "you may ask at most",
        "your budget",
        "remaining budget",
    ):
        assert phrase not in text, f"the template states a cap: {phrase!r}"
