"""The agent-visible vocabulary for the convlog adapter. Frozen, slotted, stdlib-only.

Two invariants mirror pinq.types (see that module's docstring) and are re-asserted here
independently rather than inherited, because inheritance would let ConvState pick up a base
class GoldNode could also share:

  1. NO BUDGET FIELD. `ConvState` carries no remaining-turns, cap or spend field, for the same
     reason `pinq.types.State` doesn't: a budget-aware policy makes prefix-k of a long
     conversation non-exchangeable with a true short one and confounds STOP with the announced
     cap. tests/test_convlog_state_no_budget.py enumerates the forbidden names and fails the
     build if one appears.
  2. NO FIELD NAME SHARED WITH GoldNode. `pi_eval.gold.GoldNode` prefixes every field
     `gold_*`; `ConvState` prefixes none. The two field-name sets are disjoint by construction,
     so `dataclasses.asdict()` can never smuggle a gold key across under a name that happens to
     already be trusted here.

STDLIB-ONLY is enforced by import-linter's "pinq is stdlib-only" contract's sibling — nothing
here may import pi_eval, pinq, or any third-party package, so this module has no dependency
surface a suite could accidentally widen.

SEQUENCE FIELDS ARE TUPLES, not lists, so `ConvState` (and `ToolStep`) stay hashable: the
Drafter's purity contract keys on `(view, evidence.subset_hash, seed)`-shaped tuples, and an
unhashable state would silently fall out of that contract's reach.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolStep:
    """One tool call, recorded as a TRACE entry, not its output.

    Only `ok`, `n_chars` and `head` of the result survive; the full tool_result text is
    exactly the kind of large, un-vetted string a policy should not be handed verbatim, and
    `head` (already scrubbed by the caller) is enough to show what happened without smuggling
    the rest of a paste or a file dump into the state.
    """

    tool: str
    arg_summary: str
    ok: bool
    n_chars: int
    head: str


@dataclass(frozen=True, slots=True)
class ConvState:
    """What the policy sees at one decision point of one Claude Code session.

    `dp_index` is the position of this decision point within its session, not a countdown —
    it identifies WHICH decision this is, the same way a turn index would, without ever
    telling the policy how many are left. `after_compaction` marks that `task_statement` came
    from a compaction summary rather than the session's first human prompt; the turns before
    the compaction are gone from `prior_user_turns`/`prior_assistant_text` on purpose, because
    the policy at that point in the real conversation could not see them either.
    """

    session_id: str
    dp_index: int
    cwd_norm: str
    git_branch_norm: str
    after_compaction: bool
    task_statement: str
    prior_user_turns: tuple[str, ...]
    prior_assistant_text: tuple[str, ...]
    tool_trace: tuple[ToolStep, ...]
