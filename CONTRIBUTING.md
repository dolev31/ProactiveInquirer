# Contributing

The working rules for this repository. Code comments cite them as "CONTRIBUTING.md rule N".

This is the codebase for a paper. A silent bug here does not produce a stack trace; it
produces a **wrong published number**. Everything below follows from that.

## The four rules

1. **A number without provenance is not a result.** Every reported value traces to a
   `run_id`, a `scorer_hash` and a `graph_version`. If you cannot name all three, you cannot
   put it in a table.
2. **A test that passes without the fix is not a test.** Before fixing a bug, write the test
   that fails. `test_turn_usage_is_real_not_a_zero_stamp` exists because reconciliation was
   once tautological and no existing test could tell.
3. **Never state a measurement you did not take.** Paste the command and its output. "Should
   work" and "looks right" are not results.
4. **Never weaken a test to make it pass.** If a test fails, either the code is wrong or the
   test encodes a wrong belief — say which, and fix that.

## The firewall

`pi_eval` is gold-only. Nothing outside it may import it. This is enforced four ways, and
all four must stay alive:

- an `import-linter` forbidden contract (`lint-imports` in CI);
- a type wall: `TaskView` shares no field name and no base class with `GoldNode`, so a leak
  cannot typecheck;
- a process split: rollout workers run with `PI_GOLD_ROOT` unset, so `gold_root()` raises —
  **that raise is the firewall working**, not a bug;
- canary nonces on gold records, scanned in every serialized request.

Layers 1–3 stop leakage through *code*. Only layer 4 catches leakage through a *string*.

## Things that look like details and are not

- **`Inquirer.act(s: State)` takes no ledger, no context, no budget.** A budget-aware policy
  makes prefix-k of a long rollout non-exchangeable with a true short run and confounds STOP
  with the cap you announced. Do not add a parameter here.
- **`Drafter.draft()` must be pure in `(view, evidence.subset_hash, seed)`.** phi_LOO, the
  prefix ladder and the stop test are all undefined without it.
- **`EvidenceUnit.uid` is `(corpus_id, doc_id, span)` and nothing else.** Hashing any
  model-produced string into it makes every memoization key nondeterministic.
- **The per-call cache key and `RunManifest.semantic_hash` are different on purpose.** One
  is request bytes, the other is run identity. Merging them means a prompt typo evicts a
  whole sweep.
- **`wall_ms` and `usd` are machine artifacts.** They are recorded, never compared across
  arms as evidence. An arm with more turns takes more wall time; asserting parity there
  fails on day one and teaches you to disable the check.

## Metrics that were deleted, and why

Do not reintroduce these without reading the reasoning:
Anticipation Depth (non-monotone), Proactive Precision as originally stated (circular),
`P(QU>0)` (a property of ordering), Stop Regret in quality units (a perfect stopper exhibits
regret), FutureConversationCoverage (scores a miss on exactly the success case).

## Before you open a pull request

```bash
uv run --no-sync ruff check . && uv run --no-sync ruff format --check .
.venv/bin/lint-imports        # must print: Contracts: 4 kept, 0 broken
.venv/bin/pi suites audit     # must print: registry audit: clean
.venv/bin/python -m pytest -q -m "not integration and not slow"
bash scripts/check_no_home_paths.sh
```

## Which data trains this, and which measures it

`docs/SUITES.md` is the decision record; `pi suites map` prints the roles and
`pi suites validate` constructs every adapter and joins its gold. Two rules there are worth
knowing before touching a suite: a suite may never be both a training source and eval-only,
and `pi_eval.report.ELIGIBLE` filters on `split = 'test'` -- decided 2026-09-15, which moved
every previously reported n. `pi suites eligibility` prints what that clause removes, per
suite, and names any suite it leaves reporting nothing.
