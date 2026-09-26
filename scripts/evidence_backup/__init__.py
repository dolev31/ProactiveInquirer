"""Lane L0.10: back up the isolated per-campaign score stores that published paper cells
depend on, and make the backups verifiable without the parquet itself.

CLAUDE.md rule 1 ("a number without provenance is not a result") names the triple a reported
value must trace to: `run_id`, `scorer_hash`, `graph_version`. Those live inside a campaign's
`scores_parquet`-shaped store (a directory of fixed-name tables: `runs.parquet`,
`scores.parquet`, and siblings), and `.gitignore` deliberately never tracks that store --
"content-addressed and regenerable" is the stated reason, but a store built once from a live
sweep is regenerable only while the raw rollouts it was compacted from still exist, and several
of tonight's do not. Until it is copied somewhere git does not reach but a backup does, each
store exists in exactly one place on one machine.

This package does three things, kept in separate modules so each is independently testable:

- `discover.find_score_stores` finds every directory that is itself a store, by the only
  property that actually defines one: it directly holds at least one `*.parquet` file. Matching
  on names like `scores_parquet*` or `*farm*` (the convention, and a reasonable first pass) is
  not sufficient by itself -- `artifacts/gate_wiki2_stacked_20260918/farm` is a 400-symlink
  *input* farm pointing back at the repo's shared `runs/`, not a store, while the store it feeds
  is named `parquet_qwen3-8b-dpo-stacked-notdone-both` and matches neither pattern.
- `manifest.sha256_manifest` hashes every file under a directory, used twice: once to verify a
  backup copy matches its source file-for-file, and once inside a census so a *restored* copy
  can be checked against the census without needing the original again.
- `census.compute_census` reads (never writes) a store's tables and records what CLAUDE.md rule
  1 asks for -- distinct `arm_id` / `suite_id` / `split` / `code_version` (from `runs.parquet`)
  and `scorer_hash` / `graph_version` (from `scores.parquet`), each with counts -- plus a row
  count and sha256 per table. The census is small enough to commit even when the parquet is not,
  so a future reader can tell whether a restored store is the one a cell was computed from
  without re-deriving anything.

`backup.py` wires these into the copy-then-verify-then-census pipeline the RESULT.md for this
lane describes, and is the only module here with a CLI.
"""

from __future__ import annotations
