# scripts/

Tools that sit beside the `pi` CLI, and the analysis behind individual studies. Every script
documents itself in its module docstring; the one-line summaries below are taken from those
docstrings.

The per-study folders read experiment outputs (`artifacts/`, `runs/`, `scores/`) that this
repository does not ship. They are kept so that each reported analysis has its code; they run
once those outputs are regenerated with the `pi` pipeline.

## Tools

| Script | What it does |
|---|---|
| `a7_artifacts.py` | Turn the A7 pass into things a trainer can consume, and refuse to build on a bad pass. |
| `annotation_fidelity.py` | Does a worksheet dump actually show the annotator everything they are judging? |
| `annotation_index.py` | Collect every annotation record into one place, with provenance for each rater. |
| `annotation_sizing.py` | How many annotations each gate and each human-centred metric actually needs. |
| `annotation_worksheet.py` | Dump a slice of bundle items as readable, BLINDED text for an annotator to work from. |
| `build_frames_corpus.py` | Build the FRAMES corpus: the FRAMES task file + every linked Wikipedia article. |
| `check_frames_contamination.py` | Is any FRAMES question also a training task of ours? |
| `check_no_home_paths.sh` | A committed absolute home path is the single most common way a research repo becomes unreproducible for anyone but its author. Fail loudly rather ... |
| `curate_pairs_with_a2.py` | Attach A2 rater verdicts to `pairs.jsonl` as METADATA. Never a filter. |
| `curate_pairs_with_diversity.py` | Attach a semantic-distance field to `pairs.jsonl` as METADATA. Never a filter. |
| `depth1_contrast_20260918.py` | Depth-one kill-switch contrast: inquirer_trained vs inquirer_depth1. |
| `dev_samples.py` | The dev slices the per-checkpoint Tier-A evaluator scores. |
| `heal_error_runs.py` | Delete run directories that FAILED, so a resume actually re-runs them. |
| `heldout_depth_structural_20260918.py` | Structural coarseness check for the two held-out matched-cost depth quantities. |
| `layer4_leakbound.py` | An upper bound on VERBATIM gold-text echo on the five suites layer 4 cannot cover. |
| `matched_cost.py` | P13 MATCHED COST: the tier-B gate re-read at equal retrieval cost. |
| `measure_sigma_j.py` | Measure sigma_J -- the noise floor of the judge -- by TEST-RETEST, and freeze it. |
| `paper_figures_iclr.py` | ICLR 2027 figures and tables, generated from the artifact records and never by hand. |
| `power_analysis.py` | Minimum detectable effect for every preregistered endpoint, THROUGH THE ACTUAL TEST. |
| `preflight_tau2_channel.py` | Did the policy's questions actually get ANSWERED? A precondition for any tau2 sweep. |
| `probe_retrieval.py` | Run the retrieval-sensitivity probe. Zero tokens, zero dollars, seconds. |
| `recover_graph_version.py` | Recover `graph_version`, the third element of the provenance triple CONTRIBUTING.md rule 1 requires (beside `run_id` and `scorer_hash`), for gate ... |
| `replay_budget_gate.py` | When would `BudgetGate` have refused a call in a recorded tau2 fork campaign? |
| `replay_post_hoc_harvest.py` | What the pre-change post-simulate harvest would do to a recorded fork campaign. |
| `report_forks.py` | Paired-fork engagement report over runs on disk: the paper's headline, by command. |
| `restore_corpus.py` | Rebuild `runs/` from the parquet compaction plus the response cache. No API calls. |
| `run_clean.sh` | Run a sweep with a CLEAN code_version while the working tree is dirty. |
| `run_tau2_forks.py` | Run forked tau2 rollouts from imported public trace prefixes. |
| `rung0_length_control.py` | Does the rung-0 winner's gain come from what the prompt says, or how long it is? |
| `seal_prereg.py` | Generate and seal a preregistration stage. |
| `select_checkpoint.py` | N1 SELECTION: the 08:00 stopping-round decision, plan v4 section 4.2, applied to gate dirs. |
| `size_cap32_sweeps.py` | Pure sizing check for the three new cap32 frontier grids -- no network call, no spend. |
| `stop_cell_census.py` | Census of the gate stop 2x2 cells on disk, and the identity that makes one caption useless. |
| `thinking_channel_probe.py` | Was the thinking channel ON for the base arm on the unverifiable listener? |
| `validate_pairs.py` | Does the preference dataset actually teach proactivity? Re-runnable, from the artifact alone. |
| `verify_tau2_fork_grading.py` | Does the FORK grading path still reproduce a known-good grade? |

## Folders

| Folder | Files | What it holds |
|---|---|---|
| `anchor_analysis/` | 1 | Human anchor: what the returned rater sheets say, and what they do NOT license. |
| `answer_loss/` | 3 | Lane L3: independent completeness check of a reanswer rows file against the planned requests, rebuilt from the run lists and turns.jsonl WITHOUT ... |
| `answer_node_coverage/` | 5 | Lane L1.11: does the trained arm's coverage gain reach the ANSWER-BEARING node, or does it land on prerequisites while the answer node stays ... |
| `answer_node_stop/` | 11 | Lane L6.1: condition the SFT STOP label on the ANSWER-BEARING node, and read what it buys. |
| `answerer_strength/` | 3 | Lane L1.10: does the evidence advantage reach the answer when the Answerer cannot compensate from its own knowledge. |
| `baseline_repro/` | 4 | Build an isolated, read-only-source scoring store for the baseline-completion reproduction. |
| `capaware/` | 1 | Score a cap-aware comparator campaign and contrast the two comparator arms. |
| `compaction_loss/` | 1 | Which runs hold turn records on disk and have no turn rows in the parquet. |
| `composed_pairs/` | 3 | Composed MuSiQue pairs (`musique_x2`) |
| `contributions_on_test/` | 4 | Lane L1.2, contribution B on test: trained-vs-teacher dominance at every shared cap. |
| `cost_reasoning_shares/` | 1 | Per-population reasoning-token shares and dollar-overstatement factors. |
| `decomposition_test/` | 2 | Build an isolated run-directory symlink farm for `pi compact` / `pi score`. |
| `diversity_per_seed/` | 2 | Lane L1.1: is the diversity-gate flip on the test split an instrument artifact? |
| `edge_validity/` | 4 | Is the recipe's out-of-order excess TARGETED or INCIDENTAL? precedence_violation_rate, split. |
| `evidence_backup/` | 5 | Lane L0.10: back up the isolated per-campaign score stores that published paper cells depend on, and make the backups verifiable without the parquet ... |
| `frames_diagnosis/` | 1 | Why the FRAMES answer contrast is null, and why no affordable experiment fixes it. |
| `frames_seeds/` | 1 | FRAMES cap-8 accuracy: `inquirer_trained` vs `inquirer_prompted`@Qwen3-8B-base, paired on task, per seed and pooled over three seeds, task-clustered ... |
| `frontier_calls/` | 5 | Lane L1.4: the budget frontiers on the calls axis, not the cap axis. |
| `granite_family/` | 2 | Lane L2.5: the Granite-3.3-8B family contrast, seed by seed, suite by suite. |
| `graph_sensitivity/` | 1 | How far does the headline move if the need graph's edges are wrong? |
| `horizontal_axis_separability/` | 1 | Whether the two breadth-over-independent-needs instruments are a second axis, or coverage restated. Every number in ... |
| `horizontal_design/` | 2 | Audit: do this project's gold graphs ever admit ALTERNATIVE SUFFICIENT SETS of required evidence, or is every required node conjunctively required ... |
| `horizontal_instrument/` | 2 | Q1 of artifacts/horizontal_axis_instrument_20260919/RESULT.md: a branching census taken directly from gold graphs, not from a metric. |
| `hpc/` | 6 | Checkpoint registration (`register_checkpoint.py`), model routing (`router_add.py`), checkpoint selection (`keep_best.py`) and the vLLM serving launcher (`serve.sh`, `serve_map.py`). The cluster job scripts that called them are not included. |
| `killswitch_mechanism/` | 7 | Mechanism analysis behind the depth-one kill-switch contrast (`depth1_contrast_20260918.py`): coverage at matched retrieval spend, control inventories and quality checks. |
| `label_ordering_test/` | 5 | Step 1 of 4: the run selection, as a symlink runs-root. |
| `label_variants_provenance/` | 3 | Why lock (a) of `recompute.py` fails: what the label-variants record could have computed. |
| `lane_s2/` | 3 | Relative weight distance between LoRA adapters: did a resumed DPO run keep its learning? |
| `launch_controls/` | 3 | Is the balanced prompted cell single-valued on commit, cap and PIN? |
| `length_audit/` | 4 | Length-exploitation audit over every Tier-A verdict we have. |
| `length_matched/` | 9 | Turn 7 `score_checkpoint.py` output files into the ladder table, the chance test, and the non-vacuity control that ... |
| `measure_20260919/` | 4 | Item 4 of the 2026-09-19 trained-vs-teacher measurement lane: screen every needs_answer=False, non-indexed metric in pi_eval.score.METRICS that is ... |
| `order_harm/` | 1 | Does resolving needs out of prerequisite order cost anything downstream? |
| `paired_audit/` | 1 | Recompute imitation-8B vs imitation-32B on the 970 ask/ask pairs with the paired test. |
| `plan_metrics_completed/` | 2 | Intervals for the near-zero cells on the completed comparator, on the PUBLISHED asymmetric basis. |
| `plan_metrics_symmetric/` | 3 | Does the strategyqa/wiki2 reproduction lock in ladder.py have the POWER to distinguish one aggregation rule from another, or does it pass for a ... |
| `precedence_mechanism/` | 6 | Lane L1.3: mechanism behind the adverse MuSiQue `precedence_violation_rate` contrast. |
| `price_tables/` | 2 | Dated per-model price tables for cost accounting; `PI_PRICE_TABLE` selects one. |
| `random_q_all_suites/` | 2 | Lane L1.8: the retrieval-volume control (`random_q`) contrasted against `inquirer_trained` and `inquirer_prompted`, on all three test suites, at ... |
| `rehydrated_answers/` | 3 | Coordinator gate, reproducible: does adding a column to pi_eval.schema.RUNS move any identity hash? Loads `src/pi_eval/schema.py` from two git refs ... |
| `rung32b_gate/` | 7 | Build the isolated symlink farm, compact into a PRIVATE parquet, and prove the rows survived. |
| `seed0_audit/` | 8 | The eight controls per suite and pooled for the recipe, under the published controls' rule. |
| `seed_identity/` | 16 | Answer token F1 at each arm's own stop under the shared ceiling of eight, the recipe against the same weights prompted, per training seed and pooled: ... |
| `seed_replicates/` | 4 | The seed-replicate contrast: matched-cost AND cap-8 coverage deltas, pins side by side. |
| `seedrep4/` | 4 | The four-seed table: the selected recipe's final stage at seeds 1, 2, 0 (fresh) and 3. |
| `seedrep_gate/` | 7 | The seed-replicate contrast for the rung-2 DPO stage of `qwen3-8b-dpo-stacked-notdone-both`. |
| `slot_bias_scope/` | 2 | Run the A6 top-tier slot-bias statistic (`pi_run.cmd_annotate._slot_bias`, task_type "A6") on the real, immutable A6 annotation records under ... |
| `stopping_answer_test/` | 2 | Lane L1.5: does the trained policy stop right, and does the coverage gain reach the answer. |
| `strategyqa_gold_audit/` | 1 | Per-task depth->=2 coverage: each label variant against the prompted base, StrategyQA test. |
| `structured_baselines/` | 3 | Gate the clean structured-baselines grid, then read its contrasts -- and REFUSE if a gate fails. |
| `table1_consolidate/` | 1 | The two headline rows that have no symmetric reading on the completed comparator. |
| `tau2_budget/` | 1 | Does the trained questioner respect the retrieval budget more often than the prompted one? |
| `tau2_campaign/` | 8 | The instruction ablation, with the population stated and the run-id sets digested. |
| `tau2_concordance/` | 6 | Dump a tau2 runs root to a JSON array of run records, for `fork_paired.py`. |
| `tau2_failure_taxonomy/` | 7 | Classify one Inquirer ask's TEXT as addressed to the customer or to the tools. |
| `tau2_full_protocol/` | 4 | Enumerate, shard and run one tau2-bench campaign under the BENCHMARK's own protocol. |
| `tau2_rerun/` | 4 | Read a tau2 rerun PILOT root: is the comparator degenerate under the enforced cap? |
| `tau2_trained/` | 1 | Can a tau2 consequence-labelled preference set be built from the fork rollouts on disk? |
| `tau2_transfer/` | 1 | Per-arm task-level levels for the paper's tool-use figure, from the rerun's records.json. |
| `two_axis_instrument/` | 1 | The two-axis 2x2 on the synthetic suite, with COST, over a grid of (facets, depth). |
| `two_axis_recipe/` | 3 | Per-shape matched-cost analysis for the two-axis campaign. |
