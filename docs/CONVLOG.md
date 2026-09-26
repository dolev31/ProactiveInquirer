# `convlog`: real agent–user sessions as proactivity data

Every other suite in this repository is a benchmark or a simulator. None of them contains a
person reacting to an agent that asked, offered, or was interrupted. `convlog` is that source:
Claude Code session transcripts, in which the user channel was genuinely open and the log
records what the user did next.

This document is the design record. It says what was measured, what may be labelled from a log
and what may not, and why the suite is not released.

## What was measured

41 gzipped session files (88 MB compressed) plus 18 per-session subagent folders, from one
collaborator's machine. Counted on 2026-09-02.

| | count |
|---|---|
| human prompts | 1,280 |
| assistant text blocks | 8,872 |
| explicit `AskUserQuestion` question/answer pairs | 48 |
| (assistant final text → next human prompt) adjacency pairs | 1,158 |
| … where the assistant's final text ended in a question | 461 |
| … user then asked their own question | 261 |
| … user added an item ("also …") | 98 |
| … user answered short and affirmative | 81 |
| … user corrected the agent | 26 |
| human prompts carrying two or more requests | 474 |
| one-prompt system-prompt probe sessions (noise) | ~10 |

The folder listing returned exactly 50 entries, which is Google Drive's page size, so the true
file count is probably higher. `scripts/convlog_fetch.py` pages to exhaustion for that reason.

## What the data can and cannot support

| behaviour | evidence | verdict |
|---|---|---|
| stop and report vs. ask the user | 461 trailing questions + 48 tool questions + interrupts | **primary target** |
| anticipate an unstated need | 98 "also …" additions, 474 multi-request prompts | **secondary**, as misses only |
| ask the *right* question | 48 explicit pairs | too thin to train content; evaluation only |
| minimise follow-ups | 1,158 adjacency pairs | **descriptive only** |

48 explicit questions train nothing. The 461 trailing offer-questions are the corpus: each is a
stop-or-ask decision with an observed outcome.

## The counterfactual rule

`pi_eval/metrics/human.py` records why FutureConversationCoverage was deleted: in a logged
conversation the user's third turn exists *because of* the assistant's second, so a need the
agent preempts never appears as a later utterance and the metric scores a miss on exactly the
success case. That failure is a property of scoring trajectories, and it is avoided here by
scoring only **(state, one observed object)** pairs.

| label | valid on a log | why |
|---|---|---|
| the next turn contained an item that was inferable from the state | yes | item and state both exist in the log |
| the agent asked, and the answer was already in or inferable from the state | yes | same |
| the agent asked, and the answer was genuinely new information | yes | same |
| the user interrupted where asking was available | yes | same |
| a need the agent preempted, and the user therefore never raised | **no** | right-censored, unobservable |

Legal: wasted-ask rate, necessary-ask precision, anticipation-miss rate, over-action rate, and
follow-ups per task **as a descriptive number**. Illegal: any coverage-of-future-needs metric,
and any cross-arm comparison of follow-up counts on logs — a different policy would have
produced different user turns, and those turns do not exist.

A candidate policy is therefore compared by **replaying held-out states**, never by rerunning a
conversation.

## Privacy, and why nothing here is released

The measured transcripts carried 112 live GitHub tokens across 5 files, 137 Overleaf project
URLs, and thousands of addresses. `pi_eval/build/scrub.py` runs where raw is read, substitutes
stable opaque handles, and then **re-scans its own output**, aborting the build on any residue.

Two things it deliberately leaves alone, recorded as accepted residual risk in D17:

- **bare `@handles`** — `@leshem.choshen` and `@pytest.mark.parametrize` are the same string to
  a regex, and these states are full of code;
- **SLURM job ids** — a seven-digit integer is also a line number and a byte count.

Both are tolerable only because the corpus is **never redistributed** (D16). `data/raw/convlog/`
and `data/corpora/convlog/` are both git-ignored; unlike every other suite, the *corpus* is
private too, because a scrubbed transcript is still a transcript. What may be published is
labels keyed by `item_id` with no text.

Ingestion requires `--consent-sha`, a digest of the collaborator's written consent recorded in
`docs/DATA.md`. There is no flag to skip it.

## Decision points

A decision point is one of three things, and the next user turn is the answer key that never
enters the state:

- **`yield`** — the last assistant message before the user speaks. The agent chose to report;
  if its final text ends in a question, it chose to ask.
- **`ask_tool`** — an `AskUserQuestion` call, the explicit form of the same choice.
- **`interrupt`** — the user cut the agent off. The agent was acting where it should have
  checked.

Noise removed: subagent sidechains (a subagent is not the user), meta records, queue and
file-history bookkeeping, and sessions with fewer than three human prompts. A context-compaction
summary opens a new segment, because the agent at that point could not see what came before it
either.

`template_id` is the session id, so a whole conversation lands on one side of the train/test
wall. Two decision points from one session share its task, its repository and its user's habits;
letting them straddle the wall is the ordinary way a clean split leaks.

## Stage A result, measured 2026-09-02

The committed parser, run over all 41 session files. Aggregates only; no transcript content
left the machine.

| | |
|---|---|
| session files | 41 |
| kept as conversations | 25 |
| dropped as probes or empty (fewer than 3 human prompts) | 16 |
| aborted by the scrubber | 0 |
| decision points | 2,146 (median 25 per session, max 437) |

| decision point kind | n | observed action | n |
|---|---|---|---|
| yield | 2,065 | report and continue | 1,308 |
| ask_tool | 48 | ask the person | 805 |
| interrupt | 33 | acting when interrupted | 33 |

| what the person did next | n |
|---|---|
| something else | 956 |
| asked their own question | 652 |
| added an item | 324 |
| corrected the agent | 139 |
| short affirmative | 42 |
| interrupted | 33 |

Split at session granularity: 1,543 train, 600 test, **3 dev**.

### Gate A

| check | result |
|---|---|
| secrets or addresses surviving in any state field | 0 |
| a decision point's own next turn inside its state | 0 |
| states that render to a prompt | 2,146 of 2,146 |
| rendered prompts carrying the row's identity | 0 |
| sessions the scrubber refused | 0 |

Three things the run found that reading the code would not have.

1. **573 of 2,146 states carried the session id inside their own text**, in scratchpad paths
   the tool writes and people paste. `template_id` is the session id, so this put the split key
   in the prompt. Now redacted in the parser. The test asserting otherwise had been passing
   only because the synthetic fixture never mentions a session.
2. **11.3% of decision points repeat a prompt the person already sent verbatim**, and 220 of
   those come from one session that monitors a cluster on a loop, which alone is 20% of every
   decision point mined. Not a leak, but not independent draws either, so the record carries
   `repeats_earlier_turn` for an exporter to cap on.
3. **Three dev rows.** Twenty-five sessions cannot support a three-way split at session
   granularity. Either more sessions arrive (the Drive listing is paginated and was truncated
   at 50), or dev and test become one held-out set with the reason written down.

Gate A asked for at least 30 conversations and at least 800 decision points. Decision points
clear by a wide margin; conversations do not.

## Stage B: the instruments

Three instruments, each judging one state and one observed object. They extend the A1–A7
machinery rather than duplicating it: same blind bundle and separate key, same content-addressed
`item_id`, same unanimous human-only consensus, same Krippendorff alpha, same planted foils.

```
B1  was this question necessary given the state?
    necessary | answer_in_state | answer_inferable | default_existed | cant_tell
    -> wasted_ask_rate, necessary_ask_precision

B2  was this next-turn item inferable from the state?          (one unit per item)
    stated_already | inferable_from_state | user_private | new_task | cant_tell
    -> anticipation_miss_rate

B3  should the agent have stopped and reported here?
    stop_was_right | should_have_asked | should_have_continued
    | should_have_stopped_earlier | cant_tell
    -> over_action_rate, and the supervised "should have asked X" target
```

Two verdicts must name their own evidence or they are refused, on both the human and the model
path. `default_existed` says the agent should have taken an obvious default, and with no
statement of what that default was it is unfalsifiable while still landing in the wasted-ask
numerator. `should_have_asked` is the supervised target itself, and without the question text
there is no training example, only a complaint.

B2 emits one unit per item rather than one over the set, so two annotators agreeing on three of
four items do not score as total disagreement. An item nobody labelled yields nothing at all.

None of the three writes anything gold-side, for the same reason A5 and A6 do not: a judgment
about a conversation is not a claim about a task's graph.

### Decomposition is mechanical, on purpose

B2 needs the next user turn split into the separate things it asked for. That split is done by
sentence boundaries and coordinating cues, not by a model. A model-produced decomposition would
make every anticipation number partly a test of judge agreement, and 474 of the 1,280 human
prompts already carry two or more requests, so the mechanical split has something to find.

### Two rater families, and why the pinned one is not trusted alone

The repository pins `PI_MODEL_ANNOTATOR` to `azure/gpt-5.6-terra`. That is also the model the A7
campaign quarantined for answering the slot rather than the content, at z = −6.60, despite
catching 99 of 99 foils. The pilot therefore runs it alongside `openai/aws/gpt-oss-120b`,
reports each model's foil catch separately, and computes agreement across families rather than
within one. Two runs of one model agreeing says the model is deterministic, not that the
instrument is legible.

### Gate B

| criterion | bar | why it can kill the stage |
|---|---|---|
| foil catch, per rater | ≥ 0.90 | a rater missing items built to be answerable from the state is not reading the state |
| Krippendorff alpha, per unit kind, across families | ≥ 0.60 | below this the instrument is not legible enough to label with |
| wasted-ask base rate | ≥ 0.15 | if nothing is wasted there is nothing to learn |
| anticipation-miss base rate | ≥ 0.10 | same |
| coverage, per unit kind | ≥ 0.50 | a rater answering `cant_tell` everywhere clears agreement trivially |

Foils are excluded from the base rates: a foil is built to have a known answer, so counting it
measures the plant rather than the corpus. A rater shown no foil receives no foil score rather
than a perfect one.

### Running it

```
python scripts/convlog_pilot.py build --raw <sessions> --out <dir> --consent-sha <sha>
.venv/bin/pi annotate llm --bundle <dir>/bundle.json --out <dir>/records.terra.jsonl \
    --model azure/gpt-5.6-terra --concurrency 8
.venv/bin/pi annotate llm --bundle <dir>/bundle.json --out <dir>/records.oss.jsonl \
    --model openai/aws/gpt-oss-120b --concurrency 8
python scripts/convlog_pilot.py gate --out <dir>
```

The consent digest is the sha256 of `docs/consent/convlog.md`, derived at build time rather than
pasted into a constant. There is no flag that skips it.

## Stage B result: Gate B fails on agreement, measured 2026-09-02

200 items (B1 80, B2 80, B3 40) with 24 planted foils, drawn from the 2,146 decision points,
rated independently by two model families. `azure/gpt-5.6-terra` parsed 199 of 200 for $1.89;
`openai/aws/gpt-oss-120b` parsed 191 of 200 for $0.28.

| check | gpt-5.6-terra | gpt-oss-120b | bar |
|---|---|---|---|
| foil catch, by class | 0.957 | 1.000 | ≥ 0.90 |
| foil catch, exact label | 0.913 | 0.739 | reported, not gated |

| unit kind | alpha, full vocabulary | alpha, collapsed to the metric's classes | raw agreement, collapsed |
|---|---|---|---|
| B1 | −0.044 | 0.030 | 51.5% |
| B2 | 0.339 | 0.276 | 74.3% |
| B3 | 0.277 | −0.113 | 77.5% |

Coverage was 1.000 on all three, so nobody dodged with `cant_tell`, and the pooled base rates
cleared their bars. **Gate B still fails, on agreement alone, for all three instruments.**

### What that means, and what it does not

Both raters demonstrably read the state: the foil catch is 0.96 and 1.00 once a foil is scored
against the class the metric uses rather than the exact word. So the failure is not attention.
It is the judgment.

The decisive number is what each family would have reported as the headline:

| the rate a reader would quote | gpt-5.6-terra | gpt-oss-120b |
|---|---|---|
| share of questions that were not needed | 73.1% | 36.4% |
| share of next-turn items that were inferable | 88.7% | 63.3% |
| share of stops that should have been a question | 5.0% | 17.5% |

A number that halves or doubles depending on which model was asked is a property of the rater,
not of the conversations. Collapsing the five-label vocabularies to the binary each metric
actually uses does not rescue it: B1 goes to 0.030 and B3 to −0.113.

The largest single disagreement is diagnostic. On B1, the modal cell is gpt-5.6-terra saying an
obvious default existed where gpt-oss-120b says the question was necessary, 22 of 67 items.
Whether a default was "obvious" is a judgment about risk tolerance that the state does not
settle, which makes it a defect in the instrument rather than in the raters.

An external-validity check in the style of A5, asking whether a verdict predicts something the
rater was blinded from, is **underpowered and settles nothing**: only 3 of 67 B1 items drew a
short affirmative reply, and `should_have_asked` was returned twice by one family and seven
times by the other. It is reported here so that nobody re-runs it expecting an answer.

### That first reading was wrong, and two stronger raters show why

The two families above are the weakest the router serves, and the campaign's own rater
precedence puts `claude-fable-5` ahead of both. "Two weak raters disagree" does not distinguish
an illegible instrument from weak raters. The same 200 items, the same prompts and the same seed
were therefore re-rated by `aws/claude-opus-5` ($6.80, 184 of 200 parsed) and
`gcp/gemini-3.1-pro-preview` ($7.21, 200 of 200).

Every family reads the state. Foil catch by class: 0.957, 1.000, 1.000, 0.958.

Pairwise alpha, collapsed to the class each metric uses:

| pair | B1 | B2 | B3 |
|---|---|---|---|
| terra vs gpt-oss-120b | 0.030 | 0.276 | −0.113 |
| terra vs opus-5 | 0.222 | **0.584** | −0.013 |
| terra vs gemini-3.1 | 0.218 | 0.052 | −0.013 |
| gpt-oss-120b vs opus-5 | 0.126 | 0.199 | −0.069 |
| gpt-oss-120b vs gemini-3.1 | 0.278 | 0.400 | −0.082 |
| **opus-5 vs gemini-3.1** | **0.512** | 0.127 | degenerate |

The strong pair agrees on B1 at 0.512 against 0.030 for the weak pair, a seventeen-fold
difference, with 71.6% raw agreement. B1's base rates now cluster: 73.1%, 67.2%, 62.7% and one
outlier at 36.4%. So the first pilot largely measured its raters.

### Three findings from the four-family run

**1. `gpt-oss-120b` is an outlier, and the gate now says so by rule.** Median absolute
deviation, fixed in code so it cannot be chosen after seeing which answer it gives, flags it on
B1 and B3. The rule names a rater; it never drops one. Excluding data is a decision a person
makes and writes down, which is how the A campaign quarantined a model that caught 99 of 99
foils.

**2. B3 was sampled where its own behaviour cannot occur.** Its positive verdict is "the agent
acted where it should have checked", whose observable signature is the person interrupting or
correcting. Drawing from every yield put almost every item where that cannot happen, and both
strong raters returned `should_have_asked` zero times in 39 and 40 items. This is the A7
campaign's mistake in its own words: *"concentrates forks at t0, so the pipeline has been
sampling the states where its own target behaviour cannot occur."*

B3 now prefers interrupts and corrections. A fresh 60-item slice (11 interrupts, 49
corrections) was rated by three families. **The verdict did not change**: opus-5 and gemini-3.1
each returned `should_have_asked` 0 of 60, including 0 of 11 interrupts. Only gpt-5.6-terra
said it, on 11 of 49 corrections, and the outlier rule flags terra on that slice.

That is a finding about the corpus, not the instrument. **A correction is almost never a failure
to ask.** People correct what the agent did, not that it acted without checking. Over-action as
a missing question is close to absent here.

**3. The obvious fix to B1 was wrong, and a controlled pair proved it.** Fifteen of the strong
pair's nineteen disagreements sat on `default_existed` against `necessary`, so that option was
the suspect. It was tested as `B1b` — the same 80 items, the same contexts and ids, one option
removed, both record sets kept under their own task type — rather than by editing B1, because an
in-place edit produces two numbers nobody can compare.

| | raw agreement | collapsed alpha | wasted-ask rate |
|---|---|---|---|
| B1, five labels | 71.6% | 0.512 | 64.9% |
| B1b, four labels | 72.3% | 0.476 | 25.4% |

Read at the time as: removing the option leaves agreement flat, lowers alpha, and halves the
rate, so it carries signal rather than noise. **That reading was right about the conclusion and
wrong about one of its three numbers**, which the replication below establishes.

### The B1b comparison, replicated at ten times the sample

59% of the shipped training rows are `default_existed` verdicts, so the whole dataset leans on
that n=80 comparison. It was re-run paired on all 805 B1 items, the same raters, the same
prompts, one option removed. 651 items were decided under both vocabularies.

| | raw agreement | collapsed alpha | wasted-ask rate |
|---|---|---|---|
| B1, five labels | 57.7% (n=721) | 0.362 | 76.6% |
| B1b, four labels | 51.3% (n=653) | 0.383 | 53.5% |

McNemar on the paired agreement indicator: 139 items where the raters agreed under B1 but not
under B1b, against 90 the other way. **chi-squared 10.06, p=0.0015, in B1's favour.**

Three corrections follow, and the middle one matters most.

1. **The conclusion holds and is now properly supported.** At n=80 the agreement difference
   looked flat; at n=651 it is real and significant. B1 keeps `default_existed`.
2. **The alpha comparison did not replicate — it reversed.** 0.512 against 0.476 at n=80 became
   0.362 against 0.383 at scale, now favouring B1b by a hair. That number was noise, and it was
   one of the three I cited. The conclusion never rested on it; the McNemar test does.
3. **The pilot's 80 items were not representative.** Raw agreement on B1 was 71.6% there and
   57.7% across all 721, so the pilot slice was materially easier than the corpus.

A peer session prompted this check by reporting that its own single-slice effect (+0.11 reward
from inert padding, p=0.032) reversed to −0.0968 on a fresh slice. The lesson generalises: a
one-slice comparison with a small n is a hypothesis, and this project has now had two of them
fail to replicate on the same afternoon.

### Where this actually lands

Gate B does not pass: the best cross-family alpha is 0.512 on B1 and 0.584 on B2, against a bar
of 0.60. But the failure is now specific and the data is usable in a narrower form. Unanimity
between the two strong raters, which is what `annotate.consensus` already computes:

| unit kind | unanimous | of | share | target-class rate, unanimous vs all |
|---|---|---|---|---|
| B1 | 52 | 67 | 77.6% | 69.2% vs 64.9% wasted |
| B2 | 106 | 163 | 65.0% | 84.0% vs 68.3% miss |
| B3 | 31 | 39 | 79.5% | — |

The unanimous subset is mildly skewed toward the positive class on B1 and materially on B2, and
that skew has to be reported wherever the subset is used.

Scaled to the corpus, that is roughly 620 B1 labels from the 805 questions actually asked, plus
several thousand B2 units, at about $0.07 per item per rater — on the order of $150 to label all
2,146 decision points with both strong raters.

### The decision this leaves open

Two defensible routes, and the choice is a standard-of-evidence call rather than a technical one.

1. **Ship the unanimous subset**, treating strong-pair unanimity as the label and reporting the
   coverage and the skew above. Immediately actionable and gives real training data.
2. **Adjudicate the split items by hand.** 15 B1 and 57 B2 items in the pilot; roughly 20–30% of
   any full run. This is what the A campaign concluded its own instruments needed, and it is the
   only route that reaches the 0.60 bar honestly.

What is not defensible is lowering the bar to whatever the models happen to reach.

## v2: both sources, three raters, published to the Atlas

The corpus was extended from the Drive folder alone to **both sources** — the 25 usable Drive
conversations plus 10 of the owner's own local sessions, 4,114 decision points over 35 sessions
against 2,146 over 25. The local sessions are the richer half for explicit questions: 119
`AskUserQuestion` pairs against 48.

A third strong rater was added so `annotate.consensus` could resolve by majority, which it
refuses below three raters because a majority of two is one person outvoting nobody. Raters:
`aws/claude-opus-5`, `azure/gpt-5.6-sol`, `gcp/gemini-3.1-pro-preview`.

| | first shard | v2 |
|---|---|---|
| rows | 491 | **1,126** |
| sessions | 23 | **34** |
| train / dev / test | 378 / 1 / 112 | **776 / 154 / 196** |
| ASK share | 16.7% | **21.0%** |
| unanimous / majority | 491 / 0 | **538 / 588** |
| awaiting adjudication | 1,783 | 868 |

Three things improved and one did not.

**The dev split is fixed.** One row became 154, because 35 sessions
can carry a three-way split at session granularity where 23 could not.

**The balance improved**, from 16.7% ASK to 21.0%. Still STOP-dominated, and
the first binding property below still holds.

**`should_have_asked` finally reached a consensus — once, in 848 units.** Three raters found one
state in the whole corpus where the agent should have asked and did not. That is a stronger
version of the same finding, not a reversal of it: the phenomenon exists and is genuinely rare.

**Agreement did not improve.** Alpha is 0.258 on B1, 0.372
on B2 and 0.307 on B3, and Gate B still fails. Foil catch is high for every
rater (0.976, 0.967, 0.947), so
all three read the state; they disagree about the judgment. Adding a rater bought coverage, not
legibility.

The outlier rule now names a different rater on each instrument — gpt-5.6-sol on B1 and B3,
gemini-3.1-pro on B2 — which is itself informative: no single model is the problem, and the
disagreement is not one bad rater but three defensible readings.

### What the raters disagree about, and what it does to the rows

Measured on the final v2 labels, foils excluded.

| instrument | opus-5 | gpt-5.6-sol | gemini-3.1-pro |
|---|---|---|---|
| B1, share of asked questions judged unnecessary | 69.6% | **86.9%** | 64.0% |
| B2, share of next-turn items judged a miss | 84.1% | 81.5% | **54.0%** |
| B3, share judged "should have asked" | 0.4% | 2.5% | 0.7% |

**The core judgment is shared; the disagreement is at the edges.** All three call
`default_existed` the modal B1 verdict at 58% to 65% — they agree the agent usually had an
obvious default it could have taken. They differ on the residual: sol pushes another fifth into
`answer_in_state` where the other two say `necessary`. On B2 the split is one specific reading —
gemini calls 38% of items `user_private` against 10% and 13%, treating "the person wanted X" as
a preference no agent could infer.

**No single rater is the problem.** In a two-against-one split the lone voice is sol 253 times
on B1, gemini 308 times on B2, and sol 58 times on B3. A bad rater would have to be a different
model on each instrument.

**And the instinct to keep only the high-confidence rows makes the imbalance worse.**

| | rows | ASK share | dominant verdict |
|---|---|---|---|
| unanimous | 538 | 14.5% | `default_existed` (343) |
| majority | 588 | **27.0%** | `necessary` (158) |

The ASK examples live disproportionately in the rows the raters argued about. Filtering to
unanimity would drop the dataset to 538 rows **and** take the ASK share from 21.0% to 14.5%,
buying confidence with the exact signal the dataset is thinnest in. The 158 `necessary` rows
resolved only by majority are therefore the highest-value adjudication targets in the queue:
they are the ASK supervision whose label is least certain.

### B4: asking a strong model to propose the missing question

B3 asked whether a question was missing and answered yes, unanimously, once in 848 units. That
is a recognition task with a base rate near zero, so B4 inverts it: Claude Opus 5 is shown the
same state and **proposes** the question, or declines. The proposal is then put through B1
**verbatim** by two other raters — identical prompt, origin held in the key — so the acceptance
rates for proposed and real questions are directly comparable.

**The proposer declined 837 of 849 times.** A 1.41% proposal rate, from a model explicitly
asked to find a missing question and free to invent one.

| what the person actually did next | n | proposed a question |
|---|---|---|
| nothing notable | 494 | 1.2% |
| asked their own question | 200 | 1.5% |
| added an item | 70 | 1.4% |
| **corrected the agent** | 52 | **1.9%** |
| **interrupted** | 18 | **5.6%** (1 case) |

**The rate is flat where the person pushed back.** If missing questions caused corrections, the
rate would climb there; 1.9% against a 1.2% baseline is not a climb. That is a third
independent confirmation, through a generation framing rather than a recognition one, that a
correction is almost never a failure to ask.

Blind validation of the 12 proposals:

| | judged necessary |
|---|---|
| gpt-5.6-sol, on real logged questions | 13.4% |
| gpt-5.6-sol, on proposed questions | 33.3% (2.5x) |
| gemini-3.1-pro, on real logged questions | 36.0% |
| gemini-3.1-pro, on proposed questions | 75.0% (2.1x) |
| **both raters necessary** | **4 of 12** |

Both raters accept proposals at roughly twice the rate of real questions. Two readings fit and
n=12 cannot separate them: either the proposer is well calibrated, having declined 98.6% of the
time so its few proposals are pre-filtered, or a question written by a model that just read the
state naturally references that state's own tensions and reads as well-targeted. The comparison
was built to detect the second, and it cannot rule it out.

### A second proposer, because the first one wrote the trajectories

Every transcript here was produced by a Claude model, and `claude-opus-5` produced the
plurality of assistant turns. It was also the sole B4 proposer, so B4 ran single-rater with the
model judging its own work. `gemini-3.1-pro` was run over the same 850 states as a control.

| proposer | proposed | rate | both validators necessary | validated rate |
|---|---|---|---|---|
| claude-opus-5 | 12 | 1.41% | 4 | **0.47%** |
| gemini-3.1-pro | 56 | 6.59% | 13 | **1.53%** |

**The proposer moves the number by more than 3x**, and the two agree on only 6 of the states
where either saw a gap. A single proposer measures its own threshold as much as it measures the
corpus. An earlier version of this document reported 0.47% as *the* bound; that was wrong, and
the honest statement is a range whose ceiling is under 2%.

This is not self-favouring. The Anthropic model proposed *fewer* missing questions on its own
trajectories, which is the conservative direction, and it matches the within-rater test on B1
where it is 9.5 points harsher on its own questions than either control rater is.

**What survived the change of proposer is the finding worth keeping.** The proposal rate is flat
across corrections for both: 1.9% against a 1.2% baseline for one, 7.7% against 7.3% for the
other. The absolute rates differ fourfold and the correction gap appears in neither.

**Verdict: B4 is a measurement, not a source.** 1,699 proposals and $60 produced 17 rows two
validators called necessary, against 236 from B1 on questions people were really asked. The
number it produces is the useful one: **between 0.47% and 1.53% of states where the agent stayed
silent hold a question a strong proposer suggests and two independent raters endorse.** That
bounds the headroom for proactive asking in real agent sessions, and the bound is low.

### A defect this run exposed

Carrying records between two bundles left 114 to 157 items **per rater** holding two records
with two different prompts. `annotate.item_id` hashes task type, provenance and payload but
**not context**, while a B1 foil rewrites the question in context — so a planted item and its
unplanted twin shared an id, `--resume` said "already done", and the prompt had changed
underneath it. The id now carries a digest of what was shown, present on every item so it
cannot mark a foil. The stale records were dropped by matching each record's prompt against the
bundle's current one, and the affected items re-rated.

### Published

The dataset is on the Inquirer Data Atlas as its own section, in counts rather than rows:
D16 keeps the suite unredistributed, and publishing a page uploads it. The page says so where a
reader would otherwise assume the section is unfinished.

## Stage C: the curated training set, built 2026-09-02

The full corpus was labelled by the two strongest raters available, `aws/claude-opus-5` and
`azure/gpt-5.6-sol`, over 1,777 items: every one of the 805 questions the agent actually put to
a person, all 172 states where the person interrupted or corrected, 800 anticipation items, and
160 planted foils. 3,332 records, $52 of rating.

| check | opus-5 | gpt-5.6-sol | bar |
|---|---|---|---|
| foil catch, by class | 0.957 | 0.924 | ≥ 0.90 |
| coverage | 1.000 | 0.986 | ≥ 0.50 |

| unit kind | alpha, across families | n |
|---|---|---|
| B1 | 0.252 | 1,442 |
| B2 | **0.591** | 3,599 |
| B3 | 0.473 | 343 |

B2 lands at 0.591 against a 0.60 bar. B3 rose from 0.194 on the mis-sampled pilot to 0.473 once
it was drawn where over-action can occur. B1 stays low, and the reason is marginal skew rather
than disorder: the two raters agree on 77% of items, but with both calling three quarters of
questions unnecessary, 63% agreement is expected by chance, so alpha discounts almost all of it.

### The shipped dataset

`data/rl/convlog/` holds `sft.jsonl`, `adjudication.jsonl` and `manifest.json`.

| | |
|---|---|
| rows | 491, from 23 sessions |
| split | 378 train, 112 test, 1 dev |
| action | 82 ASK, 409 STOP (**16.7% ASK**) |
| source | 398 from B1, 93 from B3 |
| train/test session overlap | 0 |
| secrets or addresses surviving | 0 |

Each row is the prompt an Inquirer would see at one real decision point, and the action both
raters independently agreed was right there. Every row loads as
`pinq_train.export.dataset.Example`, so rung 1 reads it without a second loader.

### Three properties a reader has to know before training on it

**1. It is 83% STOP, and that is what the corpus says.** Of the questions the agent actually
asked, both raters call three quarters unnecessary. Trained on alone, this teaches "do not ask",
which is one half of proactivity presented as the whole of it. The share is on the manifest as
`ask_share` rather than left to be discovered. The benchmark suites are the ASK-heavy
counterweight; a mixture, not a replacement.

**2. It cannot teach the agent to ask something it did not think of.** B3 returned
`should_have_asked` for a unanimous pair **zero times in 343 units**, including every state
where the person interrupted or corrected. That held on the mis-sampled pilot and again after
the sampling was corrected. So the honest description of this dataset is: it teaches an agent
to stop asking unnecessary questions and to keep asking necessary ones. The finding behind it
is that **a correction is almost never a failure to ask** — people correct what the agent did,
not that it acted without checking.

**3. One dev row.** Twenty-three sessions cannot support a three-way split at session
granularity. Either more sessions arrive, or dev and test merge into one held-out set with the
reason written down.

### The ASK target is a question, not the message that contained it

The first export's ASK targets had a median of 2,368 characters and a maximum of 6,622, because
a `yield` decision's recorded question is the agent's entire final message, which merely happens
to end in a question mark. Training on that teaches a model to emit a status report where a
question belongs. The extraction now happens at the target rather than at parse time, so the
label stays synchronised with the block the annotator actually read. Median ASK target is now
63 characters and every one ends in a question mark.

### The 1,783 splits are the next artifact, not a loss

`adjudication.jsonl` carries every unit the two raters disagreed on, with the state attached,
because a verdict pair without the state it was given for cannot be adjudicated. 330 B1, 1,411
B2 and 42 B3. These are the items a person should read, and reading a few hundred of them is
the shortest route to clearing the 0.60 bar honestly.

## Status

Stage A is complete. Stage B ran with **four rater families across three slices** and does not
clear its 0.60 agreement bar; the best cross-family alpha is 0.512 on B1 and 0.584 on B2. The
first two-rater reading, that the instruments were defective, was wrong: it largely measured the
two weakest models the router serves.

Stage C ran on the full corpus with the two strongest raters and shipped 491 curated rows to
`data/rl/convlog/`, plus a 1,783-unit adjudication queue. The three properties above are
binding on anyone training with it, and the suite is still not released (D16).

Total spend on rating: **$424** over 14,400 records across nine slices, summed from the
per-record `usd` field rather than from the figures each pass printed. An earlier figure of $99
in this document was wrong: it added up the spend lines that happened to be captured and missed
two passes that were stopped mid-run and one whose line was not read back. The per-record sum
is the reproducible number, and it is the one this project's own pricing rule asks for -- USD
computed as tokens times pinned rates, never read off a provider response.

| slice | records | USD |
|---|---|---|
| full (B1, B2, B3 over the whole corpus) | 3,332 | 92.72 |
| full_b1b (the paired replication) | 1,591 | 43.22 |
| pilot (four families, 200 items) | 774 | 15.51 |
| pilot_b3 (targeted over-action slice) | 180 | 5.06 |
| pilot_b1b (the n=80 comparison) | 158 | 4.51 |

Per item per rater: $0.022 with gpt-5.6-sol, $0.034 with claude-opus-5.
