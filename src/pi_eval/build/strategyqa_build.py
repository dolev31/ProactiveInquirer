"""StrategyQA -> a PUBLIC paragraph corpus and a GOLD need graph.

WHAT MAKES THIS SUITE DIFFERENT FROM MuSiQue, and why it is worth the weaker gold: MuSiQue's
question is a COMPOSITION of its sub-questions, so its decomposition is recoverable from the
question text by anyone who reads carefully. StrategyQA's is not. "Did Aristotle use a
laptop?" does not mention Aristotle's dates or the invention of the laptop; the steps are
STRATEGIC, and a policy has to invent them. That is the closest thing in the public record to
the latent-need setting this project is about, which is exactly why a suite with noisier
edges still earns its place.

HONEST ACCOUNTING OF EDGE RECALL. Two mechanisms produce edges here, and they are not
equally good:

  * `#N` back-references. The annotators reused BREAK's convention, so many steps say
    "Is #2 greater than #1?". These are explicit and are extracted by the same regex
    MuSiQue uses: gold_verified="mechanical". MEASURED on the 2,290 annotated train
    questions: 2,999 of 6,720 steps (44.6%) carry at least one, and 2,272 questions (99.2%)
    contain at least one such step.

  * A prose back-reference ("the previous step", "the answer", "those ...") from a step to
    its immediate predecessor: gold_verified="observational", because nothing verified it.
    MEASURED on the full train split, 2026-08-24: it fires on 0 of 6,720 steps -- every step
    whose text matches the prose pattern already carries an explicit "#N", so the mechanical
    branch claims it first and the fallback never runs. A full build emits 4,483 edges, all
    of them gold_verified="mechanical". The branch is kept because a future annotation batch
    may not carry the placeholders, but it must not be described as a recall mechanism: it
    currently contributes nothing.

  Everything else is left EDGELESS on purpose. A step with no detectable reference becomes a
  depth-0 seed and the task degenerates to a flat facet set. That understates depth — but
  inventing a chain from step order would manufacture the exact structure this project
  claims to measure, and a fabricated prerequisite is far worse than a missing one. Edge
  recall on this suite is therefore strictly below MuSiQue's, and any depth number reported
  from it must be read as a LOWER BOUND.

OPERATION STEPS. 1,247 of 6,720 steps (18.6%) have no paragraph evidence from any annotator:
they are pure operations over earlier answers ("Is #2 before #1?"). They stay in the graph,
because they carry the dependency edges, but they are gold_partition="optional" and
gold_ablation_verdict="UNTESTABLE" — scoring them as required would build a ceiling below
1.0 into every RNR on this suite for a reason that has nothing to do with the policy.

WHAT IS PUBLIC AND WHAT IS NOT
    public   id, question, a pool of paragraphs {idx, title, text}
    gold     the yes/no answer, `decomposition`, the annotator `facts`, `evidence` paragraph
             ids, and `term`/`description` (the Wikipedia term the question was written from,
             which names the answer's subject)

THE POOL IS CONSTRUCTED, NOT SHIPPED. Unlike MuSiQue, StrategyQA distributes evidence
paragraphs but no distractors. A pool made only of gold paragraphs would make retrieval a
no-op and inflate every discovery metric to ~1.0, so we pad each task to POOL_SIZE with
paragraphs drawn from the same corpus by a per-task deterministic RNG, then shuffle so gold
never sits at a systematic index. The padding is part of the corpus hash, so a change to it
is a different corpus and cannot be confused with a change in policy behaviour.

Licence: MIT (Geva et al., TACL 2021). Source: the authors' zip, 2,290 annotated train
questions plus 490 unannotated test questions (2,780 in total). Only train is buildable:
the test file carries `qid` and `question` and nothing else.
"""

from __future__ import annotations

import json
import random
import re
import zipfile
from pathlib import Path

from pi_eval.build.common import (
    BuildResult,
    finalize_graph,
    require_raw,
    unit_uid,
    write_corpus,
    write_graphs,
)
from pi_eval.gold import GoldEdge, GoldNode

SUITE = "strategyqa"
CORPUS_ID = "strategyqa_v1"
GRAPH_VERSION = "v1"

ZIP_NAME = "strategyqa_dataset.zip"
ZIP_URL = "https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip"
SHA256 = {ZIP_NAME: "4911d85eb6721a93bed7645419df77e721808b32b9785dee14ad80e6249e0a90"}

QUESTIONS_NAME = "strategyqa_train.json"
PARAGRAPHS_NAME = "strategyqa_train_paragraphs.json"

POOL_SIZE = 20  # matched to MuSiQue so retrieval difficulty is comparable across suites

REF_RE = re.compile(r"#(\d+)")
# Deliberately narrow. "it"/"they" would fire on most English sentences and manufacture a
# chain out of nothing; these markers at least name a previous result.
PROSE_REF_RE = re.compile(
    r"\b(the previous|the above|the answer|the result|that number|this number|these|those)\b",
    re.IGNORECASE,
)


# ------------------------------------------------------------------ raw loading


def load_raw(raw_dir: Path, *, allow_download: bool = True, verify: bool = True) -> tuple:
    """Return (questions, paragraphs) from either the zip or an already-unpacked pair.

    Both shapes are accepted because the committed test fixture is the unpacked pair: a
    3-question zip in the repo would be a binary blob no reviewer can read in a diff.
    """
    loose_q, loose_p = raw_dir / QUESTIONS_NAME, raw_dir / PARAGRAPHS_NAME
    if loose_q.exists() and loose_p.exists():
        return json.loads(loose_q.read_text()), json.loads(loose_p.read_text())
    archive = require_raw(
        raw_dir / ZIP_NAME,
        ZIP_URL,
        expect_sha256=SHA256[ZIP_NAME] if verify else None,
        allow_download=allow_download,
    )
    with zipfile.ZipFile(archive) as z:
        return json.loads(z.read(QUESTIONS_NAME)), json.loads(z.read(PARAGRAPHS_NAME))


def step_evidence(rec: dict, step_i: int) -> list[str]:
    """Paragraph ids for one decomposition step: the FIRST annotator who supplied any.

    Not the union across the three annotators, and the reason is the matcher. A node counts
    as RESOLVED only when ALL of its gold uids were retrieved, so unioning three annotators'
    paragraph choices would demand that a policy retrieve every alternative any of them
    happened to pick — a requirement no correct policy could satisfy. One annotator's set is
    a coherent, sufficient account of that step, and taking the first is deterministic.
    """
    for annotator in rec.get("evidence", []):
        if step_i >= len(annotator):
            continue
        for alternative in annotator[step_i]:
            # entries are either a list of paragraph ids or the strings
            # "operation" / "no_evidence"
            if isinstance(alternative, list) and alternative:
                return list(alternative)
    return []


# ------------------------------------------------------------------ build


def build(
    *,
    raw_dir: Path | None = None,
    root: Path | None = None,
    limit: int | None = None,
    pool_size: int = POOL_SIZE,
    seed: int = 17,
    allow_download: bool = True,
    verify: bool = True,
) -> BuildResult:
    root = root or Path.cwd()
    raw_dir = raw_dir or (root / "data" / "raw" / SUITE)
    questions, paragraphs = load_raw(raw_dir, allow_download=allow_download, verify=verify)

    # Sorted so the distractor draw is a function of the corpus, not of dict iteration order.
    all_pids = sorted(paragraphs)
    public: list[dict] = []
    graphs: list[dict] = []

    for rec in questions:
        tid = rec["qid"]
        steps = rec.get("decomposition") or []
        if not steps:
            continue  # unannotated (the test split); there is no graph to build
        ev_per_step = [step_evidence(rec, i) for i in range(len(steps))]

        gold_pids: list[str] = []
        for ids in ev_per_step:
            for pid in ids:
                if pid in paragraphs and pid not in gold_pids:
                    gold_pids.append(pid)

        # Per-task RNG: reproducible, and independent of how many tasks precede this one, so
        # building a 3-task fixture and the full 2,290 gives the same pool for a given qid.
        rng = random.Random(f"{seed}:{tid}")
        pool = list(gold_pids)
        candidates = [p for p in all_pids if p not in set(gold_pids)]
        rng.shuffle(candidates)
        pool += candidates[: max(0, pool_size - len(pool))]
        rng.shuffle(pool)  # gold must not sit at a systematic index

        idx_of = {pid: i for i, pid in enumerate(pool)}
        texts = {pid: paragraphs[pid]["content"] for pid in pool}
        public.append(
            {
                "id": tid,
                "question": rec["question"],
                "paragraphs": [
                    {"idx": i, "title": paragraphs[pid]["title"], "text": texts[pid]}
                    for i, pid in enumerate(pool)
                ],
            }
        )
        graphs.append(_graph(tid, rec, steps, ev_per_step, idx_of, texts))
        if limit is not None and len(public) >= limit:
            break

    corpus, chash = write_corpus(root, SUITE, public)
    gold = write_graphs(root, SUITE, GRAPH_VERSION, graphs, corpus_hash=chash)
    return BuildResult(corpus, gold, chash, len(public), 0)


def _graph(
    tid: str,
    rec: dict,
    steps: list[str],
    ev_per_step: list[list[str]],
    idx_of: dict[str, int],
    texts: dict[str, str],
) -> dict:
    nodes: list[GoldNode] = []
    edges: list[GoldEdge] = []
    seeds: list[str] = []

    for i, text in enumerate(steps):
        nid = f"s{i + 1}"  # 1-indexed to match the #N placeholders
        ev = tuple(
            unit_uid(CORPUS_ID, tid, idx_of[pid], texts[pid])
            for pid in ev_per_step[i]
            if pid in idx_of
        )
        operation = not ev
        nodes.append(
            GoldNode(
                gold_suite=SUITE,
                gold_task_key=tid,
                gold_node_id=nid,
                gold_text=text,
                gold_kind="constraint" if operation else "fact",
                # The STEP is human-written even where the EDGE is only inferred; the two
                # provenances are tracked separately for exactly this reason.
                gold_provenance=("human_composed",),
                gold_provenance_primary="human_composed",
                gold_partition="optional" if operation else "required",
                gold_discoverability="unknown" if operation else "kb",
                gold_ablation_verdict="UNTESTABLE" if operation else "NECESSARY",
                gold_ev_uids=ev,
                gold_confidence=1.0,
                gold_graph_version=GRAPH_VERSION,
            )
        )

        refs = sorted({int(m.group(1)) for m in REF_RE.finditer(text)})
        valid = [n for n in refs if 1 <= n <= len(steps) and n != i + 1]
        for n in valid:
            edges.append(_edge(tid, f"s{n}", nid, "mechanical", 1.0))
        if not valid and i > 0 and PROSE_REF_RE.search(text):
            # Inferred, not observed: the step names "the previous answer" in prose, and we
            # attribute it to the immediately preceding step because that is the only
            # referent we can identify without a model.
            edges.append(_edge(tid, f"s{i}", nid, "observational", 0.5))
        elif not valid:
            seeds.append(nid)

    return finalize_graph(
        suite=SUITE,
        task_key=tid,
        nodes=nodes,
        edges=edges,
        seed_ids=seeds,
        # The dataset's answer is a bool; the Answerer emits text, so gold is the word.
        answer="yes" if rec.get("answer") else "no",
        aliases=("true", "yes") if rec.get("answer") else ("false", "no"),
        version=GRAPH_VERSION,
    )


def _edge(tid: str, src: str, dst: str, verified: str, confidence: float) -> GoldEdge:
    return GoldEdge(
        gold_suite=SUITE,
        gold_task_key=tid,
        gold_src_node_id=src,
        gold_dst_node_id=dst,
        gold_edge_kind="prerequisite",
        gold_verified=verified,  # type: ignore[arg-type]
        # Both mechanisms are regex extraction, so PROVENANCE is mechanical either way.
        # What separates them is VERIFICATION: a "#2" is a statement of dependency, a
        # prose marker is our reading of one.
        gold_provenance="mechanical",
        gold_confidence=confidence,
        gold_graph_version=GRAPH_VERSION,
    )
