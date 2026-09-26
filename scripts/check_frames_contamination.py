#!/usr/bin/env python3
"""Is any FRAMES question also a training task of ours?

    python scripts/check_frames_contamination.py --root . --frames-root .

FRAMES is admissible as an external benchmark only if the trained policy never saw its
questions. It was hand-written by its authors after they discarded template generation, and
none of musique, StrategyQA or 2Wiki went into its construction -- but "the authors say so" is
not a measurement, and the three suites we mine are all Wikipedia-derived multi-hop QA. This
is the measurement.

TWO TESTS, BECAUSE EXACT EQUALITY IS THE WEAK ONE. A paraphrase of a training question is
contamination just as surely as a copy of it, and a single re-worded clause defeats string
equality. So:

  * NORMALISED EXACT: lowercase, punctuation stripped, articles removed, whitespace collapsed.
    The same normalisation `pi_eval.metrics.quality.normalize` applies before scoring an
    answer -- reimplemented here rather than imported, because a script outside pi_eval may
    not import it (contract 1, the gold firewall), and because this check must be runnable
    with PI_GOLD_ROOT unset.
  * TOKEN-SET JACCARD >= 0.8 over the same normalised tokens. Set rather than sequence, so a
    reordering does not hide; 0.8 rather than something lower because at 0.5 the multi-hop
    phrasebook ("who is the spouse of the director of ...") matches everything and the report
    stops being readable.

WHERE THE QUESTIONS COME FROM, AND WHY NOT FROM GOLD. The obvious source is
data/gold/graphs/<suite>/v1.jsonl, and it does not have one: a gold record carries exactly
gold_{suite,task_key,nodes,edges,facets,seed_node_ids,graph_version,answer,aliases,canary,
corpus_hash} and no question text at all. The question lives in the PUBLIC corpus, which is
where this reads it, and the split comes from `pinq.splitting.split_of` -- the single
definition that both the runner and the exporter use. So this script touches no gold and
imports no pi_eval.

EXIT 1 if any exact match is found. Near-duplicates are reported and do not fail: whether a
Jaccard-0.84 pair is contamination or two people asking about the same famous person is a
judgement, and it has to be made by reading the pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
import sys
from pathlib import Path

TRAIN_SUITES = ("musique", "strategyqa", "wiki2")
NEAR_DUPLICATE_JACCARD = 0.8

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    """Byte-for-byte the rule in pi_eval.metrics.quality.normalize. Duplicated on purpose:
    see the module docstring. test_frames_suite.py pins that the two definitions agree."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def frames_questions(frames_root: Path) -> list[tuple[str, str]]:
    """(task_id, question) straight from the pinned upstream TSV.

    The TSV rather than the built corpus, so this check runs before the corpus is built and
    cannot be fooled by a build that dropped rows.
    """
    tsv = frames_root / "data" / "raw" / "frames" / "test.tsv"
    if not tsv.is_file():
        raise SystemExit(
            f"no FRAMES task file at {tsv}. Run scripts/build_frames_corpus.py first, or pass "
            "--frames-root at the tree that holds it."
        )
    out: list[tuple[str, str]] = []
    with tsv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            idx = str(row.get("") or row.get("Unnamed: 0") or "").strip()
            out.append((f"frames_{int(idx):04d}", str(row.get("Prompt") or "").strip()))
    return out


def _corpus_dir(root: Path, suite: str) -> Path | None:
    base = root / "data" / "corpora" / suite
    if not base.is_dir():
        return None
    cands = sorted(d for d in base.iterdir() if (d / "tasks.jsonl").is_file())
    if len(cands) != 1:
        # The same refusal `pi run` makes: two corpus hashes are two frozen corpora, and
        # picking one silently would answer the question about the wrong training set.
        raise SystemExit(
            f"{len(cands)} built corpora under {base}: {[c.name for c in cands]}. "
            "A contamination check must name which one it read."
        )
    return cands[0]


def train_questions(root: Path) -> tuple[list[tuple[str, str, str]], dict[str, dict]]:
    """(suite, task_id, question) for every TRAIN-split task, plus a per-suite census."""
    from pinq.splitting import split_of

    rows: list[tuple[str, str, str]] = []
    census: dict[str, dict] = {}
    for suite in TRAIN_SUITES:
        d = _corpus_dir(root, suite)
        if d is None:
            census[suite] = {"corpus": None, "total": 0, "train": 0}
            continue
        template_id = _template_id_fn(suite)
        total = n_train = 0
        for line in (d / "tasks.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1
            tid = str(rec["id"])
            if split_of(suite, tid, template_id(tid)) != "train":
                continue
            n_train += 1
            rows.append((suite, tid, str(rec.get("question") or "")))
        census[suite] = {"corpus": d.name, "total": total, "train": n_train}
    return rows, census


def _template_id_fn(suite: str):
    """The split hashes `template_id or task_id`, and musique is the one suite that has one:
    two permutations of the same hop set are a paraphrase and must land on one side. Getting
    this wrong would put some train tasks in the dev/test buckets and shrink what we compare
    FRAMES against -- understating contamination, which is the direction that matters."""
    if suite == "musique":
        from pinq_adapters.musique.suite import MusiqueSuite

        return lambda tid: MusiqueSuite.template_id(MusiqueSuite, tid)  # type: ignore[arg-type]
    return lambda tid: None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=".", help="tree holding data/corpora/<suite>/")
    ap.add_argument(
        "--frames-root", default=None, help="tree holding data/raw/frames/ (default: --root)"
    )
    ap.add_argument("--jaccard", type=float, default=NEAR_DUPLICATE_JACCARD)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root).resolve()
    froot = Path(a.frames_root).resolve() if a.frames_root else root

    frames = frames_questions(froot)
    train, census = train_questions(root)

    f_norm = {tid: normalize(q) for tid, q in frames}
    f_toks = {tid: frozenset(n.split()) for tid, n in f_norm.items()}
    by_norm: dict[str, list[str]] = {}
    for tid, n in f_norm.items():
        by_norm.setdefault(n, []).append(tid)

    exact: list[dict] = []
    near: list[dict] = []
    closest: list[tuple[float, str, str, str]] = []
    for suite, tid, q in train:
        n = normalize(q)
        if not n:
            continue
        for fid in by_norm.get(n, ()):
            exact.append({"suite": suite, "task_id": tid, "frames_id": fid, "question": q})
        toks = frozenset(n.split())
        if not toks:
            continue
        for fid, ftoks in f_toks.items():
            j = jaccard(toks, ftoks)
            # THE CLOSEST PAIRS, ALWAYS, THRESHOLD OR NO THRESHOLD. A near-duplicate check
            # that reports zero is indistinguishable from a near-duplicate check that is
            # broken -- a wrong corpus path, an empty train set, a normaliser that returns ""
            # -- and all three report a clean bill of health. Printing the maximum Jaccard
            # actually observed is what makes "0 near-duplicates" a measurement rather than
            # an absence of one.
            if j > 0.0:
                closest.append((j, suite, tid, fid))
            if j >= a.jaccard:
                near.append(
                    {
                        "suite": suite,
                        "task_id": tid,
                        "frames_id": fid,
                        "jaccard": round(j, 4),
                        "train_question": q,
                        "frames_question": dict(frames)[fid],
                    }
                )
    near.sort(key=lambda r: -r["jaccard"])
    closest.sort(key=lambda t: -t[0])
    top = [
        {
            "jaccard": round(j, 4),
            "suite": suite,
            "task_id": tid,
            "frames_id": fid,
            "train_question": next(q for s_, t_, q in train if s_ == suite and t_ == tid),
            "frames_question": dict(frames)[fid],
        }
        for j, suite, tid, fid in closest[:5]
    ]

    report = {
        "frames_questions": len(frames),
        "train_questions": len(train),
        "per_suite": census,
        "jaccard_threshold": a.jaccard,
        "n_exact": len(exact),
        "n_near_duplicate": len(near),
        "max_jaccard_observed": round(closest[0][0], 4) if closest else 0.0,
        "closest_pairs": top,
        "exact": exact,
        "near_duplicate": near[:50],
    }
    if a.json:
        print(json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False))
    else:
        print(f"FRAMES questions          {len(frames)}")
        for suite, c in sorted(census.items()):
            corpus = c["corpus"] or "NOT BUILT"
            print(f"  {suite:<12} corpus {corpus:<20} tasks {c['total']:>6}  train {c['train']:>6}")
        print(f"train questions compared  {len(train)}")
        print(f"normalised exact matches  {len(exact)}")
        print(f"near-duplicates (J>={a.jaccard})  {len(near)}")
        print(
            f"max Jaccard observed      {report['max_jaccard_observed']}  "
            f"(the detector fired; 0 near-duplicates is a measurement, not an absence)"
        )
        for r in top:
            print(f"  J={r['jaccard']}  {r['suite']}/{r['task_id']}  vs  {r['frames_id']}")
            print(f"      train : {r['train_question'][:104]}")
            print(f"      frames: {r['frames_question'][:104]}")
        for r in near[:20]:
            print(f"  J={r['jaccard']}  {r['suite']}/{r['task_id']}  vs  {r['frames_id']}")
            print(f"      train : {r['train_question'][:110]}")
            print(f"      frames: {r['frames_question'][:110]}")
        for r in exact:
            print(
                f"  EXACT  {r['suite']}/{r['task_id']} == {r['frames_id']}: {r['question'][:110]}"
            )
    if exact:
        print(
            f"\nCONTAMINATED: {len(exact)} FRAMES question(s) are also train-split tasks. "
            "The transfer claim does not hold over those tasks.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
