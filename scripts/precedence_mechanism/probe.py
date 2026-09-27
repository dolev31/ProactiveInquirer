"""Closed-book answerability of a skipped parent node (Lane L1.3, step 2).

OPERATOR-SIDE, NEVER A RUN. These calls test the hypothesis that a violation's parent node
is answerable from parametric knowledge alone; they are not a policy rollout, carry no
`run_id`, `semantic_hash` or `RunManifest`, and must never be written under `runs/`. They are
still gold-adjacent (the question text is built from a gold node), so every prompt is
canary-scanned before it is sent -- firewall layer 4, the one layer that catches leakage
through a string rather than through code (`CONTRIBUTING.md`).

WHY `resolve_placeholders` AND NOT THE RAW `gold_text`. MuSiQue's gold node text is scored
byte-identical to the dataset (`pi_eval.build.musique_build`'s module docstring), so a
depth>=1 node's text still carries a bare `#N` -- "When was #1 founded?" -- that names
nothing outside the graph. The MEASURED closed-book question is the one a person (or a
model) could actually be asked: `#N` resolved to node `sN`'s own answer, which the graph
already carries as `gold_aliases[0]` (`pi_eval.build.musique_build._graph`:
`gold_aliases=(step["answer"],)`). No raw MuSiQue file is read for this -- the loaded graph
is self-sufficient. Verified on the live store: 1,464 of 1,464 depth>=1 MuSiQue nodes
resolve with zero `UnresolvedPlaceholder`s (see `tests/test_precedence_mechanism.py`).

A relation-triple node ("The Collegian >> owned by") is not a question at all; `_to_question`
turns "A >> B" into "What is A's B?" so every probed node gets a fair, natural-language
prompt regardless of which MuSiQue node type it is.

Scope: MuSiQue only. StrategyQA nodes carry no `gold_aliases` (the per-node answer this
substitution needs) and the paper's adverse, CI-excluding-zero finding is on MuSiQue alone
(`paper/results.tex:309-333`); see `artifacts/precedence_mechanism_20260918/RESULT.md` for
the scope note.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pi_eval.build.musique_build import UnresolvedPlaceholder, resolve_placeholders
from pi_eval.canary import scan_text
from pi_eval.gold import GoldGraph, GoldNode
from pi_eval.metrics.quality import contains_answer

_REL = re.compile(r"^(?P<entity>.+?)\s*>>\s*(?P<relation>.+)$")

PROMPT_TEMPLATE = (
    "Answer the following question as briefly and directly as possible, using only what you "
    'already know. If you do not know, answer exactly "unknown".\n\n'
    "Question: {question}\nAnswer:"
)


class CanaryLeak(RuntimeError):
    """A constructed probe prompt contains a registered gold canary. Refuse to send it."""


def node_answers(graph: GoldGraph) -> dict[str, str]:
    """`{node_id: its own gold answer}`, straight off the loaded graph -- see module
    docstring. Every MuSiQue node carries a non-empty `gold_aliases` (measured: 0 of 2,660
    on the v1 graphs do not), so this is total over `graph.gold_nodes` in practice, but a
    node with none is simply absent here rather than mapped to an empty string."""
    return {n.gold_node_id: n.gold_aliases[0] for n in graph.gold_nodes if n.gold_aliases}


def to_question(text: str) -> str:
    """ "A >> B" (a MuSiQue relation-triple node) becomes "What is A's B?"; anything already
    phrased as a question (MuSiQue's other node shape) passes through unchanged."""
    m = _REL.match(text)
    if not m:
        return text
    return f"What is {m.group('entity').strip()}'s {m.group('relation').strip()}?"


def closed_book_question(graph: GoldGraph, node: GoldNode) -> str | None:
    """The standalone question node's gold text implies, with every `#N` resolved to that
    sub-answer. `None` if some `#N` names a node this graph has no answer for -- refuse
    rather than probe a malformed question (see `UnresolvedPlaceholder`)."""
    try:
        resolved = resolve_placeholders(node.gold_text, node_answers(graph))
    except UnresolvedPlaceholder:
        return None
    return to_question(resolved)


def assert_canary_clean(text: str, canaries: frozenset[str], *, where: str) -> None:
    hits = scan_text(text, set(canaries), where=where)
    if hits:
        raise CanaryLeak(f"{where}: {hits[0].canary} present in {hits[0].excerpt!r}")


@dataclass(frozen=True, slots=True)
class ProbeAnswer:
    model: str
    question: str
    answer_text: str
    tok_prompt: int
    tok_completion: int
    http_status: int


# gpt-oss-120b reasons for 40-700 tokens before emitting content, INSIDE max_tokens
# (`pinq_adapters.llm.litellm_client`'s `DEFAULT_REASONING_HEADROOM` docstring, measured
# there on the same model). A 64-token cap with no headroom starves the content channel and
# returns empty text -- MEASURED here too: a live smoke call at max_tokens=64 returned ''
# with tok_completion=64. qwen3-8b-base is a base model with no hidden reasoning channel and
# needs none. The CONTENT target stays 64 (the prompt already asks for a short answer); this
# is an absolute floor on the token BUDGET, above the documented 700-token worst case, so
# reasoning cannot silently eat the whole thing.
MIN_TOKEN_BUDGET = {"aws/gpt-oss-120b": 800}


def call_model(
    *,
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    temperature: float = 0.0,
    max_tokens: int = 64,
    timeout: float = 30.0,
    num_retries: int = 1,
) -> ProbeAnswer:
    """One plain chat completion, plain `httpx` POST to the LiteLLM proxy's own
    OpenAI-compatible `/chat/completions`, not the `litellm` SDK. No ledger, no cache.

    MEASURED TWICE: `litellm.completion(..., timeout=X, num_retries=N)` produced a background
    run that sat at the same CPU-time for many minutes with zero output on the SAME proxy, the
    SAME models, at two different concurrency levels -- consistent with a call hanging past its
    nominal timeout inside the SDK's own retry wrapper (`completion_with_retries`, a `tenacity`
    loop this module does not control the backoff of) rather than in the network itself.
    `httpx`'s own timeout (connect/read/write/pool, all bounded) is a narrower, better-tested
    surface, and IS what the SDK call was built on underneath -- this sends the identical wire
    request (same `model` string, same JSON body) directly instead of through that wrapper.
    Retried here, explicitly and boundedly, rather than trusting a library's retry policy this
    module does not own.
    """
    import httpx

    budget = max(max_tokens, MIN_TOKEN_BUDGET.get(model, 0))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
        "temperature": temperature,
        "max_tokens": budget,
        "seed": 0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    t = httpx.Timeout(timeout, connect=10.0)
    last_exc: Exception | None = None
    for _attempt in range(num_retries + 1):
        try:
            resp = httpx.post(
                f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=headers, timeout=t
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data["choices"][0]["message"].get("content") or "").strip()
            usage = data.get("usage") or {}
            return ProbeAnswer(
                model=model,
                question=question,
                answer_text=text,
                tok_prompt=int(usage.get("prompt_tokens") or 0),
                tok_completion=int(usage.get("completion_tokens") or 0),
                http_status=resp.status_code,
            )
        except Exception as exc:  # noqa: BLE001 - retried a bounded number of times, then raised
            last_exc = exc
    assert last_exc is not None
    raise last_exc


def score_answer(pred: str, node: GoldNode) -> float:
    """1.0 if the closed-book answer names the node's own gold span. Judge-free, the same
    matcher the paper's `answer_correct` uses (`pi_eval.metrics.quality.contains_answer`)."""
    if not node.gold_aliases:
        return float("nan")
    return contains_answer(pred, node.gold_aliases[0], node.gold_aliases[1:])
