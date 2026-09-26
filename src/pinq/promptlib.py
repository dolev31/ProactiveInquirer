"""Prompts as FILES, hashed into run identity.

Three properties, each load-bearing:

  1. A PROMPT CHANGE IS VISIBLE IN RUN IDENTITY. `sha(name)` goes into
     `RunManifest.prompt_hashes`, which `semantic_hash` covers. Editing a word in a template
     therefore produces a different `run_id`, so two rows in a table can never silently come
     from two different prompts. (It does NOT change the per-call cache key, which is the
     request bytes — see pi_run.cache: fixing a typo must evict one entry, not a sweep.)

  2. LENGTH IS NOT A CONFOUND. Arms are compared on what their scaffolding SAYS, not on how
     much of it there is, so the templates in a parity family are token-matched and an arm
     that lacks a section carries an equal-length NON-DIRECTIVE placebo instead. `parity()`
     reports the spread and tests/test_arms.py holds it to 10%.

  3. NO CAP EVER RENDERS. There is no `{{budget}}`, `{{cap}}` or `{{turns_left}}` placeholder
     in any template and `render()` refuses an unresolved one, so a budget cannot reach a
     policy through a prompt even by accident. `Inquirer.act(s)` being budget-blind by type
     is worth nothing if the string handed to the model announces the cap.

  4. A CANDIDATE PROMPT IS A REAL FILE. `PI_PROMPT_OVERLAY` names a directory whose
     templates shadow the shipped ones. Rung 0 (prompt optimisation) needs a candidate to
     reach the policy through the same path every arm uses; pointing the loader at a
     directory does that without a `if candidate:` branch anywhere, and because `sha()`
     hashes the bytes that were actually loaded, a candidate rollout carries a different
     `prompt_hashes` entry and therefore a different `run_id`. A candidate can never be
     mistaken for a baseline rollout, which is the only property that makes the search
     safe to run against the same runs/ directory.

Token counting is a stdlib word/punctuation split, not a BPE. It is used only for RELATIVE
comparison between templates, where any monotone tokenizer gives the same verdict, and pinq
takes no third-party dependency.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# Set by pi_run.serve.rollout for the duration of one rollout, so a rung-0 candidate is
# rendered by the SAME policy class, through the SAME loop, as the shipped template.
OVERLAY_ENV = "PI_PROMPT_OVERLAY"

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
_TOKEN = re.compile(r"\w+|[^\w\s]")

# Never legal in a template. A policy that could read any of these would stop being
# budget-blind, and STOP would become confounded with the cap the experiment announced.
FORBIDDEN_PLACEHOLDERS = frozenset(
    {"budget", "cap", "budget_cap", "remaining", "turns_left", "turns_remaining", "arm", "arm_id"}
)

# A template that renders another template verbatim into a hole. `self_inquire` is the whole
# reason this exists: its claim is "the Inquirer prompt, handed to one agent as a
# self-directive", and copying the text would let the copy drift from the original in a way
# no test could see. Declaring the inclusion keeps ONE source of truth and makes the token
# count of the composed prompt honest.
INCLUDES: dict[str, tuple[str, ...]] = {"self_inquire": ("inquirer_prompted",)}

# Templates that are compared to each other and must therefore be token-matched. Membership
# is the scientific claim "these two arms differ in what the prompt says, not in its size".
PARITY_FAMILIES: dict[str, tuple[str, ...]] = {
    "inquirer": (
        "inquirer_prompted",
        "inquirer_noevidence",
        "inquirer_depth1",
        "self_ask",
        "self_inquire",
        "ircot",
        "par2_rag_plan",
        "par2_rag_refine",
    ),
    "drafter": ("drafter_draft", "drafter_verbosity", "drafter_compute_matched"),
    # The two templates rendered at RESOLVE TIME. The expansion arm renders both, so
    # matching them keeps its extra call from also being an extra-long one.
    "resolve": ("drafter_resolve", "drafter_query_expansion"),
    # One paragraph and its placebo. Only `inquirer_may_ask_user` gets the real one; every
    # other arm gets the placebo, so opening the user channel changes the prompt's CONTENT
    # and not its length.
    "user_channel": ("fragment_user_channel", "fragment_user_channel_placebo"),
}


class MissingPrompt(FileNotFoundError):
    pass


class UnresolvedPlaceholder(KeyError):
    """A template rendered with a hole left in it. Loud, because a silently empty EVIDENCE
    section is an ablation nobody declared."""


def overlay_dir() -> Path | None:
    """The candidate-template directory, or None. Read per call, never cached: one server
    process serves many candidates and a cached overlay would answer the previous one."""
    raw = os.environ.get(OVERLAY_ENV, "")
    return Path(raw) if raw else None


def path(name: str) -> Path:
    """Overlay first, shipped template second. A missing overlay file is not an error: an
    overlay that varies ONE template must not have to copy the other fourteen."""
    over = overlay_dir()
    if over is not None:
        cand = over / f"{name}.txt"
        if cand.exists():
            return cand
    return PROMPT_DIR / f"{name}.txt"


def variant_id() -> str:
    """A LABEL FOR WHICH PROMPT SET PRODUCED A RUN. "v1" when no overlay is active.

    `RunManifest.prompt_variant_id` was declared, defaulted to "v1", written into every
    manifest, compacted into a `prompt_variant_id` parquet column -- and NOTHING ever set it to
    anything else. So the column was constant across every run ever made, and rows produced
    under different prompts pooled into one cell with nothing able to separate them.

    That matters for exactly one thing the paper needs and cannot currently do: report the
    between-prompt spread of the headline effect. `PI_PROMPT_OVERLAY` already exists and
    already changes the templates -- rung-0 uses it to render candidate prompts through the
    same policy class and the same loop -- so the two halves were built and never connected.

    Run IDENTITY was never at risk: `prompt_hashes` is inside `semantic_hash`, so an overlay
    that changes template bytes already changes the run_id. What was missing is the readable
    label to GROUP BY, which a hash of fifteen templates does not give you.

    The id is the overlay directory's name plus a short digest of the template bytes it
    actually overrides, so two different overlays that happen to share a directory name cannot
    collide, and re-running the same overlay reproduces the same label.
    """
    over = overlay_dir()
    if over is None:
        return "v1"
    files = sorted(p for p in Path(over).glob("*.txt") if p.is_file())
    if not files:
        return "v1"
    d = hashlib.sha256()
    for f in files:
        d.update(f.name.encode("utf-8"))
        d.update(b"\x1f")
        d.update(f.read_bytes())
        d.update(b"\x1e")
    return f"{Path(over).name}-{d.hexdigest()[:8]}"


def load(name: str) -> str:
    p = path(name)
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise MissingPrompt(f"no prompt template {name!r} in {PROMPT_DIR}") from None


def names() -> tuple[str, ...]:
    over = overlay_dir()
    extra = (p.stem for p in over.glob("*.txt")) if over is not None else ()
    return tuple(sorted({p.stem for p in PROMPT_DIR.glob("*.txt")} | set(extra)))


def sha(name: str) -> str:
    """sha256 of the template bytes. This is the value that reaches RunManifest."""
    return hashlib.sha256(load(name).encode("utf-8")).hexdigest()


def hashes(*names_: str) -> dict[str, str]:
    return {n: sha(n) for n in sorted(names_ or names())}


def render(name: str, **fields: object) -> str:
    """Substitute `{{field}}`. Unknown fields are ignored; unresolved holes RAISE.

    ONE PASS. A SUBSTITUTED VALUE IS NEVER RESCANNED.

    This looped `text = text.replace(...)` field by field, so anything a value CONTAINED was
    still live for every later field. A retrieved paragraph carrying the literal `{{question}}`
    was therefore filled in by the question:

        render(t, evidence="Doc 1: ... see also {{question}} ...", question="What is ...?")
        -> EVIDENCE: Doc 1: ... see also What is ...? ...

    The values here are corpus text and tool output -- 2Wiki paragraphs, drgym documents, tau2
    tool results -- so this is reachable from data, not only from a hostile author. And which
    fields could inject which depended on `fields` insertion order, so the same template and the
    same inputs could render differently depending on the order of keyword arguments at the call
    site.

    A single regex pass makes a value inert by construction: whatever it contains is copied out
    verbatim and never looked at again. Behaviour is otherwise identical -- a placeholder with no
    field is left in place and then caught by the unresolved check below, and a field the
    template does not mention is still ignored.
    """
    text = load(name)
    # THE UNRESOLVED CHECK READS THE TEMPLATE, NOT THE RENDERED OUTPUT. It exists to catch a
    # caller who forgot a field, which is a property of the template. Scanning the output
    # instead means a paragraph that merely CONTAINS "{{x}}" -- an article about templating, a
    # tool result quoting a config -- aborts the rollout for saying so. After the single-pass
    # change that stopped being theoretical: the injected placeholder correctly survives
    # substitution, and an output scan would then raise on it.
    missing = sorted(set(_PLACEHOLDER.findall(text)) - set(fields))
    if missing:
        raise UnresolvedPlaceholder(f"{name}: unresolved {missing}")
    return _PLACEHOLDER.sub(
        lambda m: str(fields[m.group(1)]) if m.group(1) in fields else m.group(0), text
    )


def placeholders(name: str) -> frozenset[str]:
    return frozenset(_PLACEHOLDER.findall(load(name)))


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def tokens(name: str) -> int:
    """Static scaffolding size: the template with its holes empty.

    The holes carry STATE, which differs by arm for scientific reasons (the no-evidence
    ablation sees no evidence — that is the ablation). What must not differ is the amount of
    instruction wrapped around it, and that is exactly what this measures.
    """
    own = count_tokens(_PLACEHOLDER.sub("", load(name)))
    return own + sum(tokens(inc) for inc in INCLUDES.get(name, ()))


def parity_budget(name: str, family: str, ratio: float = 1.10) -> int:
    """The largest `name` may grow before its parity family breaks. MEASURED, not assumed.

    `INCLUDES` is why this cannot be `min(family) * ratio`: `self_inquire` embeds
    `inquirer_prompted`, so growing the latter grows the former too and the cap tightens.
    With the shipped family that is the difference between a 480-token budget and the real
    440 -- and the first rung-0 search, having no budget at all, spent itself entirely in
    that gap.
    """
    fixed = {n: tokens(n) for n in PARITY_FAMILIES[family] if n != name}
    carried = {n: tokens(n) - tokens(name) for n in fixed if name in INCLUDES.get(n, ())}
    best = 0
    for cand in range(1, 4096):
        counts = [v for n, v in fixed.items() if n not in carried]
        counts += [cand] + [cand + off for off in carried.values()]
        if max(counts) / min(counts) <= ratio:
            best = cand
        elif best:
            break
    return best


def parity(family: str) -> dict[str, float | dict[str, int]]:
    counts = {n: tokens(n) for n in PARITY_FAMILIES[family]}
    lo, hi = min(counts.values()), max(counts.values())
    return {"counts": counts, "min": lo, "max": hi, "ratio": (hi / lo) if lo else float("inf")}
