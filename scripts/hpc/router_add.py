#!/usr/bin/env python3
"""Add served names to the Mac-side LiteLLM router.

    router_add.py --name qwen3-8b-sft-headline --name qwen3-8b-dpo-pooled-control --port 8000
    router_add.py --name qwen3-4b-sft-headline --port 8001     # the 4B instance

WHY. `conf/serving/litellm.yaml` is the only way one `LITELLM_BASE_URL` reaches two
endpoints -- the tunnelled vLLM for the Inquirer, the team gateway for every frozen role --
so a checkpoint is not reachable by `PI_MODEL_INQUIRER` until it has a `model_list` entry
here. Seventeen more are coming, each six lines, five of them identical.

THE LINE THAT IS NOT BOILERPLATE. `enable_thinking: false`. Qwen3 is a hybrid-reasoning model
whose chat template emits a `<think>` channel by default; the rung-1 rows were built with it
OFF, so a served prompt with it on is a DIFFERENT PROMPT than the one the adapter was fitted
to, and the dev NLL that selected the checkpoint stops describing what the policy sees.
Forgetting it costs a sweep. A generator cannot forget it.

WHY THIS EDITS TEXT INSTEAD OF ROUND-TRIPPING YAML. Two thirds of that file is the reasoning
for its own shape. `yaml.safe_dump` would delete all of it, so the entry is inserted before a
marker line and nothing else is touched. Stdlib only, for the same reason: it must run before
anything is installed.

AND WHY A TEXT EDIT NEEDS A DUPLICATE-KEY GUARD. Editing text can write a mapping with the same
key twice, and that is not a parse error: `yaml.safe_load` keeps the LAST one and says nothing.
MEASURED 2026-09-15 15:40 IDT: an entry annotated with a comment BETWEEN two of its keys was
"moved" to a new port, the new keys landed above the comment and the old ones stayed below, and
the router went on serving the old port while this script printed the new one. So an existing
entry is now replaced WHOLE (`item_span`), and no file is written until `duplicate_keys` -- a
raw-text scan, because the loader cannot see what it silently resolves -- comes back empty.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROUTER = REPO / "conf" / "serving" / "litellm.yaml"

# New local deployments go immediately above this line, which sits above the wildcard
# passthrough. The wildcard must stay LAST: a catch-all ahead of an exact name shadows it on
# some litellm versions, and every local checkpoint would be dispatched to the gateway.
MARKER = "# >>> router_add.py inserts local deployments above this line <<<"

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def wants_thinking_kwarg(name: str, base: str = "") -> bool:
    """Qwen chat templates take `enable_thinking`; other templates do not.

    Sent to a template that does not take it, the kwarg is ignored at best and rejected at
    worst -- and `extra_body` is per-entry precisely so it never reaches the frozen roles.
    """
    return "qwen" in f"{name} {base}".lower()


def entry_yaml(name: str, *, port: int, base: str = "") -> str:
    """One `model_list` entry, shaped exactly like the hand-written `qwen3-8b-sft` one."""
    if not _NAME.match(name):
        raise ValueError(
            f"{name!r} is not a served model id: it is the vLLM --served-model-name / "
            "--lora-modules key, the price-table key and the checkpoint-registry key."
        )
    lines = [
        f"  - model_name: {name}",
        "    litellm_params:",
        f"      model: hosted_vllm/{name}",
        f"      api_base: http://127.0.0.1:{port}/v1",
        '      api_key: "not-used-vllm-is-open"',
    ]
    if wants_thinking_kwarg(name, base):
        lines += [
            "      extra_body:",
            "        chat_template_kwargs:",
            "          enable_thinking: false",
        ]
    return "\n".join(lines) + "\n"


def item_span(text: str, name: str) -> tuple[int, int] | None:
    """Character span of the WHOLE `- model_name: <name>` list item, or None if it is absent.

    An item runs from its `- model_name:` line to the next non-blank line indented less deeply
    than its own keys -- the next `  - ` item, the `  #` marker, a top-level key, or EOF --
    with trailing blank lines left out so the separator between entries survives a replacement.
    Indentation is the only thing that delimits it: the earlier version stopped at the first
    COMMENT line instead, which is why a comment sitting between two keys cut the entry in half
    and the half below it survived as a second copy of every key.
    """
    lines = text.splitlines(keepends=True)
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln)
    head = f"  - model_name: {name}"
    start = next((i for i, ln in enumerate(lines) if ln.rstrip() == head), None)
    if start is None:
        return None
    last = start  # index of the last line that belongs to the item
    for i in range(start + 1, len(lines)):
        if lines[i].strip() and not lines[i].startswith("    "):
            break
        if lines[i].strip():
            last = i
    return offsets[start], offsets[last] + len(lines[last])


_KEY = re.compile(
    r"""^(?P<indent> *)(?P<dash>- )?(?P<key>"[^"]*"|'[^']*'|[^\s#][^:]*?) *:(?=\s|$)"""
)


def duplicate_keys(text: str) -> list[tuple[int, int, str]]:
    """Every key written twice in ONE mapping, as (line, first line, key), 1-based.

    THIS CANNOT BE DONE WITH THE LOADER. `yaml.safe_load` resolves a duplicate key to the last
    occurrence and raises nothing, so a file with two `api_base` lines in one mapping parses
    clean and serves a port that no printed line ever named -- the 15:40 defect exactly. The
    scan tracks a stack of open mappings by indentation (`- ` opens a fresh one), skips blank
    lines, comments and block scalars, and deliberately does NOT understand flow mappings
    (`{a: 1, a: 2}`); nothing in this file uses them, and a scan that is wrong about them would
    be a false alarm on a file that is fine, which is how a guard gets switched off.
    """
    found: list[tuple[int, int, str]] = []
    stack: list[tuple[int, dict[str, int]]] = []  # (indent of the mapping, key -> first line)
    block: int | None = None
    for no, line in enumerate(text.splitlines(), 1):
        if block is not None:
            if not line.strip() or len(line) - len(line.lstrip(" ")) > block:
                continue
            block = None
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped in ("---", "..."):
            continue
        m = _KEY.match(line)
        indent = len(line) - len(line.lstrip(" "))
        if m is None:  # a plain sequence item or a continuation line: it opens no mapping
            while stack and stack[-1][0] > indent:
                stack.pop()
            continue
        indent = len(m.group("indent")) + (2 if m.group("dash") else 0)
        while stack and stack[-1][0] > indent:
            stack.pop()
        if m.group("dash"):  # a new list item starts a mapping of its own
            while stack and stack[-1][0] >= indent:
                stack.pop()
        if not stack or stack[-1][0] < indent:
            stack.append((indent, {}))
        seen = stack[-1][1]
        key = m.group("key").strip().strip("\"'")
        if key in seen:
            found.append((no, seen[key], key))
        else:
            seen[key] = no
        if line[m.end() :].strip()[:1] in ("|", ">"):
            block = indent
    return found


def add(path: Path, names: list[str], *, port: int, base: str = "") -> list[str]:
    """Insert every name that is not already there. Returns the names actually added."""
    text = Path(path).read_text()
    if MARKER not in text:
        raise ValueError(
            f"{path} carries no insertion marker ({MARKER!r}). Refusing to append: an entry "
            "after the wildcard passthrough is dead configuration that looks alive."
        )
    blocks, changed = [], []
    for name in names:
        block = entry_yaml(name, port=port, base=base)
        span = item_span(text, name)
        if span is not None:
            # AN EXISTING NAME FOLLOWS THE PORT IT IS GIVEN, AND THE ITEM IS REPLACED WHOLE.
            # MEASURED 2026-09-15: re-adding qwen3-8b-sft-headline with another port printed
            # "added or moved to" and left the effective port alone, because the old edit
            # PREPENDED the rendered keys and kept whatever followed the first comment line --
            # two `api_base` keys in one mapping, the loser printed and the winner served.
            # Rewriting the whole item is the only shape in which the rendered entry is what
            # the file holds. DELIBERATE: a comment that was inside the item is re-emitted
            # after it, not dropped -- two thirds of this file is reasoning and a tool that
            # deletes a line of it silently is worse than one that moves a line visibly -- but
            # it can no longer sit against the key it annotated, because that key is rewritten.
            start, stop = span
            old_block = text[start:stop]
            kept = [ln for ln in old_block.splitlines() if ln.strip().startswith("#")]
            new_block = block + "".join(f"{ln}\n" for ln in kept)
            if not old_block.endswith("\n"):
                new_block = new_block.rstrip("\n")
            if old_block == new_block:
                continue
            text = text[:start] + new_block + text[stop:]
            changed.append(name)
            continue
        blocks.append(block)
        changed.append(name)
    if blocks:
        marker_line = next(ln for ln in text.splitlines() if MARKER in ln)
        text = text.replace(marker_line, "\n".join(blocks) + "\n" + marker_line, 1)
    dupes = duplicate_keys(text)
    if dupes:
        where = "; ".join(f"line {ln} repeats {k!r} from line {first}" for ln, first, k in dupes)
        raise ValueError(
            f"{path} would hold a duplicate key in one mapping ({where}). Refusing to write: "
            "yaml.safe_load keeps the LAST of two keys and raises nothing, so the file would "
            "read as valid while the router served a port this script never printed. Delete "
            "the losing line by hand, or re-run for the name whose entry holds it -- an entry "
            "this script rewrites is rebuilt from one rendering and cannot carry two."
        )
    if not changed:
        return []
    Path(path).write_text(text)
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", action="append", required=True, dest="names")
    ap.add_argument("--port", type=int, default=8000, help="the tunnelled vLLM port for this base")
    ap.add_argument("--base", default="", help="base model id, only used to detect a Qwen template")
    ap.add_argument("--router", type=Path, default=DEFAULT_ROUTER)
    a = ap.parse_args(argv)
    try:
        added = add(a.router, a.names, port=a.port, base=a.base)
    except (ValueError, OSError) as e:
        print(f"router_add: REFUSED: {e}", file=sys.stderr)
        return 1
    for name in a.names:
        state = "added or moved to" if name in added else "already present at"
        print(f"{name}: {state} http://127.0.0.1:{a.port}/v1 ({a.router})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
