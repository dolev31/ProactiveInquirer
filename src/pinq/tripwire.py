"""Canary matching on the AGENT side, at the moment a request is built.

WHY THIS EXISTS SEPARATELY FROM `pi_eval.canary`. That module MINTS and REGISTERS nonces and
lives in the gold-only package, which nothing on the rollout path may import. This one only
MATCHES, needs no gold, and is stdlib-only like the rest of `pinq` -- so it can sit on the hot
path where the leak would actually happen.

WHY AT THE BOUNDARY RATHER THAN IN THE CACHE. `pi verify firewall` scans the response cache,
and the cache record stores the response text plus the request's SHA -- never the request
itself (see `pi_run.cache._record_from` and `_DETERMINISTIC_FIELDS`, whose minimality is what
makes two racing writers produce byte-identical files). A gold answer interpolated into a
PROMPT is therefore never written anywhere the scan can see it, and the scan could only ever
catch the rarer event of a model echoing gold back in its reply. The layer was armed and
pointed at the wrong direction.

So the tripwire fires HERE, on the serialized request, before it is dispatched: the leak is
caught at the instant it happens, before the money is spent, and the leaked text is never
persisted anywhere.

COST. Every nonce shares one prefix, so the common case is a single substring scan of the
request for `PINQCANARY_` and nothing else. Only when that appears does anything else run, and
then it is a set lookup rather than a walk over sixteen thousand needles.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import NamedTuple

PREFIX = "PINQCANARY"
_TOKEN = re.compile(rf"{PREFIX}_[0-9A-F]{{16}}")

# Where `pi_eval.canary.register` appends. NOT gold: it is a list of nonces, and a rollout
# worker is allowed -- and required -- to know them in order to refuse to send one.
REGISTRY = "data/canaries/canaries.txt"

# Layer 3's variable, read here as a STRING and never by importing `pi_eval.gold`. It names the
# gold in PLAY -- what `gold_root()` returns and what the scoring process loads its graphs from.
# The trees below are the ones this process can GUESS at; this is the one it was TOLD, and the
# arming check was not asking about it. Reading an env var keeps `pinq` stdlib-only, which is
# the import-linter contract that is the firewall's first layer, so this must never become an
# import of the gold module.
GOLD_ENV = "PI_GOLD_ROOT"
GOLD_SUBDIR = "data/gold"

_cache: dict[str, frozenset[str]] = {}


def _owning_tree(start: Path) -> Path | None:
    """The checkout a git WORKTREE belongs to, or None when `start` is not one.

    `git worktree add` writes a `.git` FILE holding `gitdir: <main>/.git/worktrees/<name>`, and
    that directory holds a `commondir` pointing back at the main `.git`. Two file reads and no
    subprocess -- this package is stdlib-only -- and off the per-request path, because `scan`
    short-circuits on the shared prefix before it ever calls `load`. Deliberately NOT memoized:
    the env and the working directory move between assertions, and a stale candidate list is the
    same class of bug as a memoized empty registry.

    WHY THIS HOP EXISTS. `.gitignore` excludes `data/gold/` and `data/canaries/`, so a worktree
    NEVER carries either -- structurally, not by accident -- while every campaign is launched
    from a pinned worktree because a dirty tree stamps its runs `dev-` and `dev-` is
    training-only. A worktree is therefore not a fresh clone with nothing to leak: it is a
    checkout of a tree whose registry exists and is one hop away.
    """
    dot = start / ".git"
    try:
        if not dot.is_file():
            return None  # a normal checkout: `.git` is a directory
        _, marker, rest = dot.read_text().partition("gitdir:")
        if not marker or not rest.strip():
            return None
        gitdir = Path(rest.strip())
        if not gitdir.is_absolute():
            gitdir = (start / gitdir).resolve()
        common = (gitdir / "commondir").read_text().strip()
        return (gitdir / common).resolve().parent
    except OSError:
        return None


def _discovered_roots() -> list[Path]:
    """The trees this process can name WITHOUT being told: the repository this module was
    installed from (walking up to a pyproject.toml, the rule `pi_run.manifest.repo_root` already
    uses), the checkout that owns it if it is a worktree, and the working directory last."""
    out: list[Path] = []
    for cand in Path(__file__).resolve().parents:
        if (cand / "pyproject.toml").is_file():
            out.append(cand)
            owner = _owning_tree(cand)
            if owner is not None and owner != cand:
                out.append(owner)
            break
    cwd = Path.cwd()
    if cwd not in out:
        out.append(cwd)
    return out


def _candidate_roots(root: str | os.PathLike[str] | None = None) -> list[Path]:
    """Where the registry is looked for, in order. An explicit argument or PI_CANARY_ROOT is a
    deliberate statement and stands alone; otherwise: THE TREE THAT HOLDS THE GOLD IN PLAY, then
    the discovered trees.

    WHY THE GOLD LEADS. `pi_eval.build.common.write_graphs` calls `register(minted, root)`, which
    writes `<root>/data/canaries/canaries.txt` beside `<root>/data/gold`. So a gold tree's
    registry is reachable FROM that gold tree with no configuration at all -- which is what makes
    this work where a variable cannot: PI_CANARY_ROOT is set nowhere in this codebase and the one
    call site passed no argument, so a fix depending on someone exporting it fails the same way
    for the same reason.

    The `commondir` hop below covers a WORKTREE reading its own repository's registry; this
    covers a process pointed by PI_GOLD_ROOT at a gold tree that is not this repository at all,
    which is every scoring and judging process. Neither subsumes the other -- measured: in a real
    worktree with the gold path unset only the hop arms, and in a non-worktree pinned tree with
    the gold path set only this does.
    """
    if root is not None:
        return [Path(root)]
    env = os.environ.get("PI_CANARY_ROOT")
    if env:
        return [Path(env)]
    out: list[Path] = []
    for gold in reachable_gold():
        owner = _canary_root_for(gold)
        if owner is not None and owner not in out:
            out.append(owner)
    for cand in _discovered_roots():
        if cand not in out:
            out.append(cand)
    return out


def _gold_probe_roots(root: str | os.PathLike[str] | None = None) -> list[Path]:
    """Where "is there gold that could leak" is asked -- which is NOT the same question as
    "where are the nonces", and conflating the two is what made layer 4 inert.

    A NAMED argument keeps asking about that tree alone: `pi verify firewall --root X` is a
    question about X. PI_CANARY_ROOT does not, because it is launch-time configuration of THIS
    process rather than a question -- a mistyped value would otherwise point the gold probe at a
    directory that holds neither gold nor nonces and be absorbed as "nothing to leak", which is
    the identical fail-open the worktree case was.

    Built from PI_CANARY_ROOT and `_discovered_roots` DIRECTLY rather than through
    `_candidate_roots`, which now follows the gold and would recurse through this function. The
    set is the same one it returned before.
    """
    if root is not None:
        return [Path(root)]
    out: list[Path] = []
    env = os.environ.get("PI_CANARY_ROOT")
    if env:
        out.append(Path(env))
    for c in _discovered_roots():
        if c not in out:
            out.append(c)
    return out


def reachable_gold(root: str | os.PathLike[str] | None = None) -> tuple[Path, ...]:
    """Every gold tree THIS PROCESS can actually reach. THE REFERENT OF THE ARMING CHECK.

    Two sources, and the first is why this exists. `assert_armed` used to derive the directory it
    tested from the REGISTRY's own root, so it asked "does gold sit beside the registry?" instead
    of "is the gold in use covered?" -- and consulted nothing about the gold in play. Measured at
    a4422c0 with the real gold path exported and a registry-less root: `assert_armed()` returned,
    `load()` held 0 needles, and `scan` of a string carrying a real REGISTERED nonce returned ().
    The path arithmetic was correct; the referent was not.

    The second source is every tree this process can name, which is the wider question the
    worktree case needs: gold built in a checkout one `commondir` hop away is still gold that can
    leak. The two are unioned because they fail in different places.

    A path that is not a directory is not reachable gold. `gold_root()` only checks that the
    variable is non-empty, so PI_GOLD_ROOT pointing at nothing must read as nothing built -- NOT
    as gold to protect, which would make a fresh checkout with a stale export unrunnable.
    """
    out: list[Path] = []
    env = os.environ.get(GOLD_ENV, "").strip()
    if env:
        p = Path(env)
        if p.is_dir():
            out.append(p)
    for base in _gold_probe_roots(root):
        g = base / GOLD_SUBDIR
        if g.is_dir() and g not in out:
            out.append(g)
    return tuple(out)


def _canary_root_for(gold: Path) -> Path | None:
    """The tree holding the registry for THIS gold, found FROM the gold. None if there is none.

    `write_graphs` puts the registry at `<root>/data/canaries` beside `<root>/data/gold`, so the
    walk up from a gold tree finds its own nonces. When it finds none, that is the hole -- and it
    is reported rather than resolved into a silent fallback.
    """
    for anc in (gold, *gold.parents):
        if (anc / REGISTRY).is_file():
            return anc
    return None


def uncovered_gold(root: str | os.PathLike[str] | None = None) -> tuple[Path, ...]:
    """Reachable gold trees with no registry findable from them. Empty is the only safe answer."""
    return tuple(g for g in reachable_gold(root) if _canary_root_for(g) is None)


def _registry_path(root: str | os.PathLike[str] | None = None) -> Path:
    """Where the nonce list lives. ANCHORED TO A TREE THAT HAS ONE, NOT TO THE WORKING DIRECTORY.

    This used to resolve against `Path.cwd()`, so a worker started from anywhere else loaded
    ZERO canaries -- and because `scan` returns () on an empty set and `load` MEMOIZED the
    failure, the tripwire was silently inert for the life of the process. A firewall layer that
    fails OPEN when it cannot find its own configuration is worse than no layer at all: it
    reports clean.

    Anchoring to the installed-from repository fixed the cwd case and left the launch pattern
    this project actually uses. MEASURED 2026-09-18 from /private/tmp/gridflag-wt, a pinned
    worktree, with its own `src` on PYTHONPATH: the registry resolved to
    `<worktree>/data/canaries/canaries.txt`, `load()` returned 0, and `assert_clean` on a REAL
    registered nonce returned without raising. So the first candidate that HAS a registry wins,
    which lets a worktree arm itself from the tree that owns one. Loading more needles can only
    turn a false negative into a true positive: a nonce appears in exactly one place, a gold
    record, so a wider needle set cannot manufacture a hit.

    When no candidate has one, the primary candidate is returned anyway -- so the arming check
    names the root a reader would expect rather than the last thing tried.
    """
    cands = _candidate_roots(root)
    for cand in cands:
        p = cand / REGISTRY
        if p.exists():
            return p
    return cands[0] / REGISTRY


class CanaryLeak(RuntimeError):
    """A gold canary reached a model request. The run is void.

    Never a warning and never downgraded. There is no benign explanation for a nonce that
    appears in exactly one place -- a gold record -- turning up in a prompt, and that is
    precisely what makes the check worth having.
    """


def load(root: str | os.PathLike[str] | None = None) -> frozenset[str]:
    """The registered nonces. Memoized per resolved path: this is read on the hot path."""
    p = _registry_path(root)
    key = str(p)
    if key not in _cache:
        try:
            _cache[key] = frozenset(p.read_text().split())
        except OSError:
            _cache[key] = frozenset()
    return _cache[key]


class ArmedStatus(NamedTuple):
    """What the arming check SAW: whether layer 4 can fire, why not, and over how many needles.

    `n_canaries` exists as a NUMBER rather than only inside `why` because a run has to be able
    to state it. The count was always computed here and then thrown away into a human string, so
    the one artefact that could have proved the layer was live -- the manifest -- recorded three
    booleans that read identically whether the layer was armed or had never been armed at all.
    """

    ok: bool
    why: str
    n_canaries: int


# The last status THIS PROCESS observed, so the manifest records what actually guarded the unit
# rather than a second reading taken later from a possibly different root. `None` means no arming
# check has run here: reported as unknown, never defaulted to a number that would read as a
# measurement (0 is the alarming case -- the check ran and the layer could not fire -- and the
# two must not collapse).
_last: ArmedStatus | None = None


def status(root: str | os.PathLike[str] | None = None) -> ArmedStatus:
    """The arming check. Cheap, but call it once per unit and not per request."""
    global _last
    p = _registry_path(root)
    if not p.exists():
        _last = ArmedStatus(False, f"no canary registry at {p}", 0)
        return _last
    n = len(load(root))
    if not n:
        _last = ArmedStatus(False, f"the canary registry at {p} is empty", 0)
        return _last
    _last = ArmedStatus(True, f"{n} canaries from {p}", n)
    return _last


def last_status() -> ArmedStatus | None:
    """What the arming check last saw in this process, or None if it never ran."""
    return _last


def reset_status() -> None:
    """Forget the observed status. For tests that move `PI_CANARY_ROOT` between assertions."""
    global _last
    _last = None


def armed(root: str | os.PathLike[str] | None = None) -> tuple[bool, str]:
    """(can this layer fire, why not). Unchanged shape; `status()` carries the count too."""
    st = status(root)
    return st.ok, st.why


def assert_armed(root: str | os.PathLike[str] | None = None) -> None:
    """FAIL CLOSED: refuse to start a unit when gold is built and this layer cannot fire.

    Layers 1-3 stop leakage through CODE. This is the only one that catches leakage through a
    STRING, and an unarmed one is indistinguishable from a clean one -- `scan` returns () either
    way. So the absence of a registry is checked ONCE, loudly, at unit start, rather than being
    silently absorbed sixteen thousand times on the hot path.

    Conditioned on gold actually being BUILT rather than asserted unconditionally: a fresh
    checkout with no corpora has nothing to leak and must stay runnable, and a guard that fires
    on a clean clone is a guard someone deletes.

    WHERE THAT CONDITION IS ASKED IS THE WHOLE QUESTION, and it used to be asked at exactly one
    place -- the same root the missing registry was looked for in. A pinned worktree holds
    NEITHER (`.gitignore` excludes both `data/gold/` and `data/canaries/`), so the check read
    "no gold here, nothing to leak" and returned, on the launch pattern every campaign uses.
    MEASURED 2026-09-18: from /private/tmp/gridflag-wt this returned silently while a REAL
    registered nonce passed `assert_clean` untouched. It failed OPEN on the one layer whose
    unarmed state is indistinguishable from a clean one.

    So when no root is NAMED, gold is looked for in every tree this process can name -- the same
    candidates the registry is looked for in. An explicitly passed `root` still asks about that
    tree alone: `pi verify firewall --root X` is a question about X.

    WHAT THIS CAN AND CANNOT TELL APART. It can tell "no gold is built in any tree this process
    can name" from "gold is built somewhere this process can name and the scanner is unarmed",
    and only the first proceeds. It CANNOT tell whether the SUITE about to run carries a nonce:
    drgym and every tau2 variant have no `gold_answer`, which is the only field the nonce is
    threaded into, so their scanner has nothing to catch either way -- but establishing that
    requires READING gold, and layer 3 unsets PI_GOLD_ROOT in exactly this process. That
    per-suite distinction is available only to `pi verify firewall` (see
    `pi_run.cli._canary_carriers`), never to a worker. A worker therefore arms unconditionally
    when gold exists anywhere it can see, which costs one file read and no gold access.

    AND THE GOLD IN PLAY IS PART OF THAT QUESTION. `reachable_gold` unions the trees this process
    can NAME with the one it was TOLD through PI_GOLD_ROOT, because a scoring process pointed at a
    gold tree outside this repository was previously asked about a directory that did not exist,
    answered no, and returned silently with zero needles loaded. Two conditions therefore have to
    hold, not one: the layer can fire, AND every gold tree this process can reach is covered by a
    registry. A loaded registry that does not cover the gold in play is not coverage -- it is
    17,844 needles for the wrong haystack, and `scan` reports clean on the right one.
    """
    st = status(root)
    reachable = reachable_gold(root)
    if not reachable:
        return  # nothing built in any tree this process can name: nothing to leak
    uncovered = uncovered_gold(root)
    if st.ok and not uncovered:
        return
    where = ", ".join(str(g) for g in (uncovered or reachable))
    raise CanaryLeak(
        f"gold is reachable under {where} but firewall layer 4 cannot fire: {st.why}. "
        "It is the only layer that catches a gold string reaching a model, and an unarmed one "
        "reports clean on every request. Rebuild gold (`pi data build`) to mint and register the "
        "nonces beside it -- `write_graphs` puts the registry at <root>/data/canaries -- or point "
        "PI_GOLD_ROOT at a tree that carries both."
    )


def scan(text: str, canaries: frozenset[str] | None = None) -> tuple[str, ...]:
    """Every registered nonce appearing in `text`, cheapest-first.

    The prefix test short-circuits the overwhelmingly common case in one pass. Unregistered
    tokens that merely look like nonces are NOT reported: a canary the registry has never seen
    is not evidence of a leak from this build, and a tripwire that fires on lookalikes is one
    that gets disabled.
    """
    if PREFIX not in text:
        return ()
    known = load() if canaries is None else canaries
    if not known:
        return ()
    return tuple(sorted({t for t in _TOKEN.findall(text) if t in known}))


def assert_clean(text: str, *, where: str, canaries: frozenset[str] | None = None) -> None:
    """Raise if any registered nonce is in `text`."""
    hits = scan(text, canaries)
    if hits:
        raise CanaryLeak(
            f"a gold canary reached {where}: {hits[0][:18]}... "
            f"({len(hits)} distinct). Gold text is in a model request; this run is void. "
            "The nonce appears in exactly one place -- a gold record -- so there is no "
            "benign explanation."
        )
