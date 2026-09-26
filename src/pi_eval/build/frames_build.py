"""FRAMES -> data/corpora/frames/ (public) + data/gold/graphs/frames/ (answers only).

FRAMES (Google DeepMind, NAACL 2025, arXiv:2409.12941, Apache-2.0) ships 824 hand-written
multi-hop questions, each with the 2-11 Wikipedia articles that together answer it. The
paper's "oracle articles" setting turns that into a closed per-question paragraph pool, which
is the shape `pinq_adapters.paragraphs.ParagraphSuite` already serves -- so the suite runs
through the existing loop, retriever, Inquirer, Drafter and Answerer unchanged.

THREE CHOICES HERE ARE MEASUREMENTS, NOT PREFERENCES.

1. RENDERED HTML, NOT `prop=extracts`. The obvious route is the plain-text extract API. It
   silently drops every table. Measured on `List_of_tallest_buildings_in_New_York_City`, the
   article FRAMES question 1 needs: the extract is 17,625 bytes of lead prose, and the gold
   answer ("37th") is not in it. 236 of the 824 questions (28.6%) are tagged
   `Tabular reasoning`. A plain-text pool makes those unanswerable BY CONSTRUCTION, which is
   a wrong published number rather than a bug that stack-traces. So the build renders the
   article and emits one paragraph per table row, headers inlined.

2. THE REVISION CURRENT AT 2024-09-01, NOT TODAY'S. FRAMES was released in September 2024 and
   several of its prompts say so in words ("as of August 2024"). Measured: the current
   revision of that same article is 1371688173, dated 2026-08-27 -- two years of edits after
   the question was written. `rvstart`/`rvdir=older` resolves the revision that was live at
   the pin date in ~0.4s per title, and `action=parse&oldid=` renders exactly that revision.
   The revid is recorded per page, so the corpus is reproducible from Wikipedia alone.

3. A PER-PAGE sha256, VERIFIED AT LOAD. No other adapter in this repo checks a hash at load,
   and none needs to: their corpora are bytes we downloaded once and pinned. This is the only
   suite whose source is live and mutable, and `corpus_hash` pins the directory NAME, which
   catches a rebuild and cannot catch a byte edited in place. See
   `pinq_adapters.frames.suite.FramesSuite`.

THE ANSWER IS GOLD AND LIVES IN THE GOLD TREE. Upstream ships it in the same TSV row as the
question; this builder splits them, exactly as musique_build splits `question_decomposition`
off the question. The gold graph is NODE-FREE -- FRAMES has no need-graph and never will --
so it carries `gold_answer` and nothing else, and `pi_eval.score` takes the gold-free branch
(`GOLD_FREE_SUITES`) that emits the answer and cost metrics and no structural metric at all.
"""

from __future__ import annotations

import ast
import concurrent.futures
import csv
import hashlib
import html
import html.parser
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from pi_eval.build.common import BuildResult, require_raw, write_corpus, write_graphs

SUITE = "frames"
CORPUS_ID = "frames_v1"
GRAPH_VERSION = "v1"

# The dataset commit, not the branch head: a benchmark that can be re-uploaded under the same
# name is a benchmark whose old numbers cannot be reproduced.
DATASET = "google/frames-benchmark"
DATASET_REVISION = "429d8fd10c6ebd1e0d96a88d0ad7323d5e45f335"
TSV_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}/test.tsv"
TSV_SHA256 = "4255093c93b595b5b04c7c8dde290b48ec87d72ca0fb0b760d9dd02740d669ff"

# Just after the FRAMES release (arXiv 2409.12941, 19 Sep 2024) and just after the latest
# date any prompt refers to ("as of August 2024").
REVISION_AS_OF = "2024-09-01T00:00:00Z"

API = "https://en.wikipedia.org/w/api.php"
UA = "proactive-inquirer/0.1 (research corpus builder; https://github.com/anthropics/claude-code)"

# The one retrieval knob in this file, and it is reported rather than tuned. Below this a
# "paragraph" is a caption, a stub heading or a one-cell table row, and a pool padded with
# those makes BM25's idf a function of boilerplate rather than of content.
MIN_PARA_CHARS = 80

# Tables and blocks that carry navigation rather than facts. Infoboxes are deliberately NOT
# here: they hold birth dates, tenures and successor links, which is exactly the kind of fact
# a FRAMES chain hops through.
SKIP_CLASSES = frozenset(
    {
        "navbox",
        "navbox-inner",
        "vertical-navbox",
        "nomobile",
        "metadata",
        "ambox",
        "mbox-small",
        "sistersitebox",
        "reflist",
        "references",
        "mw-references-wrap",
        "noprint",
        "hatnote",
        "shortdescription",
        "mw-editsection",
        "toc",
    }
)

_WS = re.compile(r"\s+")
# A percent escape that survived one unquote pass; see `title_of`.
_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
# The boundary between two URLs crammed into one `wiki_links` element; see `split_links`.
_EMBEDDED_URL = re.compile(r",\s*(?=https?://)")
_BRACKET_REF = re.compile(r"\[\s*(?:\d+|citation needed|edit|note \d+)\s*\]", re.IGNORECASE)


class FramesBuildError(RuntimeError):
    """A build-time refusal. Never a silent skip: every dropped page is named."""


# ------------------------------------------------------------------ the upstream task file


@dataclass(frozen=True, slots=True)
class FramesTask:
    task_id: str
    question: str
    answer: str
    titles: tuple[str, ...]
    reasoning_types: tuple[str, ...]


HOSTS = ("en.wikipedia.org", "en.m.wikipedia.org")

# Non-article namespaces. `Special:Search` is the one that actually occurs -- the search-box
# URL in task frames_0088 carries `title=Special:Search` in its query string, so the
# index.php branch below would otherwise have resolved it, fetched the SEARCH PAGE, and put
# it in that task's evidence pool as though upstream had linked it. That is the exact failure
# mode `title_of` exists to refuse: a pool entry the benchmark never pointed at, invisible in
# every downstream number.
NON_ARTICLE_NS = (
    "special:",
    "talk:",
    "user:",
    "wikipedia:",
    "file:",
    "mediawiki:",
    "template:",
    "help:",
    "category:",
    "portal:",
    "draft:",
    "module:",
)


def title_of(url: str) -> str | None:
    """Wikipedia URL -> article title, or None for a URL this builder will not guess at.

    MEASURED over the 2,507 distinct URLs upstream ships: 2,475 on en.wikipedia.org, 28 on
    en.m.wikipedia.org (the mobile host, same titles) and 4 malformed or elsewhere. Of those
    four, two are recovered here and two are not, and the line between them is whether the
    title is STATED or INFERRED:

      recovered  `en.wikipedia.org/wiki/Grazia_Deledda`        -- a missing scheme, nothing else
      recovered  `/w/index.php?title=Bronco&redirect=no`       -- the title is in the query
      refused    `/w/index.php?search=Polytrichum+piliferum`   -- a SEARCH box, not an article
      refused    `https://w.wiki/ASFv`                         -- a short link, target unknown
                 `https://simple.wikipedia.org/wiki/...`       -- a different wiki entirely

    The two refusals are reported by name in the manifest. Resolving the search string to the
    article it probably meant would put text in a pool upstream never pointed at, and the
    difference between "the benchmark linked this" and "the builder guessed this" is not
    visible in any downstream number.
    """
    raw_url = url.strip().rstrip(",").strip()
    if not raw_url:
        return None
    p = urllib.parse.urlparse(raw_url)
    if not p.scheme and raw_url.startswith(HOSTS):
        p = urllib.parse.urlparse("https://" + raw_url)
    if p.netloc not in HOSTS:
        return None
    if p.path.endswith("/index.php"):
        q = urllib.parse.parse_qs(p.query)
        raw = (q.get("title") or [""])[0]
    elif "/wiki/" in p.path:
        raw = p.path.split("/wiki/", 1)[1]
    else:
        return None
    if not raw:
        return None
    title = urllib.parse.unquote(raw)
    # DOUBLE-ENCODED, measured on one upstream row:
    # `2021_French_Open_%E2%80%93_Men%2527s_singles`. One pass leaves `Men%27s`, which is not
    # an article. The second pass runs only when a %XX escape SURVIVED the first, so a title
    # that legitimately contains a percent sign -- `Percentage (%)`, encoded `%28%25%29` and
    # decoded to `(%)` -- is left alone, because `%)` is not an escape.
    if _ESCAPE.search(title):
        title = urllib.parse.unquote(title)
    title = title.replace("_", " ").strip()
    if not title or title.lower().startswith(NON_ARTICLE_NS):
        return None
    return title


def split_links(element: str) -> list[str]:
    """One `wiki_links` element -> the URLs it actually holds.

    MEASURED, 7 of the 824 rows: upstream packs two or three URLs into a SINGLE list element,
    comma-separated -- `.../American_Family_Field, .../LoanDepot_Park, .../Globe_Life_Field, `.
    Read as one string the whole thing becomes the path of the first URL, so `title_of` returns
    a title that does not exist and all three articles vanish from that task's pool. The task
    then runs short, and an incomplete pool is indistinguishable downstream from a policy that
    retrieved badly.
    """
    parts = [x.strip().rstrip(",").strip() for x in _EMBEDDED_URL.split(element.strip())]
    return [x for x in parts if x]


def parse_tasks(tsv: Path, limit: int | None = None) -> tuple[list[FramesTask], list[dict]]:
    """(tasks, unresolvable urls). `wiki_links` is the authoritative field.

    The numbered `wikipedia_link_N` columns are a denormalised copy that packs everything past
    the tenth into `wikipedia_link_11+`, so parsing those would need a second delimiter
    convention upstream never documents. `wiki_links` is a Python literal list and parses for
    824 of 824 rows; the numbered columns are the fallback if that ever stops being true.
    """
    tasks: list[FramesTask] = []
    bad: list[dict] = []
    with tsv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if limit is not None and len(tasks) >= limit:
                break
            idx = str(row.get("") or row.get("Unnamed: 0") or "").strip()
            tid = f"{SUITE}_{int(idx):04d}"
            try:
                urls = list(ast.literal_eval(row.get("wiki_links") or "[]"))
            except (ValueError, SyntaxError):
                urls = [row.get(f"wikipedia_link_{i}") or "" for i in range(1, 11)]
            titles: list[str] = []
            for element in urls:
                for u in split_links(str(element)):
                    t = title_of(u)
                    if t is None:
                        bad.append(
                            {"task_id": tid, "url": u, "reason": "not an en.wikipedia article url"}
                        )
                    elif t not in titles:
                        titles.append(t)
            tasks.append(
                FramesTask(
                    task_id=tid,
                    question=_WS.sub(" ", str(row.get("Prompt") or "")).strip(),
                    answer=_WS.sub(" ", str(row.get("Answer") or "")).strip(),
                    titles=tuple(titles),
                    reasoning_types=tuple(
                        s.strip()
                        for s in str(row.get("reasoning_types") or "").split("|")
                        if s.strip()
                    ),
                )
            )
    return tasks, bad


# ------------------------------------------------------------------ html -> paragraphs


class _Flattener(html.parser.HTMLParser):
    """Rendered Wikipedia HTML -> an ordered list of paragraph strings.

    Three kinds of paragraph, and the second and third are why this exists rather than a call
    to `prop=extracts`:

      * a `<p>` block, verbatim prose;
      * a `<li>` item, prefixed with the section it sits under, because a bare list item is
        usually too short to survive MIN_PARA_CHARS and loses its referent without the heading;
      * a `<tr>` row, rendered `Section | Header: cell | Header: cell`, because a table cell
        without its column name is an unlabelled number.

    Header cells are taken from the row that declares them and carried down the table, so a
    row keeps its labels even though the labels are three hundred lines of HTML away.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self._skip_depth = 0
        self._heading = ""
        self._stack: list[str] = []
        self._buf: list[str] = []
        self._capture: str | None = None  # "p" | "h" | "li" | "cell"
        self._row: list[str] = []
        self._headers: list[str] = []
        self._row_is_header = False
        self._table_depth = 0
        self._caption = ""

    # -- helpers

    @staticmethod
    def _classes(attrs: Sequence[tuple[str, str | None]]) -> set[str]:
        for k, v in attrs:
            if k == "class" and v:
                return set(v.split())
        return set()

    def _flush(self) -> str:
        text = _WS.sub(" ", "".join(self._buf)).strip()
        self._buf = []
        return _BRACKET_REF.sub("", text).strip()

    def _emit(self, text: str) -> None:
        text = _WS.sub(" ", text).strip()
        if len(text) >= MIN_PARA_CHARS:
            self.paragraphs.append(text)

    # -- parser hooks

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip_depth:
            self._skip_depth += 1
            return
        if tag in ("style", "script", "sup", "figcaption"):
            self._skip_depth = 1
            return
        if self._classes(attrs) & SKIP_CLASSES:
            self._skip_depth = 1
            return

        if tag == "table":
            self._table_depth += 1
            self._headers = []
            self._caption = ""
        elif tag == "caption":
            self._capture, self._buf = "cap", []
        elif tag == "tr":
            self._row, self._row_is_header = [], False
        elif tag in ("td", "th"):
            self._capture, self._buf = "cell", []
            self._row_is_header = self._row_is_header or tag == "th"
        elif tag == "p" and not self._table_depth:
            self._capture, self._buf = "p", []
        elif tag == "li" and not self._table_depth:
            self._capture, self._buf = "li", []
        elif tag in ("h2", "h3", "h4"):
            self._capture, self._buf = "h", []

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "table":
            self._table_depth = max(0, self._table_depth - 1)
            self._headers, self._caption = [], ""
        elif tag == "caption" and self._capture == "cap":
            self._caption, self._capture = self._flush(), None
        elif tag in ("td", "th") and self._capture == "cell":
            self._row.append(self._flush())
            self._capture = None
        elif tag == "tr":
            self._end_row()
        elif tag == "p" and self._capture == "p":
            self._emit(self._flush())
            self._capture = None
        elif tag == "li" and self._capture == "li":
            item = self._flush()
            self._emit(f"{self._heading}: {item}" if self._heading else item)
            self._capture = None
        elif tag in ("h2", "h3", "h4") and self._capture == "h":
            self._heading, self._capture = self._flush(), None

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._capture is None:
            return
        self._buf.append(data)

    def _end_row(self) -> None:
        cells = [c for c in self._row]
        self._row = []
        if not any(cells):
            return
        if self._row_is_header and not self._headers:
            self._headers = cells
            return
        label = self._caption or self._heading
        parts = []
        for i, c in enumerate(cells):
            if not c:
                continue
            head = self._headers[i] if i < len(self._headers) else ""
            parts.append(f"{head}: {c}" if head else c)
        if not parts:
            return
        self._emit(" | ".join([label, *parts]) if label else " | ".join(parts))


def flatten_html(doc: str) -> list[str]:
    """Rendered article HTML -> deduplicated paragraphs, order preserved."""
    f = _Flattener()
    f.feed(doc)
    f.close()
    seen: set[str] = set()
    out: list[str] = []
    for p in f.paragraphs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ------------------------------------------------------------------ fetching, resumably


def page_key(title: str) -> str:
    return hashlib.sha256(title.encode("utf-8")).hexdigest()[:16]


def page_sha256(paragraphs: Sequence[str]) -> str:
    """The digest the adapter re-computes at load. Over the paragraph payload, not the HTML:
    it is the paragraphs that reach the retriever, so they are what must be pinned."""
    return hashlib.sha256("\n".join(paragraphs).encode("utf-8")).hexdigest()


def _get(url: str, *, timeout: int = 90, attempts: int = 6) -> bytes:
    """GET with a 429-aware backoff.

    THE FIRST FULL BUILD LOST 84 OF 2,474 ARTICLES TO HTTP 429, at eight workers with a
    2s-linear backoff over four attempts -- a self-inflicted failure that looks exactly like a
    benchmark whose links are broken. 59 tasks came back with an incomplete pool because of
    it. Wikipedia asks clients to back off on 429 and sends `Retry-After`; honouring that
    header and growing the wait geometrically is the difference between a corpus that is
    missing 3.4% of its articles and one that is not.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:  # noqa: PERF203 -- backoff needs the loop
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            # The server's own number wins over ours when it sends one.
            retry_after = 0.0
            try:
                retry_after = float(exc.headers.get("Retry-After") or 0)
            except (TypeError, ValueError):
                retry_after = 0.0
            time.sleep(max(retry_after, min(60.0, 2.0 * (2**i))))
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(min(30.0, 2.0 * (2**i)))
    raise FramesBuildError(f"GET failed after {attempts} attempts: {url}: {last!r}")


def _revision_query(title: str, *, direction: str, start: str | None) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "revisions",
        "titles": title,
        "rvlimit": "1",
        "rvdir": direction,
        "rvprop": "ids|timestamp",
        "redirects": "1",
    }
    if start:
        params["rvstart"] = start
    data = json.loads(_get(f"{API}?{urllib.parse.urlencode(params)}"))
    pages = (data.get("query") or {}).get("pages") or []
    if not pages:
        raise FramesBuildError("no page in the API response")
    return pages[0]


def resolve_revision(title: str, as_of: str = REVISION_AS_OF) -> tuple[int, str, bool]:
    """(revid, timestamp, created_after_the_pin). Redirects are followed by the API.

    THE FALLBACK IS NAMED, NOT SILENT. MEASURED, 5 of 2,474 articles: the page exists today and
    has no revision at or before the pin date, because it was CREATED afterwards --
    `Agnieszka Kotlarska` is pageid 77891779 with nothing before 2024-09-01. Two wrong answers
    were available. Refusing drops an article the benchmark explicitly points at, leaving that
    task with a short pool that reads downstream as a policy failure rather than as a build
    gap. Taking it quietly puts a page in the corpus that postdates the pin every other page is
    held to, with nothing saying so. So it is taken AND the third element of this tuple carries
    the fact up to the manifest, where `pages_after_pin_date` lists them by name.
    """
    pg = _revision_query(title, direction="older", start=as_of)
    if pg.get("missing"):
        raise FramesBuildError("article does not exist")
    revs = pg.get("revisions") or []
    if revs:
        return int(revs[0]["revid"]), str(revs[0].get("timestamp") or ""), False

    pg = _revision_query(title, direction="newer", start=None)
    if pg.get("missing"):
        raise FramesBuildError("article does not exist")
    revs = pg.get("revisions") or []
    if not revs:
        raise FramesBuildError(f"no revision at or before {as_of}, and none after it either")
    return int(revs[0]["revid"]), str(revs[0].get("timestamp") or ""), True


def fetch_revision_html(revid: int) -> str:
    q = urllib.parse.urlencode(
        {
            "action": "parse",
            "format": "json",
            "formatversion": "2",
            "oldid": str(revid),
            "prop": "text",
            "disablelimitreport": "1",
            "disableeditsection": "1",
            "disabletoc": "1",
        }
    )
    data = json.loads(_get(f"{API}?{q}"))
    if "error" in data:
        raise FramesBuildError(str(data["error"].get("info") or data["error"]))
    return str((data.get("parse") or {}).get("text") or "")


def fetch_page(title: str, *, as_of: str = REVISION_AS_OF) -> dict:
    revid, ts, after_pin = resolve_revision(title, as_of)
    paragraphs = flatten_html(html.unescape(fetch_revision_html(revid)))
    if not paragraphs:
        raise FramesBuildError(f"rendered to zero paragraphs over {MIN_PARA_CHARS} chars")
    return {
        "title": title,
        "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
        "revid": revid,
        "rev_timestamp": ts,
        "created_after_pin": after_pin,
        "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": page_sha256(paragraphs),
        "n_paragraphs": len(paragraphs),
        "paragraphs": paragraphs,
    }


def load_cached_page(pages_dir: Path, title: str) -> dict | None:
    """A cached page, or None. A file whose stored digest disagrees with its own payload is
    treated as absent rather than trusted -- that is what makes the build resumable after a
    kill mid-write without leaving a half-page in the corpus."""
    p = pages_dir / f"{page_key(title)}.json"
    if not p.is_file():
        return None
    try:
        rec = json.loads(p.read_text())
    except ValueError:
        return None
    if rec.get("sha256") != page_sha256(rec.get("paragraphs") or []):
        return None
    return rec


def fetch_all(
    titles: Sequence[str],
    pages_dir: Path,
    *,
    as_of: str = REVISION_AS_OF,
    workers: int = 4,
    fetcher: Callable[[str], dict] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[dict[str, dict], list[dict]]:
    """(title -> page record, failures). Resumable and idempotent: a title already cached
    with a matching digest is never refetched, so a killed build resumes where it stopped."""
    pages_dir.mkdir(parents=True, exist_ok=True)
    got: dict[str, dict] = {}
    todo: list[str] = []
    for t in titles:
        rec = load_cached_page(pages_dir, t)
        if rec is None:
            todo.append(t)
        else:
            got[t] = rec

    failures: list[dict] = []
    if not todo:
        return got, failures

    call = fetcher or (lambda t: fetch_page(t, as_of=as_of))
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(call, t): t for t in todo}
        for fut in concurrent.futures.as_completed(futures):
            title = futures[fut]
            done += 1
            try:
                rec = fut.result()
            except Exception as exc:  # a failure is NAMED, never dropped
                failures.append({"title": title, "reason": f"{type(exc).__name__}: {exc}"})
            else:
                (pages_dir / f"{page_key(title)}.json").write_text(
                    json.dumps(rec, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
                )
                got[title] = rec
            if progress:
                progress(done, len(todo), title)
    return got, failures


# ------------------------------------------------------------------ corpus + gold


def build(
    *,
    root: Path,
    allow_download: bool = True,
    verify: bool = True,
    as_of: str = REVISION_AS_OF,
    workers: int = 4,
    limit: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> BuildResult:
    """Raw -> data/corpora/frames/<hash>/{tasks.jsonl,manifest.json} + data/gold/graphs/frames/.

    A task whose articles could not all be fetched is KEPT, with the pages that did arrive,
    and its id is recorded in the manifest as `tasks_with_missing_pages` -- which is what the
    registry's `gold_gap_tasks` names. Dropping it would make the denominator a function of
    Wikipedia's availability on the day of the build.
    """
    root = Path(root)
    raw_dir = root / "data" / "raw" / SUITE
    raw_dir.mkdir(parents=True, exist_ok=True)
    # `verify=False` is for the hand-written test fixture only: it is a four-row stand-in in
    # upstream's TSV shape and cannot carry upstream's digest. Every real build verifies, and
    # the flag is named rather than inferred from `allow_download` so that an offline build
    # against a genuine cached download still checks its bytes.
    tsv = require_raw(
        raw_dir / "test.tsv",
        TSV_URL,
        expect_sha256=TSV_SHA256 if verify else None,
        allow_download=allow_download,
    )

    tasks, bad_urls = parse_tasks(tsv, limit)
    wanted: list[str] = []
    for t in tasks:
        for title in t.titles:
            if title not in wanted:
                wanted.append(title)

    pages, failures = fetch_all(
        wanted,
        raw_dir / "pages",
        as_of=as_of,
        workers=workers,
        fetcher=None if allow_download else _offline_fetcher,
        progress=progress,
    )

    records: list[dict] = []
    graphs: list[dict] = []
    gaps: list[str] = []
    used: dict[str, dict] = {}
    for t in tasks:
        paragraphs: list[dict] = []
        missing = False
        for title in t.titles:
            rec = pages.get(title)
            if rec is None:
                missing = True
                continue
            used[title] = rec
            for text in rec["paragraphs"]:
                paragraphs.append({"idx": len(paragraphs), "title": title, "text": text})
        if missing:
            gaps.append(t.task_id)
        records.append({"id": t.task_id, "question": t.question, "paragraphs": paragraphs})
        graphs.append(
            {
                "gold_suite": SUITE,
                "gold_task_key": t.task_id,
                "gold_nodes": [],
                "gold_edges": [],
                "gold_facets": [],
                "gold_seed_node_ids": [],
                "gold_graph_version": GRAPH_VERSION,
                "gold_answer": t.answer,
                "gold_aliases": [],
            }
        )

    corpus, chash = write_corpus(root, SUITE, records)
    gold = write_graphs(root, SUITE, GRAPH_VERSION, graphs, corpus_hash=chash)

    manifest = {
        "suite": SUITE,
        "corpus_id": CORPUS_ID,
        "corpus_hash": chash,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "dataset_url": TSV_URL,
        "dataset_sha256": TSV_SHA256,
        "wikipedia_api": API,
        "revision_as_of": as_of,
        "min_paragraph_chars": MIN_PARA_CHARS,
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_tasks": len(records),
        "n_titles_wanted": len(wanted),
        "n_pages_fetched": len(used),
        "n_paragraphs": sum(len(r["paragraphs"]) for r in records),
        "failures": sorted(failures, key=lambda f: f["title"]),
        # Articles taken from their OLDEST revision because none existed at the pin date; see
        # `resolve_revision`. Named so that "every page is as of 2024-09-01" stays a checkable
        # statement with a listed set of exceptions rather than an approximation.
        "pages_after_pin_date": sorted(
            t for t, rec in used.items() if rec.get("created_after_pin")
        ),
        "unresolvable_urls": bad_urls,
        "tasks_with_missing_pages": sorted(gaps),
        "pages": {
            title: {
                "revid": rec["revid"],
                "rev_timestamp": rec["rev_timestamp"],
                "fetched_utc": rec["fetched_utc"],
                "sha256": rec["sha256"],
                "n_paragraphs": rec["n_paragraphs"],
                "created_after_pin": bool(rec.get("created_after_pin")),
                "url": rec["url"],
            }
            for title, rec in sorted(used.items())
        },
    }
    (corpus.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return BuildResult(
        corpus=corpus, gold=gold, corpus_hash=chash, n_tasks=len(records), n_excluded=len(gaps)
    )


def _offline_fetcher(title: str) -> dict:
    """`allow_download=False` must not reach the network, and must say which page is missing
    rather than producing a task with an empty pool and no explanation."""
    raise FramesBuildError(f"offline build: no cached page for {title!r}")


def titles_of(tasks: Iterable[FramesTask]) -> list[str]:
    out: list[str] = []
    for t in tasks:
        for title in t.titles:
            if title not in out:
                out.append(title)
    return out
