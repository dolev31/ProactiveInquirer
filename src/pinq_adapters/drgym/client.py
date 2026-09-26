"""The DeepResearchGym search API, as an offline-first HTTP client.

WHAT MAKES THIS ADAPTER DIFFERENT FROM EVERY OTHER ONE HERE
    musique, wiki2, strategyqa and synth retrieve from a CLOSED pool that ships with the
    task. This one retrieves from a HOSTED index over an open web corpus, so the retriever
    is a network call and the corpus is not a file we can hash. Two consequences run through
    the whole package: every result is written to a content-addressed cache so a rollout can
    be replayed byte-for-byte without the network (see .cache), and `corpus_hash` pins the
    QUERY SET and the endpoint contract rather than the document set, which no client can
    certify (see .suite).

THE ENVELOPE. Both endpoints answer with

    {"results": ["<base64 of one JSON document>", ...]}

so each element is base64-decoded and then json.loads'ed individually. There is no batching
and no pagination; `k` is the whole request.

THE FIELD NAMES DIFFER PER CORPUS, and that is the one thing a caller must never have to
know. ClueWeb22 writes `URL`, `ClueWeb22-ID`, `Clean-Text`, `Language`; FineWeb writes
`url`, `id`, `text`, `language`. Normalisation happens here, at the boundary, and everything
downstream sees `SearchDoc`.

Deliberately NOT normalised: ClueWeb's `URL-hash` and FineWeb's `dump`. They are not the
same concept -- one is a document key, the other a crawl snapshot -- and merging them into a
single column would invent a field that neither corpus has. They are dropped instead.

ACCESS, VERIFIED BY DIRECT CALL ON 2026-08-24:
    GET https://clueweb22.us/health           -> 200
    GET https://clueweb22.us/fineweb/search   -> 401 without X-API-Key
    GET https://clueweb22.us/search           -> 401 without X-API-Key
The public FAQ claiming the FineWeb endpoint is keyless is STALE. A 401 is therefore the
NORMAL state of a machine without a key, which is why `available()` exists and why the 401
path raises a message containing the remedy rather than a bare HTTPError: a missing key must
never read like a bug in this code.

ClueWeb22 additionally needs a signed institutional licence (weeks of lead time), so the
default corpus is FineWeb. `corpus=` stays swappable so the pin is a configuration decision
rather than a rewrite.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from pinq.types import EvidenceUnit

DEFAULT_BASE_URL = "https://clueweb22.us"
API_KEY_ENV = "DRGYM_API_KEY"
BASE_URL_ENV = "DRGYM_BASE_URL"

# The one string a user without a key must see. It names the exact env var, the exact
# remedy, and the escape hatch that needs no key at all.
KEY_REMEDY = (
    f"{API_KEY_ENV} is unset or rejected. Both {DEFAULT_BASE_URL}/search and "
    f"{DEFAULT_BASE_URL}/fineweb/search return HTTP 401 without an X-API-Key header "
    "(verified by direct call on 2026-08-24); the public FAQ claiming the FineWeb endpoint "
    "is keyless is stale. Request a free key from deepresearchgym@cmu.edu, then "
    f"export {API_KEY_ENV}=<key>. To run with no key at all, use the offline replay mode: "
    "DrGymSuite(root, offline=True) replays a content-addressed search cache and never "
    "touches the network."
)


@dataclass(frozen=True, slots=True)
class Corpus:
    """One hosted index: its endpoint path and its field spelling."""

    name: str
    path: str
    id_field: str
    url_field: str
    text_field: str
    lang_field: str


CORPORA: Mapping[str, Corpus] = {
    "fineweb": Corpus("fineweb", "/fineweb/search", "id", "url", "text", "language"),
    "clueweb22": Corpus("clueweb22", "/search", "ClueWeb22-ID", "URL", "Clean-Text", "Language"),
}

# The corpus_id stamped into every EvidenceUnit.uid. Distinct per corpus because a uid must
# never mean two different documents.
CORPUS_IDS: Mapping[str, str] = {
    "fineweb": "drgym_fineweb_v1",
    "clueweb22": "drgym_clueweb22_v1",
}


def corpus_of(name: str) -> Corpus:
    try:
        return CORPORA[name]
    except KeyError:
        raise ValueError(f"unknown corpus {name!r}; known: {sorted(CORPORA)}") from None


# --------------------------------------------------------------------------- errors


class DrGymError(RuntimeError):
    """Base for every failure this client reports."""


class DrGymAuthError(DrGymError):
    """HTTP 401/403. Carries KEY_REMEDY so a missing key never reads as a bug."""


class DrGymHTTPError(DrGymError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"DeepResearchGym returned HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


class DrGymEnvelopeError(DrGymError):
    """The response was not `{"results": [base64(json), ...]}`.

    Raised rather than skipped: a silently dropped result is a silently shortened evidence
    set, and every retrieval metric downstream is a function of that set.
    """


# --------------------------------------------------------------------------- the envelope


@dataclass(frozen=True, slots=True)
class SearchDoc:
    """One retrieved document, in the ONE spelling the rest of the system knows."""

    doc_id: str
    url: str
    text: str
    language: str = ""

    def as_unit(self, corpus_id: str) -> EvidenceUnit:
        """span is the whole document, mirroring pi_eval.build.common.unit_uid's convention.

        The URL rides in `title` because it is the only human-meaningful handle a FineWeb
        record has (there is no title field) AND because the citation judge and the Support
        rubric both key off literal URLs appearing in the report. `doc_id` stays the corpus's
        own id, so the uid is (corpus, doc, span) and nothing model-produced.
        """
        return EvidenceUnit.make(
            corpus_id=corpus_id,
            doc_id=self.doc_id,
            span=f"0:{len(self.text)}",
            title=self.url,
            text=self.text,
            # The API returns no relevance score. Synthesising one from the rank would put a
            # number into telemetry that upstream never produced; the ORDER of the returned
            # sequence is the ranking, and Turn.retrieved_uids records it.
            score=0.0,
        )


def decode_envelope(payload: Any) -> tuple[dict[str, Any], ...]:
    """`{"results": [base64(json), ...]}` -> the decoded document dicts, in order."""
    if isinstance(payload, (bytes, bytearray, str)):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DrGymEnvelopeError(f"response body is not JSON: {exc}") from exc
    if not isinstance(payload, dict) or "results" not in payload:
        raise DrGymEnvelopeError(f"expected an object with a 'results' key, got {type(payload)}")
    results = payload["results"]
    if not isinstance(results, list):
        raise DrGymEnvelopeError(f"'results' must be a list, got {type(results)}")

    out: list[dict[str, Any]] = []
    for i, item in enumerate(results):
        if not isinstance(item, (str, bytes)):
            raise DrGymEnvelopeError(f"results[{i}] is {type(item)}, expected a base64 string")
        try:
            raw = base64.b64decode(item, validate=True)
        except Exception as exc:
            raise DrGymEnvelopeError(f"results[{i}] is not valid base64: {exc}") from exc
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DrGymEnvelopeError(f"results[{i}] does not decode to JSON: {exc}") from exc
        if not isinstance(doc, dict):
            raise DrGymEnvelopeError(f"results[{i}] decodes to {type(doc)}, expected an object")
        out.append(doc)
    return tuple(out)


def normalize(doc: Mapping[str, Any], corpus: Corpus) -> SearchDoc:
    """Per-corpus field names -> SearchDoc. A missing id or text is an error, not a default.

    A document with no text cannot support a claim and a document with no id has no uid, so
    either one silently poisons every downstream number if it is allowed through as "".
    """
    missing = [f for f in (corpus.id_field, corpus.text_field) if not doc.get(f)]
    if missing:
        raise DrGymEnvelopeError(
            f"{corpus.name} document is missing {missing}; keys present: {sorted(doc)[:12]}"
        )
    return SearchDoc(
        doc_id=str(doc[corpus.id_field]),
        url=str(doc.get(corpus.url_field, "")),
        text=str(doc[corpus.text_field]),
        language=str(doc.get(corpus.lang_field, "")),
    )


def docs_from_envelope(payload: Any, corpus: Corpus) -> tuple[SearchDoc, ...]:
    return tuple(normalize(d, corpus) for d in decode_envelope(payload))


# --------------------------------------------------------------------------- transport


@dataclass(frozen=True, slots=True)
class Response:
    status: int
    body: bytes


Transport = Callable[[str, Mapping[str, str], float], Response]


def urllib_transport(url: str, headers: Mapping[str, str], timeout: float) -> Response:
    req = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return Response(int(r.status), r.read())
    except urllib.error.HTTPError as exc:  # 4xx/5xx arrive here, body and all
        return Response(int(exc.code), exc.read())


# --------------------------------------------------------------------------- the client


class DrGymClient:
    """GET {base}/{corpus path}?query=...&k=... with an X-API-Key header.

    `transport` is injectable for one reason only: every test in this repo must run with no
    network and no key, and a test that monkeypatches urllib is a test of urllib.
    """

    def __init__(
        self,
        *,
        corpus: str = "fineweb",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_attempts: int = 3,
        backoff: float = 0.5,
        env: Mapping[str, str] | None = None,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        env = env if env is not None else os.environ
        self.corpus = corpus_of(corpus)
        self.corpus_id = CORPUS_IDS[self.corpus.name]
        self.base_url = (base_url or env.get(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")
        self._api_key = api_key if api_key is not None else env.get(API_KEY_ENV, "")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff = backoff
        self._transport = transport or urllib_transport
        self._sleep = sleep

    # ------------------------------------------------------------------ probe

    def available(self) -> tuple[bool, str]:
        """(usable, reason). The reason is the REMEDY, not a diagnosis.

        No network call: the question this answers is "is this process configured to reach
        the API at all", and a machine with no key must learn that in microseconds rather
        than through a 401 traceback.
        """
        if not self._api_key:
            return False, KEY_REMEDY
        return True, f"{API_KEY_ENV} is set; corpus={self.corpus.name} base={self.base_url}"

    # ------------------------------------------------------------------ request

    def url_for(self, query: str, k: int) -> str:
        qs = urllib.parse.urlencode({"query": query, "k": int(k)})
        return f"{self.base_url}{self.corpus.path}?{qs}"

    def fetch(self, query: str, k: int) -> dict[str, Any]:
        """The raw envelope, retried on transport and 5xx/429 only.

        A 4xx other than 429 is a request bug; burning three attempts on it hides the bug
        behind a delay. A 401/403 is neither: it is a configuration state, and it raises
        DrGymAuthError carrying the remedy.
        """
        ok, why = self.available()
        if not ok:
            raise DrGymAuthError(why)

        url = self.url_for(query, k)
        headers = {"X-API-Key": self._api_key, "Accept": "application/json"}
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                resp = self._transport(url, headers, self.timeout)
            except (TimeoutError, ConnectionError, OSError) as exc:
                last = exc
            else:
                if resp.status in (401, 403):
                    raise DrGymAuthError(
                        f"HTTP {resp.status} from {self.corpus.path}. {KEY_REMEDY}"
                    )
                if resp.status == 200:
                    try:
                        return json.loads(resp.body)
                    except json.JSONDecodeError as exc:
                        # A 200 whose body is not JSON is a proxy or an error page wearing a
                        # success code. Retrying it would hide that; a bare JSONDecodeError
                        # three frames up would not name the service.
                        raise DrGymEnvelopeError(
                            f"HTTP 200 from {self.corpus.path} but the body is not JSON "
                            f"({exc}): {resp.body[:200]!r}"
                        ) from exc
                if resp.status not in (429,) and resp.status < 500:
                    raise DrGymHTTPError(resp.status, resp.body.decode("utf-8", "replace"))
                last = DrGymHTTPError(resp.status, resp.body.decode("utf-8", "replace"))
            if attempt + 1 < self.max_attempts:
                self._sleep(self.backoff * (2**attempt))
        raise DrGymError(f"{self.max_attempts} attempts failed for {self.corpus.path}: {last}")

    def search_docs(self, query: str, k: int) -> tuple[SearchDoc, ...]:
        return docs_from_envelope(self.fetch(query, k), self.corpus)

    def search(self, query: str, k: int) -> tuple[EvidenceUnit, ...]:
        return units_of(self.search_docs(query, k), self.corpus_id)


def units_of(docs: Sequence[SearchDoc], corpus_id: str) -> tuple[EvidenceUnit, ...]:
    return tuple(d.as_unit(corpus_id) for d in docs)
