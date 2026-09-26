"""Which proxy route reaches which backend, and does that backend actually serve it?

WHY THIS EXISTS. There are THREE independent layers between a model name and the weights that
answer it, and a failure in any one of them presents identically -- as the model being absent:

    1. the PROXY's target        `api_base` in conf/serving/litellm.yaml  (a local port)
    2. the TUNNEL's destination  which host:port that local port reaches
    3. the BACKEND's adapter set what vLLM over there has actually loaded

No membership test against a model listing can separate these. The listing is not even
conservative: measured 2026-09-19, it OMITTED two ids that answer 200 through a wildcard route
it does not enumerate, and LISTED a checkpoint that answered 404. This script resolves layer 1
from the shipped yaml and asks layer 3 directly, so a dead route can be attributed rather than
guessed at.

WHAT IT CANNOT SEE, and the reason to read `lsof`/`ps` alongside it: layer 2. Two ports whose
tunnels point at the SAME host:port look, from here, like two healthy ports -- which is exactly
what happened on 2026-09-19, when :8000 was re-pointed to the backend that :8302 already
reached and 23 of :8000's 24 routes went dead without anything reporting an error.

SNAPSHOT, NOT A STANDING FACT. Measured 2026-09-19, the reading was:

    :8000   24 routes,  1 live, 23 DEAD   (only qwen3-8b-base)
    :8001    5 routes,  2 live,  3 dead
    :8002    4 routes,  backend UNREACHABLE
    :8011    2 routes,  2 live,  0 dead
    :8301    4 routes,  backend UNREACHABLE
    :8302    2 routes,  2 live,  0 dead

31 dead routes across three ports, none of which the proxy's own listing reported as anything
but present. Re-run it rather than quoting it; the numbers move whenever a serve job ends.

    python scripts/tau2_campaign/route_map.py conf/serving/litellm.yaml
"""

from __future__ import annotations

import collections
import json
import re
import sys
import urllib.request
from pathlib import Path


def routes(yaml_text: str) -> list[tuple[str, int]]:
    """(model_name, local port) for every deployment with a loopback api_base.

    Deliberately not a YAML parser: the file is a flat list of `model_name` / `api_base` pairs,
    the pairing is by adjacency, and a dependency would have to earn its place. A deployment
    whose api_base is not loopback (the gateway passthroughs) is not a tunnelled route and is
    skipped rather than reported as unreachable.
    """
    out: list[tuple[str, int]] = []
    name: str | None = None
    for line in yaml_text.splitlines():
        m = re.match(r"\s*-\s*model_name:\s*[\"']?([^\"'\s]+)", line)
        if m:
            name = m.group(1)
            continue
        b = re.search(r"api_base:\s*http://127\.0\.0\.1:(\d+)", line)
        if b and name:
            out.append((name, int(b.group(1))))
            name = None
    return out


def served(port: int, timeout: int = 20) -> set[str] | None:
    """What the backend behind `port` says it serves, or None if it cannot be reached.

    None is NOT "serves nothing": an unreachable backend and an empty one are different
    failures with different owners, and collapsing them is how a dead tunnel reads as a dead
    model.
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=timeout) as r:
            return {m["id"] for m in json.load(r).get("data", [])}
    except Exception:  # noqa: BLE001 - every failure here means "cannot say", not "absent"
        return None


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    path = Path(args[0]) if args else Path("conf/serving/litellm.yaml")
    by_port: dict[int, list[str]] = collections.defaultdict(list)
    for n, p in routes(path.read_text()):
        by_port[p].append(n)

    dead_total = 0
    for port in sorted(by_port):
        names = sorted(by_port[port])
        have = served(port)
        if have is None:
            print(f"\n:{port}  {len(names)} route(s)  BACKEND UNREACHABLE")
            print(f"   cannot say whether these are served: {names}")
            dead_total += len(names)
            continue
        live = [n for n in names if n in have]
        dead = [n for n in names if n not in have]
        dead_total += len(dead)
        print(f"\n:{port}  {len(names)} route(s); backend serves {sorted(have)}")
        print(f"   LIVE ({len(live)}): {live}")
        print(f"   DEAD ({len(dead)}): {dead}")
    print(f"\ntotal routes that would not answer: {dead_total}")
    print("A DEAD row does not say which layer failed. Check the tunnel's destination")
    print("(`lsof -nP -iTCP:<port> -sTCP:LISTEN`, `ps` for the ssh) before blaming the backend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
