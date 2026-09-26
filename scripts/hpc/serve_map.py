#!/usr/bin/env python3
"""The adapter map for `scripts/hpc/serve.sh`, in a language that can be tested.

    PINQ_LORA="qwen3-8b-sft=rung1-8b/best/by_nll,qwen3-8b-sft-headline=rung1-8b-headline"

WHY THIS IS NOT IN THE SHELL. The map used to be two parallel bash arrays inside serve.sh, so
adding a checkpoint meant editing a committed script -- and on the cluster a committed edit is
an uncommitted tree, which `hpc_banner` refuses to start under. It also meant the map could
only be exercised by submitting a GPU job. Here it is stdlib Python with a unit test
(`tests/test_serve_map.py`), and serve.sh is a wrapper that runs what this prints.

WHAT IT REFUSES, AND WHY THAT IS THE POINT. The old script SKIPPED a directory that held no
adapter and served the base alone; a sweep pinned to the missing name then 404ed mid-run,
after the queue wait was already spent. Every entry here must be a real adapter directory
(`adapter_model.safetensors` + `adapter_config.json`) or nothing starts.

WHAT IT WRITES. `$ART/serve/SERVED.<jobid>.json`: every served name, the directory it was
loaded from, the sha256 of its weights, the base, the host and the port.
`conf/checkpoints.json` records what SHOULD be served; this records what WAS, and it is the
only artifact that can settle after the fact which weights answered a sweep.

STDLIB ONLY: this runs inside the vLLM venv on a compute node and must not depend on the
project's own install.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"

# Every rung-1/2 adapter is trained at LoRA r=32 (`lora.r` in rung1.manifest.json). vLLM
# refuses at load time to apply an adapter whose rank exceeds this, and raising it costs
# memory per slot, so it is pinned rather than guessed per run.
MAX_LORA_RANK = 32

# MEASURED, not transcribed. 8192 came from plan I.14 and had never been run; on the 80 GB
# card the 8B occupies 16.31 GiB of weights and leaves 54.46 GiB of KV cache, which holds a
# 16,384-token context with room for the concurrency a dev sweep asks for.
#
# THAT NUMBER IS THE 8B's AND IT DOES NOT TRANSFER. Qwen3-32B is 61.02 GiB of bf16 weights
# (the 17 safetensors shards of the cluster's own snapshot, summed 2026-09-18) and its KV is
# 256 KiB/token against the 8B's 144 KiB (2 . 64 layers . 8 kv-heads . 128 head-dim . 2 bytes,
# from that snapshot's config.json), so ONE 16,384-token sequence costs 4.00 GiB instead of
# 2.25 GiB out of what a single card has left over -- which is why `tensor_parallel` below
# exists. artifacts/rung32b_serve_20260918/RESULT.md carries the whole arithmetic.
DEFAULT_MAX_MODEL_LEN = 16384


def parse_tensor_parallel(raw: object) -> int | None:
    """`PINQ_TENSOR_PARALLEL` -> a GPU count, or `None` for "this knob was never set".

    UNSET IS NOT A VALUE, AND THAT IS THE WHOLE POINT. `None` emits no flag at all, so every
    serve that does not ask for multiple cards produces the argv it produced before this
    function existed -- the same rule `SHA_OMIT_WHEN_DEFAULT`
    (`src/pinq_train/rung1_sft/train.py`) applies to a config sha, for the same reason: a new
    knob must not move the identity of everything that predates it. An explicit `1` IS a value:
    it says a topology was considered and one card was chosen, which is a different fact from
    never having considered it, and it is recorded as one.

    The empty string is absent, not a parse error: `${PINQ_TENSOR_PARALLEL:-}` under `set -u`
    and a bare `export PINQ_TENSOR_PARALLEL=` both produce `""`, and turning that into
    `--tensor-parallel-size ''` would hand vLLM an empty argument after the queue wait is
    already spent. `0` and `-1`, by contrast, parse as integers and are refused BY NAME: a
    typo there is a job that holds its GPUs for the whole wall limit and serves nothing.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):  # `True` is an `int`; a boolean is not a GPU count
        raise ValueError(f"PINQ_TENSOR_PARALLEL={raw!r} is not a number of GPUs")
    if isinstance(raw, int):
        n = raw
    else:
        text = str(raw).strip()
        if not text:
            return None
        try:
            n = int(text)
        except ValueError as e:
            raise ValueError(
                f"PINQ_TENSOR_PARALLEL={raw!r} is not a whole number of GPUs. It becomes "
                '`--tensor-parallel-size N`, and it must match the `-gpu "num=N"` the job '
                "was submitted with."
            ) from e
    if n < 1:
        raise ValueError(
            f"PINQ_TENSOR_PARALLEL={raw!r} is not a number of GPUs: it must be at least 1. "
            "Leave it UNSET for a single-card serve -- unset emits no flag at all, which is "
            "not the same statement as `--tensor-parallel-size 1`."
        )
    return n


# A served name is an OpenAI model id on the wire, a key in scripts/price_tables/2026-09.json,
# a key in conf/checkpoints.json and a bash word in `--lora-modules name=path`. Anything
# outside this set is one of those four things breaking quietly.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def parse_spec(spec: str) -> list[tuple[str, str]]:
    """`"a=p,b=q"` (or one pair per line, `#` comments allowed) -> `[("a","p"),("b","q")]`.

    Order is preserved: it is the order vLLM is handed the adapters and the order the log
    prints, so a reader can match one to the other.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    text = "\n".join(line.split("#", 1)[0] for line in (spec or "").splitlines())
    for seg in text.replace("\n", ",").split(","):
        seg = seg.strip()
        if not seg:
            continue
        if "=" not in seg:
            raise ValueError(
                f"PINQ_LORA entry {seg!r} has no '=': the map is name=path, and a bare path "
                "would be served under no name at all."
            )
        name, _, path = seg.partition("=")
        name, path = name.strip(), path.strip()
        if not _NAME.match(name):
            raise ValueError(
                f"PINQ_LORA name {name!r} is not a served model id: it must match "
                f"{_NAME.pattern} -- the name is looked up in the price table and the "
                "checkpoint registry by this exact string."
            )
        if not path:
            raise ValueError(f"PINQ_LORA entry {seg!r} has an empty path")
        if name in seen:
            raise ValueError(
                f"PINQ_LORA names {name!r} twice. vLLM would publish one of them and the "
                "other would be unreachable, while the registry and the price table name "
                "only one set of weights."
            )
        seen.add(name)
        out.append((name, path))
    return out


def load_spec(env: Mapping[str, str]) -> str:
    """`PINQ_LORA` if set, else the contents of `PINQ_LORA_FILE`, else empty.

    The file form exists because ten adapters on one `bsub` line cannot be reviewed.
    """
    if env.get("PINQ_LORA"):
        return env["PINQ_LORA"]
    f = env.get("PINQ_LORA_FILE")
    if f:
        return Path(f).read_text()
    return ""


def resolve(
    pairs: Sequence[tuple[str, str]], art: str | os.PathLike[str]
) -> list[tuple[str, Path]]:
    """Relative paths are relative to `$ART` (the artifacts root); absolute ones are kept.

    `$ART` moved from `~/ProactiveInquirer/artifacts` to `/proj/pinq/user/artifacts` when the
    home fileset hit its quota, so a map written against one root must still work under the
    other without an edit.
    """
    root = Path(art)
    # `~` is expanded here because bash does NOT expand it inside a quoted PINQ_LORA value:
    # left alone, `~/pinq-serve/adapters/x` is neither absolute nor a real relative path and
    # would be refused under `$ART/~/pinq-serve/...`, a path nobody typed.
    out: list[tuple[str, Path]] = []
    for n, raw in pairs:
        q = Path(raw).expanduser()
        out.append((n, q if q.is_absolute() else root / q))
    return out


def check(resolved: Sequence[tuple[str, Path]], *, served_base: str) -> None:
    """Raise unless every entry is a loadable adapter under a name that is free."""
    for name, path in resolved:
        if name == served_base:
            raise ValueError(
                f"adapter {name!r} would shadow the served base {served_base!r}: both would be "
                "published under one model id and the untrained base -- the arm every trained "
                "arm is compared against -- would become unreachable."
            )
        for required in (WEIGHTS, CONFIG):
            if not (path / required).is_file():
                raise ValueError(
                    f"{name}: {path / required} is missing, so {name!r} cannot be served. "
                    "This is a refusal and not a skip: a skipped adapter 404s mid-sweep, "
                    "after the queue wait is already spent."
                )


def vllm_args(
    *,
    base_model: str,
    served_base: str,
    resolved: Sequence[tuple[str, Path]],
    port: int,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    tensor_parallel: int | None = None,
) -> list[str]:
    """Everything after `vllm serve`.

    `--max-loras` is the size of the map: vLLM holds at most that many adapters at once, so a
    fixed 4 made the fifth checkpoint an eviction nobody was told about. `--enable-lora` stays
    on with an empty map so an adapter can be added later without a restart.

    `tensor_parallel=None` APPENDS NOTHING, so the 8B's argv is byte-identical to the one
    every `ARGS.<job>.txt` on disk already records (`tests/test_serve_tensor_parallel.py`
    compares against a copy captured before this parameter existed).

    WHEN IT IS SET IT GOES BEFORE `--lora-modules`, AND THAT POSITION IS NOT COSMETIC.
    `--lora-modules` is variadic, and `serve.sh` names the adapters for `$ART/serve/HOST` by
    treating every argument after it as a `name=path` pair -- a flag emitted after it would be
    announced to the tunnel as a served model called `--tensor-parallel-size`.
    """
    argv = [
        base_model,
        "--served-model-name",
        served_base,
        "--enable-lora",
        "--max-lora-rank",
        str(MAX_LORA_RANK),
        "--max-loras",
        str(max(1, len(resolved))),
        "--max-model-len",
        str(max_model_len),
        "--port",
        str(port),
    ]
    if tensor_parallel is not None:
        argv += ["--tensor-parallel-size", str(parse_tensor_parallel(tensor_parallel))]
    if resolved:
        argv.append("--lora-modules")
        argv += [f"{n}={p}" for n, p in resolved]
    return argv


def sha256_file(path: str | os.PathLike[str]) -> str:
    """The same bytes `sha256sum` reads, so a remote `sha256sum` and this agree."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def served_record(
    *,
    resolved: Sequence[tuple[str, Path]],
    base_model: str,
    served_base: str,
    host: str,
    port: int,
    job: str,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    tensor_parallel: int | None = None,
    git_head: str = "",
) -> dict:
    """What WAS served, with the sha of every weights file that was loaded."""
    adapters: dict[str, dict] = {}
    for name, path in resolved:
        w = path / WEIGHTS
        adapters[name] = {
            "path": str(path),
            "sha256": sha256_file(w),
            "bytes": w.stat().st_size,
            "config_sha256": sha256_file(path / CONFIG),
        }
    return {
        "schema": "pinq.serve.v1",
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "job": job,
        "host": host,
        "port": port,
        "base_model": base_model,
        "served_base": served_base,
        "served": [served_base, *[n for n, _ in resolved]],
        "max_model_len": max_model_len,
        "max_lora_rank": MAX_LORA_RANK,
        "max_loras": max(1, len(resolved)),
        # `null` when the knob was never set, NOT `1`. A 32B answered by two cards and a 32B
        # answered by one are different machines with different KV budgets, and "one card was
        # chosen" is a different fact from "the question never arose" -- the record is the only
        # artifact that can settle either after the job is gone.
        "tensor_parallel_size": parse_tensor_parallel(tensor_parallel),
        "git_head": git_head,
        "adapters": adapters,
    }


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--spec", default=None, help="name=path[,...]; default from PINQ_LORA")
    ap.add_argument("--art", default=os.environ.get("ART", "artifacts"))
    ap.add_argument("--base-model", default=os.environ.get("PINQ_BASE_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--served-base", default=os.environ.get("PINQ_SERVED_BASE", "qwen3-8b-base"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PI_VLLM_PORT", "8000")))
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("PINQ_MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN)),
    )
    # NOT `type=int`, and NOT a default of 1. The value is carried as the raw string (or None)
    # and normalised inside main()'s try/except, so a bad `PINQ_TENSOR_PARALLEL` exits 2 with
    # `serve_map: ...` on stderr like every other refusal here, rather than an argparse
    # traceback raised while the parser is still being built.
    ap.add_argument(
        "--tensor-parallel",
        default=os.environ.get("PINQ_TENSOR_PARALLEL"),
        help="GPUs to shard the base across; UNSET emits no flag at all",
    )


def _resolved(a: argparse.Namespace) -> list[tuple[str, Path]]:
    spec = a.spec if a.spec is not None else load_spec(os.environ)
    resolved = resolve(parse_spec(spec), a.art)
    check(resolved, served_base=a.served_base)
    return resolved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_args = sub.add_parser("args", help="print the vllm serve argv, one argument per line")
    _common(p_args)
    p_rec = sub.add_parser("record", help="write $ART/serve/SERVED.<jobid>.json")
    _common(p_rec)
    p_rec.add_argument("--host", required=True)
    p_rec.add_argument("--job", required=True)
    p_rec.add_argument("--out", required=True, type=Path)
    p_rec.add_argument("--git-head", default="")
    a = ap.parse_args(argv)

    try:
        resolved = _resolved(a)
        tensor_parallel = parse_tensor_parallel(a.tensor_parallel)
    except (ValueError, OSError) as e:
        print(f"serve_map: {e}", file=sys.stderr)
        return 2

    if a.cmd == "args":
        for tok in vllm_args(
            base_model=a.base_model,
            served_base=a.served_base,
            resolved=resolved,
            port=a.port,
            max_model_len=a.max_model_len,
            tensor_parallel=tensor_parallel,
        ):
            print(tok)
        return 0

    rec = served_record(
        resolved=resolved,
        base_model=a.base_model,
        served_base=a.served_base,
        host=a.host,
        port=a.port,
        job=a.job,
        max_model_len=a.max_model_len,
        tensor_parallel=tensor_parallel,
        git_head=a.git_head,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2) + "\n")
    print(str(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
